"""Resnet modules taken form diffusers and slightly adapted."""

import torch
from torch import nn

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.downsampling import Downsample2D
from chuchichaestli.models.upsampling import Upsample2D


class ResnetBlock2D(nn.Module):
    r"""A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift"
            for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: int = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        output_scale_factor: float = 1.0,
        use_in_shortcut: bool = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: int = None,
    ):
        """Initialize the ResnetBlock2D."""
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear
        conv_cls = nn.Conv2d

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.conv1 = conv_cls(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = conv_cls(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1
        )

        self.nonlinearity = ACTIVATION_FUNCTIONS.get(non_linearity)

        # Simplified the code below by removing the other options for
        # up- and down-sampling.
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample2D(
                in_channels, use_conv=False, padding=1, name="op"
            )

        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Forward pass."""
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
