"""UNet model implementation."""

import torch
from torch import nn

from chuchichaestli.models.unet.unet_1d_blocks import BLOCK_MAP_1D
from chuchichaestli.models.unet.unet_2d_blocks import BLOCK_MAP_2D
from chuchichaestli.models.unet.unet_3d_blocks import BLOCK_MAP_3D
from chuchichaestli.models.unet.time_embeddings import (
    GaussianFourierProjection,
    SinusoidalTimeEmbedding,
    TimestepEmbedding,
)

DIM_TO_BLOCK_MAP = {
    1: BLOCK_MAP_1D,
    2: BLOCK_MAP_2D,
    3: BLOCK_MAP_3D,
}

DIM_TO_CONV_MAP = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}


class UNet(nn.Module):
    """UNet model implementation."""

    def __init__(
        self,
        # General parameters
        dimensions: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        # Block-related parameters
        down_block_types: tuple[str, ...] = (
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        mid_block_type: str = "MidBlock",
        up_block_types: tuple[str, ...] = (
            "UpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
        ),
        block_out_channels: tuple[int, ...] = (224, 448, 672, 896),
        num_layers_per_block: int = 1,
        dropout: float = 0.0,
        mid_block_scale_factor: float = 1.0,
        # Attention-related parameters
        add_attention: bool = False,
        attention_head_dim: int = None,
        attn_norm_num_groups: int = 16,
        # Time-related parameters
        add_time_embedding: bool = True,
        time_embedding_type: str = "positional",
        flip_sin_to_cos: bool = False,
        freq_shift: float = 0.0,
        class_embed_type: str = "timestep",
        num_class_embeds: int = None,
        # ResNet-related parameters
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        # Normalization-related parameters
        norm_num_groups: int = None,
        norm_eps: float = 1e-6,
    ):
        """UNet model implementation."""
        super(__class__, self).__init__()

        if dimensions not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, and 3D UNets are supported.")

        if len(down_block_types) != len(up_block_types):
            raise ValueError("Down and up block types must have the same length.")

        if len(down_block_types) != len(block_out_channels):
            raise ValueError(
                "Down block types and out channels must have the same length."
            )

        # input layer
        self.conv_in = DIM_TO_CONV_MAP[dimensions](
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # Time embedding
        # Removed option to learn time embeddings; c.f. HF repo.
        self.time_proj = None
        self.time_embedding = None
        time_embed_dim = None
        if add_time_embedding:
            if time_embedding_type == "fourier":
                self.time_proj = GaussianFourierProjection(
                    embedding_size=block_out_channels[0], scale=16
                )
                timestep_input_dim = 2 * block_out_channels[0]
            elif time_embedding_type == "positional":
                self.time_proj = SinusoidalTimeEmbedding(
                    block_out_channels[0], flip_sin_to_cos, freq_shift
                )
                timestep_input_dim = block_out_channels[0]

            # HF refuses to explain this
            time_embed_dim = block_out_channels[0] * 4

            if self.time_proj is not None:
                self.time_embedding = TimestepEmbedding(
                    timestep_input_dim, time_embed_dim
                )

        # Removed class embedding from here; c.f. HF repo.

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # Down blocks
        out_channel = block_out_channels[0]
        for i, block_type in enumerate(down_block_types):
            in_channel = out_channel
            out_channel = block_out_channels[i]
            is_final_block = i == len(down_block_types) - 1

            block = DIM_TO_BLOCK_MAP[dimensions][block_type](
                num_layers=num_layers_per_block,
                in_channels=in_channel,
                out_channels=out_channel,
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=resnet_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(block)

        # Mid block
        self.mid_block = DIM_TO_BLOCK_MAP[dimensions][mid_block_type](
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=mid_block_scale_factor,
            attention_head_dim=(
                attention_head_dim
                if attention_head_dim is not None
                else block_out_channels[-1]
            ),
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # Up blocks
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            block = DIM_TO_BLOCK_MAP[dimensions][up_block_type](
                num_layers=num_layers_per_block + 1,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=resnet_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(block)

        # output layer
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = DIM_TO_CONV_MAP[dimensions](
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
    ) -> torch.FloatTensor:
        """Forward pass of the UNet.

        Args:
            sample (torch.FloatTensor): The noisy input tensor. Shape depends on the dimensions.
            timestep (torch.Tensor | float | int): The timestep.
        """
        # TODO: Think about whether we want to center the input here.

        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        if self.time_proj is not None:
            t_emb = self.time_proj(timesteps)

        if self.time_embedding is not None:
            t_emb = self.time_embedding(t_emb)

        # 2. Down
        # TODO: Skip connections
        # This is for skip connections; but we currently don't use them
        # skip = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, t_emb)
            down_block_res_samples += res_samples

        # 3. Mid
        sample = self.mid_block(sample, t_emb)

        # 4. Up
        # TODO: Skip connections
        # skip = sample
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
            sample = up_block(sample, res_samples, t_emb)

        # 5. Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # TODO: Skip connections
        # if skip is not None:
        #    sample += skip

        if isinstance(self.time_proj, GaussianFourierProjection):
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        return sample
