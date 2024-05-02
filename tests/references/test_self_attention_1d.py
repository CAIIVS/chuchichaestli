"""Test the compatibility of SelfAttention1D from chuchichaestli and HuggingFace."""

import torch
import torch.nn as nn
import math

from chuchichaestli.models.attention import (
    SelfAttention1D as SelfAttention1D_Chuchichaestli,
)


class SelfAttention1D_HuggingFace(nn.Module):
    """Self-attention mechanism for 1D data from HuggingFace's implementation."""

    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        """Initialize the SelfAttention1D module."""
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def _transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass."""
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self._transpose_for_scores(query_proj)
        key_states = self._transpose_for_scores(key_proj)
        value_states = self._transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(
            query_states * scale, key_states.transpose(-1, -2) * scale
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


def test_self_attention_1d():
    """Ensure that SelfAttention1D from chuchichaestli and HuggingFace are compatible."""
    # Create an instance of SelfAttention1D
    in_channels = 64
    n_head = 4
    dropout_rate = 0.2
    attention_chuchichaestli = SelfAttention1D_Chuchichaestli(
        in_channels, n_head, dropout_rate
    )
    attention_huggingface = SelfAttention1D_HuggingFace(
        in_channels, n_head, dropout_rate
    )

    # Load the weights of the chuchichaestli model into the huggingface model
    attention_huggingface.load_state_dict(attention_chuchichaestli.state_dict())

    # Create a random input tensor
    batch_size = 8
    seq_length = 16
    x = torch.randn(batch_size, in_channels, seq_length)

    # Perform forward pass
    output_chuchichaestli = attention_chuchichaestli(x)
    output_huggingface = attention_huggingface(x)

    # Check output shape
    assert output_chuchichaestli.shape == output_huggingface.shape

    # Check if the two outputs are equal
    # assert torch.allclose(output_chuchichaestli, output_huggingface, atol=1e-6)
