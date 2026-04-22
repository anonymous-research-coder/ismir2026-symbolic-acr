import math
import torch
import torch.nn as nn
from copy import deepcopy


def sinusoidal_positional_encoding(seq_len, d_model, device):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.3,
        activation="gelu",
        batch_first=True,
        device="cpu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)

        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff_in = nn.Dropout(dropout)
        self.dropout_ff_out = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.last_attn_weights = None

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
    ):
        x = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        if not self.training:
            self.last_attn_weights = (
                attn_weights.detach() if isinstance(attn_weights, torch.Tensor) else None
            )

        x = x + self.dropout1(attn_out)

        y = self.norm2(x)
        ff = self.linear2(self.dropout_ff_in(self.activation(self.linear1(y))))
        x = x + self.dropout_ff_out(ff)

        return x


class SEModular(nn.Module):
    def __init__(
        self,
        chord_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        pianoroll_dim=13,
        grid_length=80,
        condition_dim=None,
        unmasking_stages=None,
        trainable_pos_emb=False,
        dropout=0.3,
        device="cpu",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.d_model = d_model
        self.grid_length = grid_length
        self.condition_dim = condition_dim
        self.unmasking_stages = unmasking_stages
        self.trainable_pos_emb = trainable_pos_emb

        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        self.harmony_embedding = nn.Embedding(
            chord_vocab_size,
            d_model,
            device=self.device,
        )

        if self.condition_dim is not None:
            self.condition_proj = nn.Linear(condition_dim, d_model, device=self.device)
            self.seq_len = 1 + grid_length + grid_length
        else:
            self.seq_len = grid_length + grid_length

        if self.trainable_pos_emb:
            self.full_pos = nn.Parameter(
                torch.zeros(1, self.seq_len, d_model, device=self.device)
            )
            nn.init.trunc_normal_(self.full_pos, std=0.02)
        else:
            base_len = grid_length + (1 if self.condition_dim is not None else 0)
            shared_pos = sinusoidal_positional_encoding(base_len, d_model, self.device)
            self.register_buffer(
                "full_pos",
                torch.cat(
                    [
                        shared_pos[:, :base_len, :],
                        shared_pos[:, :grid_length, :],
                    ],
                    dim=1,
                ),
                persistent=False,
            )

        if self.unmasking_stages is not None:
            if not isinstance(self.unmasking_stages, int) or self.unmasking_stages <= 0:
                raise ValueError("unmasking_stages must be a positive integer")
            self.stage_embedding_dim = 64
            self.stage_embedding = nn.Embedding(
                self.unmasking_stages,
                self.stage_embedding_dim,
                device=self.device,
            )
            self.stage_proj = nn.Linear(
                self.d_model + self.stage_embedding_dim,
                self.d_model,
                device=self.device,
            )

        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            device=self.device,
        )
        self.encoder = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])

        self.output_head = nn.Linear(
            d_model,
            chord_vocab_size,
            device=self.device,
        )

        self.input_norm = nn.LayerNorm(d_model, device=self.device)
        self.output_norm = nn.LayerNorm(d_model, device=self.device)

        self.to(self.device)

    def forward(
        self,
        melody_grid,
        harmony_tokens=None,
        conditioning_vec=None,
        stage_indices=None,
    ):
        melody_grid = melody_grid.to(self.device)
        batch_size = melody_grid.size(0)

        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_tokens = harmony_tokens.to(self.device)
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(
                batch_size,
                self.grid_length,
                self.d_model,
                device=self.device,
            )

        full_seq = torch.cat([melody_emb, harmony_emb], dim=1)

        if conditioning_vec is not None and self.condition_dim is not None:
            conditioning_vec = conditioning_vec.to(self.device)
            cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)
            full_seq = torch.cat([cond_emb, full_seq], dim=1)

        full_seq = full_seq + self.full_pos[:, : full_seq.size(1), :]

        if self.unmasking_stages is not None:
            if stage_indices is None:
                raise ValueError("stage_indices must be provided when unmasking_stages is enabled")
            stage_indices = stage_indices.to(self.device).long()
            stage_emb = self.stage_embedding(stage_indices)
            stage_emb = stage_emb.unsqueeze(1).expand(-1, full_seq.size(1), -1)
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)
            full_seq = self.stage_proj(full_seq)

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        encoded = full_seq
        for layer in self.encoder:
            encoded = layer(encoded)

        encoded = self.output_norm(encoded)
        harmony_output = self.output_head(encoded[:, -self.grid_length :, :])

        return harmony_output

    def get_attention_maps(self):
        return [layer.last_attn_weights for layer in self.encoder]