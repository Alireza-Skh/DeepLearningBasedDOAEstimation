import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        num_embeddings = 2 * max_len + 1
        self.relative_embeddings = nn.Embedding(num_embeddings, d_model)

    def _get_relative_position_indices(self, seq_len: int) -> torch.Tensor:
        indices = torch.arange(seq_len, dtype=torch.long)
        relative_distances = indices.unsqueeze(0) - indices.unsqueeze(1)
        relative_distances = torch.clamp(
            relative_distances, -self.max_len, self.max_len)
        relative_position_indices = relative_distances + self.max_len
        return relative_position_indices.to(self.relative_embeddings.weight.device)

    def forward(self, seq_len: int) -> torch.Tensor:
        relative_position_indices = self._get_relative_position_indices(
            seq_len)

        r_embeddings = self.relative_embeddings(relative_position_indices)
        return r_embeddings

# class RelativePositionalEncoding(nn.Module):
#     """
#     Relative positional embeddings for multi-head self-attention.

#     Parameters
#     ----------
#     max_len : int
#         Maximum sequence length you expect to see.
#     d_head : int
#         Dimensionality of each head's embeddings (usually d_model // num_heads).
#     num_heads : int
#         Number of attention heads.
#     shared : bool, default=True
#         If True, all heads share the same embedding table.
#         If False, each head gets its own table (parameter count increases).
#     """
#     def __init__(self, max_len: int, d_head: int, num_heads: int = 1, shared: bool = True):
#         super().__init__()
#         self.max_len = max_len
#         self.d_head = d_head
#         self.num_heads = num_heads
#         self.shared = shared

#         # Number of distinct relative positions: -(L-1) … +(L-1)
#         num_rel_positions = 2 * max_len - 1

#         if shared:
#             self.rel_emb = nn.Embedding(num_rel_positions, d_head)
#         else:
#             self.rel_emb = nn.Embedding(num_rel_positions * num_heads, d_head)

#     def forward(self, seq_len: int, device=None):
#         """
#         Return the relative-position embeddings for a batch of size *any*.

#         Return Shape
#         -------
#             (num_heads, seq_len, seq_len, d_head) if shared True
#             (seq_len, seq_len, d_head) if shared False
#         """
#         device = device or self.rel_emb.weight.device
#         seq_len = min(seq_len, self.max_len)

#         indices = torch.arange(seq_len, dtype=torch.long, device=device)
#         offset_matrix = indices.unsqueeze(1) - indices.unsqueeze(0)
#         offset_matrix = offset_matrix.clamp(-self.max_len + 1, self.max_len - 1)
#         idx = offset_matrix + (self.max_len - 1) # in [0, 2*max_len-2]

#         if self.shared:
#             rel_emb = self.rel_emb(idx)
#         else:
#             idx = idx.unsqueeze(0).expand(self.num_heads, -1, -1)
#             idx_flat = idx + torch.arange(self.num_heads, device=device) * (2 * self.max_len - 1)
#             rel_emb = self.rel_emb(idx_flat.view(-1))
#             rel_emb = rel_emb.view(self.num_heads, seq_len, seq_len, self.d_head)
#         return rel_emb


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, pe=None):
        attn = torch.bmm(q, k.transpose(1, 2))  # (BH), Q, K

        if pe is not None:
            relative_logits = torch.einsum("bid,ijd->bij", q, pe)
            attn += relative_logits

        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class ComplexScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_real, q_imag, k_real, k_imag, v_real, v_imag, mask=None, pe=None):
        attn_r = torch.bmm(q_real, k_real.transpose(1, 2)) + torch.bmm(q_imag, k_imag.transpose(1, 2))
        attn_i = torch.bmm(q_imag, k_real.transpose(1, 2)) - torch.bmm(q_real, k_imag.transpose(1, 2))

        if pe is not None:
            relative_logits_r = torch.einsum("bid,ijd->bij", q_real, pe)
            relative_logits_i = torch.einsum("bid,ijd->bij", q_imag, pe)
            attn_r += relative_logits_r
            attn_i += relative_logits_i

        attn_r /= self.temperature
        attn_i /= self.temperature

        if mask is not None:
            attn_r = attn_r.masked_fill(mask == 0, -1e9)
            attn_i = attn_i.masked_fill(mask == 0, -1e9)

        attn = F.softmax(torch.sqrt(attn_r**2 + attn_i**2), dim=-1)
        attn = self.dropout(attn)
        output_r = torch.bmm(attn, v_real)
        output_i = torch.bmm(attn, v_imag)
        return (output_r, output_i), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, use_rpe=True, perform_on_features=False, dropout=0.1):
        super().__init__()
        if n_head * d_k != d_model or n_head * d_v != d_model:
            raise ValueError("n_head * d_k and n_head * d_v must equal d_model")
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.on_f = perform_on_features
        self.use_rpe = use_rpe

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        if self.use_rpe:
            # self.rpe = RelativePositionalEncoding(d_k, d_k, n_head, True)
            self.rpe = RelativePositionalEncoding(d_k)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v, mask=mask, pe=self.rpe(len_q) if self.use_rpe else None)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))

        return output, attn


class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, use_rpe=True, perform_on_features=False, dropout=0.1):
        super().__init__()
        if n_head * d_k != d_model or n_head * d_v != d_model:
            raise ValueError(
                "n_head * d_k and n_head * d_v must equal d_model")
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.use_rpe = use_rpe
        self.on_f = perform_on_features

        self.w_qs_r = nn.Linear(d_model, d_model)
        self.w_qs_i = nn.Linear(d_model, d_model)
        self.w_ks_r = nn.Linear(d_model, d_model)
        self.w_ks_i = nn.Linear(d_model, d_model)
        self.w_vs_r = nn.Linear(d_model, d_model)
        self.w_vs_i = nn.Linear(d_model, d_model)

        self.attention = ComplexScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc_r = nn.Linear(d_model, d_model)
        self.fc_i = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        if self.use_rpe:
            self.rpe = RelativePositionalEncoding(d_k, d_k, n_head, True)

    def _split_complex(self, x, n_dim):
        if self.on_f:
            real_part = x[:, :n_dim, :]
            imag_part = x[:, n_dim:, :]
        else:
            real_part = x[:, :, :n_dim]
            imag_part = x[:, :, n_dim:]
        return real_part, imag_part

    def _wieghts(self, x_real, x_imag, w_real, w_imag, len_x):
        x_r = w_real(x_real).view(self.bs, len_x, self.n_head, self.d_k)
        x_i = w_imag(x_imag).view(self.bs, len_x, self.n_head, self.d_k)

        x_r = x_r.permute(2, 0, 1, 3).contiguous().view(-1, len_x, self.d_k)
        x_i = x_i.permute(2, 0, 1, 3).contiguous().view(-1, len_x, self.d_k)

        return x_r, x_i

    def forward(self, q, k, v, mask=None):
        self.bs = q.shape[0]

        if self.on_f:
            n_dim = q.shape[-2] // 2
        else:
            n_dim = q.shape[-1] // 2 

        q_r, q_i = self._split_complex(q, n_dim)
        k_r, k_i = self._split_complex(k, n_dim)
        v_r, v_i = self._split_complex(v, n_dim)

        len_q, len_k, len_v = q_r.shape[1], k_r.shape[1], v_r.shape[1]

        q_r, q_i = self._wieghts(q_r, q_i, self.w_qs_r, self.w_qs_i, len_q)
        k_r, k_i = self._wieghts(k_r, k_i, self.w_ks_r, self.w_ks_i, len_k)
        v_r, v_i = self._wieghts(v_r, v_i, self.w_vs_r, self.w_vs_i, len_v)

        (output_r, output_i), attn = self.attention(
            q_r, q_i, k_r, k_i, v_r, v_i, mask=mask, pe=self.rpe(
                len_q) if self.use_rpe else None
        )

        output_r = output_r.view(self.n_head, self.bs, len_q, self.d_v)
        output_i = output_i.view(self.n_head, self.bs, len_q, self.d_v)
        output_r = output_r.permute( 1, 2, 0, 3).contiguous().view(self.bs, len_q, -1)
        output_i = output_i.permute( 1, 2, 0, 3).contiguous().view(self.bs, len_q, -1)

        output_r = self.dropout(self.fc_r(output_r))
        output_i = self.dropout(self.fc_i(output_i))

        if self.on_f:
            output = torch.cat([output_r, output_i], dim=-2)
        else:
            output = torch.cat([output_r, output_i], dim=-1)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, activation_fn="gelu", dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ac_f = nn.GELU() if activation_fn == "gelu" else nn.ReLU()
        
    def forward(self, x):
        residual = x
        output = self.ac_f(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return output + residual


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_inner, n_head, d_k, d_v, use_rpe, perform_on_features, dropout=0.1):
        super().__init__()
        self.slf_attn = attention_layer(
            n_head, d_model, d_k, d_v, use_rpe, perform_on_features, dropout=dropout
        )
        self.on_f = perform_on_features
        if ComplexMultiHeadAttention == attention_layer and self.on_f == False:
            d_model = 2*d_model
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):
        if self.on_f:
            enc_input = enc_input.permute(0, 2, 1)
            residual = enc_input
            attn_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input
            )
            enc_output = self.norm1(residual + self.dropout(attn_output))
            ff_output = self.pos_ffn(enc_output)
            enc_output = self.norm2(enc_output + self.dropout(ff_output))
            enc_output = enc_output.permute(0, 2, 1)
        else:
            residual = enc_input
            attn_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input
            )
            enc_output = self.norm1(residual + self.dropout(attn_output))
            ff_output = self.pos_ffn(enc_output)
            enc_output = self.norm2(enc_output + self.dropout(ff_output))
        
        return enc_output, enc_slf_attn
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, activation_func="relu"):
        super(DenseLayer, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
        )

        if activation_func == "relu":
            self.ac_fn = nn.ReLU(inplace=True)
        elif activation_func == "gelu":
            self.ac_fn = nn.GELU()

    def forward(self, x):
        out = self.conv(self.ac_fn(self.norm(x)))
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, kernel_size, activation_func):
        super(DenseBlock, self).__init__()
        layers = []
        channels = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(channels, growth_rate,
                          kernel_size, activation_func))
            channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        if activation_func == "relu":
            self.ac_fn = nn.ReLU(inplace=True)
        elif activation_func == "gelu":
            self.ac_fn = nn.GELU()

    def forward(self, x):
        x = self.conv(self.ac_fn(self.norm(x)))
        x = self.pool(x)
        return x


class ComputeCov(nn.Module):
    """
    Note
        Input data shape as (B, F), `in_features` should be `F`,
        `out_features` is number of antennas.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.M = out_features
        self.linear = nn.Linear(in_features, 2 * self.M * self.M, bias=False)

    def forward(self, x, num_targets):
        # x shape: (B, T, F)
        cov = self.linear(x).view(x.shape[0], self.M, self.M, 2)
        cov_complex = torch.view_as_complex(cov)

        cov_hermitian = (cov_complex + cov_complex.mH) / 2
        _, evc = torch.linalg.eigh(cov_hermitian)

        noise_evc = evc[:, :, :-num_targets]
        noise_projector = noise_evc @ noise_evc.mH
        return noise_projector


# class MusicSpectrum(nn.Module):
#     def __init__(self, array, angles_list, phase_calibration=None):
#         super().__init__()
#         self.array = array
#         self.angles_list = angles_list
#         self.phase_calibration = phase_calibration
#         self.norm = nn.LayerNorm(len(angles_list))

#     def calibrated_steering_vector(self, angle):
#         base_stv = self.array.receive_steering_vector_cuda(angle)
#         if self.phase_calibration:
#             correction = torch.exp(-1j *
#                                    self.phase_calibration).to(angle.device)
#             return base_stv * correction
#         return base_stv

#     def manifold_vectors(self):
#         steering_vectors = [
#             self.calibrated_steering_vector(target) for target in self.angles_list
#         ]

#         A = torch.stack(steering_vectors, dim=0).to(
#             torch.complex64)  # Shape: (num_angles, M)

#         self.register_buffer("A", A)

#     def forward(self, x):
#         # x shape: (B, M, M)
#         spec = torch.einsum("im,bmn,in->bi", self.A, x, self.A.conj())
#         music_sp = 1.0 / (torch.abs(spec) + 1e-6)
#         music_sp = self.norm(music_sp)
#         return music_sp


class MusicSpectrum(nn.Module):
    def __init__(self, array, angles_list):
        super().__init__()
        self.array = array
        self.angles_list = angles_list.to(torch.device("cuda"))
        self.norm = nn.LayerNorm(len(angles_list))

        self.register_buffer('base_steering_vectors',
                             self._compute_base_steering_vectors(),
                             persistent=False)

    def _compute_base_steering_vectors(self):
        steering_vectors = [
            self.array.receive_steering_vector_cuda(target) for target in self.angles_list
        ]

        return torch.stack(steering_vectors, dim=0).to(torch.complex64)

    def compute_manifold(self, phase_calibration=None):
        if phase_calibration is None:
            return self.base_steering_vectors

        correction = torch.exp(-1j * phase_calibration).unsqueeze(0)  # (1, M)
        return self.base_steering_vectors * correction  # (num_angles, M)

    def forward(self, x, phase_calibration=None):
        # x shape: (B, M, M)
        A = self.compute_manifold(phase_calibration)
        spec = torch.einsum("im,bmn,in->bi", A, x, A.conj())
        music_sp = 1.0 / (torch.abs(spec) + 1e-6)
        music_sp = self.norm(music_sp)
        return music_sp


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x
    
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=12,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=3,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            bias=True
        )
        
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, num_antennas: int = 16, embed_dim: int = 128):
        super().__init__()
        self.num_antennas = num_antennas
        self.embed_dim = embed_dim
        self.num_elements = num_antennas * (num_antennas - 1) // 2
        
        self.embed_conv = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(1, 6),
            stride=(1, 6),
            padding=(0, 0),
            bias=True
        )
        
        self.num_patches = (self.num_elements + 2 * 0 - 6) // 6 + 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_conv(x)
        x = x.squeeze(2).transpose(1, 2)
        return x
    
    