import torch.nn.functional as F
from torch import nn
import torch
from deep.utils import *


###########################################
# FeatureAttention
###########################################
class FeatureAttention(nn.Module):
    def __init__(self, d_model, seq_len, n_targets, n_head_t, n_head_f, n_layers_t, n_layers_f, d_inner_t, d_inner_f, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(seq_len)
        d_k_t = d_model // n_head_t
        d_v_t = d_model // n_head_t
        self.time_mha = nn.ModuleList([
            EncoderLayer(MultiHeadAttention, d_model, d_inner_t, n_head_t, d_k_t, d_v_t, False, False, dropout=dropout)
            for _ in range(n_layers_t)])

        # self.ape = SinusoidalPE(d_model)
        self.ape = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.mha_norm = nn.LayerNorm(d_model)

        d_k_f = seq_len // n_head_f
        d_v_f = seq_len // n_head_f
        self.feature_mha = nn.ModuleList([
            EncoderLayer(MultiHeadAttention, seq_len, d_inner_f, n_head_f, d_k_f, d_v_f, True, True, dropout=dropout)
            for _ in range(n_layers_f)])

        self.ffn = nn.Sequential(
            nn.Linear(seq_len, n_targets*2),
            nn.LayerNorm(n_targets*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_targets*2, n_targets),
            nn.LayerNorm(n_targets),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, return_attns=False) -> torch.Tensor:
        x = self.norm(x)
        # x_time = self.ape(x)
        x_time = x + self.ape
        for enc_layer in self.time_mha:
            x_time, _ = enc_layer(x_time)

        enc_slf_attn_list = []

        x_feat = self.mha_norm(x_time)
        for enc_layer in self.feature_mha:
            x_feat, enc_slf_attn = enc_layer(x_feat)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return x_feat, enc_slf_attn_list

        x_feat = x_feat.mean(-1)
        enc_output = self.ffn(x_feat)
        return enc_output,


###########################################
# ComplexFeatureAttention
###########################################
class ComplexFeatureAttention(nn.Module):
    def __init__(self, d_model, seq_len, n_targets, n_head_t, n_head_f, n_layers_t, n_layers_f, d_inner_t, d_inner_f, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(seq_len)
        d_k_t = d_model // (2*n_head_t)
        d_v_t = d_model // (2*n_head_t)
        self.time_mha = nn.ModuleList([
            EncoderLayer(ComplexMultiHeadAttention, d_model//2, d_inner_t, n_head_t, d_k_t, d_v_t, False, False, dropout=dropout)
            for _ in range(n_layers_t)])

        # self.ape = SinusoidalPE(d_model)
        self.ape = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.mha_norm = nn.LayerNorm(d_model)

        d_k_f = seq_len // n_head_f
        d_v_f = seq_len // n_head_f
        self.feature_mha = nn.ModuleList([
            EncoderLayer(ComplexMultiHeadAttention, seq_len, d_inner_f, n_head_f, d_k_f, d_v_f, True, True, dropout=dropout)
            for _ in range(n_layers_f)])

        self.ffn = nn.Sequential(
            nn.Linear(seq_len, n_targets*2),
            nn.LayerNorm(n_targets*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_targets*2, n_targets),
            nn.LayerNorm(n_targets),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_targets, n_targets),
            nn.Sigmoid())

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, return_attns=False) -> torch.Tensor:
        x = self.norm(x)
        # x_time = self.ape(x)
        x_time = x + self.ape
        for enc_layer in self.time_mha:
            x_time, _ = enc_layer(x_time)

        enc_slf_attn_list = []

        x_feat = self.mha_norm(x_time)
        for enc_layer in self.feature_mha:
            x_feat, enc_slf_attn = enc_layer(x_feat)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return x_feat, enc_slf_attn_list

        x_feat = x_feat.mean(-1)
        enc_output = self.ffn(x_feat)
        return enc_output,


###########################################
# ComplexFeatureAttention (attention map)
###########################################
# class ComplexFeatureAttention2(nn.Module):
#     def __init__(self, array, angles_list, d_model, seq_len, n_head, n_layers, d_inner, n_target, dropout=0.1):
#         super().__init__()
#         self.seq_len = seq_len
#         self.n_dim = d_model // 2
#         self.n_target = n_target
#         self.dropout = dropout

#         d_k_f = self.seq_len // n_head
#         d_v_f = self.seq_len // n_head
#         self.feature_csmha = nn.ModuleList([
#             ComplexEncoderLayer(
#                 self.seq_len, d_inner, n_head, d_k_f, d_v_f, dropout=dropout)
#             for _ in range(n_layers)])

#         self.music_spec = MusicSpectrum(array, angles_list)

#         self.phase_error_model = self._dense_net(
#             d_model, [2*self.n_dim, 4*self.n_dim, self.n_dim])

#         # self.norm = nn.LayerNorm(self.n_dim)

#         self.phase_error_model_2 = self._dense_net(
#             self.n_dim, [2*self.n_dim, 4*self.n_dim, self.n_dim])

#         out_dim = len(angles_list)
#         self.fc = self._dense_net(out_dim, [2*out_dim, out_dim])

#     def _dense_net(self, in_s, d_layers):
#         layers = []
#         h_dim = in_s

#         for dim_l in d_layers:
#             layers.append(nn.Linear(h_dim, dim_l))
#             layers.append(nn.Tanh())
#             layers.append(nn.LayerNorm(dim_l))
#             layers.append(nn.Dropout(self.dropout))
#             h_dim = dim_l

#         return nn.Sequential(*layers)

#     def _herm(self, z: torch.Tensor) -> torch.Tensor:
#         return z.conj().transpose(1, 2)

#     def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
#         real = x[..., :self.n_dim]
#         imag = x[..., self.n_dim:]
#         return torch.complex(real, imag)

#     def _noise_supspace(self, z: torch.Tensor) -> torch.Tensor:
#         C = torch.matmul(self._herm(z), z) / float(self.seq_len)
#         Qn = torch.linalg.svd(C)[0]
#         Qn = Qn[:, :, self.n_target:]
#         NSS = torch.matmul(Qn, self._herm(Qn))
#         return NSS

#     def forward(self, x: torch.Tensor, return_attns=False) -> torch.Tensor:
#         enc_slf_attn_list = []

#         x_feat = x
#         for enc_layer in self.feature_csmha:
#             x_feat, enc_slf_attn = enc_layer(x_feat)
#             if return_attns:
#                 enc_slf_attn_list += [enc_slf_attn]

#         if return_attns:
#             return x_feat, enc_slf_attn_list

#         zf = self._to_complex(x)
#         R_x_feat = self._noise_supspace(zf)

#         estimated_error = self.phase_error_model(x_feat.mean(1))
#         estimated_error = self.phase_error_model_2(
#             estimated_error.mean(0, True)).squeeze()
#         spec = self.music_spec(R_x_feat, estimated_error)
#         # spec = F.sigmoid(self.fc(spec))
#         spec = F.sigmoid(spec)

#         return spec,


###########################################
# TransMUSIC
###########################################
class TransMUSIC(nn.Module):
    def __init__(self, array, d_model, n_targets, d_inner, n_head, angles_list, dropout=0.1):
        super().__init__()
        self.pe = SinusoidalPE(d_model)
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.mha = EncoderLayer(MultiHeadAttention, d_model, d_inner, n_head, d_k, d_v, False, False, dropout)
        self.cov_ = ComputeCov(d_model, d_model//2)
        self.music = MusicSpectrum(array, angles_list)

        self.fc = nn.Sequential(
            nn.Linear(len(angles_list), n_targets//8),
            nn.LayerNorm(n_targets//8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_targets//8, n_targets//4),
            nn.LayerNorm(n_targets//4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_targets//4, n_targets),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        x, _ = self.mha(x)
        x = x.mean(-2)
        x = self.cov_(x, 1)
        x = self.music(x)
        x = self.fc(x)
        return x
    

###########################################
# DeepMUSIC
###########################################
class DeepMUSIC(nn.Module):
    def __init__(self, n_targets, n_stacked, split_output, dropout=0.1):
        super().__init__()
        self.n_targets = n_targets
        self.n_stacked = n_stacked
        self.split_output = split_output
        self.dropout = dropout

        self.norm = nn.BatchNorm2d(3)

        self.dnn = self._create_dnn()
        self.initialize_weights()

    def _create_dnn(self):
        layers = []
        for _ in range(self.n_stacked):
            layers.append(nn.Sequential(
                nn.Conv2d(3, 256, 5, 1, 'same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 5, 1, 'same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv2d(256, 256, 3, 1, 'same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv2d(256, 256, 3, 1, 'same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(), 
                nn.Linear(256, self.n_targets*2),
                nn.LayerNorm(self.n_targets*2),
                nn.Softmax(dim=-1),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_targets*2, self.n_targets),
                nn.Sigmoid()
            ))
        return nn.ModuleList(layers)

    def initialize_weights(self):
        for layer in self.dnn:
            for module in layer:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        out = []
        for layer in self.dnn:
            out.append(layer(x))

        if not self.split_output:
            out = torch.cat(out, dim=-1)
        return out


###########################################
# ViT-DOA
###########################################
class ViTDoANetwork(nn.Module):
    def __init__(
        self,
        num_antennas: int = 32,
        output_dim: int = 1024,
        num_heads: int = 8,
        num_encoder_blocks: int = 3,
        dropout=0.1
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.output_dim = output_dim
        self.embed_dim = 2 * num_antennas
        
        self.norm = nn.BatchNorm1d(num_antennas)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, num_antennas, self.embed_dim))

        d_k = self.embed_dim // num_heads
        self.encoder_blocks = nn.ModuleList([
            EncoderLayer(
                MultiHeadAttention, self.embed_dim, self.embed_dim*4, num_heads, d_k, d_k, False, False, dropout
            )
            for _ in range(num_encoder_blocks)
        ])
        
        self.conv1 = Conv1DBlock(num_antennas, 64)
        self.conv2 = Conv1DBlock(64, 128)
        
        conv_output_dim = 128 * num_antennas * 2 
        
        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, output_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim*2, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=-1)
        x = self.norm(x)

        x += self.pos_encoding
        
        for encoder in self.encoder_blocks:
            x, _ = encoder(x)
        
        x_conv = x
        x_conv = self.conv1(x_conv)
        x_conv = self.conv2(x_conv)
        
        x_flat = x_conv.reshape(batch_size, -1)
        output = self.mlp(x_flat)
        
        return output


###########################################
# HMC-ViT
###########################################
class HMCViT(nn.Module):
    def __init__(
        self,
        num_antennas: int = 16,
        num_class_tokens: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        output_dim: int = 1024,
        grid_size: float = 1.0,
        num_fc_layers: int = 8  # Number of separate FC layers for class tokens
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_class_tokens = num_class_tokens
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.output_dim = output_dim
        
        self.cnn_extractor = CNNFeatureExtractor()
        
        self.patch_embedding = PatchEmbedding(num_antennas, embed_dim)
        num_patches = self.patch_embedding.num_patches  # Should be 20
        
        self.class_tokens = nn.Parameter(torch.randn(1, num_class_tokens, embed_dim))
        
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches + num_class_tokens, embed_dim))
        
        d_k = embed_dim / num_heads
        self.encoder_blocks = nn.ModuleList([
            EncoderLayer(
                MultiHeadAttention, self.embed_dim, self.embed_dim*4, num_heads, d_k, d_k, False, False, 0.1
            )
            for _ in range(num_layers)
        ])
        
        self.num_fc_layers = num_fc_layers
        self.fc_layers = nn.ModuleList()
        
        if num_fc_layers == 8:
            for i in range(7):
                self.fc_layers.append(nn.Linear(embed_dim, 15))
            self.fc_layers.append(nn.Linear(embed_dim, 16))
        elif num_fc_layers == 1:
            self.fc_layers.append(nn.Linear(embed_dim * num_class_tokens, self.output_dim))
        else:
            raise NotImplementedError("Only 8 or 1 FC layers are implemented")
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            # Initialize class tokens and positional encoding
            if module.data.dim() > 1:
                nn.init.normal_(module.data, std=0.02)
    
    def forward(self, R: torch.Tensor) -> torch.Tensor:
        batch_size = R.shape[0]
        
        x = self.cnn_extractor(x)  # [batch, 3, 1, 120]
        patches = self.patch_embedding(x)  # [batch, 20, 128]
        class_tokens = self.class_tokens.expand(batch_size, -1, -1)  # [batch, 8, 128]
        
        num_tokens_before = self.num_class_tokens // 2  # 4
        num_tokens_after = self.num_class_tokens // 2    # 4
        
        sequence = torch.cat([
            class_tokens[:, :num_tokens_before, :],  # First 4 tokens
            patches,                                  # 20 patches
            class_tokens[:, num_tokens_before:, :]    # Last 4 tokens
        ], dim=1)  # [batch, 28, 128]
        
        sequence = sequence + self.pos_encoding
        
        for layer in self.transformer_layers:
            sequence, _ = layer(sequence)

        tokens_before = sequence[:, :num_tokens_before, :]  # [batch, 4, 128]
        tokens_after = sequence[:, -num_tokens_after:, :]   # [batch, 4, 128]

        all_tokens = torch.cat([tokens_before, tokens_after], dim=1)  # [batch, 8, 128]
        
        if self.num_fc_layers == 8:
            outputs = []
            for i, fc_layer in enumerate(self.fc_layers):
                token_output = fc_layer(all_tokens[:, i, :])  # [batch, 15 or 16]
                outputs.append(token_output)
            
            output = torch.cat(outputs, dim=1)  # [batch, 121]
        elif self.num_fc_layers == 1:
            flattened = all_tokens.reshape(batch_size, -1)  # [batch, 8*128]
            output = self.fc_layers[0](flattened)  # [batch, 121]
        
        return output

###########################################
# FCDNN-MUSIC
###########################################
class FCDNNMUSIC(nn.Module):
    """
    input shape: (B, 2, M, M)
    """

    def __init__(self, in_shape, array, angles_list, num_antenna, dropout=0.1):
        super().__init__()
        self.num_antenna = num_antenna
        self.h1 = nn.Linear(2*in_shape, 128)
        self.h2 = nn.Linear(128, 256)
        self.dropout_l = nn.Dropout(dropout)
        self.out_l = nn.Linear(256, num_antenna)

        self.music_spec = MusicSpectrum(array, angles_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.tanh(self.h1(x.flatten(1, -1)))
        y = F.tanh(self.h2(y))
        y = self.dropout_l(y)
        est_error = self.out_l(y)

        R = torch.complex(x[:, :1, :], x[:, 1:, :]).squeeze()
        error_cor = torch.diag_embed(torch.exp(-1j * est_error)).to(R.dtype)
        calib_R = torch.matmul(error_cor, R)
        calib_R = torch.matmul(calib_R, error_cor.mH)

        spec = F.sigmoid(self.music_spec(calib_R))

        return spec,


class FCDNN(nn.Module):
    def __init__(self, array, angles_list, M=6, hidden1=256, hidden2=128, dropout_rate=0.3):
        super(FCDNN, self).__init__()
        self.array = array
        self.angles_list = angles_list
        self.M = M

        self.num_upper_tri = (M * (M - 1)) // 2
        self.input_dim = 2 * self.num_upper_tri

        self.fc1 = nn.Linear(self.input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, M - 1)

        self.dropout = nn.Dropout(dropout_rate)

        self.music_spec = MusicSpectrum(self.array, self.angles_list)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def preprocess_covariance(self, R_hat):
        upper_tri_indices = torch.triu_indices(self.M, self.M, offset=1)
        z = R_hat[upper_tri_indices[0], upper_tri_indices[1]]

        z_real = torch.real(z).to(torch.float32)
        z_imag = torch.imag(z).to(torch.float32)

        z_concat = torch.cat([z_real, z_imag])
        norm = torch.norm(z_concat, p=2)

        if norm > 0:
            y = z_concat / norm
        else:
            y = z_concat

        return y

    def forward(self, R_hat_batch):
        batch_size = R_hat_batch.shape[0]
        processed_inputs = []

        # Preprocess each covariance matrix in the batch
        for i in range(batch_size):
            y = self.preprocess_covariance(R_hat_batch[i])
            processed_inputs.append(y)

        # Stack to form batch
        x = torch.stack(processed_inputs)

        # Forward pass through network
        # Hidden layer 1 with tanh activation
        x = torch.tanh(self.fc1(x))

        # Hidden layer 2 with tanh activation and dropout
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)

        # Output layer (no activation for regression)
        phase_errors = self.fc3(x)

        R = self.create_calibration_matrix(phase_errors).to(x.device)
        spec = F.sigmoid(self.music_spec(R))

        return spec,

    def create_calibration_matrix(self, phase_errors):
        """
        Create calibration matrix from estimated phase errors.

        Args:
            phase_errors: Estimated phase errors [batch_size, M-1] or [M-1]

        Returns:
            Calibration matrix [batch_size, M, M] or [M, M]
        """
        if len(phase_errors.shape) == 1:
            # Single sample
            phase_errors = phase_errors.unsqueeze(0)
            return_single = True
        else:
            return_single = False

        batch_size = phase_errors.shape[0]
        M = self.M

        # Create diagonal matrices with phase corrections
        calibration_matrices = torch.zeros(
            batch_size, M, M, dtype=torch.complex64)

        for b in range(batch_size):
            # Reference element has no phase error
            diag_elements = torch.ones(M, dtype=torch.complex64)

            # Apply phase corrections to elements 2..M
            for m in range(1, M):
                # Negative phase for calibration (see Eq. 14 in paper)
                phase_correction = -phase_errors[b, m-1]
                diag_elements[m] = torch.exp(1j * phase_correction)

            calibration_matrices[b] = torch.diag(diag_elements)

        if return_single:
            return calibration_matrices[0]
        return calibration_matrices


###########################################
# HDNN
###########################################
class MultiTaskAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_subregions):
        """
        Args:
            input_dim: Dimension of input vector r (real-valued)
            hidden_dim: Dimension of hidden layer c1 (typically input_dim/2)
            num_subregions: Number of spatial subregions P
        """
        super(MultiTaskAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_subregions = num_subregions

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        self.decoders = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim, bias=True)
            for _ in range(num_subregions)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        """
        Forward pass of autoencoder
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            outputs: List of decoder outputs, each of shape (batch_size, input_dim)
            hidden: Encoder hidden representation (batch_size, hidden_dim)
        """
        # Encode
        hidden = self.encoder(x)

        outputs = []
        for decoder in self.decoders:
            output = decoder(hidden)
            outputs.append(output)

        return outputs, hidden


class SubregionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: Dimension of decoder output (same as autoencoder input_dim)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of directional grids in subregion (I0)
        """
        super(SubregionClassifier, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        """
        Forward pass of classifier
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            output: Spatial spectrum for subregion (batch_size, output_dim)
        """
        return self.network(x)


class DNN_DOA_Estimator(nn.Module):
    def __init__(self, M, P, I0, hidden_dims_classifier):
        """
        Args:
            M: Number of array elements
            P: Number of spatial subregions
            I0: Number of directional grids per subregion
            hidden_dims_classifier: List of hidden layer dimensions for classifiers
        """
        super(DNN_DOA_Estimator, self).__init__()

        self.M = M
        self.P = P
        self.I0 = I0
        self.total_grids = P * I0

        self.input_dim = M * (M - 1)
        # self.input_dim = M * (M - 1) // 2

        self.hidden_dim_autoencoder = self.input_dim // 2

        self.autoencoder = MultiTaskAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim_autoencoder,
            num_subregions=P
        )

        self.classifiers = nn.ModuleList([
            SubregionClassifier(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims_classifier,
                output_dim=I0
            ) for _ in range(P)
        ])

        self.train_autoencoder_only = False
        self.train_classifiers_only = False

    def forward(self, r):
        """
        Complete forward pass through autoencoder and classifiers
        Args:
            r: Input covariance vector of shape (batch_size, input_dim)
               Preprocessed according to equations (11-12)
        Returns:
            spatial_spectrum: Concatenated spatial spectrum of shape (batch_size, total_grids)
            decoder_outputs: List of decoder outputs for debugging/analysis
        """
        decoder_outputs, hidden = self.autoencoder(r)

        # If only training autoencoder, return decoder outputs
        if self.train_autoencoder_only:
            return decoder_outputs

        # Classifiers forward pass for each subregion
        classifier_outputs = []
        for i, (decoder_out, classifier) in enumerate(zip(decoder_outputs, self.classifiers)):
            subregion_spectrum = classifier(decoder_out)
            classifier_outputs.append(subregion_spectrum)

        # Concatenate classifier outputs to form complete spatial spectrum
        spatial_spectrum = F.sigmoid(torch.cat(classifier_outputs, dim=1))

        return spatial_spectrum, decoder_outputs

    def set_train_autoencoder_only(self):
        self.train_autoencoder_only = True
        self.train_classifiers_only = False

        for param in self.classifiers.parameters():
            param.requires_grad = False

        for param in self.autoencoder.parameters():
            param.requires_grad = True

    def set_train_classifiers_only(self):
        self.train_autoencoder_only = False
        self.train_classifiers_only = True

        for param in self.autoencoder.parameters():
            param.requires_grad = False

        for param in self.classifiers.parameters():
            param.requires_grad = True

    def set_train_all(self):
        self.train_autoencoder_only = False
        self.train_classifiers_only = False

        for param in self.parameters():
            param.requires_grad = True

    def get_autoencoder_parameters(self):
        return list(self.autoencoder.parameters())

    def get_classifier_parameters(self):
        params = []
        for classifier in self.classifiers:
            params.extend(list(classifier.parameters()))
        return params
