# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:24:19 2025

@author: WiCi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

 # Spatial Attention Module
class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 修改: 使用 AdaptiveAvgPool1d 和 AdaptiveMaxPool1d
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # 修改: 使用 Conv1d
        self.conv = nn.Conv1d(2 * in_channels, in_channels, kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()
 

    def forward(self, x):
        """
        Args:
        x: Input feature map of shape [batch_size, in_channels, sequence_length]
        Returns:
        Attended feature map of shape [batch_size, in_channels, sequence_length]
        """
        x = x.permute(0, 2, 1)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attention = self.sigmoid(out)
        x = x * attention
        x = x.permute(0, 2, 1)
        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
        batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
 

    def forward(self, src):
        # src shape: [batch_size, seq_len, d_model]
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
 

 # Spatial Attention Module with Transformer
class SpatialAttentionTransformerModule(nn.Module):
    def __init__(self, in_channels, transformer_heads=4,transformer_layers=1):
        super().__init__()
        self.in_channels = in_channels
        # Spatial Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(2 * in_channels, in_channels, kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()
     
    
        # Transformer
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(in_channels, transformer_heads)
            for _ in range(transformer_layers)
        ])
 

    def forward(self, x):
        """
        Args:
        x: Input feature map of shape [batch_size, in_channels, sequence_length]
        Returns:
        Attended feature map of shape [batch_size, in_channels, sequence_length]
        """
        # Spatial Attention
        x_perm = x.permute(0, 2, 1)  # [batch_size, sequence_length, in_channels]
        avg_out = self.avg_pool(x_perm)
        max_out = self.max_pool(x_perm)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attention = self.sigmoid(out)
        x = x_perm * attention
         
        
        # Transformer
        x = x.permute(0, 2, 1)  # Prepare for Transformer: [batch_size, sequence_length, in_channels]
        for layer in self.transformer_layers:
            x = layer(x)
            x = x.permute(0, 2, 1)  # Back to original shape
         
        x = x.permute(0, 2, 1)
        return x

class PASA(nn.Module):
    """基于 MoE 的多模态信道估计模型"""
    def __init__(self, max_antennas=16, sig_dim=4*2, hidden_dim=64, mlp_dim=128, num_blocks=4, num_experts=3, output_dim=2):
        super().__init__()
        self.max_antennas = max_antennas
        self.sig_dim = sig_dim
        self.num_sa_layers = 3
        
        # 模态编码器
        # self.pos_encoder = nn.Linear(1, hidden_dim)  # 天线位置编码
        self.sig_encoder = nn.Linear(sig_dim, hidden_dim)  # 导频信号编码
        
        self.gating = GatingMechanism(input_dim=hidden_dim, hidden_dim=64)
        
        # 多专家 MLP-Mixer
        # self.moe_mixer = SpatialAttentionModule(64)
        
        self.sa_layers = nn.ModuleList([SpatialAttentionModule(64) for _ in range(self.num_sa_layers)])
                                
        self.pos_encoder = FourierPositionalEmbedding(input_dim=1, embed_dim=hidden_dim, num_frequencies=4)
        
        # 动态填充向量
        self.learned_padding1 = nn.Parameter(torch.randn(1, 1, 1))
        self.learned_padding2 = nn.Parameter(torch.randn(1, 1, sig_dim))
        
        # 动态输出层
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, positions, signals):
        """
        Args:
            positions: [batch, current_antennas, 1]
            signals:   [batch, current_antennas, slots*2]
            current_antennas: 当前天线数量
        Returns:
            Tensor: [batch, current_antennas, output_dim]
        """        
        batch_size, current_antennas, _ = positions.size()
        
        padding1 = self.learned_padding1.repeat(batch_size, self.max_antennas - current_antennas, 1)
        padding2 = self.learned_padding2.repeat(batch_size, self.max_antennas - current_antennas, 1)
        
        # 填充到最大天线数量
        if current_antennas < self.max_antennas:
            pad_size = self.max_antennas - current_antennas
            positions = torch.cat([positions, torch.zeros((batch_size, pad_size, 1), dtype=positions.dtype, device=positions.device)], dim=1)
            signals = torch.cat([signals, torch.zeros((batch_size, pad_size, self.sig_dim), dtype=signals.dtype, device=signals.device)], dim=1)
        
            positions[:, current_antennas:, :] = padding1
            signals[:, current_antennas:, :] = padding2
        
        
        # 编码特征
        pos_features = self.pos_encoder(positions)  # [batch, max_antennas, hidden_dim]
        sig_features = self.sig_encoder(signals)    # [batch, max_antennas, hidden_dim]
        
        # 门控机制融合
        fused_features = self.gating(
            control_features=pos_features,  # 使用天线位置生成门控信号
            target_features=sig_features    # 对导频信号应用门控
        )  # [batch, max_antennas, hidden_dim]
        
        # 使用多专家 MLP-Mixer 进行特征提取
        # expert_features = self.moe_mixer(fused_features, current_patches=self.max_antennas)  # [batch, max_antennas, hidden_dim]
        
        for i in range(self.num_sa_layers):
            expert_features = self.sa_layers[i](fused_features)
        # 特征拼接
        combined = torch.cat([pos_features, expert_features], dim=-1)  # [batch, max_antennas, hidden_dim * 2]
        
        # 输出层
        output = self.output_fc(combined)  # [batch, max_antennas, output_dim]
        
        # 裁剪到当前天线数量
        return output[:, :current_antennas, :]
    
class PADA(nn.Module):
    """基于 MoE 的多模态信道估计模型"""
    def __init__(self, max_antennas=16, sig_dim=4*2, hidden_dim=64, mlp_dim=128, num_blocks=4, num_experts=3, output_dim=2):
        super().__init__()
        self.max_antennas = max_antennas
        self.sig_dim = sig_dim
        self.num_sa_layers = 3
        
        # 模态编码器
        # self.pos_encoder = nn.Linear(1, hidden_dim)  # 天线位置编码
        self.sig_encoder = nn.Linear(sig_dim, hidden_dim)  # 导频信号编码
        
        self.gating = GatingMechanism(input_dim=hidden_dim, hidden_dim=64)
        
        # 多专家 MLP-Mixer
        
        self.sa_layers = nn.ModuleList([SpatialAttentionTransformerModule(64) for _ in range(self.num_sa_layers)])
                                
        self.pos_encoder = FourierPositionalEmbedding(input_dim=1, embed_dim=hidden_dim, num_frequencies=4)
        
        # 动态填充向量
        self.learned_padding1 = nn.Parameter(torch.randn(1, 1, 1))
        self.learned_padding2 = nn.Parameter(torch.randn(1, 1, sig_dim))
        
        # 动态输出层
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, positions, signals):
        """
        Args:
            positions: [batch, current_antennas, 1]
            signals:   [batch, current_antennas, slots*2]
            current_antennas: 当前天线数量
        Returns:
            Tensor: [batch, current_antennas, output_dim]
        """        
        batch_size, current_antennas, _ = positions.size()
        
        padding1 = self.learned_padding1.repeat(batch_size, self.max_antennas - current_antennas, 1)
        padding2 = self.learned_padding2.repeat(batch_size, self.max_antennas - current_antennas, 1)
        
        # 填充到最大天线数量
        if current_antennas < self.max_antennas:
            pad_size = self.max_antennas - current_antennas
            positions = torch.cat([positions, torch.zeros((batch_size, pad_size, 1), dtype=positions.dtype, device=positions.device)], dim=1)
            signals = torch.cat([signals, torch.zeros((batch_size, pad_size, self.sig_dim), dtype=signals.dtype, device=signals.device)], dim=1)
        
            positions[:, current_antennas:, :] = padding1
            signals[:, current_antennas:, :] = padding2
        
        
        # 编码特征
        pos_features = self.pos_encoder(positions)  # [batch, max_antennas, hidden_dim]
        sig_features = self.sig_encoder(signals)    # [batch, max_antennas, hidden_dim]
        
        # 门控机制融合
        fused_features = self.gating(
            control_features=pos_features,  # 使用天线位置生成门控信号
            target_features=sig_features    # 对导频信号应用门控
        )  # [batch, max_antennas, hidden_dim]
        
        # 使用多专家 MLP-Mixer 进行特征提取
        # expert_features = self.moe_mixer(fused_features, current_patches=self.max_antennas)  # [batch, max_antennas, hidden_dim]
        
        for i in range(self.num_sa_layers):
            expert_features = self.sa_layers[i](fused_features)
        # 特征拼接
        combined = torch.cat([pos_features, expert_features], dim=-1)  # [batch, max_antennas, hidden_dim * 2]
        
        # 输出层
        output = self.output_fc(combined)  # [batch, max_antennas, output_dim]
        
        # 裁剪到当前天线数量
        return output[:, :current_antennas, :]

class SAB(nn.Module):
    """
    Simplified Set Attention Block
    做多头自注意力后，再接一个前馈网络 (FeedForward), 类似Transformer Encoder。
    """
    def __init__(self, dim, num_heads, ff_hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * dim
        
        # Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim)
        )
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [batch, n, dim]
        输出: [batch, n, dim]
        """
        # Self-Attention + 残差
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)  # [B, n, dim]
        x = x + attn_out
        
        # FFN + 残差
        x_norm = self.ln2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out
        
        return x
    
class SetTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, ff_hidden_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([
            SAB(dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiModalSetTransformerModel(nn.Module):
    def __init__(self, pos_dim=1, sig_dim=8, hidden_dim=64, output_dim=2):
        super().__init__()
        self.pos_encoder = nn.Linear(pos_dim, hidden_dim)
        self.sig_encoder = nn.Linear(sig_dim, hidden_dim)
        self.gating = GatingMechanism(hidden_dim, 64)  # 示例
        self.encoder = SetTransformerEncoder(dim=hidden_dim, num_heads=4, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, positions, signals, return_embedding=False):
        pos_feat = self.pos_encoder(positions)
        sig_feat = self.sig_encoder(signals)
        fused = self.gating(pos_feat, sig_feat) + pos_feat
        x = self.encoder(fused)  # [B, N, hidden_dim]
        
        # 取mean pooling作为全局embedding
        emb = x.mean(dim=1)  # [B, hidden_dim]
        
        if return_embedding:
            return emb
        
        out = torch.cat([x, pos_feat], dim=-1)
        out = self.output_fc(out)  # [B, N, 2]
        return out



class SetBasedChannelEstimationModel(nn.Module):
    def __init__(self, pos_dim=1, sig_dim=8, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2):
        """
        pos_dim:    天线位置维度 (1D, 或 2D等)
        sig_dim:    导频信号维度 (实部+虚部 或更多)
        hidden_dim: 注意力模型中每个 token 的内部表示维度
        num_heads:  Multi-Head Attention 的头数
        num_layers: 堆叠的 SAB 层数
        output_dim: 最终要估计的信道输出维度 (一般是 2 表示实部+虚部)
        """
        super().__init__()
        
        # 先把 pos+sig 合并到embedding
        self.input_dim = pos_dim + sig_dim
        self.hidden_dim = hidden_dim
        
        # Embedding 层: 把 (pos_dim + sig_dim) -> hidden_dim
        self.emb = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 多层 SAB (Set Attention Block)
        self.encoder = SetTransformerEncoder(dim=hidden_dim,
                                             num_heads=num_heads,
                                             num_layers=num_layers)
        
        # 输出层: 每个元素 (天线) 输出 2 维 (实部, 虚部)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, positions, signals):
        """
        positions: [B, n, pos_dim]
        signals:   [B, n, sig_dim]
        return:    [B, n, output_dim]  (对每根天线输出一个信道向量)
        """
        # 拼接 pos + sig
        x = torch.cat([positions, signals], dim=-1)  # [B, n, (pos_dim+sig_dim)]
        
        # 映射到 hidden_dim
        x = self.emb(x)  # [B, n, hidden_dim]
        
        # 使用 Set Transformer Encoder
        x = self.encoder(x)  # [B, n, hidden_dim]
        
        # 输出层
        out = self.out_fc(x)  # [B, n, output_dim]
        return out

def pinching_consistency_loss(pred_full, pred_sub, sub_idx):
    """
    pred_full: [N, 2]   => 全阵列对应的预测结果 (N根天线, 每根天线输出2维(实部,虚部))
    pred_sub:  [M, 2]   => 子阵列预测 (M根天线, 2维)
    sub_idx:   长度为 M 的索引列表或张量(从[0..N-1]中挑选)

    return: 标量, L2距离的平均值
    """
    # 取出 pred_full 在子集索引 sub_idx 处的预测 => [M, 2]
    full_sub = pred_full[:,sub_idx,:]  # shape [M,2]

    # L2差异: (full_sub - pred_sub)^2 然后按行和，再取平均
    diff = full_sub - pred_sub     # [M,2]
    return torch.mean(torch.sum(diff**2, dim=1))  # => 标量


class GatingMechanism(nn.Module):
    """门控机制模块"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 门控信号范围 [0, 1]
        )
        
    def forward(self, control_features, target_features):
        """
        Args:
            control_features: 用于生成门控信号的特征 [batch, antennas, input_dim]
            target_features: 被门控的特征 [batch, antennas, input_dim]
        Returns:
            Tensor: 融合后的特征 [batch, antennas, input_dim]
        """
        gate = self.fc(control_features)  # 生成门控信号
        fused_features = gate * target_features  # 应用门控
        return fused_features

class DynamicMLPMixerBlock(nn.Module):
    """动态调整的 MLP-Mixer 模块"""
    def __init__(self, max_patches, hidden_dim, mlp_dim):
        super().__init__()
        self.max_patches = max_patches
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        
        # 动态调整的 MLP 参数
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        
        self.LR = nn.LeakyReLU(0.3)
        
    def forward(self, x, current_patches):
        """
        Args:
            x: Tensor [batch, current_patches, hidden_dim]
            current_patches: 当前天线数量
        Returns:
            Tensor: [batch, current_patches, hidden_dim]
        """
        # 对特征维度的 MLP
        residual = x
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, current_patches]
        x = self.norm1(x)
        x = x.permute(0, 2, 1)  # [batch, current_patches, hidden_dim]
        x = self.LR(self.fc1(x))  # [batch, current_patches, mlp_dim]
        x = self.LR(self.fc2(x))  # [batch, current_patches, hidden_dim]
        x = x + residual  # 残差连接
        
        # 对空间维度的 MLP
        residual = x
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, current_patches]
        x = self.norm2(x)
        x = x.permute(0, 2, 1)  # [batch, current_patches, hidden_dim]
        x = self.LR(self.fc1(x))  # [batch, current_patches, mlp_dim]
        x = self.LR(self.fc2(x))  # [batch, current_patches, hidden_dim]
        x = x + residual  # 残差连接
        
        return x
    
class DynamicMLPMixer(nn.Module):
    """动态调整的 MLP-Mixer 模型"""
    def __init__(self, max_patches, hidden_dim, mlp_dim, num_blocks):
        super().__init__()
        self.max_patches = max_patches
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.blocks = nn.ModuleList([
            DynamicMLPMixerBlock(max_patches, hidden_dim, mlp_dim)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, current_patches):
        """
        Args:
            x: Tensor [batch, current_patches, hidden_dim]
            current_patches: 当前天线数量
        Returns:
            Tensor: [batch, current_patches, hidden_dim]
        """
        for block in self.blocks:
            x = block(x, current_patches)
        return x

class ExpertNetwork(nn.Module):
    """专家网络"""
    def __init__(self, max_patches, hidden_dim, mlp_dim, num_blocks):
        super().__init__()
        self.mixer = DynamicMLPMixer(max_patches, hidden_dim, mlp_dim, num_blocks)
        
    def forward(self, x, current_patches):
        return self.mixer(x, current_patches)

class GatingNetwork(nn.Module):
    """门控网络"""
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        return self.softmax(self.fc(x))

class MultiExpertMLPMixer(nn.Module):
    """多专家 MLP-Mixer 模型"""
    def __init__(self, max_patches, hidden_dim, mlp_dim, num_blocks, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertNetwork(max_patches, hidden_dim, mlp_dim, num_blocks)
            for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(hidden_dim, num_experts)
        
    def forward(self, x, current_patches):
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x, current_patches)
            expert_outputs.append(expert_output.unsqueeze(-1))
        expert_outputs = torch.cat(expert_outputs, dim=-1)
        
        # 计算门控权重
        gate_input = x.mean(dim=1)
        gate_weights = self.gating_network(gate_input)
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(1)
        
        # 加权平均专家输出
        output = torch.sum(expert_outputs * gate_weights, dim=-1)
        return output
    
class FourierPositionalEmbedding(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.linear = nn.Linear(num_frequencies*2, embed_dim)
        
    def forward(self, x):
        """
        x: [batch, antennas, 1] (假设天线1D坐标)
        """
        batch_size, n_ant, _ = x.size()
        # 生成 [sin(2^k * π * x), cos(2^k * π * x)]
        freq_bases = 2.0 ** torch.arange(self.num_frequencies, device=x.device).float() * math.pi
        freq_bases = freq_bases.view(1, 1, -1)  # [1,1,num_frequencies]
        
        x_expanded = x  # [batch, antennas, 1]
        angles = x_expanded * freq_bases  # [batch, antennas, num_frequencies]
        
        sin_embed = torch.sin(angles)  # [batch, antennas, num_frequencies]
        cos_embed = torch.cos(angles)  # [batch, antennas, num_frequencies]
        
        embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [batch, antennas, num_frequencies*2]
        embed = self.linear(embed)  # [batch, antennas, embed_dim]
        
        return embed

class MultiModalChannelEstimationModel(nn.Module):
    """基于 MoE 的多模态信道估计模型"""
    def __init__(self, max_antennas=16, sig_dim=4*2, hidden_dim=64, mlp_dim=128, num_blocks=4, num_experts=3, output_dim=2):
        super().__init__()
        self.max_antennas = max_antennas
        self.sig_dim = sig_dim
        
        # 模态编码器
        # self.pos_encoder = nn.Linear(1, hidden_dim)  # 天线位置编码
        self.sig_encoder = nn.Linear(sig_dim, hidden_dim)  # 导频信号编码
        
        self.gating = GatingMechanism(input_dim=hidden_dim, hidden_dim=64)
        
        # 多专家 MLP-Mixer
        self.moe_mixer = MultiExpertMLPMixer(max_antennas, hidden_dim, mlp_dim, num_blocks, num_experts)
        
        self.pos_encoder = FourierPositionalEmbedding(input_dim=1, embed_dim=hidden_dim, num_frequencies=4)
        
        # 动态填充向量
        self.learned_padding1 = nn.Parameter(torch.randn(1, 1, 1))
        self.learned_padding2 = nn.Parameter(torch.randn(1, 1, sig_dim))
        
        # 动态输出层
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, positions, signals):
        """
        Args:
            positions: [batch, current_antennas, 1]
            signals:   [batch, current_antennas, slots*2]
            current_antennas: 当前天线数量
        Returns:
            Tensor: [batch, current_antennas, output_dim]
        """        
        batch_size, current_antennas, _ = positions.size()
        
        padding1 = self.learned_padding1.repeat(batch_size, self.max_antennas - current_antennas, 1)
        padding2 = self.learned_padding2.repeat(batch_size, self.max_antennas - current_antennas, 1)
        
        # 填充到最大天线数量
        if current_antennas < self.max_antennas:
            pad_size = self.max_antennas - current_antennas
            positions = torch.cat([positions, torch.zeros((batch_size, pad_size, 1), dtype=positions.dtype, device=positions.device)], dim=1)
            signals = torch.cat([signals, torch.zeros((batch_size, pad_size, self.sig_dim), dtype=signals.dtype, device=signals.device)], dim=1)
        
            positions[:, current_antennas:, :] = padding1
            signals[:, current_antennas:, :] = padding2
        
        
        # 编码特征
        pos_features = self.pos_encoder(positions)  # [batch, max_antennas, hidden_dim]
        sig_features = self.sig_encoder(signals)    # [batch, max_antennas, hidden_dim]
        
        # 门控机制融合
        fused_features = self.gating(
            control_features=pos_features,  # 使用天线位置生成门控信号
            target_features=sig_features    # 对导频信号应用门控
        )  # [batch, max_antennas, hidden_dim]
        
        # 使用多专家 MLP-Mixer 进行特征提取
        expert_features = self.moe_mixer(fused_features, current_patches=self.max_antennas)  # [batch, max_antennas, hidden_dim]
        
        # 特征拼接
        combined = torch.cat([pos_features, expert_features], dim=-1)  # [batch, max_antennas, hidden_dim * 2]
        
        # 输出层
        output = self.output_fc(combined)  # [batch, max_antennas, output_dim]
        
        # 裁剪到当前天线数量
        return output[:, :current_antennas, :]

class MLPChannelEstimationModel(nn.Module):
    def __init__(self, pos_dim=1, sig_dim=8, hidden_dim=64, num_layers=4, output_dim=2):
        """
        pos_dim:    天线位置维度 (1D, 或 2D等)
        sig_dim:    导频信号维度 (实部+虚部 或更多)
        hidden_dim: MLP 中每个层的隐藏维度
        num_layers: MLP 中的隐藏层数
        output_dim: 最终要估计的信道输出维度 (一般是 2 表示实部+虚部)
        """
        super().__init__()
        
        # 输入维度是位置和信号维度的拼接
        self.input_dim = pos_dim + sig_dim
        
        # 构建 MLP 网络，使用多个隐藏层
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))  # 输入层
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):  # 多个隐藏层
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))  # 输出 2 维 (实部，虚部)
        
        # 将所有层组合成一个序列
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, positions, signals):
        """
        positions: [B, n, pos_dim]
        signals:   [B, n, sig_dim]
        return:    [B, n, output_dim]  (对每根天线输出一个信道向量)
        """
        # 拼接位置和信号
        x = torch.cat([positions, signals], dim=-1)  # [B, n, (pos_dim + sig_dim)]
        
        # 展平维度，适配 MLP 输入 [B * n, input_dim]
        batch_size, n_antennas, _ = x.size()
        x = x.view(batch_size * n_antennas, -1)  # [B * n, input_dim]
        
        # 通过 MLP 进行前向传播
        out = self.mlp(x)  # [B * n, output_dim]
        
        # 恢复输出为原始的天线数量
        out = out.view(batch_size, n_antennas, -1)  # [B, n, output_dim]
        
        return out

    
class SAB(nn.Module):
    """
    Simplified Set Attention Block
    做多头自注意力后，再接一个前馈网络 (FeedForward), 类似Transformer Encoder。
    """
    def __init__(self, dim, num_heads, ff_hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * dim
        
        # Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(ff_hidden_dim, dim)
        )
        
        # LayerNorm
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [batch, n, dim]
        输出: [batch, n, dim]
        """
        # Self-Attention + 残差
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)  # [B, n, dim]
        x = x + attn_out
        
        # FFN + 残差
        x_norm = self.ln2(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out
        
        return x
    
class SetTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, ff_hidden_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([
            SAB(dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiModalSetTransformerModel(nn.Module):
    def __init__(self, pos_dim=1, sig_dim=8, hidden_dim=64, output_dim=2):
        super().__init__()
        self.pos_encoder = nn.Linear(pos_dim, hidden_dim)
        self.sig_encoder = nn.Linear(sig_dim, hidden_dim)
        self.gating = GatingMechanism(hidden_dim, 64)  # 示例
        self.encoder = SetTransformerEncoder(dim=hidden_dim, num_heads=4, num_layers=2)
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, positions, signals, return_embedding=False):
        pos_feat = self.pos_encoder(positions)
        sig_feat = self.sig_encoder(signals)
        fused = self.gating(pos_feat, sig_feat) + pos_feat
        x = self.encoder(fused)  # [B, N, hidden_dim]
        
        # 取mean pooling作为全局embedding
        emb = x.mean(dim=1)  # [B, hidden_dim]
        
        if return_embedding:
            return emb
        
        out = torch.cat([x, pos_feat], dim=-1)
        out = self.output_fc(out)  # [B, N, 2]
        return out



class SetBasedChannelEstimationModel(nn.Module):
    def __init__(self, pos_dim=1, sig_dim=8, hidden_dim=64, num_heads=4, num_layers=2, output_dim=2):
        """
        pos_dim:    天线位置维度 (1D, 或 2D等)
        sig_dim:    导频信号维度 (实部+虚部 或更多)
        hidden_dim: 注意力模型中每个 token 的内部表示维度
        num_heads:  Multi-Head Attention 的头数
        num_layers: 堆叠的 SAB 层数
        output_dim: 最终要估计的信道输出维度 (一般是 2 表示实部+虚部)
        """
        super().__init__()
        
        # 先把 pos+sig 合并到embedding
        self.input_dim = pos_dim + sig_dim
        self.hidden_dim = hidden_dim
        
        # Embedding 层: 把 (pos_dim + sig_dim) -> hidden_dim
        self.emb = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 多层 SAB (Set Attention Block)
        self.encoder = SetTransformerEncoder(dim=hidden_dim,
                                             num_heads=num_heads,
                                             num_layers=num_layers)
        
        # 输出层: 每个元素 (天线) 输出 2 维 (实部, 虚部)
        self.out_fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, positions, signals):
        """
        positions: [B, n, pos_dim]
        signals:   [B, n, sig_dim]
        return:    [B, n, output_dim]  (对每根天线输出一个信道向量)
        """
        # 拼接 pos + sig
        x = torch.cat([positions, signals], dim=-1)  # [B, n, (pos_dim+sig_dim)]
        
        # 映射到 hidden_dim
        x = self.emb(x)  # [B, n, hidden_dim]
        
        # 使用 Set Transformer Encoder
        x = self.encoder(x)  # [B, n, hidden_dim]
        
        # 输出层
        out = self.out_fc(x)  # [B, n, output_dim]
        return out