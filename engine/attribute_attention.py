"""
序列化属性注意力模块（无mask消融版）
充分利用6类属性信息来增强语义分割和属性识别

【消融实验版本】：移除了boundary_mask和surface_mask的注意力增强
用于验证mask对模型性能的贡献

属性类别定义（基于实际数据）：
- 类别0: 平面
- 类别1: 曲面
- 类别2: 内部直边
- 类别3: 内部曲边
- 类别4: 边界直边
- 类别5: 边界曲边

属性层次结构：
Level 1 - 位置: 表面(0,1) vs 边缘(2,3,4,5)
Level 2 - 表面: 平面(0) vs 曲面(1)
Level 3 - 边界: 边界(4,5) vs 内部(2,3)
Level 4 - 几何: 直边(2,4) vs 曲边(3,5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class AttributeAwareAttention(nn.Module):
    """
    属性感知注意力 (Attribute-Aware Attention)
    
    核心改进：
    1. 使用完整的6类属性信息，而非仅边界概率
    2. 区分平面/曲面、边界/内部、直边/曲边
    3. 属性引导的多头注意力调制
    4. 跨层属性特征传递
    
    属性分组（基于实际数据定义）：
    - 位置类型：表面(0,1) vs 边缘(2,3,4,5)
    - 表面类型：平面(0) vs 曲面(1)
    - 边界类型：边界(4,5) vs 内部(2,3)
    - 几何类型：直边(2,4) vs 曲边(3,5)
    
    Args:
        channels: 特征通道数
        num_heads: 注意力头数
        patch_size: patch大小
        num_att_classes: 属性类别数（默认6）
        attribute_enhance: 属性增强强度
        use_attribute_prediction: 是否使用属性预测（否则用特征估计）
    """
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=512,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_att_classes=6,
        attribute_enhance=0.5,
        use_attribute_prediction=False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.patch_size = patch_size
        self.num_att_classes = num_att_classes
        self.attribute_enhance = attribute_enhance
        self.use_attribute_prediction = use_attribute_prediction
        
        # 标准QKV投影
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        
        # ========== 属性预测器（轻量级）==========
        # 从特征预测属性分布
        self.attribute_predictor = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels // 2, num_att_classes),
        )
        
        # ========== 属性嵌入 ==========
        # 将属性概率转换为可学习的嵌入
        self.attribute_embedding = nn.Sequential(
            nn.Linear(num_att_classes, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
        )
        
        # ========== 多组属性调制器 ==========
        # 1. 表面类型调制（平面 vs 曲面）
        self.position_type_modulator = nn.Sequential(
            nn.Linear(2, num_heads),
            nn.Sigmoid(),
        )
        self.surface_type_modulator = nn.Sequential(
            nn.Linear(2, num_heads),  # [平面概率, 曲面概率]
            nn.Sigmoid(),
        )
        
        # 2. 边界类型调制（边界 vs 内部）
        self.boundary_type_modulator = nn.Sequential(
            nn.Linear(2, num_heads),  # [边界概率, 内部概率]
            nn.Sigmoid(),
        )
        
        # 3. 几何类型调制（直边 vs 曲边）
        self.geometry_type_modulator = nn.Sequential(
            nn.Linear(2, num_heads),  # [直边概率, 曲边概率]
            nn.Sigmoid(),
        )
        
        # ========== 属性特征提取器 ==========
        # 针对不同属性类型提取特定特征
        self.surface_feature_extractor = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels // 2),
        )
        
        self.boundary_feature_extractor = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels // 2),
        )
        
        # ========== 多尺度融合门控 ==========
        # 融合注意力输出、属性嵌入、表面特征、边界特征
        self.fusion_gate = nn.Sequential(
            nn.Linear(channels * 3, channels),  # attn_out + surface_feat + boundary_feat
            nn.Sigmoid(),
        )
        
        # ========== 跨层特征传递 ==========
        # 用于将属性信息传递到更深层
        self.cross_layer_projection = nn.Sequential(
            nn.Linear(channels + num_att_classes, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        
        # 输出投影
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
    
    def compute_attribute_distribution(self, features, att_logits=None):
        """
        计算属性分布
        
        Args:
            features: [N, C] - 输入特征
            att_logits: [N, num_att_classes] - 外部属性预测（可选）
        
        Returns:
            att_prob: [N, num_att_classes] - 属性概率分布
        """
        if att_logits is not None and self.use_attribute_prediction:
            # 使用外部属性预测
            att_prob = F.softmax(att_logits, dim=-1)
        else:
            # 使用内部属性预测器
            att_logits = self.attribute_predictor(features)
            att_prob = F.softmax(att_logits, dim=-1)
        
        return att_prob
    
    def extract_attribute_groups(self, att_prob):
        """
        从属性分布中提取不同的属性组（修正版）
        
        Args:
            att_prob: [N, 6] - 属性概率分布
            类别0: 平面, 类别1: 曲面
            类别2: 内部直边, 类别3: 内部曲边
            类别4: 边界直边, 类别5: 边界曲边
        
        Returns:
            Dict包含：
            - position_type: [N, 2] - [表面概率, 边缘概率]
            - surface_type: [N, 2] - [平面概率, 曲面概率]
            - boundary_type: [N, 2] - [边界概率, 内部概率]
            - geometry_type: [N, 2] - [直边概率, 曲边概率]
        """
        # 位置类型：表面(0,1) vs 边缘(2,3,4,5)
        surface_prob = att_prob[:, 0:2].sum(dim=-1, keepdim=True)  # [N, 1]
        edge_prob = att_prob[:, 2:6].sum(dim=-1, keepdim=True)  # [N, 1]
        position_type = torch.cat([surface_prob, edge_prob], dim=-1)  # [N, 2]
        
        # 表面类型：平面(0) vs 曲面(1)
        plane_prob = att_prob[:, 0:1]  # [N, 1]
        curved_surface_prob = att_prob[:, 1:2]  # [N, 1]
        surface_type = torch.cat([plane_prob, curved_surface_prob], dim=-1)  # [N, 2]
        
        # 边界类型：边界(4,5) vs 内部(2,3)
        boundary_prob = att_prob[:, 4:6].sum(dim=-1, keepdim=True)  # [N, 1]
        interior_prob = att_prob[:, 2:4].sum(dim=-1, keepdim=True)  # [N, 1]
        boundary_type = torch.cat([boundary_prob, interior_prob], dim=-1)  # [N, 2]
        
        # 几何类型：直边(2,4) vs 曲边(3,5)
        straight_prob = att_prob[:, 2:3] + att_prob[:, 4:5]  # [N, 1]
        curved_edge_prob = att_prob[:, 3:4] + att_prob[:, 5:6]  # [N, 1]
        geometry_type = torch.cat([straight_prob, curved_edge_prob], dim=-1)  # [N, 2]
        
        return {
            'position_type': position_type,    # 新增：表面 vs 边缘
            'surface_type': surface_type,      # 平面 vs 曲面
            'boundary_type': boundary_type,    # 边界 vs 内部
            'geometry_type': geometry_type,    # 直边 vs 曲边
        }
    
    def forward(self, point, att_logits=None):
        """前向传播 - 兼容PTv3的Point对象接口"""
        # 从Point对象中提取特征
        x = point.feat  # [N, C]
        N, C = x.shape
        H = self.num_heads
        K = self.patch_size

        # ========== 阶段1: 属性预测 ==========
        att_prob = self.compute_attribute_distribution(x, att_logits)  # [N, 6]

        # 提取属性组
        att_groups = self.extract_attribute_groups(att_prob)
        position_type = att_groups['position_type']
        surface_type = att_groups['surface_type']
        boundary_type = att_groups['boundary_type']
        geometry_type = att_groups['geometry_type']

        # ========== 阶段2: Padding和Patch化 ==========
        num_patches = N // K
        if N % K != 0:
            pad_size = K - (N % K)
            x = F.pad(x, (0, 0, 0, pad_size))
            att_prob = F.pad(att_prob, (0, 0, 0, pad_size))
            position_type = F.pad(position_type, (0, 0, 0, pad_size))
            surface_type = F.pad(surface_type, (0, 0, 0, pad_size))
            boundary_type = F.pad(boundary_type, (0, 0, 0, pad_size))
            geometry_type = F.pad(geometry_type, (0, 0, 0, pad_size))
            N_padded = N + pad_size
            num_patches = N_padded // K
        else:
            N_padded = N

        x_patches = x.reshape(num_patches, K, C)
        att_prob_patches = att_prob.reshape(num_patches, K, self.num_att_classes)
        position_patches = position_type.reshape(num_patches, K, 2)
        surface_patches = surface_type.reshape(num_patches, K, 2)
        boundary_patches = boundary_type.reshape(num_patches, K, 2)
        geometry_patches = geometry_type.reshape(num_patches, K, 2)

        # ========== 阶段3: QKV投影 ==========
        qkv = self.qkv(x_patches)  # [N', K, 3*C]
        qkv = qkv.reshape(num_patches, K, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, N', H, K, C//H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ========== 阶段4: 自适应属性调制 ==========
        # 计算属性置信度（逐patch）
        att_confidence = att_prob.max(dim=-1)[0]  # [N_padded] - 每个点的最大属性概率
        att_confidence_patches = att_confidence.reshape(num_patches, K).mean(dim=1)  # [N'] - 每个patch的平均置信度

        # 将置信度扩展为 [N', 1, 1, 1] 用于广播
        adaptive_enhance = att_confidence_patches.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [N', 1, 1, 1]

        # 计算四种属性调制器
        position_patch_mean = position_patches.mean(dim=1)  # [N', 2]
        surface_patch_mean = surface_patches.mean(dim=1)    # [N', 2]
        boundary_patch_mean = boundary_patches.mean(dim=1)  # [N', 2]
        geometry_patch_mean = geometry_patches.mean(dim=1)  # [N', 2]

        position_mod = self.position_type_modulator(position_patch_mean)   # [N', H]
        surface_mod = self.surface_type_modulator(surface_patch_mean)      # [N', H]
        boundary_mod = self.boundary_type_modulator(boundary_patch_mean)   # [N', H]
        geometry_mod = self.geometry_type_modulator(geometry_patch_mean)   # [N', H]

        # 加权组合四种调制（边界权重更高）
        combined_mod = (
            position_mod * 1.0
            + surface_mod * 1.0
            + boundary_mod * 1.0
            + geometry_mod * 1.0
        ) / 4.0
        combined_mod = combined_mod.unsqueeze(2).unsqueeze(3)  # [N', H, 1, 1]

        # 应用自适应增强：置信度高的patch增强更强，置信度低的patch增强更弱
        # adaptive_enhance: [N', 1, 1, 1], combined_mod: [N', H, 1, 1]
        q = q * (1.0 + self.attribute_enhance * adaptive_enhance * combined_mod)
        k = k * (1.0 + self.attribute_enhance * adaptive_enhance * combined_mod)

        # ========== 阶段5: 标准注意力（无mask消融版）==========
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [N', H, K, K]

        # 【消融实验】：移除boundary_mask和surface_mask的计算和应用
        # 原始代码中的mask增强被注释掉，使用标准attention
        # boundary_mask = boundary_patches[:, :, 0:1] @ boundary_patches[:, :, 0:1].transpose(1, 2)
        # boundary_mask = boundary_mask.unsqueeze(1)  # [N', 1, K, K]
        #
        # plane_mask = surface_patches[:, :, 0:1] @ surface_patches[:, :, 0:1].transpose(1, 2)
        # curved_mask = surface_patches[:, :, 1:2] @ surface_patches[:, :, 1:2].transpose(1, 2)
        # surface_mask = (plane_mask + curved_mask).unsqueeze(1)  # [N', 1, K, K]
        #
        # combined_mask = boundary_mask * 1.5 + surface_mask * 0.5
        # attn = attn + self.attribute_enhance * combined_mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(num_patches, K, C)
        out = out.reshape(N_padded, C)

        # ========== 阶段6: 移除Padding ==========
        if N_padded != N:
            out = out[:N]
            x = x[:N]
            att_prob = att_prob[:N]
            surface_type = surface_type[:N]
            boundary_type = boundary_type[:N]

        # ========== 阶段7: 属性特征提取和融合 ==========
        surface_feat = self.surface_feature_extractor(x)  # [N, C//2]
        boundary_feat = self.boundary_feature_extractor(x)  # [N, C//2]
        combined_feat = torch.cat([surface_feat, boundary_feat], dim=-1)  # [N, C]

        gate_input = torch.cat([out, combined_feat], dim=-1)  # [N, 2*C]
        att_embed = self.attribute_embedding(att_prob)  # [N, C]
        gate_input = torch.cat([gate_input, att_embed], dim=-1)  # [N, 3*C]
        gate = self.fusion_gate(gate_input)  # [N, C]

        out = gate * (combined_feat + att_embed) + (1 - gate) * out

        # ========== 阶段8: 跨层特征传递 ==========
        cross_layer_input = torch.cat([out, att_prob], dim=-1)  # [N, C+6]
        out = self.cross_layer_projection(cross_layer_input)  # [N, C]

        out = self.proj(out)
        out = self.proj_drop(out)

        # ========== 阶段9: 更新Point对象 ==========
        point.feat = out
        point['attribute_prob'] = att_prob
        point['position_type'] = position_type
        point['surface_type'] = surface_type
        point['boundary_type'] = boundary_type
        point['geometry_type'] = geometry_type
        point['boundary_prob'] = boundary_type[:, 0:1]

        return point


class AttributeAwareBlock(nn.Module):
    """
    属性感知Block，包含AttributeAwareAttention和FFN
    """
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=512,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        num_att_classes=6,
        attribute_enhance=0.5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = AttributeAwareAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            num_att_classes=num_att_classes,
            attribute_enhance=attribute_enhance,
        )
        
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, channels),
            nn.Dropout(proj_drop),
        )
        
        # DropPath
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
    
    def forward(self, point, att_logits=None):
        # 注意力分支
        point_attn = self.attn(point, att_logits)
        point.feat = point.feat + self.drop_path(point_attn.feat - point.feat)
        
        # FFN分支
        feat = self.norm2(point.feat)
        point.feat = point.feat + self.drop_path(self.mlp(feat))
        
        return point


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
