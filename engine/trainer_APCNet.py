import os
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassJaccardIndex)
import wandb
import sys
import warnings
import gc

# 过滤掉学习率调度器的deprecation警告
warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')


_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataloader.point import PointCloudDataset
from utils.misc import seed_torch, init_logger, print_num_params
from engine.attribute_attention import AttributeAwareAttention


# 导入 PointTransformerV3 相关模块
ptv3_path = os.path.join(_current_dir, 'PointTransformerV3-main')
if ptv3_path not in sys.path:
    sys.path.insert(0, ptv3_path)
from model_no_spconv import (
    PointTransformerV3,
    SerializedAttention,
)

# 数据格式说明：
# 输入数据：(x, y, z, nx, ny, nz, sem, ins, att)
# - sem: 语义标签，范围1-25（25类加工特征）
# - ins: 实例标签，相同sem下的不同实例编号（0,1,2,...）
# - att: 属性标签，范围0-5（平面0、曲面1、内部直边2、内部曲边3、边界直边4、边界曲边5）


class PTV3Backbone_MultiTask_Ab(nn.Module):
    """
    PointTransformerV3 多任务骨干网络（属性感知注意力版本）
    - 任务1: 语义分割（25类加工特征，使用属性增强）
    - 任务2: 实例分割（使用Pull-Push损失）
    - 任务3: 属性识别（6类几何属性：平面、曲面、内部直边、内部曲边、边界直边、边界曲边）
    
    架构设计：共享编码器-解码器 + 独立任务头 + 属性感知注意力
    - 共享主干提取通用特征
    - 在编码器和解码器的浅层（Stage 0,1）使用属性感知注意力
    - 属性感知注意力充分利用6类属性信息进行多维度调制
    - 语义分割头融合属性概率信息以提升性能
    - 适合中等规模数据集（~15k训练样本）
    
    核心改进：
    - 使用完整的6类属性信息（平面、曲面、内部直/曲边、边界直/曲边）
    - 多维度属性分组：位置、表面、边界、几何
    - 应用在浅层（Stage 0,1）以捕获局部几何特征
    - 跨层属性特征传递
    """
    def __init__(self, 
                 num_sem_classes: int = 25,
                 num_att_classes: int = 6,
                 dropout_rate: float = 0.3,
                 attribute_enhance: float = 0.5,
                 apply_to_stages: list = None):
        super().__init__()
        self.num_sem_classes = num_sem_classes
        self.num_att_classes = num_att_classes
        self.dropout_rate = dropout_rate
        self.attribute_enhance = attribute_enhance
        self.apply_to_stages = apply_to_stages if apply_to_stages is not None else [0, 1]  # 默认只用在浅层
        
        # ========== 共享编码器-解码器主干 ==========
        # 相比5M版本适度增强：6层→7层编码器
        # 通道数适度增加以支持多任务学习
        self.backbone = PointTransformerV3(
            in_channels=6,  # xyz + normals
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            
            # 编码器：7层（比单任务多1层）
            enc_depths=(2, 2, 2, 2, 2),  # 总共7层
            enc_channels=(32, 64, 128, 256, 512),  # 峰值512（比单任务256增强）
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(512, 512, 512, 512, 512),
            
            # 解码器：4层（保持轻量）
            dec_depths=(2, 2, 2, 2),
            dec_channels=(32, 64, 128, 256),
            dec_num_head=(2, 4, 8, 16),
            dec_patch_size=(512, 512, 512, 512),
            
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=dropout_rate,
            drop_path=0,  # 适度正则化
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=True,
            upcast_softmax=True,
            cls_mode=False,  # 使用解码器
        )
        
        # ========== 替换编码器和解码器指定stage为属性感知注意力 ==========
        print(f"Replacing attention layers with Attribute-Aware Attention in stages {self.apply_to_stages}...")
        self._replace_attention_layers()
        
        print(f"Attribute-Aware Attention applied with enhancement strength: {attribute_enhance}")
        
        # 共享特征维度
        shared_dim = 32

        # ========== 任务1: 语义分割头（优化版本：简化结构，减少冗余参数）==========
        # 【优化】去掉中间的Linear(128→128)层，节省16,768个参数
        # 单层128维隐层足够，与其他任务头保持一致
        self.semantic_head = nn.Sequential(
            nn.Linear(shared_dim, 128),  # 32 → 128
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_sem_classes),  # 128 → 25
        )

        # ========== 任务2: 实例分割头 ==========
        # 预测每个点的嵌入向量（用于Pull-Push损失）
        # 使用更高维度的嵌入空间（10D），直接使用共享特征
        self.instance_embedding_dim = 10  # 嵌入空间维度
        self.instance_head = nn.Sequential(
            nn.Linear(shared_dim, 128),  # 纯共享特征
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.instance_embedding_dim),  # 10D embedding
        )
        
        # ========== 任务3: 属性识别头 ==========
        # 预测几何属性（平面/曲面/直边/曲边/边界直边/边界曲边）
        self.attribute_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_att_classes),
        )
    
    def _replace_attention_layers(self):
        """
        替换编码器和解码器指定stage的序列化注意力为AttributeAwareAttention
        应用在编码器和解码器的指定stage（默认0,1,2）
        """
        # 编码器配置
        enc_depths = [2, 2, 2, 2, 2]
        enc_channels = [32, 64, 128, 256, 512]
        enc_num_heads = [2, 4, 8, 16, 32]
        enc_patch_size = [512, 512, 512, 512, 512]

        # 解码器配置
        dec_depths = [2, 2, 2, 2]
        dec_channels = [32, 64, 128, 256]
        dec_num_heads = [2, 4, 8, 16]
        dec_patch_size = [512, 512, 512, 512]
        
        replaced_count = 0
        
        # 替换编码器指定stage
        for stage_idx in self.apply_to_stages:
            if hasattr(self.backbone, 'enc'):
                stage_name = f'enc{stage_idx}'
                if hasattr(self.backbone.enc, stage_name):
                    stage = getattr(self.backbone.enc, stage_name)
                    # 遍历该stage的所有block
                    for block_idx in range(enc_depths[stage_idx]):
                        block_name = f'block{block_idx}'
                        if hasattr(stage, block_name):
                            block = getattr(stage, block_name)
                            # 查找并替换SerializedAttention
                            count = self._replace_serialized_attention_in_block(
                                block, 
                                enc_channels[stage_idx],
                                enc_num_heads[stage_idx],
                                enc_patch_size[stage_idx]
                            )
                            replaced_count += count
                    print(f"  Encoder stage {stage_idx} ({stage_name}): checked {enc_depths[stage_idx]} blocks")
        
        # 替换解码器指定stage
        for stage_idx in self.apply_to_stages:
            if hasattr(self.backbone, 'dec'):
                stage_name = f'dec{stage_idx}'
                if hasattr(self.backbone.dec, stage_name):
                    stage = getattr(self.backbone.dec, stage_name)
                    # 遍历该stage的所有block
                    for block_idx in range(dec_depths[stage_idx]):
                        block_name = f'block{block_idx}'
                        if hasattr(stage, block_name):
                            block = getattr(stage, block_name)
                            # 查找并替换SerializedAttention
                            count = self._replace_serialized_attention_in_block(
                                block,
                                dec_channels[stage_idx],
                                dec_num_heads[stage_idx],
                                dec_patch_size[stage_idx]
                            )
                            replaced_count += count
                    print(f"  Decoder stage {stage_idx} ({stage_name}): checked {dec_depths[stage_idx]} blocks")
        
        print(f"  Total SerializedAttention modules replaced: {replaced_count}")
    
    def _replace_serialized_attention_in_block(self, block, channels, num_heads, patch_size):
        """
        在一个Block中查找并替换SerializedAttention为AttributeAwareAttention
        
        Returns:
            int: 替换的模块数量
        """
        replaced = 0
        # 遍历block的所有子模块
        for name, module in block.named_children():
            if isinstance(module, SerializedAttention):
                # 创建新的AttributeAwareAttention
                new_attn = AttributeAwareAttention(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    qkv_bias=True,
                    qk_scale=None,
                    attn_drop=0.0,
                    proj_drop=self.dropout_rate,
                    num_att_classes=self.num_att_classes,
                    attribute_enhance=self.attribute_enhance,
                    use_attribute_prediction=False,
                )
                # 替换
                setattr(block, name, new_attn)
                replaced += 1
                print(f"    → Replaced {name} in {type(block).__name__}")
        return replaced
    
    def forward(self, batched_graph=None, points=None):
        """
        Args:
            points: [B, 6, N] - (xyz + normals)
        Returns:
            sem_pred: [B, N, num_sem_classes] - 语义分割logits
            inst_pred: [B, N, embedding_dim] - 实例嵌入
            att_pred: [B, N, num_att_classes] - 属性分类logits
        """
        assert points is not None, "points tensor is required"
        
        B, C, N = points.shape
        assert C == 6, f"Expected 6 channels (xyz + normals), got {C}"
        
        # 准备PTv3输入格式
        points_flat = points.permute(0, 2, 1).reshape(B * N, C)  # [B*N, 6]
        
        coord = points_flat[:, :3]  # [B*N, 3]
        feat = points_flat          # [B*N, 6]
        
        batch = torch.arange(B, device=points.device).repeat_interleave(N)
        offset = torch.arange(1, B + 1, device=points.device) * N
        
        data_dict = {
            'coord': coord,
            'feat': feat,
            'batch': batch,
            'offset': offset,
            'grid_size': 0.01,
        }
        
        # 共享主干前向传播
        point_output = self.backbone(data_dict)
        shared_features = point_output.feat  # [B*N, 32]

        # 【消融实验】不再提取和使用属性统计特征
        # 原代码：
        # if 'attribute_stats' in point_output.keys():
        #     att_stats = point_output['attribute_stats']
        # else:
        #     att_stats = torch.zeros(shared_features.shape[0], 19, ...)

        # 三个任务头
        # 任务1: 语义分割（消融版本：仅使用共享特征，不拼接属性统计）
        sem_logits = self.semantic_head(shared_features)  # [B*N, 25]

        # 任务2: 实例分割（使用纯共享特征）
        inst_output = self.instance_head(shared_features)    # [B*N, instance_embedding_dim]

        # 任务3: 属性识别（使用纯共享特征）
        att_logits = self.attribute_head(shared_features)    # [B*N, 6]

        # 重塑为 [B, N, *]
        sem_pred = sem_logits.view(B, N, self.num_sem_classes)
        inst_pred = inst_output.view(B, N, self.instance_embedding_dim)
        att_pred = att_logits.view(B, N, self.num_att_classes)

        return sem_pred, inst_pred, att_pred


def discriminative_loss(embeddings, instance_labels, semantic_labels,
                       attribute_labels=None, coords=None,
                       delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
    """实例分割的判别损失（Pull-Push Loss，带 size-aware + 轻量多连通实例加权）

    Args:
        embeddings: [B, N, D] - 点的嵌入向量
        instance_labels: [B, N] - 实例标签（相同sem下的实例编号）
        semantic_labels: [B, N] - 语义标签（1-25）
        attribute_labels: [B, N] or None - 属性标签（用于边界点降权）
        coords: [B, N, 3] or None - 点坐标（用于近似识别“空间上较分散/多连通”的实例并加大权重）
        delta_v: variance损失的margin
        delta_d: distance损失的margin
        alpha: variance损失权重
        beta: distance损失权重
        gamma: regularization损失权重

    Returns:
        total_loss, total_var_loss, total_dist_loss, total_reg_loss
    """
    batch_size, num_points, embedding_dim = embeddings.shape
    device = embeddings.device
    
    total_loss = embeddings.new_tensor(0.0)
    total_var_loss = embeddings.new_tensor(0.0)
    total_dist_loss = embeddings.new_tensor(0.0)
    total_reg_loss = embeddings.new_tensor(0.0)
    total_boundary_loss = embeddings.new_tensor(0.0)
    num_valid_batches = 0
    
    for b in range(batch_size):
        embed = embeddings[b]  # [N, D]
        inst_label = instance_labels[b]  # [N]
        sem_label = semantic_labels[b]   # [N]
        att_label = None
        if attribute_labels is not None:
            att_label = attribute_labels[b]
        coord_b = None
        if coords is not None:
            coord_b = coords[b]
        
        # 只对需要实例结构的语义标签（1-24）计算损失，屏蔽语义25（stock毛坯）
        valid_mask = (sem_label >= 1) & (sem_label < 25)
        
        if valid_mask.sum() == 0:
            continue
        
        multi_inst_mask = torch.zeros_like(valid_mask)
        sem_valid = sem_label[valid_mask]
        unique_sems = torch.unique(sem_valid)
        for sem_id in unique_sems:
            sem_val = int(sem_id.item())
            if sem_val < 1 or sem_val >= 25:
                continue
            sem_class_mask = (sem_label == sem_val) & valid_mask
            if not sem_class_mask.any():
                continue
            inst_ids_cls = inst_label[sem_class_mask]
            valid_inst_cls = inst_ids_cls != -1
            if not valid_inst_cls.any():
                continue
            unique_inst_cls = torch.unique(inst_ids_cls[valid_inst_cls])
            if unique_inst_cls.numel() >= 2:
                multi_inst_mask[sem_class_mask] = True

        if not multi_inst_mask.any():
            continue

        embed = embed[multi_inst_mask]
        inst_label = inst_label[multi_inst_mask]
        sem_label = sem_label[multi_inst_mask]
        if att_label is not None:
            att_label = att_label[multi_inst_mask]
        if coord_b is not None:
            coord_b = coord_b[multi_inst_mask]
        
        # 为每个语义类别分别计算实例损失
        unique_sems = torch.unique(sem_label)
        
        # variance: per-instance 平均（每个实例内部先平均，再跨实例等权平均）
        batch_var_numer = 0.0  # 实例级 variance 累积
        batch_var_denom = 0.0  # 实例数量
        batch_dist_loss = 0.0
        batch_reg_loss = 0.0
        num_clusters = 0
        
        # 记录每个实例聚类中心及其对应的语义标签，便于后续在同一语义内部计算 Push
        cluster_means = []
        cluster_sem_labels = []
        
        for sem_id in unique_sems:
            if sem_id == 0:  # 跳过背景
                continue
            
            # 获取该语义类别的所有点
            sem_mask = (sem_label == sem_id)
            sem_embed = embed[sem_mask]
            sem_inst = inst_label[sem_mask]
            sem_att = None
            if att_label is not None:
                sem_att = att_label[sem_mask]
            sem_coord = None
            if coord_b is not None:
                sem_coord = coord_b[sem_mask]
            
            # 获取该语义类别下的所有实例
            unique_insts = torch.unique(sem_inst)
            
            for inst_id in unique_insts:
                inst_mask = (sem_inst == inst_id)
                inst_embed = sem_embed[inst_mask]  # [M, D]
                inst_att = None
                if sem_att is not None:
                    inst_att = sem_att[inst_mask]
                inst_coord = None
                if sem_coord is not None:
                    inst_coord = sem_coord[inst_mask]
                
                if inst_embed.shape[0] == 0:
                    continue
                
                # 计算聚类中心
                cluster_mean = inst_embed.mean(dim=0)  # [D]
                cluster_means.append(cluster_mean)
                cluster_sem_labels.append(sem_id)
                multi_inst_weight = 1.0
                
                # Variance Loss (Pull): 让同一实例的点靠近中心
                var = torch.norm(inst_embed - cluster_mean, dim=1) - delta_v
                var = F.relu(var)  # hinge loss
                weights = torch.ones_like(var)

                # per-instance 累积：先在该实例内部按点平均，再跨实例等权平均
                inst_weight_sum = weights.sum()
                if inst_weight_sum > 0:
                    inst_var = (var.pow(2) * weights).sum() / inst_weight_sum
                    batch_var_numer += inst_var
                    batch_var_denom += 1.0
                
                # Regularization Loss: 防止中心点过大
                batch_reg_loss += torch.norm(cluster_mean)
                
                num_clusters += 1

        # 边界邻域对比损失：在边界点附近，同一实例的点拉近，不同实例的点推远
        # 这里的邻域搜索限制在“同一语义”内部，使边界对比更专注于区分同语义下的不同实例
        boundary_contrast_loss = embeddings.new_tensor(0.0)
        boundary_contrast_count = 0
        if (att_label is not None) and (coord_b is not None):
            boundary_mask_all = (att_label == 4) | (att_label == 5)
            if boundary_mask_all.any():
                # 只在存在边界点的语义类别内部进行 kNN 搜索
                boundary_sems = sem_label[boundary_mask_all]
                unique_boundary_sems = torch.unique(boundary_sems)
                for sem_id in unique_boundary_sems:
                    # 当前语义下的边界点和所有同语义点
                    sem_mask = (sem_label == sem_id)
                    boundary_mask = boundary_mask_all & sem_mask
                    if not boundary_mask.any():
                        continue
                    boundary_idx = boundary_mask.nonzero(as_tuple=False).squeeze(1)
                    neighbor_mask = sem_mask
                    neighbor_idx_all = neighbor_mask.nonzero(as_tuple=False).squeeze(1)

                    coords_bd = coord_b[boundary_idx]           # [M, 3]
                    coords_sem = coord_b[neighbor_idx_all]      # [N_sem, 3]
                    M = coords_bd.shape[0]
                    if M == 0 or coords_sem.shape[0] == 0:
                        continue

                    # 使用轻量级 kNN：在同一语义内部，边界点到所有同语义点的距离
                    dist_full = torch.cdist(coords_bd, coords_sem, p=2)  # [M, N_sem]
                    k_neighbors = min(4, coords_sem.shape[0])
                    knn_local = dist_full.topk(k_neighbors, largest=False).indices  # [M, K]

                    i_idx = torch.arange(M, device=device).unsqueeze(1).expand(-1, k_neighbors).reshape(-1)
                    j_idx_local = knn_local.reshape(-1)
                    anchor_idx = boundary_idx[i_idx]
                    neighbor_idx = neighbor_idx_all[j_idx_local]

                    emb_i = embed[anchor_idx]
                    emb_j = embed[neighbor_idx]

                    same_mask = (inst_label[anchor_idx] == inst_label[neighbor_idx]) & \
                                (sem_label[anchor_idx] == sem_label[neighbor_idx])
                    # 同一实例：embedding 应该接近（pull）
                    if same_mask.any():
                        diff_same = emb_i[same_mask] - emb_j[same_mask]
                        loss_same = diff_same.pow(2).sum(dim=1).mean()
                        boundary_contrast_loss = boundary_contrast_loss + loss_same
                        boundary_contrast_count += 1
                    # 不同实例（同语义下的不同实例）：embedding 应该拉开（push）
                    diff_mask = ~same_mask
                    if diff_mask.any():
                        diff_diff = emb_i[diff_mask] - emb_j[diff_mask]
                        dist_ij = diff_diff.norm(dim=1)
                        margin = 0.5
                        loss_diff = (margin - dist_ij).clamp(min=0).pow(2).mean()
                        boundary_contrast_loss = boundary_contrast_loss + loss_diff
                        boundary_contrast_count += 1
        if boundary_contrast_count > 0:
            boundary_contrast_loss = boundary_contrast_loss / boundary_contrast_count
        else:
            boundary_contrast_loss = embeddings.new_tensor(0.0)
        
        # Distance Loss (Push): 让同一语义下不同实例的中心互相远离
        if len(cluster_means) > 1:
            cluster_means = torch.stack(cluster_means)  # [K, D]
            cluster_sems = torch.stack(cluster_sem_labels).to(device)  # [K]
            num_centers = cluster_means.shape[0]

            # 计算所有中心点对之间的距离
            means_1 = cluster_means.unsqueeze(1).expand(num_centers, num_centers, embedding_dim)
            means_2 = cluster_means.unsqueeze(0).expand(num_centers, num_centers, embedding_dim)
            dist = torch.norm(means_1 - means_2, dim=2)  # [K, K]

            # 只在同一语义内部、且不同聚类中心之间计算 Push
            sems_1 = cluster_sems.unsqueeze(1).expand(num_centers, num_centers)
            sems_2 = cluster_sems.unsqueeze(0).expand(num_centers, num_centers)
            same_sem_mask = sems_1 == sems_2
            diff_center_mask = ~torch.eye(num_centers, device=device, dtype=torch.bool)
            pair_mask = same_sem_mask & diff_center_mask

            if pair_mask.any():
                dist = dist[pair_mask]
                # Hinge loss: 距离小于delta_d时才有损失
                dist = delta_d - dist
                dist = F.relu(dist)
                batch_dist_loss = dist.pow(2).mean()
            else:
                batch_dist_loss = embeddings.new_tensor(0.0)
        
        # 归一化（per-instance 平均）
        if batch_var_denom != 0:
            batch_var_loss = batch_var_numer / batch_var_denom
        else:
            batch_var_loss = embeddings.new_tensor(0.0)

        if num_clusters > 0:
            batch_reg_loss = batch_reg_loss / num_clusters
        else:
            batch_reg_loss = embeddings.new_tensor(0.0)
        
        # 加权求和（暂时关闭边界对比损失在总实例损失中的权重）
        batch_loss = (alpha * batch_var_loss
                      + beta * batch_dist_loss
                      + gamma * batch_reg_loss)
        
        # 累积到 batch 级统计
        total_loss += batch_loss
        total_var_loss += batch_var_loss
        total_dist_loss += batch_dist_loss
        total_reg_loss += batch_reg_loss
        total_boundary_loss += boundary_contrast_loss
        num_valid_batches += 1
    
    if num_valid_batches > 0:
        total_loss = total_loss / num_valid_batches
        total_var_loss = total_var_loss / num_valid_batches
        total_dist_loss = total_dist_loss / num_valid_batches
        total_reg_loss = total_reg_loss / num_valid_batches
        total_boundary_loss = total_boundary_loss / num_valid_batches
    
    return total_loss, total_var_loss, total_dist_loss, total_reg_loss, total_boundary_loss


class FocalLoss(nn.Module):
    """多分类 Focal Loss，用于语义分割（基于 softmax 的版本）。

    Args:
        gamma (float): focusing parameter，常用取值 2.0。
        alpha (Tensor | None): 按类别的权重系数，形状为 [C]，默认为 None 不加权。
        reduction (str): 'none' | 'mean' | 'sum'。
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """inputs: [N, C] logits, targets: [N] long labels"""
        if inputs.numel() == 0:
            return inputs.new_zeros(())

        # log_softmax + gather 真实类别
        logp = F.log_softmax(inputs, dim=1)  # [N, C]
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        p_t = logp_t.exp()  # [N]

        # (1 - p_t)^gamma * logp_t
        loss = - (1 - p_t) ** self.gamma * logp_t

        # 可选类别权重 alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def validate(model, val_loader, device, n_sem_classes, n_att_classes, 
             val_sem_acc, val_sem_iou, val_att_acc, val_att_iou, logger=None,
             inst_delta_v: float = 0.5, inst_delta_d: float = 1.5,
             inst_alpha: float = 1.0, inst_beta: float = 1.0, inst_gamma: float = 0.001):
    """
    多任务验证函数
    """
    model.eval()
    
    val_losses = []
    val_sem_losses = []
    val_inst_losses = []
    val_inst_var_losses = []
    val_inst_dist_losses = []
    val_inst_reg_losses = []
    val_inst_boundary_losses = []
    val_att_losses = []
    
    # 损失函数：语义分割使用带类别权重的 Focal Loss，属性识别保持交叉熵
    sem_loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

    # 属性识别验证损失：使用带类别权重的 CrossEntropy，与训练阶段保持一致
    att_class_weights_val = torch.ones(n_att_classes, device=device)
    att_boundary_weight = getattr(wandb.config, 'att_boundary_weight', 5.0)
    att_class_weights_val[4] = att_boundary_weight  # 边界直边
    att_class_weights_val[5] = att_boundary_weight  # 边界曲边
    att_loss_fn = nn.CrossEntropyLoss(weight=att_class_weights_val, reduction='mean')
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating"):
            points = data["points"].to(device, non_blocking=True)  # [B, N, 6]
            sem_label = data["sem_labels"].to(device, non_blocking=True)  # [B, N]
            ins_label = data["ins_labels"].to(device, non_blocking=True)  # [B, N]
            att_label = data["att_labels"].to(device, non_blocking=True)  # [B, N]
            
            # Forward pass
            points_input = points.permute(0, 2, 1).contiguous()  # [B, 6, N]
            sem_pred, inst_pred, att_pred = model(batched_graph=None, points=points_input)
            
            # ===== 任务1: 语义分割损失 =====
            sem_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            sem_label_mapped = sem_label - 1  # 映射到0-24
            sem_pred_masked = sem_pred[sem_mask]
            sem_label_masked = sem_label_mapped[sem_mask]
            
            valid_sem_mask = (sem_label_masked >= 0) & (sem_label_masked < n_sem_classes)
            sem_pred_valid = sem_pred_masked[valid_sem_mask]
            sem_label_valid = sem_label_masked[valid_sem_mask]
            
            loss_sem = sem_loss_fn(sem_pred_valid, sem_label_valid)

            # ===== 任务2: 实例分割损失 =====
            inst_embedding = inst_pred  # [B, N, D]

            loss_inst, loss_inst_var, loss_inst_dist, loss_inst_reg, loss_inst_boundary = discriminative_loss(
                inst_embedding, ins_label, sem_label,
                attribute_labels=att_label,
                coords=points[..., :3],
                delta_v=inst_delta_v,
                delta_d=inst_delta_d,
                alpha=inst_alpha,
                beta=inst_beta,
                gamma=inst_gamma,
            )
            
            # ===== 任务3: 属性识别损失 =====
            # 只对有效语义标签的点计算属性损失
            att_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            att_pred_masked = att_pred[att_mask]
            att_label_masked = att_label[att_mask].long()
            
            valid_att_mask = (att_label_masked >= 0) & (att_label_masked < n_att_classes)
            att_pred_valid = att_pred_masked[valid_att_mask]
            att_label_valid = att_label_masked[valid_att_mask]
            
            loss_att = att_loss_fn(att_pred_valid, att_label_valid)
            
            # 总损失（等权重）
            loss = loss_sem + loss_inst + loss_att
            
            val_losses.append(loss.detach())
            val_sem_losses.append(loss_sem.detach())
            val_inst_losses.append(loss_inst.detach())
            val_inst_var_losses.append(loss_inst_var.detach())
            val_inst_dist_losses.append(loss_inst_dist.detach())
            val_inst_reg_losses.append(loss_inst_reg.detach())
            val_inst_boundary_losses.append(loss_inst_boundary.detach())
            val_att_losses.append(loss_att.detach())
            
            # 更新指标
            val_sem_acc.update(sem_pred_valid, sem_label_valid)
            val_sem_iou.update(sem_pred_valid, sem_label_valid)
            val_att_acc.update(att_pred_valid, att_label_valid)
            val_att_iou.update(att_pred_valid, att_label_valid)
    
    # 计算平均指标
    mean_val_loss = torch.stack(val_losses).mean().item() if val_losses else 0.0
    mean_val_sem_loss = torch.stack(val_sem_losses).mean().item() if val_sem_losses else 0.0
    mean_val_inst_loss = torch.stack(val_inst_losses).mean().item() if val_inst_losses else 0.0
    mean_val_inst_var_loss = torch.stack(val_inst_var_losses).mean().item() if val_inst_var_losses else 0.0
    mean_val_inst_dist_loss = torch.stack(val_inst_dist_losses).mean().item() if val_inst_dist_losses else 0.0
    mean_val_inst_reg_loss = torch.stack(val_inst_reg_losses).mean().item() if val_inst_reg_losses else 0.0
    mean_val_inst_boundary_loss = torch.stack(val_inst_boundary_losses).mean().item() if val_inst_boundary_losses else 0.0
    mean_val_att_loss = torch.stack(val_att_losses).mean().item() if val_att_losses else 0.0
    
    mean_val_sem_acc = val_sem_acc.compute().item()
    mean_val_sem_iou = val_sem_iou.compute().item()
    mean_val_att_acc = val_att_acc.compute().item()
    mean_val_att_iou = val_att_iou.compute().item()
    
    return (mean_val_loss, mean_val_sem_loss, mean_val_inst_loss,
            mean_val_inst_var_loss, mean_val_inst_dist_loss, mean_val_inst_reg_loss,
            mean_val_inst_boundary_loss,
            mean_val_att_loss,
            mean_val_sem_acc, mean_val_sem_iou, mean_val_att_acc, mean_val_att_iou)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_API_KEY"] = '##################'  
    os.environ["WANDB_MODE"] = "offline"
    
    # start a new wandb run
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
        project="PTV3_MultiTask_Ab",
        config={
            "seed": 42,
            "num_sem_class": 25,  # 语义分割：25类加工特征
            "num_att_class": 6,   # 属性识别：6类几何属性
            "num_point": 1024,
            "device": 'cuda',
            "architecture": "BSPNet",
            "dataset": "dataset/data",
            "epochs": 100,
            "lr": 2.5e-3,  # 多任务学习建议较低学习率
            "weight_decay": 1e-2,
            "batch_size": 64,  # 多任务显存占用更大，适当减小
            "grad_accum_steps": 1,  # 梯度累计步数（1表示不使用梯度累计）
            "dropout_rate": 0.3, 
            "ema_decay_per_epoch": 1. / 2.,
            "resume_checkpoint": "none",
            "augmentation_mode": "none",  # 数据增强
            # 属性感知注意力参数
            "attribute_enhance": 0.5,  # 属性增强强度
            "apply_to_stages": [0,1,2],  # 应用到的stage（0,1,2 内全部block替换为属性注意力）
            # 多任务损失权重
            "loss_weight_sem": 1.0,    # 语义分割权重
            "loss_weight_inst": 0.5,   # 实例分割权重
            "loss_weight_att": 1,    # 属性识别权重（略高，因为边界类别重要）
            # 实例分割参数
            "inst_delta_v": 0.35,   # Pull margin
            "inst_delta_d": 1.5,   # Push margin
            "inst_alpha": 3.0,     # Variance loss weight
            "inst_beta": 0.5,      # Distance loss weight
            "inst_gamma": 0.001,   # Regularization weight
            # 属性识别边界类别权重增强
            "att_boundary_weight": 1.0,  # 边界类别（4,5）的损失权重
        }
    )
    
    # 初始化
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    dataset = wandb.config['dataset']
    n_sem_classes = wandb.config['num_sem_class']
    n_att_classes = wandb.config['num_att_class']
    
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"模型: PTV3Backbone_MultiTask_Ab (dropout={wandb.config['dropout_rate']})")
    print(f"属性感知注意力: 应用于Stage {wandb.config['apply_to_stages']}")
    print(f"属性增强强度: {wandb.config['attribute_enhance']}")
    print(f"任务1: 语义分割 - {n_sem_classes}类加工特征（属性增强）")
    print(f"任务2: 实例分割 - Pull-Push判别损失")
    print(f"任务3: 属性识别 - {n_att_classes}类（平面/曲面/内部直曲边/边界直曲边）")
    print(f"训练轮数: {wandb.config['epochs']} | Batch: {wandb.config['batch_size']}")
    print(f"学习率: {wandb.config['lr']} | 权重衰减: {wandb.config['weight_decay']}")
    
    # 使用多任务模型（属性感知注意力版本）
    model = PTV3Backbone_MultiTask_Ab(
        num_sem_classes=n_sem_classes,
        num_att_classes=n_att_classes,
        dropout_rate=wandb.config['dropout_rate'],
        attribute_enhance=wandb.config['attribute_enhance'],
        apply_to_stages=wandb.config['apply_to_stages']
    ).to(device)

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params
    print(f"{'='*70}")

    # 数据加载
    print(f"\n{'='*70}")
    print(f"加载数据集")
    print(f"{'='*70}")
    train_dataset = PointCloudDataset(
        root_dir=dataset, 
        split='train', 
        center_and_scale=False, 
        normalize=False, 
        random_rotate=False,
        augmentation_mode=wandb.config['augmentation_mode']
    )
    val_dataset = PointCloudDataset(
        root_dir=dataset, 
        split='val', 
        center_and_scale=False, 
        normalize=False, 
        random_rotate=False,
        augmentation_mode='none'
    )
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=wandb.config['batch_size'], 
        shuffle=True, 
        num_workers=16,
        pin_memory=True, 
        collate_fn=train_dataset._collate,

    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=wandb.config['batch_size'], 
        shuffle=False, 
        num_workers=16,
        pin_memory=True, 
        collate_fn=val_dataset._collate,
    )
    
    print(f"\n{'='*70}")
    print(f"训练配置")
    print(f"{'='*70}")
    print(f"数据集: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"DataLoader: workers=24/16, batch={wandb.config['batch_size']}")
    print(f"优化器: AdamW (lr={wandb.config['lr']}, wd={wandb.config['weight_decay']})")
    print(f"损失权重: Sem={wandb.config['loss_weight_sem']}, "
          f"Inst={wandb.config['loss_weight_inst']}, "
          f"Att={wandb.config['loss_weight_att']}")
    print(f"{'='*70}\n")
    
    # 初始化EMA
    iters = len(train_loader)
    ema_decay = wandb.config['ema_decay_per_epoch']**(1/iters)
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    # 梯度累计设置
    grad_accum_steps = int(getattr(wandb.config, 'grad_accum_steps', 1))
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    # 损失函数：语义分割使用带类别权重的 Focal Loss（权重来自训练集统计）
    sem_loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
    
    # 属性识别损失（边界类别增强权重）
    att_class_weights = torch.ones(n_att_classes, device=device)
    att_class_weights[4] = wandb.config['att_boundary_weight']  # 边界直边
    att_class_weights[5] = wandb.config['att_boundary_weight']  # 边界曲边
    att_loss_fn = nn.CrossEntropyLoss(weight=att_class_weights, reduction='mean')
    
    opt = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], 
                           weight_decay=wandb.config['weight_decay'])
    
    # Warmup设置
    warmup_epochs = 4  # 多任务学习建议更长warmup
    warmup_lr_start = wandb.config['lr'] * 0.1
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=wandb.config['epochs'] - warmup_epochs, eta_min=0
    )

    # 评估指标
    # 训练集
    train_sem_acc = MulticlassAccuracy(num_classes=n_sem_classes).to(device)
    train_sem_iou = MulticlassJaccardIndex(num_classes=n_sem_classes).to(device)
    train_att_acc = MulticlassAccuracy(num_classes=n_att_classes).to(device)
    train_att_iou = MulticlassJaccardIndex(num_classes=n_att_classes).to(device)
    
    # 验证集
    val_sem_acc = MulticlassAccuracy(num_classes=n_sem_classes).to(device)
    val_sem_iou = MulticlassJaccardIndex(num_classes=n_sem_classes).to(device)
    val_att_acc = MulticlassAccuracy(num_classes=n_att_classes).to(device)
    val_att_iou = MulticlassJaccardIndex(num_classes=n_att_classes).to(device)
    
    # 最佳指标跟踪
    best_val_loss = float('inf')
    best_val_sem_iou = 0.
    best_val_att_iou = 0.
    start_epoch = 0
    
    save_path = 'output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'log.txt'))
    
    # 加载checkpoint恢复训练
    if wandb.config['resume_checkpoint'] is not None and os.path.exists(wandb.config['resume_checkpoint']):
        logger.info(f'============= Resuming from checkpoint =============')
        logger.info(f'Checkpoint path: {wandb.config["resume_checkpoint"]}')
        checkpoint = torch.load(wandb.config['resume_checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            logger.info('EMA state restored')
        start_epoch = checkpoint['epoch'] + 1
        best_val_sem_iou = checkpoint.get('best_val_sem_iou', 0.)
        best_val_att_iou = checkpoint.get('best_val_att_iou', 0.)
        logger.info(f'Resumed from epoch {start_epoch}')
        logger.info(f'Best val sem IoU: {best_val_sem_iou:.6f}, Best val att IoU: {best_val_att_iou:.6f}')
        logger.info('====================================================')
        logger.info('')
    
    # 记录训练超参数
    logger.info('============= Training Hyperparameters =============')
    logger.info(f'Model Architecture: {wandb.config["architecture"]}')
    logger.info(f'Dataset: {wandb.config["dataset"]}')
    logger.info(f'Tasks: Semantic({n_sem_classes}), Instance, Attribute({n_att_classes})')
    logger.info(f'Number of Points: {wandb.config["num_point"]}')
    logger.info(f'Total Parameters: {wandb.config["total_params"]}')
    logger.info('----------- Multi-Task Settings -----------')
    logger.info(f'Loss Weights: Sem={wandb.config["loss_weight_sem"]}, '
                f'Inst={wandb.config["loss_weight_inst"]}, '
                f'Att={wandb.config["loss_weight_att"]}')
    logger.info(f'Instance Loss: delta_v={wandb.config["inst_delta_v"]}, '
                f'delta_d={wandb.config["inst_delta_d"]}')
    logger.info(f'Attribute Boundary Weight: {wandb.config["att_boundary_weight"]}x')
    logger.info('----------- Training Settings -----------')
    logger.info(f'Epochs: {wandb.config["epochs"]}')
    logger.info(f'Batch Size: {wandb.config["batch_size"]}')
    logger.info(f'Learning Rate: {wandb.config["lr"]}')
    logger.info(f'Warmup Epochs: {warmup_epochs}')
    logger.info(f'Weight Decay: {wandb.config["weight_decay"]}')
    logger.info(f'Dropout Rate: {wandb.config["dropout_rate"]}')
    logger.info(f'Augmentation Mode: {wandb.config["augmentation_mode"]}')
    logger.info('====================================================')
    logger.info('')
    
    for epoch in range(start_epoch, wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        
        # Warmup学习率调整
        if epoch < warmup_epochs:
            warmup_lr = warmup_lr_start + (wandb.config['lr'] - warmup_lr_start) * (epoch / warmup_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = warmup_lr
            logger.info(f'Warmup phase: epoch {epoch}/{warmup_epochs}, lr = {warmup_lr:.6f}')
        
        model.train()
        train_losses = []
        train_sem_losses = []
        train_inst_losses = []
        train_inst_var_losses = []
        train_inst_dist_losses = []
        train_inst_reg_losses = []
        train_inst_boundary_losses = []
        train_att_losses = []
        train_bar = tqdm(train_loader)
        current_avg_loss = 0.0
        
        for batch_idx, data in enumerate(train_bar):
            points = data["points"].to(device, non_blocking=True)  # [B, N, 6]
            sem_label = data["sem_labels"].to(device, non_blocking=True)  # [B, N]
            ins_label = data["ins_labels"].to(device, non_blocking=True)  # [B, N]
            att_label = data["att_labels"].to(device, non_blocking=True)  # [B, N]
            
            # 梯度清零：仅在一个累计周期的起始 batch 进行
            if batch_idx % grad_accum_steps == 0:
                opt.zero_grad(set_to_none=True)
            
            # Forward pass
            points_input = points.permute(0, 2, 1).contiguous()  # [B, 6, N]
            sem_pred, inst_pred, att_pred = model(batched_graph=None, points=points_input)
            
            # ===== 任务1: 语义分割损失 =====
            sem_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            sem_label_mapped = sem_label - 1
            sem_pred_masked = sem_pred[sem_mask]
            sem_label_masked = sem_label_mapped[sem_mask]
            
            valid_sem_mask = (sem_label_masked >= 0) & (sem_label_masked < n_sem_classes)
            sem_pred_valid = sem_pred_masked[valid_sem_mask]
            sem_label_valid = sem_label_masked[valid_sem_mask]
            
            loss_sem = sem_loss_fn(sem_pred_valid, sem_label_valid)

            # ===== 任务2: 实例分割损失 =====
            inst_embedding = inst_pred  # [B, N, D]

            loss_inst, loss_inst_var, loss_inst_dist, loss_inst_reg, loss_inst_boundary = discriminative_loss(
                inst_embedding, ins_label, sem_label,
                attribute_labels=att_label,
                coords=points[..., :3],
                delta_v=wandb.config['inst_delta_v'],
                delta_d=wandb.config['inst_delta_d'],
                alpha=wandb.config['inst_alpha'],
                beta=wandb.config['inst_beta'],
                gamma=wandb.config['inst_gamma']
            )
            
            # ===== 任务3: 属性识别损失（边界类别增强权重）=====
            att_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            att_pred_masked = att_pred[att_mask]
            att_label_masked = att_label[att_mask].long()
            
            valid_att_mask = (att_label_masked >= 0) & (att_label_masked < n_att_classes)
            att_pred_valid = att_pred_masked[valid_att_mask]
            att_label_valid = att_label_masked[valid_att_mask]
            
            # 仅使用加权多分类交叉熵作为属性损失（4/5类具有更高权重）
            loss_att_ce = att_loss_fn(att_pred_valid, att_label_valid)
            loss_att = loss_att_ce
            
            # 加权总损失
            loss = (wandb.config['loss_weight_sem'] * loss_sem + 
                   wandb.config['loss_weight_inst'] * loss_inst + 
                   wandb.config['loss_weight_att'] * loss_att)
            
            # 记录损失
            train_losses.append(loss.detach())
            train_sem_losses.append(loss_sem.detach())
            train_inst_losses.append(loss_inst.detach())
            train_inst_var_losses.append(loss_inst_var.detach())
            train_inst_dist_losses.append(loss_inst_dist.detach())
            train_inst_reg_losses.append(loss_inst_reg.detach())
            train_inst_boundary_losses.append(loss_inst_boundary.detach())
            train_att_losses.append(loss_att.detach())

            # 更新进度条
            lr = opt.param_groups[0]["lr"]
            if batch_idx % 10 == 0:
                current_avg_loss = torch.stack(train_losses).mean().item()
            info = "E:%d LR:%.6f Loss:%.3f (S:%.3f I:%.3f A:%.3f)" % (
                epoch, lr, current_avg_loss,
                loss_sem.item(), loss_inst.item(), loss_att.item()
            )
            train_bar.set_description(info)

            # Backward pass（梯度累计）
            (loss / grad_accum_steps).backward()

            # 仅在累计了 grad_accum_steps 个 mini-batch 或到达 epoch 末尾时更新参数
            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                opt.step()
                ema.update()
            
            # 减少指标更新频率
            if batch_idx % 5 == 0:
                train_sem_acc.update(sem_pred_valid.detach(), sem_label_valid)
                train_sem_iou.update(sem_pred_valid.detach(), sem_label_valid)
                train_att_acc.update(att_pred_valid.detach(), att_label_valid)
                train_att_iou.update(att_pred_valid.detach(), att_label_valid)
        
        # 计算训练集指标
        mean_train_loss = torch.stack(train_losses).mean().item()
        mean_train_sem_loss = torch.stack(train_sem_losses).mean().item()
        mean_train_inst_loss = torch.stack(train_inst_losses).mean().item()
        mean_train_inst_var_loss = torch.stack(train_inst_var_losses).mean().item()
        mean_train_inst_dist_loss = torch.stack(train_inst_dist_losses).mean().item()
        mean_train_inst_reg_loss = torch.stack(train_inst_reg_losses).mean().item()
        mean_train_inst_boundary_loss = torch.stack(train_inst_boundary_losses).mean().item()
        mean_train_att_loss = torch.stack(train_att_losses).mean().item()
        
        mean_train_sem_acc = train_sem_acc.compute().item()
        mean_train_sem_iou = train_sem_iou.compute().item()
        mean_train_att_acc = train_att_acc.compute().item()
        mean_train_att_iou = train_att_iou.compute().item()
        
        logger.info(f'Train - Loss: {mean_train_loss:.6f} '
                   f'(Sem: {mean_train_sem_loss:.6f}, '
                   f'Inst: {mean_train_inst_loss:.6f} '
                   f'[Var: {mean_train_inst_var_loss:.6f}, '
                   f'Dist: {mean_train_inst_dist_loss:.6f}, '
                   f'Reg: {mean_train_inst_reg_loss:.6f}, '
                   f'Bnd: {mean_train_inst_boundary_loss:.6f}], '
                   f'Att: {mean_train_att_loss:.6f})')
        logger.info(f'Train - Sem: Acc={mean_train_sem_acc:.6f}, IoU={mean_train_sem_iou:.6f}')
        logger.info(f'Train - Att: Acc={mean_train_att_acc:.6f}, IoU={mean_train_att_iou:.6f}')
        
        # 重置训练指标
        train_sem_acc.reset()
        train_sem_iou.reset()
        train_att_acc.reset()
        train_att_iou.reset()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # ========== 验证阶段 ==========
        logger.info('---------- Validation ----------')
        val_results = validate(
            model, val_loader, device, n_sem_classes, n_att_classes,
            val_sem_acc, val_sem_iou, val_att_acc, val_att_iou, logger,
            inst_delta_v=wandb.config['inst_delta_v'],
            inst_delta_d=wandb.config['inst_delta_d'],
            inst_alpha=wandb.config['inst_alpha'],
            inst_beta=wandb.config['inst_beta'],
            inst_gamma=wandb.config['inst_gamma'],
        )
        
        (mean_val_loss, mean_val_sem_loss, mean_val_inst_loss,
         mean_val_inst_var_loss, mean_val_inst_dist_loss, mean_val_inst_reg_loss,
         mean_val_inst_boundary_loss,
         mean_val_att_loss,
         mean_val_sem_acc, mean_val_sem_iou, mean_val_att_acc, mean_val_att_iou) = val_results
        
        logger.info(f'Val - Loss: {mean_val_loss:.6f} '
                   f'(Sem: {mean_val_sem_loss:.6f}, '
                   f'Inst: {mean_val_inst_loss:.6f} '
                   f'[Var: {mean_val_inst_var_loss:.6f}, '
                   f'Dist: {mean_val_inst_dist_loss:.6f}, '
                   f'Reg: {mean_val_inst_reg_loss:.6f}, '
                   f'Bnd: {mean_val_inst_boundary_loss:.6f}], '
                   f'Att: {mean_val_att_loss:.6f})')
        logger.info(f'Val - Sem: Acc={mean_val_sem_acc:.6f}, IoU={mean_val_sem_iou:.6f}')
        logger.info(f'Val - Att: Acc={mean_val_att_acc:.6f}, IoU={mean_val_att_iou:.6f}')
        
        # 重置验证指标
        val_sem_acc.reset()
        val_sem_iou.reset()
        val_att_acc.reset()
        val_att_iou.reset()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'train_sem_loss': mean_train_sem_loss,
            'train_inst_loss': mean_train_inst_loss,
            'train_inst_var_loss': mean_train_inst_var_loss,
            'train_inst_dist_loss': mean_train_inst_dist_loss,
            'train_inst_reg_loss': mean_train_inst_reg_loss,
            'train_inst_boundary_loss': mean_train_inst_boundary_loss,
            'train_att_loss': mean_train_att_loss,
            'train_sem_acc': mean_train_sem_acc,
            'train_sem_iou': mean_train_sem_iou,
            'train_att_acc': mean_train_att_acc,
            'train_att_iou': mean_train_att_iou,
            'val_loss': mean_val_loss,
            'val_sem_loss': mean_val_sem_loss,
            'val_inst_loss': mean_val_inst_loss,
            'val_inst_var_loss': mean_val_inst_var_loss,
            'val_inst_dist_loss': mean_val_inst_dist_loss,
            'val_inst_reg_loss': mean_val_inst_reg_loss,
            'val_inst_boundary_loss': mean_val_inst_boundary_loss,
            'val_att_loss': mean_val_att_loss,
            'val_sem_acc': mean_val_sem_acc,
            'val_sem_iou': mean_val_sem_iou,
            'val_att_acc': mean_val_att_acc,
            'val_att_iou': mean_val_att_iou,
            'learning_rate': opt.param_groups[0]['lr'],
            'best_val_loss': best_val_loss,
            'best_val_sem_iou': best_val_sem_iou,
            'best_val_att_iou': best_val_att_iou,
        })
        
        # 保存最佳模型（基于验证集语义IoU最大）
        if mean_val_sem_iou > best_val_sem_iou:
            best_val_sem_iou = mean_val_sem_iou
            logger.info(f'New best validation Semantic IoU: {best_val_sem_iou:.6f}, saving best model...')

            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_sem_iou': best_val_sem_iou,
                'best_val_att_iou': best_val_att_iou,
                'val_loss': mean_val_loss,
                'val_sem_acc': mean_val_sem_acc,
                'val_att_acc': mean_val_att_acc,
            }, best_model_path)
            logger.info(f'Best model saved to {best_model_path}')
        
        # 更新最佳属性IoU
        if mean_val_att_iou > best_val_att_iou:
            best_val_att_iou = mean_val_att_iou
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 保存checkpoint
        checkpoint_path = os.path.join(save_path, "checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_sem_iou': best_val_sem_iou,
            'best_val_att_iou': best_val_att_iou,
        }, checkpoint_path)
        
        # 每10个epoch额外保存
        if (epoch + 1) % 10 == 0:
            numbered_checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_sem_iou': best_val_sem_iou,
                'best_val_att_iou': best_val_att_iou,
            }, numbered_checkpoint_path)
            logger.info(f'Checkpoint saved to {numbered_checkpoint_path}')
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
    
    logger.info('============= Training Completed =============')
    logger.info(f'Best validation Loss: {best_val_loss:.6f}')
    logger.info(f'Best validation Semantic IoU: {best_val_sem_iou:.6f}')
    logger.info(f'Best validation Attribute IoU: {best_val_att_iou:.6f}')
    logger.info(f'Best model saved at: {os.path.join(save_path, "best_model.pth")}')
    logger.info('===============================================')

