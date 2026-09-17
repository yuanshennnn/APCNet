import os
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from torchmetrics.classification import (
    MulticlassAccuracy, 
    BinaryAccuracy, 
    BinaryJaccardIndex, 
    MulticlassJaccardIndex,
    BinaryAUROC)
import sys


_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataloader.point import PointCloudDataset
from utils.misc import seed_torch, print_num_params
import importlib.util
from engine.trainer_APCNet import PTV3Backbone_MultiTask_Ab


# 实例聚类和评估函数（优化版本）
def mean_shift_clustering_optimized(embeddings, bandwidth=0.5, max_iter=50, boundary_mask=None):
    """
    优化的Mean-shift聚类算法（向量化版本，支持边界约束）
    Args:
        embeddings: [N, D] embedding向量
        bandwidth: 带宽参数
        max_iter: 最大迭代次数
        boundary_mask: [N] bool tensor, True表示边界点（不参与聚类）
    Returns:
        cluster_labels: [N] 聚类标签
    """
    N, D = embeddings.shape
    device = embeddings.device
    
    # 初始化聚类标签
    cluster_labels = torch.full((N,), -1, dtype=torch.long, device=device)
    
    # 处理边界约束
    if boundary_mask is not None:
        # 只对非边界点进行聚类
        non_boundary_mask = ~boundary_mask
        if not non_boundary_mask.any():
            return cluster_labels
        
        # 只使用非边界点
        valid_embeddings = embeddings[non_boundary_mask]
        N_valid = valid_embeddings.shape[0]
    else:
        valid_embeddings = embeddings
        N_valid = N
    
    # 改进的采样策略：使用K-means++风格的初始化，确保覆盖所有区域
    max_seeds = min(2000, N_valid)
    if N_valid > max_seeds:
        # K-means++风格采样：优先选择距离已选点较远的点
        sample_indices = [torch.randint(0, N_valid, (1,), device=device).item()]
        min_distances = torch.full((N_valid,), float('inf'), device=device)
        
        for _ in range(max_seeds - 1):
            # 更新到最近已选点的距离
            last_center = valid_embeddings[sample_indices[-1]]
            distances = torch.norm(valid_embeddings - last_center, dim=1)
            min_distances = torch.minimum(min_distances, distances)
            
            # 以距离平方为概率采样下一个点
            probs = min_distances ** 2
            probs = probs / probs.sum()
            next_idx = torch.multinomial(probs, 1).item()
            sample_indices.append(next_idx)
        
        sample_indices = torch.tensor(sample_indices, device=device)
        seeds = valid_embeddings[sample_indices]
    else:
        seeds = valid_embeddings.clone()
        sample_indices = torch.arange(N_valid, device=device)
    
    centers = seeds.clone()
    n_seeds = centers.shape[0]
    
    # Mean-shift迭代（向量化）
    for iteration in range(max_iter):
        old_centers = centers.clone()
        
        # 批量处理：每次处理一批中心点
        batch_size = 256
        for i in range(0, n_seeds, batch_size):
            batch_centers = centers[i:i+batch_size]  # [batch, D]
            
            # 计算这批中心点到所有有效embedding的距离 [batch, N_valid]
            distances = torch.cdist(batch_centers, valid_embeddings, p=2)
            
            # 找到在bandwidth内的点
            weights = (distances < bandwidth).float()  # [batch, N_valid]
            
            # 更新中心为加权平均
            sum_weights = weights.sum(dim=1, keepdim=True)  # [batch, 1]
            sum_weights = torch.clamp(sum_weights, min=1.0)  # 避免除以0
            
            # [batch, N_valid] @ [N_valid, D] = [batch, D]
            new_centers = (weights @ valid_embeddings) / sum_weights
            centers[i:i+batch_size] = new_centers
        
        # 检查收敛
        center_shift = torch.norm(centers - old_centers, dim=1).mean()
        if center_shift < 1e-4:
            break
    
    # 合并相近的中心（向量化）
    # 计算中心之间的距离矩阵
    center_distances = torch.cdist(centers, centers, p=2)
    
    # 使用贪心方法合并，调整合并阈值为bandwidth * 0.8（更宽松）
    merge_threshold = bandwidth * 0.8
    unique_centers = []
    center_to_cluster = torch.full((n_seeds,), -1, dtype=torch.long, device=device)
    
    for i in range(n_seeds):
        if center_to_cluster[i] == -1:
            # 创建新簇
            cluster_id = len(unique_centers)
            # 找到所有与当前中心接近的中心
            close_centers = (center_distances[i] < merge_threshold).nonzero(as_tuple=True)[0]
            center_to_cluster[close_centers] = cluster_id
            unique_centers.append(centers[i])
    
    # 为有效点分配簇标签（找到最近的中心，但需要在bandwidth范围内）
    if len(unique_centers) > 0:
        unique_centers_tensor = torch.stack(unique_centers)  # [n_clusters, D]
        distances_to_centers = torch.cdist(valid_embeddings, unique_centers_tensor, p=2)  # [N_valid, n_clusters]
        
        # 找到最近的中心
        min_distances, valid_cluster_labels = distances_to_centers.min(dim=1)  # [N_valid]
        
        # 只分配距离在bandwidth * 1.5范围内的点，其他点标记为噪声
        noise_mask = min_distances > (bandwidth * 1.5)
        valid_cluster_labels[noise_mask] = -1
        
        # 将聚类结果映射回原始索引
        if boundary_mask is not None:
            cluster_labels[non_boundary_mask] = valid_cluster_labels
        else:
            cluster_labels = valid_cluster_labels
    
    return cluster_labels


def boundary_guided_region_growing(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    boundary_mask: torch.Tensor,
    k: int = 16,
    sim_thresh: float = None,
    bandwidth: float = 0.5,
) -> torch.Tensor:
    """简化版 BGPC：在点云 kNN 图上做 BFS 区域生长 + 边界约束。

    仅使用测试阶段的 embedding 和坐标信息，纯 Python 实现。
    - 只从未分配的非边界点作为种子开始生长；
    - 非边界邻居若与种子在 embedding 空间足够接近，则加入簇并继续扩张；
    - 边界点若满足相似度阈值，则挂接到簇，但不作为扩张节点。
    """
    device = embeddings.device
    M, D = embeddings.shape
    if M == 0:
        return torch.full((0,), -1, dtype=torch.long, device=device)

    # 若该语义类别下点数极少，特殊处理，避免 topk(k+1) 越界
    if M == 1:
        # 只有一个点：若为非边界点则单独成簇，否则标记为 -1
        labels = torch.full((1,), -1, dtype=torch.long, device=device)
        if not boundary_mask[0]:
            labels[0] = 0
        return labels

    # 默认相似度阈值与 bandwidth 同量级
    if sim_thresh is None:
        sim_thresh = bandwidth * 1.0

    # 构建基于坐标的 kNN 图
    # 注意：当 M 较小时，需要收缩 k，避免 topk(k+1) 越界
    k_eff = min(k, max(M - 1, 1))
    # dist_space: [M, M]
    dist_space = torch.cdist(coords, coords, p=2)
    # 每个点最近的 k_eff+1 个邻居（包含自身），去掉自身
    knn_idx = dist_space.topk(k_eff + 1, largest=False).indices[:, 1:]  # [M, k_eff]

    cluster_labels = torch.full((M,), -1, dtype=torch.long, device=device)
    next_cluster_id = 0

    for seed in range(M):
        # 仅从未分配的非边界点作为种子
        if boundary_mask[seed] or cluster_labels[seed] != -1:
            continue

        seed_feat = embeddings[seed]
        q = deque()
        q.append(seed)
        cluster_labels[seed] = next_cluster_id

        while q:
            cur = q.popleft()
            neighbors = knn_idx[cur]

            for nb in neighbors:
                nb = nb.item()
                if cluster_labels[nb] != -1:
                    continue

                # 与种子点的特征距离作为相似度判定
                dist_feat = torch.norm(embeddings[nb] - seed_feat, p=2)
                if dist_feat > sim_thresh:
                    continue

                if boundary_mask[nb]:
                    # 边界点：只挂接，不扩张
                    cluster_labels[nb] = next_cluster_id
                else:
                    # 非边界点：加入簇并继续 BFS
                    cluster_labels[nb] = next_cluster_id
                    q.append(nb)

        next_cluster_id += 1

    return cluster_labels


def compute_instance_iou(pred_labels, gt_labels, ignore_label=-1):
    """
    计算实例分割的IoU
    Args:
        pred_labels: [N] 预测的实例标签
        gt_labels: [N] ground truth实例标签
        ignore_label: 忽略的标签值
    Returns:
        mean_iou: 平均IoU
        matched_ious: 每个GT实例的最佳IoU列表
    """
    # 过滤掉ignore_label
    valid_mask = gt_labels != ignore_label
    pred_labels = pred_labels[valid_mask]
    gt_labels = gt_labels[valid_mask]
    
    if len(gt_labels) == 0:
        return 0.0, []
    
    # 获取唯一的GT实例
    unique_gt = torch.unique(gt_labels)
    unique_pred = torch.unique(pred_labels)
    
    if len(unique_gt) == 0 or len(unique_pred) == 0:
        return 0.0, []
    
    # 为每个GT实例找到最佳匹配的预测实例
    matched_ious = []
    
    for gt_id in unique_gt:
        if gt_id == ignore_label:
            continue
            
        gt_mask = (gt_labels == gt_id)
        best_iou = 0.0
        
        # 找到与当前GT实例重叠最多的预测实例
        for pred_id in unique_pred:
            pred_mask = (pred_labels == pred_id)
            
            # 计算IoU
            intersection = (gt_mask & pred_mask).sum().float()
            union = (gt_mask | pred_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou.item())
        
        matched_ious.append(best_iou)
    
    # 计算平均IoU
    if len(matched_ious) > 0:
        mean_iou = np.mean(matched_ious)
    else:
        mean_iou = 0.0
    
    return mean_iou, matched_ious


def compute_instance_iou_hungarian(pred_labels, gt_labels, ignore_label=-1):
    pred_np = pred_labels.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()

    # 仅根据 gt 过滤 ignore_label，对齐 Test_fine_tuning 中的行为
    valid_mask = gt_np != ignore_label
    pred_np = pred_np[valid_mask]
    gt_np = gt_np[valid_mask]

    # 若无有效点或无实例，直接返回 0
    if gt_np.size == 0:
        return 0.0, []

    unique_gt = np.unique(gt_np)
    unique_pred = np.unique(pred_np)

    if unique_gt.size == 0 or unique_pred.size == 0:
        return 0.0, []

    num_gt = unique_gt.shape[0]
    num_pred = unique_pred.shape[0]

    # IoU 矩阵：[num_gt, num_pred]
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)

    for gi, gt_id in enumerate(unique_gt):
        gt_mask = gt_np == gt_id
        for pi, pred_id in enumerate(unique_pred):
            pred_mask = pred_np == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                continue
            iou_matrix[gi, pi] = intersection / union

    # 与 Test_fine_tuning 中 calculate_multiple_iou 一致：
    # 直接在矩形 IoU 矩阵上做 Hungarian，匹配对数 = min(num_gt, num_pred)，
    # 仅在匹配到的实例对上取平均，不对未匹配实例显式加 0。
    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = [iou_matrix[r, c] for r, c in zip(row_ind, col_ind)]

    if len(matched_ious) == 0:
        mean_iou = 0.0
    else:
        mean_iou = float(np.mean(matched_ious))

    return mean_iou, matched_ious


def compute_instance_f1(pred_labels, gt_labels, ignore_label=-1, iou_threshold=0.5):
    pred_np = pred_labels.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()

    valid_mask = gt_np != ignore_label
    pred_np = pred_np[valid_mask]
    gt_np = gt_np[valid_mask]

    unique_gt = np.unique(gt_np) if gt_np.size > 0 else np.array([])
    unique_pred = np.unique(pred_np) if pred_np.size > 0 else np.array([])

    num_gt = unique_gt.shape[0]
    num_pred = unique_pred.shape[0]

    if num_gt == 0 and num_pred == 0:
        return 0.0, 0.0, 0.0

    if num_gt == 0 and num_pred > 0:
        return 0.0, 0.0, 0.0

    if num_gt > 0 and num_pred == 0:
        return 0.0, 0.0, 0.0

    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)

    for gi, gt_id in enumerate(unique_gt):
        gt_mask = gt_np == gt_id
        for pi, pred_id in enumerate(unique_pred):
            pred_mask = pred_np == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            if intersection == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                continue
            iou_matrix[gi, pi] = intersection / union

    TP = 0
    FP = 0
    FN = 0

    if num_gt <= num_pred:
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_preds = set()
        for r, c in zip(row_ind, col_ind):
            if num_pred == 0:
                break
            assigned_preds.add(c)
            iou = iou_matrix[r, c]
            if iou >= iou_threshold:
                TP += 1
            else:
                FN += 1
                FP += 1
        for j in range(num_pred):
            if j not in assigned_preds:
                FP += 1
    else:
        cost = 1.0 - iou_matrix
        pad_width = num_gt - num_pred
        pad_block = np.ones((num_gt, pad_width), dtype=cost.dtype)
        cost_padded = np.concatenate([cost, pad_block], axis=1)
        row_ind, col_ind = linear_sum_assignment(cost_padded)
        assigned_preds = set()
        for r, c in zip(row_ind, col_ind):
            if r >= num_gt:
                continue
            if c < num_pred:
                assigned_preds.add(c)
                iou = iou_matrix[r, c]
                if iou >= iou_threshold:
                    TP += 1
                else:
                    FN += 1
                    FP += 1
            else:
                FN += 1
        for j in range(num_pred):
            if j not in assigned_preds:
                FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return f1, precision, recall


def analyze_over_under_segmentation_counts(pred_labels, gt_labels, ignore_label=-1, iou_threshold=0.3):
    valid_mask = gt_labels != ignore_label
    pred_labels = pred_labels[valid_mask]
    gt_labels = gt_labels[valid_mask]

    if pred_labels.numel() == 0 or gt_labels.numel() == 0:
        return 0, 0, 0, 0

    unique_gt = torch.unique(gt_labels)
    unique_pred = torch.unique(pred_labels)

    total_gt = 0
    total_gt_split = 0
    total_pred = 0
    total_pred_merge = 0

    # 过分割：一个 GT 被多个预测实例覆盖
    for gt_id in unique_gt:
        if gt_id == ignore_label:
            continue
        gt_mask = gt_labels == gt_id
        if not gt_mask.any():
            continue
        total_gt += 1
        overlap_count = 0
        for pred_id in unique_pred:
            if pred_id == ignore_label:
                continue
            pred_mask = pred_labels == pred_id
            if not pred_mask.any():
                continue
            intersection = (gt_mask & pred_mask).sum().float()
            if intersection == 0:
                continue
            union = (gt_mask | pred_mask).sum().float()
            if union == 0:
                continue
            iou = intersection / union
            if iou > iou_threshold:
                overlap_count += 1
        if overlap_count > 1:
            total_gt_split += 1

    # 欠分割：一个预测实例覆盖多个 GT
    for pred_id in unique_pred:
        if pred_id == ignore_label:
            continue
        pred_mask = pred_labels == pred_id
        if not pred_mask.any():
            continue
        total_pred += 1
        overlap_count = 0
        for gt_id in unique_gt:
            if gt_id == ignore_label:
                continue
            gt_mask = gt_labels == gt_id
            if not gt_mask.any():
                continue
            intersection = (gt_mask & pred_mask).sum().float()
            if intersection == 0:
                continue
            union = (gt_mask | pred_mask).sum().float()
            if union == 0:
                continue
            iou = intersection / union
            if iou > iou_threshold:
                overlap_count += 1
        if overlap_count > 1:
            total_pred_merge += 1

    return total_gt, total_gt_split, total_pred, total_pred_merge


# ============================================================
# 进阶指标（对齐 engine/test.py 的 recognition / localization）
# ============================================================
feat_names = [
    'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
    'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
    'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
    'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
    'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
    'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round',
]


class FeatureInstance():
    """与 engine/test.py 中的 FeatureInstance 对齐，用于识别/定位指标。"""
    def __init__(self, name: int = None, faces: np.ndarray = None):
        self.name = name
        self.faces = faces


def print_class_metric(metric, names=feat_names):
    string = ''
    for i in range(len(metric)):
        cname = names[i] if i < len(names) else f'class_{i}'
        string += f'{cname}: {metric[i]:.4f}, '
    print(string)


def build_features_from_labels(inst_labels, sem_labels, num_sem_classes=25):
    """从 [N] 的实例 ID 和 [N] 的语义标签构造 FeatureInstance 列表。

    - 忽略 inst_id < 0 的点；
    - 每个实例的语义类别由其内部前景点（1..num_sem_classes-1）多数投票决定；
    - name 从 1-based 语义类映射到 0-based 索引（与 test.py 一致）。
    """
    inst_np = inst_labels.detach().cpu().numpy() if isinstance(inst_labels, torch.Tensor) else np.asarray(inst_labels)
    sem_np = sem_labels.detach().cpu().numpy() if isinstance(sem_labels, torch.Tensor) else np.asarray(sem_labels)

    features = []
    unique_ids = np.unique(inst_np)
    for iid in unique_ids:
        if iid < 0:
            continue
        mask = inst_np == iid
        if not mask.any():
            continue
        sems = sem_np[mask]
        valid = (sems >= 1) & (sems < num_sem_classes)
        if not valid.any():
            continue
        vals, counts = np.unique(sems[valid], return_counts=True)
        sem_id = int(vals[np.argmax(counts)])
        features.append(FeatureInstance(name=sem_id - 1, faces=np.where(mask)[0]))
    return features


def cal_recognition_performance(feature_list, label_list, num_classes=24):
    pred = np.zeros(num_classes, dtype=int)
    gt = np.zeros(num_classes, dtype=int)
    for feature in feature_list:
        if 0 <= feature.name < num_classes:
            pred[feature.name] += 1
    for label in label_list:
        if 0 <= label.name < num_classes:
            gt[label.name] += 1
    tp = np.minimum(gt, pred)
    return pred, gt, tp


def cal_localization_performance(feature_list, label_list, num_classes=24, iou_thresh=0.5):
    """与 test.py 中同名函数一致的匹配逻辑，但 IoU 阈值可配置。

    test.py 中 CAD B-Rep 面粒度使用 IoU>=1.0 严格匹配，点云通常放宽到 0.5。
    """
    pred = np.zeros(num_classes, dtype=int)
    gt = np.zeros(num_classes, dtype=int)
    for feature in feature_list:
        if 0 <= feature.name < num_classes:
            pred[feature.name] += 1
    for label in label_list:
        if 0 <= label.name < num_classes:
            gt[label.name] += 1

    tp = np.zeros(num_classes, dtype=int)
    found_lbl = np.zeros(len(label_list), dtype=bool)
    for feat in feature_list:
        pname = feat.name
        if not (0 <= pname < num_classes):
            continue
        best_iou = -1.0
        best_lbl = -1
        for lbl_i, lbl in enumerate(label_list):
            if found_lbl[lbl_i] or lbl.name != pname:
                continue
            inter = np.intersect1d(feat.faces, lbl.faces).size
            union = np.union1d(feat.faces, lbl.faces).size
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_lbl = lbl_i
        if best_lbl >= 0 and best_iou >= iou_thresh:
            found_lbl[best_lbl] = True
            tp[pname] += 1
    return pred, gt, tp


def eval_metric(pre, trul, tp):
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(pre > 0, tp / pre, 0.0)
        recall = np.where(trul > 0, tp / trul, 0.0)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    precision[trul == 0] = 1.0
    recall[trul == 0] = 1.0
    return precision, recall


def merge_small_clusters(pred_labels, embeddings, sem_labels=None, min_cluster_size=20):
    unique_ids = torch.unique(pred_labels)
    valid_ids = unique_ids[unique_ids >= 0]
    if valid_ids.numel() == 0:
        return pred_labels
    device = pred_labels.device
    N, D = embeddings.shape
    K = valid_ids.shape[0]
    cluster_sizes = torch.zeros(K, dtype=torch.long, device=device)
    cluster_centers = torch.zeros(K, D, device=device)
    cluster_sems = None
    if sem_labels is not None:
        cluster_sems = torch.full((K,), -1, dtype=torch.long, device=device)
    for i in range(K):
        cid = valid_ids[i]
        mask = pred_labels == cid
        size = mask.sum()
        cluster_sizes[i] = size
        if size > 0:
            cluster_centers[i] = embeddings[mask].mean(dim=0)
        if cluster_sems is not None:
            sem_cluster = sem_labels[mask]
            if sem_cluster.numel() > 0:
                vals, counts = torch.unique(sem_cluster, return_counts=True)
                cluster_sems[i] = vals[counts.argmax()]
    small_mask = cluster_sizes < min_cluster_size
    large_mask = cluster_sizes >= min_cluster_size
    if not small_mask.any() or not large_mask.any():
        return pred_labels
    small_indices = torch.nonzero(small_mask, as_tuple=True)[0]
    large_indices = torch.nonzero(large_mask, as_tuple=True)[0]
    new_pred = pred_labels.clone()
    for idx in small_indices:
        cid = valid_ids[idx]
        mask = pred_labels == cid
        if not mask.any():
            continue
        center_small = embeddings[mask].mean(dim=0, keepdim=True)
        candidate_indices = large_indices
        if cluster_sems is not None:
            sem_small = cluster_sems[idx]
            same_sem_mask = cluster_sems[large_indices] == sem_small
            if same_sem_mask.any():
                candidate_indices = large_indices[same_sem_mask]
        if candidate_indices.numel() == 0:
            continue
        centers_candidate = cluster_centers[candidate_indices]
        dists = torch.cdist(center_small, centers_candidate, p=2).squeeze(0)
        target_local = torch.argmin(dists)
        target_idx = candidate_indices[target_local]
        target_cid = valid_ids[target_idx]
        new_pred[mask] = target_cid
    return new_pred


def merge_boundary_fragments(
    pred_labels: torch.Tensor,
    embeddings: torch.Tensor,
    sem_labels: torch.Tensor | None = None,
    boundary_mask: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
    min_cluster_size: int = 80,
    center_dist_thresh: float = 0.9,
    boundary_ratio_thresh: float = 0.2,
    k_neighbors: int = 6,
):
    """在推理阶段基于边界信息做一次簇级合并，缓解过分割。

    思路：
      - 仅针对“尺寸较小且包含边界点”的簇（疑似碎片）；
      - 查看这些簇在边界上的点，其 kNN 邻居主要连接到哪些其他簇；
      - 若某个相邻簇与其同语义、embedding 中心距离足够近，且有足够多边界邻接，
        则将该小簇并入该大簇。
    """
    if coords is None or boundary_mask is None:
        return pred_labels

    device = embeddings.device
    N, D = embeddings.shape

    new_pred = pred_labels.clone()
    unique_ids = torch.unique(new_pred)
    valid_ids = unique_ids[unique_ids >= 0]
    if valid_ids.numel() == 0:
        return new_pred

    K = valid_ids.shape[0]
    cluster_sizes = torch.zeros(K, dtype=torch.long, device=device)
    cluster_centers = torch.zeros(K, D, device=device)
    cluster_sems = None
    if sem_labels is not None:
        cluster_sems = torch.full((K,), -1, dtype=torch.long, device=device)
    cluster_boundary_counts = torch.zeros(K, dtype=torch.long, device=device)

    # 统计每个簇的尺寸、中心、语义和边界点数量
    for i in range(K):
        cid = valid_ids[i]
        mask = new_pred == cid
        size = mask.sum()
        cluster_sizes[i] = size
        if size > 0:
            cluster_centers[i] = embeddings[mask].mean(dim=0)
            if cluster_sems is not None:
                sem_cluster = sem_labels[mask]
                if sem_cluster.numel() > 0:
                    vals, counts = torch.unique(sem_cluster, return_counts=True)
                    cluster_sems[i] = vals[counts.argmax()]
            cluster_boundary_counts[i] = (mask & boundary_mask).sum()

    # 仅在边界点上计算 kNN，用于统计簇间在边界上的邻接关系
    # coords 形状 [N, 3]
    if not boundary_mask.any():
        return new_pred

    boundary_idx = boundary_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
    coords_bd = coords[boundary_idx]  # [M, 3]
    if coords_bd.shape[0] == 0:
        return new_pred

    dist_bd_all = torch.cdist(coords_bd, coords, p=2)  # [M, N]
    k_eff = min(k_neighbors, max(N, 1))
    knn_idx = dist_bd_all.topk(k_eff, largest=False).indices  # [M, K]

    # 建立簇 id 到索引的映射
    cid_to_index = {int(cid.item()): idx for idx, cid in enumerate(valid_ids)}

    # adjacency[i, j] 统计：簇 i 的边界点，其 kNN 邻居落在簇 j 上的次数
    adjacency = torch.zeros((K, K), dtype=torch.long, device=device)
    M = boundary_idx.shape[0]
    for bi in range(M):
        i = boundary_idx[bi]
        ci = int(new_pred[i].item())
        if ci < 0:
            continue
        ci_idx = cid_to_index.get(ci, None)
        if ci_idx is None:
            continue
        neighbors = knn_idx[bi]
        for j in neighbors:
            cj = int(new_pred[j].item())
            if cj < 0 or cj == ci:
                continue
            cj_idx = cid_to_index.get(cj, None)
            if cj_idx is None:
                continue
            # 若提供语义标签，则仅在同语义簇之间考虑合并
            if cluster_sems is not None:
                sem_i = cluster_sems[ci_idx]
                sem_j = cluster_sems[cj_idx]
                if (sem_i >= 0) and (sem_j >= 0) and (sem_i != sem_j):
                    continue
            adjacency[ci_idx, cj_idx] += 1

    # 仅对“小且含边界”的簇尝试合并
    for idx in range(K):
        size = int(cluster_sizes[idx].item())
        bd_count = int(cluster_boundary_counts[idx].item())
        if size == 0 or bd_count == 0:
            continue
        if size >= min_cluster_size:
            continue

        neighbor_counts = adjacency[idx]  # [K]
        if neighbor_counts.max() == 0:
            continue

        ratios = neighbor_counts.float() / float(bd_count)
        candidate_mask = (ratios >= boundary_ratio_thresh) & (cluster_sizes > size)
        if not candidate_mask.any():
            continue

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
        centers_frag = cluster_centers[idx].unsqueeze(0)  # [1, D]
        centers_candidate = cluster_centers[candidate_indices]  # [C, D]
        dists = torch.cdist(centers_frag, centers_candidate, p=2).squeeze(0)  # [C]

        # 仅保留中心距离足够近的候选簇
        dist_mask = dists <= center_dist_thresh
        if not dist_mask.any():
            continue

        candidate_indices_valid = candidate_indices[dist_mask]
        dists_valid = dists[dist_mask]
        target_local = torch.argmin(dists_valid)
        target_idx = candidate_indices_valid[target_local]
        target_cid = valid_ids[target_idx]

        # 将当前小簇整体并入目标簇
        src_cid = valid_ids[idx]
        new_pred[new_pred == src_cid] = target_cid

    return new_pred


def cluster_and_evaluate_instances(inst_embeddings, inst_labels_gt, sem_labels_gt=None, 
                                  sem_labels_pred=None, att_labels_pred=None, att_labels_gt=None,
                                  points=None,
                                  bandwidth=0.5, use_boundary_constraint=True,
                                  use_semantic_grouping=True, use_bgpc_lite=True,
                                  num_sem_classes=25, k_neighbors=16,
                                  sim_thresh_factor=1.0, max_bgpc_points=None):
    """
    对实例embeddings进行聚类并评估（优化版，支持语义引导和边界约束）
    Args:
        inst_embeddings: [B, N, D] 实例embedding
        inst_labels_gt: [B, N] ground truth实例标签
        sem_labels_gt: [B, N] ground truth语义标签（仅用于评估）
        sem_labels_pred: [B, N] 预测的语义标签（用于语义分组和整体F1计算）
        att_labels_pred: [B, N] 预测的属性标签（用于边界检测，类别4和5为边界）
        att_labels_gt: [B, N] ground truth 属性标签（用于评估时忽略边界点）
        points: [B, N, 3] 点坐标，用于构建 kNN 图（BGPC-lite 时必需）
        bandwidth: Mean-Shift带宽参数
        use_boundary_constraint: 是否使用边界约束（推荐True）
        use_semantic_grouping: 是否使用语义分组，即先按语义类别分组再聚类（推荐True）
        use_bgpc_lite: 是否使用简化版 BGPC（基于 kNN 图 + 边界约束的区域生长）
        num_sem_classes: 语义类别数量（默认25）
        k_neighbors: 构图时每个点的近邻数量（默认16）
        sim_thresh_factor: 相似度阈值系数，最终阈值为 bandwidth * sim_thresh_factor
        max_bgpc_points: 单个语义簇使用 BGPC-lite 的最大点数阈值；若为 None 则不限制
    Returns:
        mean_inner_f1: 仅针对语义1-24（前景类别）的实例F1分数（忽略边界点）
        mean_boundary_f1: 仅在边界点上的实例F1分数
    """
    B, N, D = inst_embeddings.shape
    batch_f1_inner = []
    batch_f1_boundary = []
    total_gt_instances = 0
    total_gt_split_instances = 0
    total_pred_instances = 0
    total_pred_merge_instances = 0

    # ---------- 进阶指标累计器：recognition / localization（对齐 test.py） ----------
    num_rec_classes = num_sem_classes - 1  # 语义 1..num_sem_classes-1 → 索引 0..num_rec_classes-1
    rec_predictions = np.zeros(num_rec_classes, dtype=int)
    rec_truelabels = np.zeros(num_rec_classes, dtype=int)
    rec_truepositives = np.zeros(num_rec_classes, dtype=int)
    loc_predictions = np.zeros(num_rec_classes, dtype=int)
    loc_truelabels = np.zeros(num_rec_classes, dtype=int)
    loc_truepositives = np.zeros(num_rec_classes, dtype=int)
    
    # 在聚类阶段显示样本级进度条
    from tqdm import tqdm
    for b in tqdm(range(B), desc="Clustering samples", total=B):
        embeddings = inst_embeddings[b]  # [N, D]
        gt_labels = inst_labels_gt[b]    # [N]
        coords_b = points[b] if points is not None else None  # [N, 3]
        
        # 创建完整的预测标签数组，初始化为-1
        pred_labels = torch.full((N,), -1, dtype=torch.long, device=embeddings.device)

        # 边界mask：聚类与评估阶段都优先使用GT属性标签
        if att_labels_gt is not None and att_labels_gt.shape[0] == inst_embeddings.shape[0]:
            att_label_gt = att_labels_gt[b]  # [N]
            # 属性类别4（边界直边）和类别5（边界曲边）为边界点
            boundary_mask = (att_label_gt == 4) | (att_label_gt == 5)
            boundary_eval_mask = boundary_mask
        elif att_labels_pred is not None:
            att_label_pred = att_labels_pred[b]  # [N]
            boundary_mask = (att_label_pred == 4) | (att_label_pred == 5)
            boundary_eval_mask = boundary_mask
        else:
            # 如果没有属性信息，则不使用边界约束
            boundary_mask = torch.zeros(N, dtype=torch.bool, device=embeddings.device)
            boundary_eval_mask = boundary_mask
        
        current_cluster_id = 0  # 全局簇ID

        # 语义引导聚类：优先使用GT语义标签，其次使用预测语义标签
        sem_label_for_group = None
        if sem_labels_gt is not None and sem_labels_gt.shape[0] == inst_embeddings.shape[0]:
            sem_label_for_group = sem_labels_gt[b]
        elif sem_labels_pred is not None:
            sem_label_for_group = sem_labels_pred[b]
        
        # 【核心改进】：使用语义标签引导的聚类（此处仅使用Mean-Shift，不再启用BGPC-lite）
        if use_semantic_grouping and sem_label_for_group is not None:
            sem_label = sem_label_for_group  # [N]

            multi_inst_sem_flags = None
            if sem_labels_gt is not None and sem_labels_gt.shape[0] == inst_embeddings.shape[0]:
                sem_gt_cluster = sem_labels_gt[b]
                multi_inst_sem_flags = torch.zeros(num_sem_classes, dtype=torch.bool, device=embeddings.device)
                for semantic_class_tmp in range(1, num_sem_classes):
                    sem_mask_gt = (sem_gt_cluster == semantic_class_tmp)
                    if not sem_mask_gt.any():
                        continue
                    inst_ids_cls = gt_labels[sem_mask_gt]
                    valid_inst_cls = inst_ids_cls != -1
                    if not valid_inst_cls.any():
                        continue
                    unique_inst_cls = torch.unique(inst_ids_cls[valid_inst_cls])
                    if unique_inst_cls.numel() >= 2:
                        multi_inst_sem_flags[semantic_class_tmp] = True
            
            # 对每个语义类别（1到num_sem_classes-1）分别聚类，跳过类别0（边界）和类别num_sem_classes（背景/其他）
            for semantic_class in range(1, num_sem_classes):  # 类别1到24（如果num_sem_classes=25）
                if multi_inst_sem_flags is not None and not multi_inst_sem_flags[semantic_class]:
                    continue
                # 找到属于当前语义类别的点（基于预测结果）
                semantic_mask = (sem_label == semantic_class)
                
                if not semantic_mask.any():
                    continue  # 如果没有该类别的点，跳过
                
                # 提取该语义类别的embedding和坐标
                class_embeddings = embeddings[semantic_mask]
                class_boundary_mask = boundary_mask[semantic_mask]
                class_coords = coords_b[semantic_mask] if coords_b is not None else None
                num_points_class = class_embeddings.shape[0]
                
                # 对该语义类别内的点进行聚类
                try:
                    # 仅使用 Mean-Shift 聚类（纯均值漂移聚类），可选带边界约束
                    if use_boundary_constraint:
                        class_pred_labels = mean_shift_clustering_optimized(
                            class_embeddings,
                            bandwidth=bandwidth,
                            boundary_mask=class_boundary_mask
                        )
                    else:
                        class_pred_labels = mean_shift_clustering_optimized(
                            class_embeddings,
                            bandwidth=bandwidth
                        )
                    
                    # 将聚类结果映射回全局标签（更新全局簇ID）
                    valid_cluster_mask = class_pred_labels != -1
                    if valid_cluster_mask.any():
                        # 为该语义类别的簇分配全局唯一ID
                        unique_class_clusters = torch.unique(class_pred_labels[valid_cluster_mask])
                        for local_cluster_id in unique_class_clusters:
                            local_mask = class_pred_labels == local_cluster_id
                            # 找到在原始数组中的位置
                            class_indices = semantic_mask.nonzero(as_tuple=True)[0]
                            global_indices = class_indices[local_mask]
                            pred_labels[global_indices] = current_cluster_id
                            current_cluster_id += 1
                        
                        # 【改进】：将该语义类别中的边界点分配给最近的非边界簇
                        if use_boundary_constraint and class_boundary_mask.any():
                            boundary_indices_local = class_boundary_mask.nonzero(as_tuple=True)[0]
                            non_boundary_indices_local = (~class_boundary_mask).nonzero(as_tuple=True)[0]
                            
                            if len(non_boundary_indices_local) > 0 and len(boundary_indices_local) > 0:
                                # 计算边界点到非边界点的距离
                                boundary_embeddings = class_embeddings[boundary_indices_local]
                                non_boundary_embeddings = class_embeddings[non_boundary_indices_local]
                                non_boundary_labels = class_pred_labels[non_boundary_indices_local]
                                
                                # 找到每个边界点最近的非边界点
                                distances = torch.cdist(boundary_embeddings, non_boundary_embeddings, p=2)
                                nearest_non_boundary = distances.argmin(dim=1)
                                
                                # 将边界点分配给最近非边界点的簇（如果距离足够近）
                                for i, nearest_idx in enumerate(nearest_non_boundary):
                                    if distances[i, nearest_idx] < bandwidth * 1.0:  # 在bandwidth范围内
                                        boundary_label = non_boundary_labels[nearest_idx]
                                        if boundary_label != -1:
                                            # 映射回全局索引
                                            global_boundary_idx = class_indices[boundary_indices_local[i]]
                                            pred_labels[global_boundary_idx] = boundary_label
                    
                except Exception as e:
                    print(f"Clustering failed for semantic class {semantic_class}: {e}")
                    continue
        
        else:
            # 【原始方法】：不使用语义分组，所有点一起聚类
            # 过滤掉边界类别（0）和背景类别（num_sem_classes）
            if sem_label_for_group is not None:
                sem_label = sem_label_for_group
                valid_mask = (sem_label > 0) & (sem_label < num_sem_classes)
            else:
                valid_mask = torch.ones(N, dtype=torch.bool, device=embeddings.device)
            
            valid_embeddings = embeddings[valid_mask]
            valid_boundary_mask = boundary_mask[valid_mask]
            valid_coords = coords_b[valid_mask] if coords_b is not None else None
            
            if valid_embeddings.shape[0] > 0:
                try:
                    # 使用Mean-Shift聚类（不再启用BGPC-lite）
                    if use_boundary_constraint:
                        pred_labels_valid = mean_shift_clustering_optimized(
                            valid_embeddings,
                            bandwidth=bandwidth,
                            boundary_mask=valid_boundary_mask
                        )
                    else:
                        pred_labels_valid = mean_shift_clustering_optimized(
                            valid_embeddings,
                            bandwidth=bandwidth
                        )
                    
                    pred_labels[valid_mask] = pred_labels_valid
                except Exception as e:
                    print(f"Clustering failed: {e}")

        # 基于边界的后处理：合并疑似碎片簇
        if coords_b is not None and boundary_mask.any():
            sem_for_merge = None
            if sem_labels_gt is not None and sem_labels_gt.shape[0] == inst_embeddings.shape[0]:
                sem_for_merge = sem_labels_gt[b]
            elif sem_label_for_group is not None:
                sem_for_merge = sem_label_for_group

            # 使用 merge_boundary_fragments 内部的默认超参（更适合当前数据分布）
            pred_labels = merge_boundary_fragments(
                pred_labels,
                embeddings,
                sem_labels=sem_for_merge,
                boundary_mask=boundary_mask,
                coords=coords_b,
            )

        # 计算IoU（不再区分内部/边界），只评估语义1-24实例（语义25忽略）
        try:
            overall_f1 = None

            if sem_labels_gt is not None and sem_labels_gt.shape[0] == inst_embeddings.shape[0]:
                sem_gt = sem_labels_gt[b]  # [N]

                if sem_gt.shape[0] == gt_labels.shape[0]:
                    # ---------- Debug: 仅前几个 batch 打印语义和实例分布 ----------
                    if b < 3:
                        sem_unique, sem_counts = torch.unique(sem_gt, return_counts=True)
                        print(f"[Debug][batch {b}] sem_gt distribution (label:count): "
                              f"{list(zip(sem_unique.cpu().tolist(), sem_counts.cpu().tolist()))}")
                        valid_inst_mask = gt_labels != -1
                        if valid_inst_mask.any():
                            inst_unique, inst_counts = torch.unique(gt_labels[valid_inst_mask], return_counts=True)
                            print(f"[Debug][batch {b}] gt instance ids: {inst_unique.cpu().tolist()}")
                            print(f"[Debug][batch {b}] gt instance sizes: {inst_counts.cpu().tolist()}")
                        fg_mask = (sem_gt >= 1) & (sem_gt < num_sem_classes)
                        bg_mask_local = (sem_gt == num_sem_classes)
                        fg_points = int(fg_mask.sum().item())
                        bg_points = int(bg_mask_local.sum().item())
                        print(f"[Debug][batch {b}] num_fg_points={fg_points}, num_bg_points={bg_points}")

                    # ---------- 实例 F1（只评估语义1-24，忽略背景，不再区分边界/内部） ----------
                    overall_gt_labels = gt_labels.clone()
                    overall_pred_labels = pred_labels.clone()
                    bg_mask = (sem_gt == num_sem_classes)
                    if bg_mask.any():
                        overall_gt_labels[bg_mask] = -1
                        overall_pred_labels[bg_mask] = -1

                    # 点级别过滤：仅保留其语义类别下具有至少两个不同实例 ID 的点
                    # 对每个前景语义类别分别检查在该类别下 GT 实例 ID 的种类数
                    for semantic_class in range(1, num_sem_classes):
                        sem_mask_cls = (sem_gt == semantic_class)
                        if not sem_mask_cls.any():
                            continue
                        # 仅考虑当前语义类别下、尚未被忽略的点
                        cls_valid_mask = sem_mask_cls & (overall_gt_labels != -1)
                        if not cls_valid_mask.any():
                            continue
                        inst_ids_cls = overall_gt_labels[cls_valid_mask]
                        unique_inst_cls = torch.unique(inst_ids_cls)
                        # 若该语义类别内实例种类数 <= 1，则将该语义类别下所有有效点标记为忽略
                        if unique_inst_cls.numel() <= 1:
                            overall_gt_labels[cls_valid_mask] = -1
                            overall_pred_labels[cls_valid_mask] = -1

                    valid_inst_mask = overall_gt_labels != -1
                    if valid_inst_mask.any():
                        unique_inst = torch.unique(overall_gt_labels[valid_inst_mask])
                        if unique_inst.numel() <= 1:
                            continue
                    else:
                        continue
                    overall_f1, _, _ = compute_instance_f1(overall_pred_labels, overall_gt_labels, ignore_label=-1, iou_threshold=0.5)
                    gt_total, gt_split, pred_total, pred_merge = analyze_over_under_segmentation_counts(
                        overall_pred_labels, overall_gt_labels, ignore_label=-1, iou_threshold=0.3
                    )
                    total_gt_instances += gt_total
                    total_gt_split_instances += gt_split
                    total_pred_instances += pred_total
                    total_pred_merge_instances += pred_merge

                    # ---------- 进阶指标：recognition / localization ----------
                    sem_for_feat = sem_labels_pred[b] if sem_labels_pred is not None else sem_gt
                    pred_features = build_features_from_labels(
                        overall_pred_labels, sem_for_feat, num_sem_classes=num_sem_classes
                    )
                    gt_features = build_features_from_labels(
                        overall_gt_labels, sem_gt, num_sem_classes=num_sem_classes
                    )
                    p_rec, g_rec, tp_rec = cal_recognition_performance(
                        pred_features, gt_features, num_classes=num_rec_classes
                    )
                    rec_predictions += p_rec
                    rec_truelabels += g_rec
                    rec_truepositives += tp_rec
                    p_loc, g_loc, tp_loc = cal_localization_performance(
                        pred_features, gt_features, num_classes=num_rec_classes, iou_thresh=0.5
                    )
                    loc_predictions += p_loc
                    loc_truelabels += g_loc
                    loc_truepositives += tp_loc

            # 回退策略：如果未能使用语义标签限制到1-24，则在所有点上计算实例F1
            if overall_f1 is None:
                valid_inst_mask = gt_labels != -1
                if valid_inst_mask.any():
                    unique_inst = torch.unique(gt_labels[valid_inst_mask])
                    if unique_inst.numel() <= 1:
                        continue
                else:
                    continue
                overall_f1, _, _ = compute_instance_f1(pred_labels, gt_labels, ignore_label=-1, iou_threshold=0.5)
                gt_total, gt_split, pred_total, pred_merge = analyze_over_under_segmentation_counts(
                    pred_labels, gt_labels, ignore_label=-1, iou_threshold=0.3
                )
                total_gt_instances += gt_total
                total_gt_split_instances += gt_split
                total_pred_instances += pred_total
                total_pred_merge_instances += pred_merge

            # 为了与外部接口兼容，将整体实例F1同时记录到inner/boundary两个列表
            batch_f1_inner.append(overall_f1)
            batch_f1_boundary.append(overall_f1)
        except Exception as e:
            print(f"F1 computation failed for batch {b}: {e}")
            continue
    
    if len(batch_f1_inner) > 0:
        mean_inner_f1 = np.mean(batch_f1_inner)
    else:
        mean_inner_f1 = 0.0

    if len(batch_f1_boundary) > 0:
        mean_boundary_f1 = np.mean(batch_f1_boundary)
    else:
        mean_boundary_f1 = 0.0

    if total_gt_instances > 0:
        over_seg_ratio = total_gt_split_instances / total_gt_instances
    else:
        over_seg_ratio = 0.0

    if total_pred_instances > 0:
        under_seg_ratio = total_pred_merge_instances / total_pred_instances
    else:
        under_seg_ratio = 0.0

    print(f"Instance Over-Segmentation (GT被多个预测实例覆盖的比例): {over_seg_ratio:.4f}")
    print(f"Instance Under-Segmentation (预测实例覆盖多个GT的比例): {under_seg_ratio:.4f}")

    # ---------- 进阶指标输出（recognition / localization） ----------
    rec_stats = (rec_predictions, rec_truelabels, rec_truepositives)
    loc_stats = (loc_predictions, loc_truelabels, loc_truepositives)

    print('\n------------- recognition performance -------------')
    print('rec_pred', rec_predictions)
    print('rec_true', rec_truelabels)
    print('rec_trpo', rec_truepositives)
    rec_precision, rec_recall = eval_metric(rec_predictions, rec_truelabels, rec_truepositives)
    print('recognition Precision scores')
    print_class_metric(rec_precision)
    rec_precision_mean = rec_precision.mean()
    print('AVG recognition Precision:', rec_precision_mean)
    print('recognition Recall scores')
    print_class_metric(rec_recall)
    rec_recall_mean = rec_recall.mean()
    print('AVG recognition Recall:', rec_recall_mean)
    denom = rec_recall_mean + rec_precision_mean
    rec_F = (2 * rec_recall_mean * rec_precision_mean / denom) if denom > 0 else 0.0
    print('recognition F score:', rec_F)

    print('\n------------- localization performance -------------')
    print('loc_pred', loc_predictions)
    print('loc_true', loc_truelabels)
    print('loc_trpo', loc_truepositives)
    loc_precision, loc_recall = eval_metric(loc_predictions, loc_truelabels, loc_truepositives)
    print('localization Precision scores')
    print_class_metric(loc_precision)
    loc_precision_mean = loc_precision.mean()
    print('AVG localization Precision:', loc_precision_mean)
    print('localization Recall scores')
    print_class_metric(loc_recall)
    loc_recall_mean = loc_recall.mean()
    print('AVG localization Recall:', loc_recall_mean)
    denom = loc_recall_mean + loc_precision_mean
    loc_F = (2 * loc_recall_mean * loc_precision_mean / denom) if denom > 0 else 0.0
    print('localization F score:', loc_F)

    return mean_inner_f1, mean_boundary_f1, rec_stats, loc_stats, rec_F, loc_F


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    
    # 测试配置
    config = {
        "seed": 42,
        "num_sem_class": 25,
        "num_att_class": 6,
        "num_point": 1024,
        "device": 'cuda',
        "dataset": "dataset/data",
        "batch_size": 64,
        "dropout_rate": 0,
        "model_path": "output/2026_08_26_16_09_16/best_model.pth", 
        "bandwidth": 0.6,  # 聚类带宽参数，建议0.3-0.8
        "use_boundary_constraint": False,  # 使用边界约束：属性类别4和5为边界，不跨边界聚类
        "use_semantic_grouping": True,  # 使用语义分组：先按类别1-24分组，再在每组内聚类
        "eval_all_classes": True,  # 是否评估所有25类（包含背景），False则只评估1-24类
        "use_bgpc_lite": True,     # 使用 BGPC-lite：基于 kNN 图 + 边界约束的区域生长聚类
        "k_neighbors": 8,          # kNN 图的近邻数量（减小以加速）
        "sim_thresh_factor": 1.5,  # 特征相似度阈值系数（更严格，相当于阈值=bandwidth*1.5）
        "max_bgpc_points": 256,    # 单个语义簇使用 BGPC-lite 的最大点数（更偏向小物体）
    }
    
    print("=" * 60)
    print("Starting PTV3 Ab Multi-Task Test (Semantic / Instance / Attribute)")
    print("=" * 60)
    
    seed_torch(config['seed'])
    device = config['device']
    dataset = config['dataset']
    n_sem_classes = config['num_sem_class']
    n_att_classes = config['num_att_class']
    
    # 加载模型 
    print(f"Loading Ab model from: {config['model_path']}")
    model = PTV3Backbone_MultiTask_Ab(
        num_sem_classes=n_sem_classes,
        num_att_classes=n_att_classes,
        dropout_rate=config['dropout_rate'],
        attribute_enhance=0.5,
        apply_to_stages=[0,1,2]
    ).to(device)
    
    # 加载权重（兼容直接state_dict或包含model_state_dict的checkpoint）
    if os.path.exists(config['model_path']):
        checkpoint = torch.load(config['model_path'], map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {config['model_path']}")
        print("Using randomly initialized model for testing structure...")
    
    total_params = print_num_params(model)
    print(f"Total parameters: {total_params}")
    
    # 加载测试数据集
    print(f"\nLoading test dataset from: {dataset}")
    test_dataset = PointCloudDataset(
        root_dir=dataset,
        split='test',
        center_and_scale=False,
        normalize=False,
        random_rotate=False,
        augmentation_mode='none'
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=test_dataset._collate
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 评估指标：语义分割 + 属性识别
    test_sem_acc = MulticlassAccuracy(num_classes=n_sem_classes).to(device)
    test_sem_iou = MulticlassJaccardIndex(num_classes=n_sem_classes).to(device)
    test_att_acc = BinaryAccuracy().to(device)
    test_att_iou = BinaryJaccardIndex().to(device)
    
    print("\n" + "=" * 60)
    print("Starting Inference and Clustering Evaluation")
    print("=" * 60)
    
    with torch.no_grad():
        model.eval()
        
        # 用于收集实例聚类评估的数据
        test_inst_embeddings = []
        test_inst_labels = []
        test_sem_labels = []  # ground truth语义标签（用于评估）
        test_sem_preds = []   # 预测的语义标签（用于语义分组）
        test_att_preds = []   # 预测的属性标签（用于边界检测）
        test_att_gts = []     # ground truth 属性标签（用于评估时忽略边界点）
        test_points = []      # 点坐标（用于 BGPC-lite 构图）
        
        test_bar = tqdm(test_loader, desc="Testing")
        
        for data in test_bar:
            points = data["points"].to(device)          # [B, N, 6]
            sem_label = data["sem_labels"].to(device)   # [B, N]
            inst_label = data["ins_labels"].to(device)  # [B, N]
            att_label = data["att_labels"].to(device)   # [B, N]
            
            # 前向推理
            points_input = points.permute(0, 2, 1).contiguous()  # [B, 6, N]
            sem_pred, inst_pred, att_pred = model(batched_graph=None, points=points_input)
            
            # ===== 语义分割指标 =====
            # 数据标签范围：1-25 (类别1-24为有效类别，类别25为背景)
            # 模型输出：25个logits，对应类别0-24
            # 需要将标签从1-25映射到0-24
            # 【修复】：与验证代码保持一致的处理逻辑

            # 先过滤有效的语义标签范围（与验证代码一致）
            sem_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            sem_label_mapped = sem_label - 1  # 映射到0-24
            sem_pred_masked = sem_pred[sem_mask]
            sem_label_masked = sem_label_mapped[sem_mask]

            # 再次验证映射后的标签范围
            valid_sem_mask = (sem_label_masked >= 0) & (sem_label_masked < n_sem_classes)
            sem_pred_valid = sem_pred_masked[valid_sem_mask]
            sem_label_valid = sem_label_masked[valid_sem_mask]
            
            if sem_pred_valid.numel() > 0:
                test_sem_acc.update(sem_pred_valid, sem_label_valid)
                test_sem_iou.update(sem_pred_valid, sem_label_valid)
            
            # ===== 属性识别指标（与训练时相同的mask） =====
            att_mask = (sem_label >= 1) & (sem_label <= n_sem_classes)
            att_pred_masked = att_pred[att_mask]
            att_label_masked = att_label[att_mask].long()
            
            valid_att_mask = (att_label_masked >= 0) & (att_label_masked < n_att_classes)
            att_pred_valid = att_pred_masked[valid_att_mask]
            att_label_valid = att_label_masked[valid_att_mask]
            
            if att_pred_valid.numel() > 0:
                # 使用与训练相同的 log-sum-exp 方式，把6类属性聚合成边界/非边界二分类
                z_boundary = torch.logsumexp(att_pred_valid[:, 4:6], dim=1)
                z_nonboundary = torch.logsumexp(att_pred_valid[:, 0:4], dim=1)
                boundary_logit = z_boundary - z_nonboundary
                boundary_target = ((att_label_valid == 4) | (att_label_valid == 5)).int()
                boundary_prob = torch.sigmoid(boundary_logit)
                test_att_acc.update(boundary_prob, boundary_target)
                test_att_iou.update(boundary_prob, boundary_target)
            
            # ===== 收集实例嵌入用于聚类评估 =====
            inst_dim = getattr(model, 'instance_embedding_dim', inst_pred.shape[-1] - 3)
            test_inst_embeddings.append(inst_pred[:, :, :inst_dim].cpu())
            test_inst_labels.append(inst_label.cpu())
            test_sem_labels.append(sem_label.cpu())
            # 收集语义预测结果（映射回1-25的范围）
            # sem_pred: [B, N, 25], argmax在dim=2上取最大值类别
            sem_pred_labels = sem_pred.argmax(dim=2) + 1  # [B, N]，从0-24映射到1-25
            test_sem_preds.append(sem_pred_labels.cpu())
            # 收集属性预测结果（用于边界检测）
            # att_pred: [B, N, 6], argmax在dim=2上取最大值类别
            att_pred_labels = att_pred.argmax(dim=2)  # [B, N]，类别0-5
            test_att_preds.append(att_pred_labels.cpu())

            # 收集属性 GT（用于在评估阶段忽略边界点）
            test_att_gts.append(att_label.cpu())

            # 收集点坐标（仅使用 xyz）
            test_points.append(points[:, :, :3].cpu())
        
        # 计算各项分类指标
        mean_test_sem_acc = test_sem_acc.compute().item()
        mean_test_sem_iou = test_sem_iou.compute().item()
        mean_test_att_acc = test_att_acc.compute().item()
        mean_test_att_iou = test_att_iou.compute().item()
        
        # 执行实例聚类评估
        print("\n" + "=" * 60)
        print("Computing Instance Clustering F1 Score...")
        print("=" * 60)
        
        mean_test_inst_f1_fg = 0.0
        mean_test_inst_f1_all = 0.0
        if len(test_inst_embeddings) > 0:
            try:
                # 合并所有batch的数据
                all_inst_embeddings = torch.cat(test_inst_embeddings, dim=0)  # [Total_B, N, D]
                all_inst_labels = torch.cat(test_inst_labels, dim=0)          # [Total_B, N]
                all_sem_labels = torch.cat(test_sem_labels, dim=0)            # [Total_B, N] GT标签
                all_sem_preds = torch.cat(test_sem_preds, dim=0)              # [Total_B, N] 语义预测
                all_att_preds = torch.cat(test_att_preds, dim=0)              # [Total_B, N] 属性预测
                all_att_gts = torch.cat(test_att_gts, dim=0)                  # [Total_B, N] 属性 GT
                all_points = torch.cat(test_points, dim=0)                    # [Total_B, N, 3]
                
                print(f"Total samples for clustering: {all_inst_embeddings.shape[0]}")
                print(f"Embedding dimension: {all_inst_embeddings.shape[2]}")
                print(f"Bandwidth: {config['bandwidth']}")
                clustering_method = 'Mean-Shift (GT-guided, with optional boundary constraint)'
                print(f"Clustering method: {clustering_method}")
                print(f"Semantic classes: {n_sem_classes} (类别1-{n_sem_classes-1}用于实例分割)")
                print(f"Semantic grouping: {'Enabled (基于语义预测结果先分组再聚类)' if config['use_semantic_grouping'] else 'Disabled (所有点一起聚类)'}")
                print(f"Boundary constraint: {'Enabled (属性类别5和6为边界，不跨边界聚类)' if config['use_boundary_constraint'] else 'Disabled'}")
                
                # 执行聚类和评估 (使用GPU加速)
                all_inst_embeddings = all_inst_embeddings.to(device)
                all_inst_labels = all_inst_labels.to(device)
                all_sem_labels = all_sem_labels.to(device)
                all_sem_preds = all_sem_preds.to(device)
                all_att_preds = all_att_preds.to(device)
                all_att_gts = all_att_gts.to(device)
                all_points = all_points.to(device)
                
                start_time = time.time()
                (mean_test_inst_f1_inner,
                 mean_test_inst_f1_boundary,
                 rec_stats, loc_stats,
                 rec_F, loc_F) = cluster_and_evaluate_instances(
                    all_inst_embeddings,
                    all_inst_labels,
                    sem_labels_gt=all_sem_labels,
                    sem_labels_pred=all_sem_preds,  # 使用预测的语义标签进行语义分组
                    att_labels_pred=all_att_preds,  # 使用预测的属性标签进行边界检测
                    att_labels_gt=all_att_gts,      # 使用GT属性标签在评估时忽略边界点
                    points=all_points,
                    bandwidth=config['bandwidth'],
                    use_boundary_constraint=config['use_boundary_constraint'],
                    use_semantic_grouping=config['use_semantic_grouping'],
                    use_bgpc_lite=False,
                    num_sem_classes=n_sem_classes,
                    k_neighbors=config.get('k_neighbors', 16),
                    sim_thresh_factor=config.get('sim_thresh_factor', 1.0),
                    max_bgpc_points=config.get('max_bgpc_points', None),
                )
                clustering_time = time.time() - start_time

                print(f"Clustering completed in {clustering_time:.2f} seconds")
                print(f"Instance F1 (semantic 1-24, interior only): {mean_test_inst_f1_inner:.6f}")
                print(f"Instance F1 (semantic 1-24, boundary only): {mean_test_inst_f1_boundary:.6f}")
                print(f"Recognition F score: {rec_F:.6f}")
                print(f"Localization F score: {loc_F:.6f}")
            except Exception as e:
                print(f"Failed to compute instance F1: {e}")
                import traceback
                traceback.print_exc()
                mean_test_inst_f1 = 0.0
        else:
            print("No instance labels found in test dataset")
        
        # 输出测试结果
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Semantic Segmentation:")
        print(f"  - Accuracy:        {mean_test_sem_acc:.6f}")
        print(f"  - mIoU:            {mean_test_sem_iou:.6f}")
        print(f"\nInstance Segmentation:")
        print(f"  - F1 Score (semantic 1-24, all points): {mean_test_inst_f1_inner:.6f}")
        print(f"  - Recognition F score (%): {rec_F * 100:.4f}")
        print(f"  - Localization F score (%): {loc_F * 100:.4f}")
        print(f"\nAttribute Recognition:")
        print(f"  - Accuracy:        {mean_test_att_acc:.6f}")
        print(f"  - mIoU:            {mean_test_att_iou:.6f}")
        print("=" * 60)
