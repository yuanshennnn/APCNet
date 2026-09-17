"""
数据适配器：将BSPNet的点云数据转换为ASIN模型所需的面片格式
"""
import numpy as np
import os
import sys

# 添加BSPNet路径
sys.path.insert(0, '/root/github/BSPNet-main')

def convert_pointcloud_to_faces(points, sem_labels, ins_labels, att_labels=None, num_faces=64, points_per_face=32):
    """
    将点云数据转换为面片格式

    Args:
        points: (N, 6) - xyz + normals
        sem_labels: (N,) - semantic labels (1-25)
        ins_labels: (N,) - instance labels
        att_labels: (N,) - attribute labels (0-5) [可选]
        num_faces: 目标面片数量
        points_per_face: 每个面片的点数

    Returns:
        faces_data: (num_faces, points_per_face, 6)
        faces_sem_labels: (num_faces,)
        similarity_matrix: (num_faces, num_faces) - 相同实例为1，不同为0
        num_instances: int
        bottom_labels: (num_faces,) - 如果有att_labels，则为平面(att=0)识别；否则全0
    """
    N = points.shape[0]

    # 如果点数不足，进行上采样
    if N < num_faces * points_per_face:
        indices = np.random.choice(N, num_faces * points_per_face, replace=True)
        points = points[indices]
        sem_labels = sem_labels[indices]
        ins_labels = ins_labels[indices]
        if att_labels is not None:
            att_labels = att_labels[indices]
        N = num_faces * points_per_face

    # 将点云分成num_faces个面片
    indices = np.random.permutation(N)[:num_faces * points_per_face]
    selected_points = points[indices]
    selected_sem = sem_labels[indices]
    selected_ins = ins_labels[indices]
    if att_labels is not None:
        selected_att = att_labels[indices]

    # 重塑为面片格式
    faces_data = selected_points.reshape(num_faces, points_per_face, 6)
    faces_sem = selected_sem.reshape(num_faces, points_per_face)
    faces_ins = selected_ins.reshape(num_faces, points_per_face)
    if att_labels is not None:
        faces_att = selected_att.reshape(num_faces, points_per_face)

    # 每个面片的标签取多数投票
    faces_sem_labels = np.zeros(num_faces, dtype=np.int32)
    faces_ins_labels = np.zeros(num_faces, dtype=np.int32)
    if att_labels is not None:
        faces_att_labels = np.zeros(num_faces, dtype=np.int32)

    for i in range(num_faces):
        # 语义标签：取众数
        unique_sem, counts_sem = np.unique(faces_sem[i], return_counts=True)
        faces_sem_labels[i] = unique_sem[np.argmax(counts_sem)]

        # 实例标签：取众数
        unique_ins, counts_ins = np.unique(faces_ins[i], return_counts=True)
        faces_ins_labels[i] = unique_ins[np.argmax(counts_ins)]

        # 属性标签：取众数
        if att_labels is not None:
            unique_att, counts_att = np.unique(faces_att[i], return_counts=True)
            faces_att_labels[i] = unique_att[np.argmax(counts_att)]

    # 构建相似度矩阵（相同语义+相同实例为1，不同为0）
    similarity_matrix = np.zeros((num_faces, num_faces), dtype=np.float32)
    for i in range(num_faces):
        for j in range(num_faces):
            # 必须同时满足：相同语义 AND 相同实例 AND 实例非0
            if (faces_sem_labels[i] == faces_sem_labels[j] and
                faces_ins_labels[i] == faces_ins_labels[j] and
                faces_ins_labels[i] != 0):
                similarity_matrix[i, j] = 1.0

    # 计算实例数量（每个语义类别内独立计数）
    unique_sem_ids = np.unique(faces_sem_labels[faces_sem_labels != 0])
    num_instances = 0
    for sem_id in unique_sem_ids:
        sem_mask = (faces_sem_labels == sem_id)
        unique_instances_in_sem = np.unique(faces_ins_labels[sem_mask])
        unique_instances_in_sem = unique_instances_in_sem[unique_instances_in_sem != 0]
        num_instances += len(unique_instances_in_sem)

    # 将语义标签转换为one-hot编码 (假设25类)
    num_classes = 25
    faces_sem_onehot = np.zeros((num_faces, num_classes), dtype=np.float32)
    for i in range(num_faces):
        if 1 <= faces_sem_labels[i] <= num_classes:
            faces_sem_onehot[i, faces_sem_labels[i] - 1] = 1.0

    # 底面标签：使用属性标签att=0（平面）作为二分类目标
    if att_labels is not None:
        # att=0为平面，标记为1；其他为0
        bottom_labels = (faces_att_labels == 0).astype(np.float32)
    else:
        # 如果没有属性标签，全部设为0
        bottom_labels = np.zeros(num_faces, dtype=np.float32)

    return faces_data, faces_sem_onehot, similarity_matrix, num_instances, bottom_labels


def load_bspnet_data(data_root, split='train', num_samples=-1):
    """
    加载BSPNet数据集

    Args:
        data_root: 数据根目录
        split: 'train', 'val', or 'test'
        num_samples: 加载的样本数量，-1表示全部

    Returns:
        faces_data: (N, 64, 32, 6)
        sem_labels: (N, 64, 25)
        similarity_matrices: (N, 64, 64)
        num_instances: (N, 1)
        bottom_labels: (N, 64)
    """
    # 读取文件列表
    partition_file = os.path.join(data_root, 'processed_npys', f'{split}.txt')

    if not os.path.exists(partition_file):
        print(f"Warning: {partition_file} not found, using all files in points directory")
        points_dir = os.path.join(data_root, 'processed_npys', 'points')
        filenames = [f for f in os.listdir(points_dir) if f.endswith('.npy')]
    else:
        with open(partition_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

    if num_samples > 0:
        filenames = filenames[:num_samples]

    print(f"Loading {len(filenames)} samples from {split} split...")

    all_faces = []
    all_sem_labels = []
    all_similarity = []
    all_num_instances = []
    all_bottom_labels = []

    for idx, filename in enumerate(filenames):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(filenames)}...")

        try:
            # 加载点云数据
            if not filename.endswith('.npy'):
                filename = filename + '.npy'

            filepath = os.path.join(data_root, 'processed_npys', 'points', filename)
            data = np.load(filepath).astype(np.float32)

            # 解析数据 (N, 9): xyz, normals, sem, ins, att
            points = data[:, :6]  # xyz + normals
            sem_labels = data[:, 6].astype(np.int32)
            ins_labels = data[:, 7].astype(np.int32)
            att_labels = data[:, 8].astype(np.int32)  # 属性标签

            # 转换为面片格式（包含属性标签）
            faces_data, faces_sem, sim_matrix, num_inst, bottom_lab = convert_pointcloud_to_faces(
                points, sem_labels, ins_labels, att_labels
            )

            all_faces.append(faces_data)
            all_sem_labels.append(faces_sem)
            all_similarity.append(sim_matrix)
            all_num_instances.append(num_inst)
            all_bottom_labels.append(bottom_lab)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # 转换为numpy数组
    all_faces = np.array(all_faces, dtype=np.float32)
    all_sem_labels = np.array(all_sem_labels, dtype=np.float32)
    all_similarity = np.array(all_similarity, dtype=np.float32)
    all_num_instances = np.array(all_num_instances, dtype=np.int32).reshape(-1, 1)
    all_bottom_labels = np.array(all_bottom_labels, dtype=np.float32)

    print(f"Loaded {len(all_faces)} samples")
    print(f"  Faces shape: {all_faces.shape}")
    print(f"  Semantic labels shape: {all_sem_labels.shape}")
    print(f"  Similarity matrix shape: {all_similarity.shape}")
    print(f"  Num instances shape: {all_num_instances.shape}")
    print(f"  Bottom labels shape: {all_bottom_labels.shape}")

    return all_faces, all_sem_labels, all_similarity, all_num_instances, all_bottom_labels


if __name__ == '__main__':
    # 测试数据转换
    data_root = '/root/github/BSPNet-main/dataset/data'

    print("Testing data conversion...")
    faces, sem_labels, similarity, num_inst, bottom = load_bspnet_data(
        data_root, split='train', num_samples=10
    )

    print("\nSample statistics:")
    print(f"  Face coordinates range: [{faces[:,:,:,:3].min():.3f}, {faces[:,:,:,:3].max():.3f}]")
    print(f"  Face normals range: [{faces[:,:,:,3:].min():.3f}, {faces[:,:,:,3:].max():.3f}]")
    print(f"  Semantic label sum per sample: {sem_labels.sum(axis=1).mean():.3f}")
    print(f"  Similarity matrix range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    print(f"  Average num instances: {num_inst.mean():.2f}")
