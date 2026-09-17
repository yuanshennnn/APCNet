"""
预处理BSPNet数据并保存为HDF5格式
只需要运行一次，之后训练可以直接加载
"""
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/github/BSPNet-main')
sys.path.insert(0, '/root/github/ASIN-master-main')

from data_adapter import convert_pointcloud_to_faces

def preprocess_and_save(data_root, split='train', output_file=None):
    """
    预处理数据并保存为HDF5

    Args:
        data_root: BSPNet数据根目录
        split: 'train', 'val', or 'test'
        output_file: 输出HDF5文件路径
    """
    print(f"=" * 70)
    print(f"Preprocessing {split} split...")
    print(f"=" * 70)

    # 读取文件列表
    partition_file = os.path.join(data_root, 'processed_npys', f'{split}.txt')

    if not os.path.exists(partition_file):
        print(f"Warning: {partition_file} not found, using all files in points directory")
        points_dir = os.path.join(data_root, 'processed_npys', 'points')
        filenames = [f for f in os.listdir(points_dir) if f.endswith('.npy')]
    else:
        with open(partition_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

    print(f"Found {len(filenames)} samples")

    # 预分配数组
    num_samples = len(filenames)
    all_faces = np.zeros((num_samples, 64, 32, 6), dtype=np.float32)
    all_sem_labels = np.zeros((num_samples, 64, 25), dtype=np.float32)
    all_similarity = np.zeros((num_samples, 64, 64), dtype=np.float32)
    all_num_instances = np.zeros((num_samples, 1), dtype=np.int32)
    all_bottom_labels = np.zeros((num_samples, 64), dtype=np.float32)

    # 处理每个样本
    failed_count = 0
    for idx, filename in enumerate(tqdm(filenames, desc="Processing")):
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
            att_labels = data[:, 8].astype(np.int32)

            # 转换为面片格式
            faces_data, faces_sem, sim_matrix, num_inst, bottom_lab = convert_pointcloud_to_faces(
                points, sem_labels, ins_labels, att_labels
            )

            all_faces[idx] = faces_data
            all_sem_labels[idx] = faces_sem
            all_similarity[idx] = sim_matrix
            all_num_instances[idx] = num_inst
            all_bottom_labels[idx] = bottom_lab

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            failed_count += 1
            continue

    # 如果有失败的样本，移除空行
    if failed_count > 0:
        valid_indices = np.where(all_num_instances.flatten() > 0)[0]
        all_faces = all_faces[valid_indices]
        all_sem_labels = all_sem_labels[valid_indices]
        all_similarity = all_similarity[valid_indices]
        all_num_instances = all_num_instances[valid_indices]
        all_bottom_labels = all_bottom_labels[valid_indices]
        print(f"\nRemoved {failed_count} failed samples")

    print(f"\n" + "=" * 70)
    print(f"Preprocessing completed!")
    print(f"=" * 70)
    print(f"Successfully processed: {len(all_faces)} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Data shapes:")
    print(f"  Faces: {all_faces.shape}")
    print(f"  Semantic labels: {all_sem_labels.shape}")
    print(f"  Similarity matrix: {all_similarity.shape}")
    print(f"  Num instances: {all_num_instances.shape}")
    print(f"  Bottom labels: {all_bottom_labels.shape}")

    # 保存为HDF5
    if output_file is None:
        output_file = f'data/{split}data_bspnet.h5'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('data', data=all_faces, compression='gzip', compression_opts=4)
        hf.create_dataset('label', data=all_sem_labels, compression='gzip', compression_opts=4)
        hf.create_dataset('similarity', data=all_similarity, compression='gzip', compression_opts=4)
        hf.create_dataset('numofinstances', data=all_num_instances, compression='gzip', compression_opts=4)
        hf.create_dataset('bottomfacelabel', data=all_bottom_labels, compression='gzip', compression_opts=4)

    print(f"✓ Data saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print("=" * 70)

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess BSPNet data for ASIN training')
    parser.add_argument('--data_root', type=str,
                        default='/root/github/BSPNet-main/dataset/data',
                        help='BSPNet data root directory')
    parser.add_argument('--output_dir', type=str,
                        default='/root/github/ASIN-master-main/data',
                        help='Output directory for HDF5 files')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("BSPNet Data Preprocessing for ASIN")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # 预处理训练集
    print("\n[1/2] Processing training data...")
    train_file = preprocess_and_save(
        args.data_root,
        split='train',
        output_file=os.path.join(args.output_dir, 'traindata_bspnet.h5')
    )

    # 预处理验证集
    print("\n[2/2] Processing validation data...")
    val_file = preprocess_and_save(
        args.data_root,
        split='val',
        output_file=os.path.join(args.output_dir, 'validationdata_bspnet.h5')
    )

    print("\n" + "=" * 70)
    print("All preprocessing completed!")
    print("=" * 70)
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    print("\nYou can now run training with:")
    print("  python train_bspnet_fast.py")
    print("=" * 70)
