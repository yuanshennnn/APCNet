import pathlib
import os

import torch
import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 normalize=True, 
                 center_and_scale=False, 
                 random_rotate=False, 
                 num_train_data=-1, 
                 transform=None,
                 augmentation_mode='none'):
        """
        Load the Point Cloud Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the point cloud. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the point cloud. Defaults to False.
            num_train_data (int, optional): Number of training examples to use. Defaults to -1 (all training examples will be used).
            transform (callable, optional): Transformation to apply to the data.
            augmentation_mode (str, optional): Data augmentation mode. Options: 'none', 'minimal', 'moderate', 'aggressive'. 
                                               Only applied to training set. Defaults to 'none'.
        """
        super().__init__()
        path = pathlib.Path(root_dir)
        self.path = path
        self.transform = transform
        self.random_rotate = random_rotate
        self.center_and_scale = center_and_scale
        self.normalize = normalize
        # 标记是否使用扁平目录(root_dir/points/*.npy)而不是 processed_npys 结构
        self.use_flat_points_dir = False
        assert split in ("train", "val", "test", "all")
        
        # 数据增强已禁用
        self.split = split
        self.augmentation = None
        if split == 'train' and augmentation_mode != 'none':
            print(f"  ⚠️  数据增强模块未安装，跳过增强（augmentation_mode={augmentation_mode}）")

        # load data partition from train.txt / val.txt / test.txt if存在；
        # 否则回退到直接遍历 root_dir/points/*.npy（全部视为该split）
        partition_dir = os.path.join(root_dir, 'processed_npys')
        train_txt = os.path.join(partition_dir, 'train.txt')
        val_txt = os.path.join(partition_dir, 'val.txt')
        test_txt = os.path.join(partition_dir, 'test.txt')

        if split == "all" and os.path.exists(train_txt) and os.path.exists(val_txt) and os.path.exists(test_txt):
            with open(train_txt, 'r') as f:
                train_filelist = [x.strip() for x in f.readlines()]
            with open(val_txt, 'r') as f:
                valid_filelist = [x.strip() for x in f.readlines()]
            with open(test_txt, 'r') as f:
                test_filelist = [x.strip() for x in f.readlines()]
            assert len(train_filelist) != 0 and len(valid_filelist) != 0 and len(test_filelist) != 0, \
                'have empty partition file'
            split_filelist = train_filelist + valid_filelist + test_filelist
        elif split != "all" and os.path.exists(os.path.join(partition_dir, split + '.txt')):
            with open(os.path.join(partition_dir, split + '.txt'), 'r') as f:
                split_filelist = f.readlines()
        else:
            # 回退模式：没有 processed_npys/xxx.txt，直接使用 root_dir/points/*.npy
            points_dir = path.joinpath('points')
            if not points_dir.exists():
                raise FileNotFoundError(
                    f"Neither partition file {os.path.join(partition_dir, split + '.txt')} "
                    f"nor flat points directory {points_dir} exist."
                )
            self.use_flat_points_dir = True
            split_filelist = [p.name for p in points_dir.glob('*.npy')]

        if split == "train":
            split_filelist = split_filelist[:num_train_data]
            
        assert len(split_filelist) != 0, 'have empty partition file'
        split_filelist = [x.strip() for x in split_filelist]
        
        # 存储文件路径而不是预加载所有数据（减少内存占用，提高启动速度）
        self.filenames = split_filelist
        
        # 强制使用按需加载模式（预加载模式有bug会卡住）
        self.load_in_memory = False
        self.data = None
        
        # 注释掉预加载逻辑，直接跳过
        # if len(split_filelist) < 1000:
        #     self.load_in_memory = True
        #     self.data = []
        
        # 只在预加载模式下加载数据
        if self.load_in_memory:
            print(f"预加载模式：加载 {len(split_filelist)} 个样本到内存...")
        else:
            # 按需加载模式：从磁盘读取
            # 按需加载模式：静默验证文件（不打印详细信息）
            valid_files = []
            for filename in split_filelist:
                # 确保文件名有.npy后缀
                if not filename.endswith('.npy'):
                    filename_with_ext = filename + '.npy'
                else:
                    filename_with_ext = filename

                if self.use_flat_points_dir:
                    data_path = path.joinpath("points", filename_with_ext)
                else:
                    data_path = path.joinpath("processed_npys", "points", filename_with_ext)
                if data_path.exists():
                    valid_files.append(filename)  # 保存原始文件名（可能没有.npy）
            self.filenames = valid_files
            return
        
        for filename in split_filelist:
            try:
                # Try to load complete data with new format (x, y, z, xn, yn, zn, sem, ins, att)
                data_path = path.joinpath("processed_npys", "points", filename)
                full_data = np.load(str(data_path)).astype(np.float32)
                
                # Check data format and handle both old and new formats
                if full_data.ndim != 2:
                    print(f"Error: {filename} has incorrect dimensions {full_data.ndim}, expected 2D array")
                    continue
                
                if full_data.shape[1] == 9:
                    # New format: (x, y, z, xn, yn, zn, sem, ins, att)
                    # 注意：att 为 0-5 的多类属性标签（平面/曲面/内部直曲边/边界直曲边），不能转为 bool
                    points_data = full_data[:, :6]  # (x, y, z, xn, yn, zn)
                    sem_labels = full_data[:, 6].astype(np.uint8)  # sem
                    ins_labels = full_data[:, 7].astype(np.uint8)  # ins（语义内部编号）
                    att_labels = full_data[:, 8].astype(np.uint8)  # att (0-5 attribute labels)
                    
                elif full_data.shape[1] == 6:
                    # Old format: only (x, y, z, xn, yn, zn), need to load labels separately
                    points_data = full_data
                    
                    # Load labels from separate file
                    labels_path = path.joinpath("processed_npys", "labels", filename)
                    raw_labels = np.load(str(labels_path), allow_pickle=True)
                    
                    # Convert structured array to dictionary
                    if raw_labels.dtype.names is not None:
                        labels_data = {}
                        for field in raw_labels.dtype.names:
                            labels_data[field] = raw_labels[field]
                    else:
                        # Handle flat array format
                        num_points = points_data.shape[0]
                        if raw_labels.size == num_points * 3:
                            labels_data = {
                                'sem': raw_labels[:num_points],
                                'ins': raw_labels[num_points:num_points*2],
                                'att': raw_labels[num_points*2:]
                            }
                        else:
                            print(f"Error: {filename} labels have incorrect size {raw_labels.size}")
                            continue
                    
                    # Extract labels
                    sem_labels = np.array(labels_data.get('sem', labels_data.get('Sem', np.zeros(points_data.shape[0]))), dtype=np.uint8)
                    ins_labels = np.array(labels_data.get('ins', labels_data.get('Ins', np.zeros(points_data.shape[0]))), dtype=np.uint8)
                    
                    # Try to get attachment labels (try different possible field names)
                    # att_data 期望为 0-5 多类属性；若为布尔则会自动变为 0/1
                    att_data = labels_data.get('att', labels_data.get('Att', labels_data.get('bot', labels_data.get('Bot', np.zeros(points_data.shape[0], dtype=np.uint8)))))
                    att_labels = np.array(att_data, dtype=np.uint8)
                else:
                    print(f"Error: {filename} has incorrect shape {full_data.shape}, expected (N, 6) or (N, 9)")
                    continue

                # 将 (sem, ins) 组合为全局唯一的实例 ID：sem * K + ins
                # 这里不改变 sem_labels，本身仍在 1-25 范围内，仅在 ins_labels 中编码语义信息
                sem_int = sem_labels.astype(np.int32)
                ins_int = ins_labels.astype(np.int32)
                global_factor = 1000  # 假定单一语义下实例数远小于该值
                ins_labels = sem_int * global_factor + ins_int
                
                # Ensure point cloud has correct shape (N, 6) - 3 coords + 3 normals
                assert points_data.shape[1] == 6, f"Point cloud should have 6 features (3 coords + 3 normals), got {points_data.shape[1]}"
                
                # 检查原始数据是否包含NaN或Inf（在预处理之前）
                if np.isnan(points_data).any() or np.isinf(points_data).any():
                    nan_count = np.isnan(points_data).sum()
                    inf_count = np.isinf(points_data).sum()
                    print(f"Warning: {filename} contains NaN ({nan_count}) or Inf ({inf_count}) in raw data. Skipping...")
                    continue
                
                # Ensure labels have correct length
                num_points = points_data.shape[0]
                assert len(sem_labels) == num_points, f"Semantic labels length {len(sem_labels)} != number of points {num_points}"
                assert len(ins_labels) == num_points, f"Instance labels length {len(ins_labels)} != number of points {num_points}"
                assert len(att_labels) == num_points, f"Attachment labels length {len(att_labels)} != number of points {num_points}"
                
                # Apply preprocessing
                processed_points = self._preprocess_point_cloud(points_data)
                
                # 检查处理后的数据是否包含NaN或Inf
                if np.isnan(processed_points).any() or np.isinf(processed_points).any():
                    nan_count = np.isnan(processed_points).sum()
                    inf_count = np.isinf(processed_points).sum()
                    print(f"Warning: {filename} contains NaN ({nan_count}) or Inf ({inf_count}) after preprocessing. Skipping...")
                    continue
                
                # 检查标签是否有效（语义标签应该在1-25范围内）
                if (sem_labels < 0).any() or (sem_labels > 25).any():
                    invalid_count = ((sem_labels < 0) | (sem_labels > 25)).sum()
                    print(f"Warning: {filename} contains {invalid_count} invalid semantic labels (out of range 0-25). Skipping...")
                    continue
                
                # Store data
                sample = {
                    'points': torch.tensor(processed_points, dtype=torch.float32),
                    'sem_labels': torch.tensor(sem_labels, dtype=torch.long),
                    'ins_labels': torch.tensor(ins_labels, dtype=torch.long),
                    # 属性标签保持为 long (0-5)，便于多类交叉熵和边界类别(4/5)判断
                    'att_labels': torch.tensor(att_labels, dtype=torch.long),
                    'filename': filename
                }
                
                self.data.append(sample)
                self.filenames.append(filename)
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
        
        # 打印数据集加载统计信息
        skipped_files = len(split_filelist) - len(self.data)
        print(f"Dataset loaded: {len(self.data)} samples from {len(split_filelist)} files ({split} split)")
        if skipped_files > 0:
            print(f"  - Skipped {skipped_files} files due to errors or invalid data")

    def _preprocess_point_cloud(self, points_data):
        """
        Preprocess the point cloud data.
        
        Args:
            points_data (np.ndarray): Raw point cloud data with shape (N, 6)
            
        Returns:
            np.ndarray: Preprocessed point cloud data
        """
        # Separate coordinates and normals
        coords = points_data[:, :3]
        normals = points_data[:, 3:6]
        
        if self.center_and_scale:
            # Center the point cloud
            center = np.mean(coords, axis=0)
            coords = coords - center
            
            # Scale to unit sphere
            max_dist = np.max(np.linalg.norm(coords, axis=1))
            if max_dist > 0:
                coords = coords / max_dist
        
        if self.normalize:
            # Normalize normals to unit length
            norm_lengths = np.linalg.norm(normals, axis=1)
            norm_lengths[norm_lengths == 0] = 1  # Avoid division by zero
            normals = normals / norm_lengths[:, np.newaxis]
        
        # Combine coordinates and normals
        processed_points = np.concatenate([coords, normals], axis=1)
        return processed_points

    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.load_in_memory:
            return len(self.data)
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.load_in_memory:
            # 预加载模式：从内存读取
            sample = self.data[idx].copy()
        else:
            # 按需加载模式：从磁盘读取
            sample = self._load_single_file(self.filenames[idx])
        
        # 数据增强已禁用
        # if self.augmentation is not None:
        #     sample = self.augmentation(sample)

        # 应用自定义transform（如果有）
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _load_single_file(self, filename):
        """按需加载单个文件（用于大数据集）"""
        try:
            # 确保文件名有.npy后缀
            if not filename.endswith('.npy'):
                filename = filename + '.npy'

            # 加载数据文件
            if getattr(self, "use_flat_points_dir", False):
                # 扁平目录模式: root_dir/points/filename
                data_path = self.path.joinpath("points", filename)
            else:
                # 默认模式: root_dir/processed_npys/points/filename
                data_path = self.path.joinpath("processed_npys", "points", filename)
            full_data = np.load(str(data_path)).astype(np.float32)
            
            # 检查格式
            if full_data.ndim != 2:
                raise ValueError(f"Incorrect dimensions {full_data.ndim}, expected 2D array")
            
            if full_data.shape[1] == 9:
                # 9列格式: (x,y,z,nx,ny,nz,sem,ins,att)
                points_data = full_data[:, :6]
                sem_labels = full_data[:, 6].astype(np.uint8)
                ins_labels = full_data[:, 7].astype(np.uint8)
                # 0-5 多类属性标签
                att_labels = full_data[:, 8].astype(np.uint8)
                pid_labels = None
            elif full_data.shape[1] == 10:
                # 10列格式: (x,y,z,nx,ny,nz,sem,ins,att,pid)
                points_data = full_data[:, :6]
                sem_labels = full_data[:, 6].astype(np.uint8)
                ins_labels = full_data[:, 7].astype(np.uint8)
                att_labels = full_data[:, 8].astype(np.uint8)
                pid_labels = full_data[:, 9].astype(np.int32)
            elif full_data.shape[1] == 6:
                # 6列格式（向后兼容）
                points_data = full_data
                labels_path = self.path.joinpath("processed_npys", "labels", filename)
                raw_labels = np.load(str(labels_path), allow_pickle=True)
                
                if raw_labels.dtype.names is not None:
                    sem_labels = np.array(raw_labels['sem'], dtype=np.uint8)
                    ins_labels = np.array(raw_labels['ins'], dtype=np.uint8)
                    # 期望 att 为 0-5；若为布尔则会变成 0/1
                    att_labels = np.array(raw_labels.get('att', np.zeros(len(sem_labels))), dtype=np.uint8)
                else:
                    raise ValueError("Unsupported label format")
            else:
                raise ValueError(f"Incorrect shape {full_data.shape}, expected (N, 6) or (N, 9)")

            # 将 (sem, ins) 组合为全局唯一的实例 ID：sem * K + ins
            # 这里不改变 sem_labels，本身仍在 1-25 范围内，仅在 ins_labels 中编码语义信息
            sem_int = sem_labels.astype(np.int32)
            ins_int = ins_labels.astype(np.int32)
            global_factor = 1000  # 假定单一语义下实例数远小于该值
            ins_labels = sem_int * global_factor + ins_int

            # 预处理
            processed_points = self._preprocess_point_cloud(points_data)
            
            # 返回样本
            sample = {
                'points': torch.tensor(processed_points, dtype=torch.float32),
                'sem_labels': torch.tensor(sem_labels, dtype=torch.long),
                'ins_labels': torch.tensor(ins_labels, dtype=torch.long),
                # 属性标签保持为 long (0-5)
                'att_labels': torch.tensor(att_labels, dtype=torch.long),
                'filename': filename
            }
            # 可选的 pid（仅当10列格式时存在）
            if 'pid_labels' in locals() and pid_labels is not None:
                sample['pid'] = torch.tensor(pid_labels, dtype=torch.long)
            
            return sample
            
        except Exception as e:
            # 出错时返回空样本
            print(f"Error loading {filename}: {e}")
            # 返回一个占位符样本
            return {
                'points': torch.zeros((1024, 6), dtype=torch.float32),
                'sem_labels': torch.zeros(1024, dtype=torch.long),
                'ins_labels': torch.zeros(1024, dtype=torch.long),
                'att_labels': torch.zeros(1024, dtype=torch.long),
                'filename': filename
            }

    def _collate(self, batch):
        """
        Collate a batch of data samples together into a single batch.
        优化版本：减少循环和内存分配，提高速度

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data.
        """
        batch_size = len(batch)
        # 假设所有样本都是1024点（预处理后固定大小）
        # 这样可以直接stack而不需要padding，大幅提升速度
        first_sample = batch[0]
        num_points = first_sample["points"].shape[0]
        
        # 快速检查：如果所有样本点数相同，直接stack（避免padding）
        all_same_size = all(sample["points"].shape[0] == num_points for sample in batch)
        
        if all_same_size:
            # 快速路径：直接stack（比循环padding快10倍）
            collated = {
                "points": torch.stack([sample["points"] for sample in batch], dim=0),
                "sem_labels": torch.stack([sample["sem_labels"] for sample in batch], dim=0),
                "ins_labels": torch.stack([sample["ins_labels"] for sample in batch], dim=0),
                "att_labels": torch.stack([sample["att_labels"] for sample in batch], dim=0),
                "filename": [sample["filename"] for sample in batch]
            }
            # 如果样本中包含 pid，则一并打包
            if "pid" in batch[0]:
                collated["pid"] = torch.stack([sample["pid"] for sample in batch], dim=0)
            return collated
        else:
            # 慢速路径：需要padding（保留兼容性）
            max_points = max([sample["points"].shape[0] for sample in batch])
            
            # 预分配tensor（比逐个append快）
            batched_points = torch.zeros(batch_size, max_points, 6, dtype=torch.float32)
            batched_sem_labels = torch.zeros(batch_size, max_points, dtype=torch.long)
            batched_ins_labels = torch.zeros(batch_size, max_points, dtype=torch.long)
            batched_att_labels = torch.zeros(batch_size, max_points, dtype=torch.long)
            has_pid = "pid" in first_sample
            if has_pid:
                batched_pid = torch.zeros(batch_size, max_points, dtype=torch.long)
            batched_filenames = []
            
            for i, sample in enumerate(batch):
                num_points = sample["points"].shape[0]
                batched_points[i, :num_points] = sample["points"]
                batched_sem_labels[i, :num_points] = sample["sem_labels"]
                batched_ins_labels[i, :num_points] = sample["ins_labels"]
                batched_att_labels[i, :num_points] = sample["att_labels"]
                if has_pid:
                    batched_pid[i, :num_points] = sample["pid"]
                batched_filenames.append(sample["filename"])
            
            collated = {
                "points": batched_points,
                "sem_labels": batched_sem_labels,
                "ins_labels": batched_ins_labels,
                "att_labels": batched_att_labels,
                "filename": batched_filenames
            }
            if has_pid:
                collated["pid"] = batched_pid
            return collated


if __name__ == '__main__':
    dataset = PointCloudDataset(root_dir='/root/github/AAGNet/dataset/data_10000', split='test', center_and_scale=True, normalize=True)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[1]
        print(f"Sample keys: {sample.keys()}")
        print(f"Points shape: {sample['points'].shape}")
        print(f"Semantic labels shape: {sample['sem_labels'].shape}")
        print(f"Instance labels shape: {sample['ins_labels'].shape}")
        
        print(torch.unique(sample['sem_labels']))
        print(torch.unique(sample['ins_labels'])[:20])

        print(f"Attachment labels shape: {sample['att_labels'].shape}")
        print(f"Filename: {sample['filename']}")
    