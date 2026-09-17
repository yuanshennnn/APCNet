"""
使用预处理的HDF5数据训练ASIN模型（快速版本）
"""
import os
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.python.keras import backend as K
import logging

# 导入ASIN模型组件
from models import model
from loss import Loss_Function
from acc import Instance_Segmentation_Accuracy

# ==================== 配置参数 ====================
# 预处理数据路径
TRAIN_DATA_H5 = 'data/traindata_bspnet.h5'
VAL_DATA_H5 = 'data/validationdata_bspnet.h5'

# 模型保存路径
MODEL_SAVE_DIR = 'models_bspnet'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_save_weights_file = os.path.join(MODEL_SAVE_DIR, 'ASIN_bspnet_weights.h5')
best_model_save_weights_file = os.path.join(MODEL_SAVE_DIR, 'ASIN_bspnet_best_weights.h5')

# 日志目录
LOG_DIR = 'logdir_bspnet'
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== 设置日志 ====================
log_file = os.path.join(MODEL_SAVE_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("ASIN Model Training on BSPNet Dataset (Fast Version)")
logger.info("=" * 70)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 0.001

logger.info(f"Training Configuration:")
logger.info(f"  Batch size: {BATCH_SIZE}")
logger.info(f"  Epochs: {EPOCHS}")
logger.info(f"  Initial learning rate: {INITIAL_LR}")

# ==================== 加载预处理数据 ====================
def load_h5_data(h5_file):
    """加载HDF5格式的预处理数据"""
    logger.info(f"Loading data from {h5_file}...")
    with h5py.File(h5_file, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        similarity = f['similarity'][:]
        numofinstances = f['numofinstances'][:]
        bottomlabel = f['bottomfacelabel'][:]
    logger.info(f"  Loaded {len(data)} samples")
    return data, label, similarity, numofinstances, bottomlabel

logger.info("=" * 70)
logger.info("Loading preprocessed data...")
logger.info("=" * 70)

train_points_OnFace, train_labels, train_similar_matrix, train_numofinstances, train_bottom_labels = \
    load_h5_data(TRAIN_DATA_H5)

validation_points_OnFace, validation_labels, validation_similar_matrix, validation_numofinstances, validation_bottom_labels = \
    load_h5_data(VAL_DATA_H5)

logger.info("=" * 70)
logger.info("Dataset loaded successfully!")
logger.info("=" * 70)
logger.info(f"Training samples: {len(train_points_OnFace)}")
logger.info(f"Validation samples: {len(validation_points_OnFace)}")
logger.info(f"Input shape: {train_points_OnFace.shape}")
logger.info(f"Semantic labels shape: {train_labels.shape}")
logger.info(f"Similarity matrix shape: {train_similar_matrix.shape}")
logger.info("=" * 70)

# ==================== 构建模型 ====================
logger.info("Building ASIN model...")
mymodel = model.ASIN_model()

# 将模型摘要写入日志
import io
stream = io.StringIO()
mymodel.summary(print_fn=lambda x: stream.write(x + '\n'))
model_summary = stream.getvalue()
logger.info("Model Architecture:")
logger.info("\n" + model_summary)

# ==================== 学习率调度器 ====================
def scheduler(epoch):
    """动态调整学习率"""
    if epoch < 100:
        K.set_value(mymodel.optimizer.lr, INITIAL_LR)
    else:
        if (epoch - 100) % 30 == 0 or epoch == 100:
            lr = K.get_value(mymodel.optimizer.lr)
            K.set_value(mymodel.optimizer.lr, lr * 0.5)
            logger.info(f"Learning rate changed to {lr * 0.5}")
    return K.get_value(mymodel.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

# ==================== 编译模型 ====================
optimizer_adam = optimizers.Adam(lr=INITIAL_LR)
mymodel.compile(
    optimizer=optimizer_adam,
    loss=[
        losses.categorical_crossentropy,
        losses.binary_crossentropy,
        Loss_Function
    ],
    loss_weights=[1, 1, 1],  # 降低实例分割权重从3到1
    metrics=['accuracy', Instance_Segmentation_Accuracy],
)

logger.info("Model compiled successfully")
logger.info(f"  Optimizer: Adam (lr={INITIAL_LR})")
logger.info(f"  Loss weights: [1, 1, 1] - Balanced weights to reduce instance overfitting")

# ==================== 回调函数 ====================
csv_logger = CSVLogger(
    os.path.join(MODEL_SAVE_DIR, 'training_log.csv'),
    separator=',',
    append=False
)

tbcallback = TensorBoard(log_dir=LOG_DIR)

checkpoint = ModelCheckpoint(
    best_model_save_weights_file,
    monitor='val_loss',
    save_best_only=True,
    mode='auto',
    period=1,
    verbose=1
)

logger.info("Callbacks configured:")
logger.info(f"  - CSV Logger: {MODEL_SAVE_DIR}/training_log.csv")
logger.info(f"  - TensorBoard: {LOG_DIR}")
logger.info(f"  - ModelCheckpoint: {best_model_save_weights_file}")

# ==================== 开始训练 ====================
logger.info("=" * 70)
logger.info("Starting training...")
logger.info("=" * 70)
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Initial learning rate: {INITIAL_LR}")
logger.info("=" * 70)

time_start = time.time()

# 检查GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"Using GPU: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth setting failed: {e}")
else:
    logger.info("No GPU found, using CPU")

# 训练模型
history = mymodel.fit(
    train_points_OnFace,
    [train_labels, train_bottom_labels, train_similar_matrix],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=(
        validation_points_OnFace,
        [validation_labels, validation_bottom_labels, validation_similar_matrix]
    ),
    callbacks=[tbcallback, reduce_lr, checkpoint, csv_logger],
    verbose=1
)

# 保存最终模型
mymodel.save_weights(model_save_weights_file)
logger.info(f"Final model saved to: {model_save_weights_file}")
logger.info(f"Best model saved to: {best_model_save_weights_file}")

time_end = time.time()
training_time = time_end - time_start

# ==================== 训练总结 ====================
logger.info("=" * 70)
logger.info("Training completed!")
logger.info("=" * 70)
logger.info(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
logger.info(f"Average time per epoch: {training_time/EPOCHS:.2f} seconds")
logger.info("")
logger.info("Final metrics:")
logger.info(f"  Training loss: {history.history['loss'][-1]:.4f}")
logger.info(f"  Validation loss: {history.history['val_loss'][-1]:.4f}")
if 'accuracy' in history.history:
    logger.info(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
logger.info("=" * 70)

# 找出最佳epoch
best_val_loss = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(best_val_loss) + 1
logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

# 保存训练历史
import pickle
history_file = os.path.join(MODEL_SAVE_DIR, 'training_history.pkl')
with open(history_file, 'wb') as f:
    pickle.dump(history.history, f)
logger.info(f"Training history saved to: {history_file}")

logger.info("=" * 70)
logger.info("Log files saved:")
logger.info(f"  - Training log: {log_file}")
logger.info(f"  - CSV log: {MODEL_SAVE_DIR}/training_log.csv")
logger.info(f"  - TensorBoard logs: {LOG_DIR}")
logger.info("=" * 70)
logger.info("Training completed successfully!")
logger.info("=" * 70)
