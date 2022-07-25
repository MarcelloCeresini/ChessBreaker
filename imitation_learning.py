import chess, os, chess.pgn
import numpy as np
import tensorflow as tf

from model import ResNet, create_model
import utils
conf = utils.Config()

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

val_file_path = "/home/marcello/github/ChessBreaker/data/Database/lichess_elite_2015-12.pgn"    # 446'402 samples

output_signature=(
    tf.TensorSpec((8,8,119), dtype=tf.dtypes.float16),
    (
        tf.TensorSpec((8,8,73), dtype=tf.dtypes.float32),
        tf.TensorSpec((1,), dtype=tf.dtypes.float32)
    )
)

dataset = tf.data.Dataset.from_generator(utils.gen, output_signature=output_signature)                            # training is 313'831'972 moves

val_dataset = tf.data.Dataset.from_generator(utils.gen, output_signature=output_signature, args=[val_file_path])  # validation is 446'402 moves --> we don't want to loose to much time

# oss: total training is 5,4 milion total steps, so 150k shuffle buffer should be enough 
# (also because a match is ~200 steps, you want just to remove the correlation of moves inside games)
ds = dataset.shuffle(conf.BATCH_DIM*10000) \
    .batch(conf.IMITATION_LEARNING_BATCH, num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = val_dataset.batch(conf.IMITATION_LEARNING_BATCH, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

model = create_model()

optimizer = tf.keras.optimizers.Adam()

losses = {
	"action_v": tf.keras.losses.BinaryCrossentropy(from_logits = True, label_smoothing=0.1),
	"state_v": tf.keras.losses.MeanSquaredError()
}

metrics = {
    "action_v": tf.keras.metrics.BinaryCrossentropy(from_logits = True),
    "state_v": tf.keras.metrics.MeanSquaredError()
}

model.compile(
    optimizer = optimizer,
    loss = losses,
    metrics = metrics
)

# oss: the epochs are just for convenience, it's just ONE PASS through almost all of the data
# so no overfitting is possible --> 
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-4, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath="tmp/checkpoint-{epoch:02d}.hdf5", save_freq='epoch', verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)
]

history = model.fit(
    ds,
    validation_data = val_ds,
    epochs = 60,
    steps_per_epoch = 10000,    # 10000*60*512 = 307'200'000 ~ all the available data
    callbacks = callbacks,
    workers = 16,
    use_multiprocessing = True
)