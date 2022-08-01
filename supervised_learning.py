import chess, os, chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics

from model import ResNet, create_model_v2
import utils
conf = utils.Config()

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

val_file_path = "/home/marcello/github/ChessBreaker/data/Database/lichess_elite_2015-12.pgn"    # 446'402 samples

output_signature=(
    tf.TensorSpec((8,8,119), dtype=tf.dtypes.float16),
    (
        tf.TensorSpec((8,8,73), dtype=tf.dtypes.float16),
        tf.TensorSpec((1,), dtype=tf.dtypes.float16)
    )
)

dataset = tf.data.Dataset.from_generator(utils.gen, output_signature=output_signature)                            # training is 313'831'972 moves

val_dataset = tf.data.Dataset.from_generator(utils.gen, output_signature=output_signature, args=[val_file_path])  # validation is 446'402 moves --> we don't want to loose to much time

# oss: total training is 5,4 milion total steps
# (also because a match is ~200 steps, you want just to remove the correlation of moves inside games)
ds = dataset.shuffle(conf.BATCH_DIM*10000) \
    .batch(conf.IMITATION_LEARNING_BATCH, num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = val_dataset.batch(conf.IMITATION_LEARNING_BATCH, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

model = create_model_v2()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

loss_dict = {
	"policy": tf.keras.losses.BinaryCrossentropy(from_logits = True), # tried *label smoothing* but it makes NO SENSE 99% of the moves are NOT legal moves, so it's useless to give them backprop (you are effectively just reducing the right move backprop)
	"value": tf.keras.losses.MeanSquaredError()
}

metric_dict = {
    "policy": tf.keras.metrics.TopKCategoricalAccuracy(k=1),
    "value": tf.keras.metrics.MeanSquaredError()
}

model.compile(
    optimizer = optimizer,
    loss = loss_dict,
    metrics = metric_dict
)

# oss: the epochs are just for convenience, it's just ONE PASS through almost all of the data
# so no overfitting is possible --> 
callbacks = [
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-4, verbose=1), # not needed since we don't repeat the database
    # learning rate was set to 0.2 for each game, and was dropped three times during the course of training to 0.02, 0.002 and 0.0002 respectively, after 100, 300 and 500 thousands of steps
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