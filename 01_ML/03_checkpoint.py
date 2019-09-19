"""
Checkpoints : Save and restore models
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

# ----------------------------------------------------------------
# Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# ----------------------------------------------------------------
# Define a model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# ----------------------------------------------------------------
# Save checkpoints during training
# ----------------------------------------------------------------

checkpoint_path = "ckpt/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = callbacks.ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        verbose=0)

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.fit(train_images, train_labels,
          epochs = 10,
          validation_data = (test_images,test_labels),
          verbose = 0,
          callbacks = [cp_callback])  # pass callback to training

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# ----------------------------------------------------------------
# Checkpoint callback options

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "ckpt/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = callbacks.ModelCheckpoint(checkpoint_path,
                                        save_weights_only=True,
                                        verbose=0,
                                        period=5)   # Save weights, every 5-epochs.

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose = 0)

latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_ckpt)


# ----------------------------------------------------------------
# Manually save weights
# ----------------------------------------------------------------

# Save the weights
model.save_weights('ckpt/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('ckpt/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Save the entire model
model.save('ckpt/my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('ckpt/my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Reloaded model, accuracy: {:5.2f}%".format(100*acc))


# ----------------------------------------------------------------
