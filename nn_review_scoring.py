import tensorflow as tf
import numpy as np
import os
import datetime
from dataset import ReviewDataset


class Network:
    def __init__(self, args, num_words, logdir, ch):
        word_ids = tf.keras.layers.Input(shape=(None,))
        layeri = tf.keras.layers.Embedding(
            num_words+1, args['we_dim'], mask_zero=True)(word_ids)
        rnn_layer = tf.keras.layers.LSTM(256, return_sequences=True)
        layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode='concat', weights=None)(layeri)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
        predictions_scaled = tf.math.multiply(predictions, 5)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        self.model = tf.keras.Model(
            inputs=word_ids, outputs=predictions_scaled)
        print(self.model.summary())
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.keras.losses.MSE,
                           metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
        self._writer = tf.summary.create_file_writer(
            logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args, num_batches):
        for x, y, u in dataset.batches(args['batch_size'], num_batches=num_batches):
            metrics = self.model.train_on_batch(
                x, y, reset_metrics=True)
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            with self._writer.as_default():
                for name, value in zip(self.model.metrics_names, metrics):
                    tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, args, num_batches):
        for x, y, u in dataset.batches(args['batch_size'], train=False, num_batches=num_batches):
            metrics = self.model.test_on_batch(
                x, y, reset_metrics=False)
        self.model.reset_metrics()
        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("test/{}".format(name), value)
        return metrics


args = {
    'batch_size': 256,
    'we_dim': 100
}
now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
LOG_DIR = os.path.join('log', f'{now}')
epochs = 10
train_data = ReviewDataset('train.npy', test_ratio=0.1)
train_data.find_out_num_words()
train_data.num_words = 40116
net = Network(args, num_words=train_data.num_words, logdir=LOG_DIR)
for epoch in range(epochs):
    net.train_epoch(train_data, args, 100)
    metrics = net.evaluate(train_data, args, 30)
    print(f'Epoch {epoch}:{metrics}')
    net.save_model()
