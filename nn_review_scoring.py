import tensorflow as tf
import numpy as np
import os
import datetime
from dataset import ReviewDataset


class ReviewStarsNetwork:
    def __init__(self, args, num_words, logdir):
        word_ids = tf.keras.layers.Input(shape=(None,))
        layeri = tf.keras.layers.Embedding(
            num_words+1, args['we_dim'], mask_zero=True)(word_ids)
        rnn_layer = tf.keras.layers.LSTM(256, return_sequences=True)
        layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode='concat', weights=None)(layeri)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
        predictions_scaled = tf.math.multiply(predictions, 4)
        self.model = tf.keras.Model(
            inputs=word_ids, outputs=predictions_scaled)
        print(self.model.summary())

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)

        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                           loss=tf.keras.losses.MSE,
                           metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
        self._writer = tf.summary.create_file_writer(
            logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args, num_batches):
        for x, y, u in dataset.batches(args['batch_size'], num_batches=num_batches):
            metrics = self.model.train_on_batch(
                x, u, reset_metrics=True)
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            with self._writer.as_default():
                for name, value in zip(self.model.metrics_names, metrics):
                    tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, args, num_batches):
        for x, y, u in dataset.batches(args['batch_size'], train=False, num_batches=num_batches):
            metrics = self.model.test_on_batch(
                x, u, reset_metrics=False)
        self.model.reset_metrics()
        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("test/{}".format(name), value)
        return metrics


class CategoricalStarsNetwork(ReviewStarsNetwork):
    def __init__(self, args, num_words, logdir):
        word_ids = tf.keras.layers.Input(shape=(None,))
        layeri = tf.keras.layers.Embedding(
            num_words+1, args['we_dim'], mask_zero=True)(word_ids)
        rnn_layer = tf.keras.layers.LSTM(256, return_sequences=False)
        layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode='concat', weights=None)(layeri)
        predictions = tf.keras.layers.Dense(5, activation='softmax')(layer)
        self.model = tf.keras.Model(
            inputs=word_ids, outputs=predictions)
        print(self.model.summary())
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=[tf.keras.losses.CategoricalCrossentropy()])
        self._writer = tf.summary.create_file_writer(
            logdir, flush_millis=10 * 1000)


args = {
    'batch_size': 256,
    'we_dim': 100
}
now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
LOG_DIR = os.path.join('log', f'useful_mse{now}')
epochs = 30
train_data = ReviewDataset('train.npy', test_ratio=0.1)
train_data.find_out_num_words()
train_data.num_words = 40116
net = ReviewStarsNetwork(args, num_words=train_data.num_words, logdir=LOG_DIR)
for epoch in range(epochs):
    net.train_epoch(train_data, args, 100)
    metrics = net.evaluate(train_data, args, 30)
    print(f'Epoch {epoch}:{metrics}')
    if epoch % 10 == 0:
        net.model.save_weights(f'models/useful_mse_epoch{epoch}')
