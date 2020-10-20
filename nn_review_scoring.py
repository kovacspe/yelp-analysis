import tensorflow as tf
import numpy as np
import os
import datetime
from dataset import ReviewDataset


class Network:
    def __init__(self, args, num_words, logdir):
        word_ids = tf.keras.layers.Input(shape=(None,))
        embed = tf.keras.layers.Embedding(
            num_words+1, args['we_dim'], mask_zero=True)(word_ids)
        rnn_layer = tf.keras.layers.LSTM(256, return_sequences=False)
        layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode='concat', weights=None)(embed)
        if args['network_type'] == 'classification':
            # classification
            output = tf.keras.layers.Dense(5, activation='softmax')(layer)
            loss = tf.keras.losses.CategoricalCrossentropy()
            metrics = [tf.keras.metrics.CategoricalCrossentropy(
                name="crossentropy")]
        elif args['network_type'] == 'binary_classification':
            # binary classification
            output = tf.keras.layers.Dense(2, activation='softmax')(layer)
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = [tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()]
        elif args['network_type'] == 'regression':
            # regression
            predictions = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
            output = tf.math.multiply(predictions, 4)
            loss = tf.keras.losses.MSE()
            metrics = [tf.keras.metrics.MeanSquaredError(name="mse")]
        else:
            raise AttributeError('Not valid network type')

        # Compile model
        self.model = tf.keras.Model(inputs=word_ids, outputs=output)
        print(self.model.summary())
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=args['lr']),
                           loss=loss,
                           metrics=metrics)
        self._writer = tf.summary.create_file_writer(
            logdir, flush_millis=10 * 1000)

    def train_epoch(self, dataset, args, num_batches):
        for x, y in dataset.batches(args['batch_size'], num_batches=num_batches):
            metrics = self.model.train_on_batch(
                x, y, reset_metrics=True)
            tf.summary.experimental.set_step(self.model.optimizer.iterations)
            with self._writer.as_default():
                for name, value in zip(self.model.metrics_names, metrics):
                    tf.summary.scalar("train/{}".format(name), value)

    def evaluate(self, dataset, args, num_batches):
        for x, y in dataset.batches(args['batch_size'], train=False, num_batches=num_batches):
            metrics = self.model.test_on_batch(
                x, y, reset_metrics=False)
        self.model.reset_metrics()
        metrics = dict(zip(self.model.metrics_names, metrics))
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("test/{}".format(name), value)
        return metrics


output_type = 'stars'
epochs = 10
args = {
    'batch_size': 256,
    'we_dim': 100,
    'lr': tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9),
    'network_type': 'binary_classification',
    'name': 'network_name'
}


name = args['name']
now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
LOG_DIR = os.path.join('log', f'{name}-{now}')

train_data = ReviewDataset(
    'train.npy', test_ratio=0.1, output_type=output_type, data_type=args['network_type'])
train_data.find_out_num_words()
train_data.num_words = 40116

net = Network(args, num_words=train_data.num_words, logdir=LOG_DIR)
for epoch in range(epochs):
    net.train_epoch(train_data, args, 100)
    metrics = net.evaluate(train_data, args, 30)
    print(f'Epoch {epoch}:{metrics}')
    if epoch % 10 == 9:
        net.model.save_weights(f'models/{name}-{epoch}')
