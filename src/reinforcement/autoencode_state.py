
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

states = np.load(
    "play_with_proba_without_score/" + os.listdir("play_with_proba_without_score/")[1],
)
n, p = states.shape
p = p - 6


class Coder(Model):
    def __init__(self) -> None:
        super().__init__()
        self.d0 = Dense(50)
        self.d1 = Dense(30, activation="relu")
        self.d2 = Dense(10, activation="relu")

    def call(self, x):
        x = self.d0(x)
        y = self.d1(x)
        return self.d2(y)


class Decoder(Model):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = Dense(10, activation="relu")
        self.d11 = Dense(30, activation="relu")
        self.d2 = Dense(p)

    def call(self, x):
        y = self.d1(x)
        y = self.d11(y)
        return self.d2(y)


class AutoEncode(Model):
    def __init__(self) -> None:
        super().__init__()
        self.c = Coder()
        self.d = Decoder()

    def call(self, x):
        return self.d(self.c(x))


# class Split(Model):
#     def __init__(self):
#         super(Split, self).__init__()
#         self.walls_rep = AutoEncode()
#         self.players_rep = AutoEncode()


# Create an instance of the model
autoencode = AutoEncode()

loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(learning_rate=10**-4)

train_loss = tf.keras.metrics.Mean(name="train_loss")
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name="test_loss")
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(
#     name='test_accuracy')


@tf.function
def accuracy(x, y):
    true_tab = tf.reduce_all(tf.equal(tf.math.round(x), tf.cast(y, tf.float32)), axis=1)
    return tf.shape(true_tab[true_tab])[0] / tf.shape(x)[0]


@tf.function
def train_step(s) -> None:
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = autoencode(s)
        loss = loss_object(s, predictions)
    train_loss(loss)
    gradients = tape.gradient(loss, autoencode.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencode.trainable_variables))


@tf.function
def test_step(s):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = autoencode(s)
    t_loss = loss_object(s, predictions)
    test_loss(t_loss)
    return predictions


template = "Epoch {}, Loss: {}, Test Loss: {}"
train_ratio = 0.8

for file in os.listdir("play_with_proba_without_score/")[:100]:
    print("File : " + file)
    states = np.load("play_with_proba_without_score/" + file)
    n, _ = states.shape
    np.random.shuffle(states)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(states[: int(train_ratio * n), :-6])
        .shuffle(100)
        .batch(32)
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        states[int(train_ratio * n) :, :-6],
    ).batch(32)
    # TRAINING LOOP
    EPOCHS = 10
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for state in train_ds:
            train_step(state)

        for state in test_ds:
            test_step(state)

        if epoch % 10 == 0:
            print(
                template.format(
                    epoch,
                    train_loss.result(),
                    test_loss.result(),
                ),
            )
            test_state = states[int(train_ratio * n) :, :-6]
            pred = test_step(test_state)
            # print(pred)
            # print(test_state)
            print("Accuracy : %.2f %%" % (accuracy(pred, test_state) * 100))

for file in os.listdir("play_with_proba_without_score/")[100:200]:
    print("File : " + file)
    states = np.load("play_with_proba_without_score/" + file)
    test_state = states[:, :-6]
    pred = test_step(test_state)
    print("Accuracy : %.2f %%" % (accuracy(pred, test_state) * 100))
