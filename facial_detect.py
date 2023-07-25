import functools
import tensorflow as tf  # The main deep learning library
import tqdm
from datasetloader import TrainingDatasetLoader
from utils import LossHistory, PeriodicPlotter

# Data Collection
path_to_training_data = tf.keras.utils.get_file(
    "train_face.h5", "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
)
batch_size = 8
loader = TrainingDatasetLoader(path_to_training_data)

# Defining the cnn

n_filters = 12  # The base number for convolutional filters


def make_standard_classifier(n_outputs=1):  # One output in the final layer
    # Creating partial functions to help in defining the CNN layers
    Conv2D = functools.partial(
        tf.keras.layers.Conv2D, padding="same", activation="relu"
    )
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(tf.keras.layers.Dense, activation="relu")

    # Creating an instance of the sequential model
    model = tf.keras.Sequential(
        [
            Conv2D(filters=1 * n_filters, kernel_size=5, strides=2),
            BatchNormalization(),
            Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
            BatchNormalization(),
            Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
            BatchNormalization(),
            Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
            BatchNormalization(),
            Flatten(),
            Dense(512),
            Dense(n_outputs, activation=None),
        ]
    )
    return model


standard_classifier = make_standard_classifier()

# Training the CNN
# Setting the training hyperparams
batch_size = 8  # Samples to be processed per iteration
num_epochs = 4  #  Number of times the entire dataset will be passed through the model during training
learning_rate = 0.00004  # The step size at each iteration during optimization process
train_size = len(loader.pos_train_inds) + len(loader.neg_train_inds)

optimizer = tf.keras.optimizers.Adam(
    learning_rate
)  # The optimizer usd to train the model
loss_history = LossHistory(smoothing_factor=0.99)  # To record the loss values
plotter = PeriodicPlotter(sec=2, scale="semilogy")
if hasattr(tqdm, "_instances"):
    tqdm._instances.clear()  # Clears an instance of the progress bar if they exist


@tf.function
def standard_train_step(lgt, lbl):  # Training step
    with tf.GradientTape() as tape:
        # Feed the images into the model
        lgt = tf.cast(lgt, tf.float32)
        lbl = tf.cast(lbl, tf.float32)
        logits = standard_classifier(lgt)
        # Compute the loss
        lss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=lbl, logits=logits
        )  # Applies a sigmoid activation function and compues the cross-entropy loss between the true labels and the precicted logits
    # Back propagation
    grads = tape.gradient(
        lss, standard_classifier.trainable_variables
    )  # Calculates the gradient loss wrt the trainable variables of the standard classifier model
    optimizer.apply_gradients(
        zip(grads, standard_classifier.trainable_variables)
    )  # Use the computed gradients with the optimizer defined above to update the trainable variables
    return lss


# The training loss
for epoch in range(num_epochs):
    for idx in range(
        train_size // batch_size
    ):  # This ensures that the entire training dataset is processed in batches.
        x, y = loader.get_batch(
            batch_size
        )  # Retrieving a batch of training data. x and y store image and corresponding labels respectively
        loss = standard_train_step(
            x, y
        )  # Call the training function defined above to perform the training
        loss_history.append(
            loss.numpy().mean()
        )  # add the loss values to the loss objects
        plotter.plot(loss_history.get())  # Plotting the evolution of the loss

# Pickling the model
(batch_x, batch_y) = loader.get_batch(5)
y_pred_standard = tf.round(tf.nn.sigmoid(standard_classifier.predict(batch_x)))
acc_standard = tf.reduce_mean(tf.cast(tf.equal(batch_y, y_pred_standard), tf.float32))

print(f"Standard CNN accuracy on (potentially biased) training set: {acc_standard}")

# Recompile the model with the appropriate optimizer, loss function, and metrics
standard_classifier.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)


# Saving the trained model in the native Keras format
standard_classifier.save("model/trained_model.keras")
print("Model saved")
