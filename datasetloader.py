import h5py
import numpy as np
import sys
import glob
import tensorflow as tf  # The main deep learning library
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TrainingDatasetLoader(tf.keras.utils.Sequence):
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    # Allows the class to be used as a data generator for training models
    def __init__(self, data_path, batch_size=20, training=True):
        # Accessing the path to the data
        print(f"Opening {data_path}")
        sys.stdout.flush()  # Ensures that the message is immediately printed in the console
        self.cache = h5py.File(
            data_path, "r"
        )  # Openning the HDF5 file in read only mode

        # Loading the dataset
        print("Loading data...")
        sys.stdout.flush()
        self.images = self.cache["images"][:10]
        self.labels = self.cache["labels"][:10]  # Convert the labels into float32 type
        self.image_dims = self.images.shape  # Assigning the shape of the loaded images

        # Creating the index arrays for the negative and positive samples
        train_inds = np.arange(len(self.images))
        pos_train_inds = train_inds[
            self.labels[train_inds, 0] == 1.0
        ]  # Contain indices where the labels are 1
        neg_train_inds = train_inds[
            self.labels[train_inds, 0] != 1.0
        ]  # Contain indices where the labels are not 1

        # Splitting the positive and negative indices into a 80:20 ratio for training:non-training
        if training:  # During training, assign 80% of the pos and neg indices
            self.pos_train_inds = pos_train_inds[: int(0.8 * len(pos_train_inds))]
            self.neg_train_inds = neg_train_inds[: int(0.8 * len(neg_train_inds))]
        else:  # During training, assign 20% of the pos and neg indices
            self.pos_train_inds = pos_train_inds[-1 * int(0.2 * len(pos_train_inds))]
            self.neg_train_inds = neg_train_inds[-1 * int(0.2 * len(neg_train_inds))]

        # Randomizing the training samples in each epoch
        np.random.shuffle(self.pos_train_inds)
        np.random.shuffle(self.neg_train_inds)

        self.train_inds = np.concatenate((self.pos_train_inds, self.neg_train_inds))
        self.train_batch_size = batch_size
        self.p_pos = np.ones(self.pos_train_inds.shape) / len(self.pos_train_inds)

    def get_train_size(self):
        return len(self.train_inds)

    def get_all_train_faces(self):
        return self.images[
            self.pos_train_inds
        ]  # Retrieve the training faces using the training indices

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            selected_inds = np.random.choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos
            )
        else:
            num_pos_samples = min(n // 2, len(self.pos_train_inds))
            num_neg_samples = n - num_pos_samples
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=num_pos_samples, replace=False, p=p_pos
            )
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=num_neg_samples, replace=False, p=p_neg
            )
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :, :, ::-1] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]
        return (
            (train_img, train_label, sorted_inds)
            if return_inds
            else (train_img, train_label)
        )
