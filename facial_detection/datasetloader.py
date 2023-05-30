import h5py
import numpy as np
import sys
import glob
import tensorflow as tf #The main deep learning library

class TrainingDatasetLoader(tf.keras.utils.Sequence): # Allows the class to be used as a data generator for training models 
    def __init__(self,data_path, batch_size=200, training=True):
        
        # Accessing the path to the data
        print(f"Opening {data_path}")
        sys.stdout.flush() # Ensures that the message is immediately printed in the console
        self.cache = h5py.File(data_path, "r") # Openning the HDF5 file in read only mode
        
        # Loading the dataset
        print("Loading data...")
        sys.stdout.flush()
        self.images = self.cache["images"][:]
        self.labels = self.cache['labels'][:].astype(np.float32) # Convert the labels into float32 type
        self.image_dims = self.images.shape # Assigning the shape of the loaded images
        
        # Creating the index arrays for the negative and positive samples
        train_inds = np.arange(len(self.images))
        pos_train_inds = train_inds[self.labels[train_inds, 0] == 1.0] # Contain indices where the labels are 1
        neg_train_inds = train_inds[self.labels[train_inds, 0] != 1.0] # Contain indices where the labels are not 1
        
        # Splitting the positive and negative indices into a 80:20 ratio for training:non-training
        if training: # During training, assign 80% of the pos and neg indices
            self.pos_train_inds = pos_train_inds[:int(.8 * len(pos_train_inds))]
            self.neg_train_inds = neg_train_inds[:int(.8 * len(neg_train_inds))]
        else: # During training, assign 20% of the pos and neg indices
            self.pos_train_inds = pos_train_inds[-1 * int(.2 * len(pos_train_inds))]
            self.neg_train_inds = neg_train_inds[-1 * int(.2 * len(neg_train_inds))]
        
        # Randomizing the training samples in each epoch
        np.random.shuffle(self.pos_train_inds)
        np.random.shuffle(self.neg_train_inds)
        
        self.train_inds = np.concatenate ((self.pos_train_inds, self.neg_train_inds))
        self.train_batch_size = batch_size
        self.p_pos = np.ones(self.pos_train_inds.shape) / len(self.pos_train_inds)
        
    def get_batch(self, batch_size):
        selected_pos_inds = np.random.choice(self.pos_train_inds, size=batch_size // 2, replace=False, p=self.p_pos) # Ensures an equal number of positive samples in the batch
        selected_neg_inds = np.random.choice(self.neg_train_inds, size=batch_size // 2, replace=False) # Ensures an equal number of negative samples in the batch
        selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds)) # concatenates the selected positive and negative indices into a single array
        
        sorted_inds = np.sort(selected_inds).flatten() # Sorts the selected indices in ascending order
        sorted_inds = sorted_inds.astype(int) 
        train_img = (self.images[sorted_inds] / 255.0).astype(np.float32) # Normalizing the images
        train_label = (self.labels[sorted_inds]).astype(np.float32)
        
        return train_img, train_label
        
    def get_train_size(self):
        return len(self.train_inds)
    
    def get_all_train_faces(self):
        train_faces = self.images[self.train_inds] # Retrieve the training faces using the training indices
        train_faces = train_faces.astype(np.float32) / 255.0 # Normalize the training faces 
        return train_faces 
    
        
        
    
    