import time
import numpy as np
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt

# Plots a sample input and the corresponding reconstruction from the VAE Model 
def plot_sample(x, y, vae):
    plt.figure(figsize=(2,1))
    plt.subplot(1, 2, 1)
    idx = np.where(y==1)[0][0] # Finds the first instance where the y is equal to 1
    plt.imshow(x[idx]) # Displays the image at the `idx` index from the `x` input  data
    plt.grid(True)
    plt.subplot(1, 2, 2) # Plots a new subplot within the figure 
    _, _, _, recon = vae(x) # Calls the vae function to reconstruct the output while ignoring all other outputs (_, _, _, _, _, _, _, _, _)
    recon = np.clip(recon, 0, 1) # Limits the values between 0 and 1
    plt.imshow(recon[idx])
    plt.grid(True)
    
    plt.show()     

# LossHistory: Used to track the  evolution of the loss during training
class LossHistory: 
    def __init__(self,smoothing_factor=0.0): # No smoothing applied in the data. Each loss appended to the list without any smoothing calculations
        self.alpha = smoothing_factor 
        self.loss = [] # Initialize an empty list to append the loss
    def append(self, value): # Appending the loss value to the loss list
        self.loss.append(self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss)>0 else value) # the smoothing calculation
    def get(self):
        return self.loss # Return the current list of the loss values  
    
# PeriodicPlotter: Used to create periodic plots for tge training process
class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale
        self.tic = time.time()
    def plot(self, data): # Plots the data
        if time.time() - self.tic > self.sec: # Checking if the last time interval has passed since the last plot
            plt.cla() # If yes, the current plot is cleared
            
            # Creates a new plot using a specified scale, either none, semilogx, semilogy or loglog
            if self.scale is None:
                plt.plot(data)
            elif self.scale == 'semilogx':
                plt.semilogx(data)
            elif self.scale == 'semilogy':
                plt.semilogy(data)
            elif self.scale == 'loglog':
                plt.loglog(data)
            else: 
                raise ValueError(f"{self.scale} is not recognized. Please use a recognized one!)")
                
            plt.xlabel(self.xlabel), plt.ylabel(self.ylabel)
            ipythondisplay.clear_output(wait=True) # clears the output displayed in the IPython environment, ensuring that only the current plot is visible
            ipythondisplay.display(plt.gcf()) # displays the current plot
            
            
            
    def create_grid_of_images(xs, size=(5,5)): # Combines a list of images into a single grid. xs: List of images. size: desired size of the grid
        grid = [] # Initalizing an empty list that will house the rows of the image grid
        counter = 0 # a counter variable to keep track of the position in the xs list        
       
        for i in range(size[0]):  # Iterating over the desired number of rows
            row = [] # Initialize an empty list of rows
            for j in range(size[1]): #Iterating over the desired number of columns
                row.append(xs[counter]) # Append the image into the current position in the image list to the row list
                counter += 1 # Incrementing the counter to move to the next image in the image (xs) list
            row = np.hstack(row) # Horizontally stack the images creating a single row of images
            grid.append(row) # Append the single row to the grid list
        grid = np.vstack(grid) # vertically stack the riws in the grid list into a single grid
        return grid
        
            
                
    