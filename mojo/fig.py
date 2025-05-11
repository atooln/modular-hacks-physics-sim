import matplotlib.pyplot as plt
import numpy as np

def plot_surf(arr, width, height):
    """
    Plot a 3D surface using matplotlib.

    Parameters:
    - arr: 2D numpy array to plot
    - width: width of the plot
    - height: height of the plot
    """
    # reshape the arr into a 2D width and height
    arr = arr.reshape((width, height))

    # fig = plt.figure(figsize=(width, height))
    # ax = fig.add_subplot(111, projection='3d')
    
    plt.imshow(arr, cmap='viridis')
    # Create a meshgrid for the x and y coordinates
    # x = np.arange(arr.shape[0])
    # y = np.arange(arr.shape[1])
    # X, Y = np.meshgrid(x, y)
    
    # # Plot the surface
    # surf = ax.plot_surface(X, Y, arr, cmap='viridis')
    
    # # Add color bar
    # fig.colorbar(surf)
    
    # # plt.show()
    plt.savefig('surface_plot.png')
    # plt.show()