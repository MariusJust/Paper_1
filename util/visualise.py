import matplotlib.pyplot as plt
import numpy as np

def Visualise(Time, GDP, Temp):
# Create meshgrid for 3D plotting
    time_grid, temp_grid = np.meshgrid(Time, Temp)

    # Creating the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(time_grid, temp_grid, GDP, cmap='viridis', edgecolor='none')

    # Add title and labels
    ax.set_title('Temperature vs GDP vs Time (Global Model)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature')
    ax.set_zlabel('Temperature')

    # Optional: Customize appearance (grid, transparency, etc.)
    ax.view_init(elev=20, azim=120)  # Adjust camera angle for better view
    ax.grid(False)                   # Remove grid lines for cleaner visualization

    # Add a color bar which maps values to colors
    fig.colorbar(surface, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()
