import numpy as np
import matplotlib.pyplot as plt
import os

def plot_grid(grid_size, grid_res, start_pos, end_pos, path, obstacles, hazard_radius, open_set, closed_set, colors):
    # Define color variables
    background_color = (colors.COLOR_PATH_BG[0]/255,colors.COLOR_PATH_BG[1]/255,colors.COLOR_PATH_BG[2]/255)  # Dark navy background
    hazard_color = (colors.COLOR_PATH_HAZARD_CIRCLE[0]/255,colors.COLOR_PATH_HAZARD_CIRCLE[1]/255,colors.COLOR_PATH_HAZARD_CIRCLE[2]/255)  # Red for hazard circles
    border_color = (colors.COLOR_PATH_BORDER[0]/255,colors.COLOR_PATH_BORDER[1]/255,colors.COLOR_PATH_BORDER[2]/255)  # White for the arena border
    path_color = (colors.COLOR_PATH_WPS[0]/255,colors.COLOR_PATH_WPS[1]/255,colors.COLOR_PATH_WPS[2]/255)  # Green for the path
    grid_color = (colors.COLOR_PATH_GRID[0]/255,colors.COLOR_PATH_GRID[1]/255,colors.COLOR_PATH_GRID[2]/255)  # White for the grid lines
    start_color = (colors.COLOR_PATH_START[0]/255,colors.COLOR_PATH_START[1]/255,colors.COLOR_PATH_START[2]/255)  # Green for the start point
    end_color = (colors.COLOR_PATH_END[0]/255,colors.COLOR_PATH_END[1]/255,colors.COLOR_PATH_END[2]/255)  # Red for the end point
    open_color = (colors.COLOR_PATH_OPEN_NODE[0]/255,colors.COLOR_PATH_OPEN_NODE[1]/255,colors.COLOR_PATH_OPEN_NODE[2]/255)  # Yellow for open nodes
    closed_color = (colors.COLOR_PATH_CLOSE_NODE[0]/255,colors.COLOR_PATH_CLOSE_NODE[1]/255,colors.COLOR_PATH_CLOSE_NODE[2]/255)  # Cyan for closed nodes

    grid_half = grid_size / 2
    grid_dim = int(grid_size / grid_res) + 1  # One extra line for the intersections

    def to_grid_coords(pos):
        return ((pos[0] + grid_half) / grid_res,
                (pos[1] + grid_half) / grid_res)

    fig, ax = plt.subplots()
    
    # Set the background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)  # Dark navy background

    # Plot obstacles with hazard radius
    for obs in obstacles:
        obs_x, obs_y = to_grid_coords(obs)
        hazard_circle = plt.Circle((obs_x, obs_y), hazard_radius / grid_res, color=hazard_color, alpha=0.5)
        ax.add_patch(hazard_circle)

    # Plot start and end points
    start_grid = to_grid_coords(start_pos)
    end_grid = to_grid_coords(end_pos)

    # Mark the start and end points in the grid
    ax.text(start_grid[0], start_grid[1], 'S', fontsize=12, color=start_color, ha='center')
    ax.text(end_grid[0], end_grid[1], 'E', fontsize=12, color=end_color, ha='center')

    # Plot open and closed nodes
    open_x = [node[0] for node in open_set]
    open_y = [node[1] for node in open_set]
    closed_x = [node[0] for node in closed_set]
    closed_y = [node[1] for node in closed_set]

    ax.scatter(open_x, open_y, color=open_color, label='Open Nodes', alpha=0.5, s=10)
    ax.scatter(closed_x, closed_y, color=closed_color, label='Closed Nodes', alpha=0.5, s=10)

    # Plot path
    if path:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, marker='o', color=path_color, label='Path')

    # Adjust ticks to reflect real-world coordinates where (15, 15) is (0, 0)
    ticks = np.linspace(-grid_half, grid_half, grid_dim)
    ax.set_xticks(np.arange(0, grid_dim, 1))
    ax.set_xticklabels(np.round(ticks[::-1], 2))  # Reflect correct tick labels (real-world values)
    ax.set_yticks(np.arange(0, grid_dim, 1))
    ax.set_yticklabels(np.round(ticks[::], 2))  # Reverse Y-axis for proper display
    ax.grid(which="both", color=grid_color)  # Set grid color

    # Crop the plot to the edge of the arena (defined by grid_size)
    ax.set_xlim(0, grid_dim - 1)  # X-axis limits to match grid size
    ax.set_ylim(0, grid_dim - 1)  # Y-axis limits to match grid size

    # Draw bold lines around the arena's edges
    arena_edges_x = [0, grid_dim - 1, grid_dim - 1, 0, 0]
    arena_edges_y = [0, 0, grid_dim - 1, grid_dim - 1, 0]
    ax.plot(arena_edges_x, arena_edges_y, color=border_color, linewidth=3)  # Bold line for border

    # Remove the axis for cleaner visualization
    plt.axis('off')

    # Create a directory for saving plots
    current_dir = os.getcwd()
    pics_dir = os.path.join(current_dir, 'pics')
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)

    # Save the cropped plot inside the 'pics' folder
    plot_path = os.path.join(pics_dir, 'path_plot.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())  # Include background color
    #print("path plotted")
    plt.close()