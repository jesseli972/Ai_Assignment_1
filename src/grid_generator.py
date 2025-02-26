import random
import numpy as np
import os
import matplotlib.pyplot as plt

GRID_SIZE = 101  # Size of the grid
NUM_GRIDWORLDS = 50  # Number of gridworlds to generate
BLOCKED_PROBABILITY = 0.3  # Probability that a cell is blocked
UNBLOCKED = 0  # Unblocked cell
BLOCKED = 1  # Blocked cell

def generate_gridworld():
    # Initialize the grid with all unvisited cells
    grid = np.full((GRID_SIZE, GRID_SIZE), -1)
    stack = []

    # Get a random start point and append to the stack
    start_x, start_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    grid[start_x, start_y] = UNBLOCKED
    stack.append((start_x, start_y))

    # Perform DFS to generate the grid
    while True:
        if not stack:
            unvisited_cells = np.argwhere(grid == -1)
            if len(unvisited_cells) == 0:
                break  # All cells are visited
            start_x, start_y = unvisited_cells[random.randint(0, len(unvisited_cells) - 1)]
            grid[start_x, start_y] = UNBLOCKED
            stack.append((start_x, start_y))

        x, y = stack[-1]
        neighbors = []

        # Get unvisited neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == -1:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            if random.random() < BLOCKED_PROBABILITY:
                grid[nx, ny] = BLOCKED  # Mark as blocked
            else:
                grid[nx, ny] = UNBLOCKED  # Mark as unblocked and continue DFS
                stack.append((nx, ny))
        else:
            stack.pop()  # No unvisited neighbors, backtrack

    return grid

# Function to save grids in text files
def save_gridworld(grid, filename):
    np.savetxt(filename, grid, fmt='%d')

# Function to visualize the gridworld
def visualize_gridworld(grid, filename=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray_r", origin="upper")  # Black = blocked, white = unblocked
    plt.xticks([])
    plt.yticks([])
    plt.title("Gridworld Visualization")
    
    if filename:
        plt.savefig(filename, bbox_inches="tight")  # Save as image
    else:
        plt.show()  # Show the plot

def generate_and_save_gridworlds():
    if not os.path.exists("gridworlds"):
        os.makedirs("gridworlds")
    if not os.path.exists("gridworlds/images"):
        os.makedirs("gridworlds/images")

    for i in range(NUM_GRIDWORLDS):
        grid = generate_gridworld()
        filename_txt = f"gridworlds/gridworld{i+1}.txt"
        filename_img = f"gridworlds/images/gridworld{i+1}.png"
        
        save_gridworld(grid, filename_txt)
        visualize_gridworld(grid, filename_img)  # Save visualization

if __name__ == "__main__":
    generate_and_save_gridworlds()
