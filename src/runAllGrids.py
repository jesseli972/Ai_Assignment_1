import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from repeated_Forward_astar import repeated_forward_astar
from repeated_backward_astar import repeated_backward_astar
from adaptive_astar import adaptive_astar

def find_nearest_unblocked(grid, position):
    """
    Finds the nearest unblocked cell to the given position.
    Uses BFS to find the closest unblocked cell.
    """

    rows, cols = grid.shape
    queue = deque([position])
    visited = set([position])

    while queue:
        x, y = queue.popleft()

        if grid[x, y] == 0:
            return (x, y)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None

def visualize_grid_with_path(grid, path, filename):
    """
    Saves an image of the grid with the path highlighted in green.
    """
    plt.figure(figsize=(6, 6))
    grid_img = np.zeros((*grid.shape, 3))

    # Set colors
    grid_img[grid == 1] = [0, 0, 0]  # Blocked cells Black
    grid_img[grid == 0] = [1, 1, 1]  # Unblocked cells White

    # Draw path in Green
    for x, y in path:
        grid_img[x, y] = [0, 1, 0]

    plt.imshow(grid_img, origin="upper")
    plt.xticks([])
    plt.yticks([])
    plt.title("Gridworld with Path")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def runAll50Grids():
    """
    Loads all 50 gridworlds, validates start/goal positions, and runs the three pathfinding algorithms.
    Saves an image of the grid path.
    """
    grid_dir = "gridworlds/"
    image_dir = "gridworlds/images/"
    os.makedirs(image_dir, exist_ok=True)

    total_results = []

    for i in range(1, 51): 
        filename = os.path.join(grid_dir, f"gridworld{i}.txt")
        
        if not os.path.exists(filename):
            print(f"Grid file {filename} not found. Skipping...")
            continue

        print(f"\nCurrent Grid: {filename}...")
        grid = np.loadtxt(filename, dtype=int)
        start, goal = (0, 0), (grid.shape[0] - 1, grid.shape[1] - 1)

        # Ensure start/goal are unblocked
        if grid[start] == 1:
            start = find_nearest_unblocked(grid, start)
        if grid[goal] == 1:
            goal = find_nearest_unblocked(grid, goal)

        if start is None or goal is None:
            print(f"Grid {i}: No valid start/goal found. Skipping...")
            continue

        print(f"Grid {i}: Start = {start}, Goal = {goal}")

        # Run each A* variant
        path_fwd, nodes_expanded_fwd = repeated_forward_astar(grid, start, goal)
        path_bwd, nodes_expanded_bwd = repeated_backward_astar(grid, start, goal)
        path_adaptive, nodes_expanded_adaptive = adaptive_astar(grid, start, goal)

        # Store results
        total_results.append((i, nodes_expanded_fwd, nodes_expanded_bwd, nodes_expanded_adaptive))
        print(f"Grid {i}: Nodes Expanded - Forward A*: {nodes_expanded_fwd}, Backward A*: {nodes_expanded_bwd}, Adaptive A*: {nodes_expanded_adaptive}")

        
        #Save grid visualization with paths
        if path_fwd:
            visualize_grid_with_path(grid, path_fwd, os.path.join(image_dir, f"grid{i}_forward.png"))
        if path_bwd:
            visualize_grid_with_path(grid, path_bwd, os.path.join(image_dir, f"grid{i}_backward.png"))
        if path_adaptive:
            visualize_grid_with_path(grid, path_adaptive, os.path.join(image_dir, f"grid{i}_adaptive.png"))

    # Display summary of results
    print("\nSUMMARY OF RESULTS")
    print("Grid | Forward A* | Backward A* | Adaptive A*")
    for res in total_results:
        print(f"{res[0]:4} | {res[1]:10} | {res[2]:10} | {res[3]:10}")

if __name__ == "__main__":
    runAll50Grids()
