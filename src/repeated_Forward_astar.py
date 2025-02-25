import heapq
import numpy as np

UNBLOCKED = 0
BLOCKED = 1

def manhattan_distance(a, b):
    """
    Compute the Manhattan distance between two cells a and b.
    a, b: tuples (x, y)
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(cell, grid):
    """
    For a given cell (x, y), return all neighboring cells that are in bounds and unblocked.
    """
    neighbors = []
    x, y = cell
    # Four possible moves: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Check bounds
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == UNBLOCKED:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    """
    Reconstruct the path from the start to the current cell by backtracking through came_from.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(grid, start, goal):
    """
    Perform an A* search on the grid from start to goal.
    
    Parameters:
      grid  : a 2D numpy array where 0 represents unblocked and 1 represents blocked.
      start : tuple (x, y) indicating the start cell.
      goal  : tuple (x, y) indicating the goal cell.
      
    Returns:
      A list of cells (tuples) representing the path from start to goal, or None if no path is found.
    """
    open_set = []
    # Each element in the open set is a tuple: (f, g, cell)
    start_h = manhattan_distance(start, goal)
    heapq.heappush(open_set, (start_h, 0, start))
    
    came_from = {}       # For reconstructing the path.
    g_score = {start: 0} # Cost from start to each cell.

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        # Check if we have reached the goal.
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Expand neighbors.
        for neighbor in get_neighbors(current, grid):
            temp_g = g_score[current] + 1  # Cost for a move is 1.
            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                f_score = temp_g + manhattan_distance(neighbor, goal)
                # Make tentative_g negative to favor a larger g
                heapq.heappush(open_set, (f_score, temp_g, neighbor))
                came_from[neighbor] = current

    # If we exit the loop without finding the goal, no path exists.
    return None

# Example
if __name__ == "__main__":
    grid = np.loadtxt("gridworlds/gridworld4.txt", dtype=int)
    
    # Define start and goal positions.
    # (Ensure that the start and goal cells are unblocked!)
    start = (0, 0)
    goal = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    path = astar(grid, start, goal)
    if path:
        print("Path found:")
        print(path)
    else:
        print("No path found.")
