import heapq
import numpy as np

# Constants
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
    path.reverse()  # Ensure the path is from start to goal
    return path

def astar(grid, start, goal, tie_breaking="larger_g"):
    """
    Perform an A* search on the grid from start to goal.
    Parameters:
      grid: a 2D numpy array where 0 represents unblocked and 1 represents blocked.
      start: tuple (x, y) indicating the start cell.
      goal: tuple (x, y) indicating the goal cell.
    Returns:
      A list of cells (tuples) representing the path from start to goal, or None if no path is found.
    """
    open_set = []
    # Each element in the open set is a tuple: (f, g, cell)
    start_h = manhattan_distance(start, goal)
    heapq.heappush(open_set, (start_h, 0, start))
    
    came_from = {}       # For reconstructing the path.
    g_score = {start: 0} # Cost from start to each cell.
    nodes_expanded = 0  # Counter for nodes expanded
    closed_set = set()

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        # Check if we have reached the goal.
        if current == goal:
            return reconstruct_path(came_from, current), nodes_expanded
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        nodes_expanded += 1  # Increment the counter
        # Expand neighbors.
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                if tie_breaking == "smaller_g":
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                elif tie_breaking == "larger_g":
                    heapq.heappush(open_set, (f_score, -tentative_g, neighbor))
                came_from[neighbor] = current

    # If we exit the loop without finding the goal, no path exists.
    return None, nodes_expanded

def repeated_forward_astar(grid, start, goal):
    """
    Repeated Forward A* algorithm.
    The agent searches from its current position to the goal and moves along the path.
    """
    agent_position = start
    agent_knowledge = np.full_like(grid, UNBLOCKED)  # Initially, assume all cells are unblocked
    returnedPath = []
    total_nodes_expanded = 0

    while agent_position != goal:
        # Run A* from the agent's current position to the goal
        path, nodes_expanded = astar(agent_knowledge, agent_position, goal)
        total_nodes_expanded += nodes_expanded  # Accumulate nodes expanded
        if not path:
            return None
        
        # Move the agent along the path (from start to goal)
        for next_cell in path:  # Skip the first cell (current position)
            if grid[next_cell[0], next_cell[1]] == BLOCKED:
                # Update the agent's knowledge: mark this cell as blocked
                agent_knowledge[next_cell[0], next_cell[1]] = BLOCKED
                break  # Stop moving and replan
            else:
                # Move to the next cell
                agent_position = next_cell
                returnedPath.append(agent_position)
        else:
            # If the loop completes without breaking, the agent has reached the goal
            print("Reached the goal!")
            print(f"Total nodes expanded: {total_nodes_expanded}")
            return returnedPath

    print("Reached the goal!")
    print(f"Total nodes expanded: {total_nodes_expanded}")
    return returnedPath

# Example usage
if __name__ == "__main__":
    # Load a gridworld from a file
    grid = np.loadtxt("gridworlds/gridworld4.txt", dtype=int)
    
    # Define start and goal positions
    start = (0, 0)
    goal = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    # Run Repeated Forward A*
    path = repeated_forward_astar(grid, start, goal)
    if path:
        print("Path found:")
        print(path)
    else:
        print("No path found.")