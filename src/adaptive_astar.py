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

def astar_adaptive(grid, start, goal, h_values):
    """
    Perform an A* search on the grid from start to goal with adaptive heuristic values.
    Parameters:
      grid: a 2D numpy array where 0 represents unblocked and 1 represents blocked.
      start: tuple (x, y) indicating the start cell.
      goal: tuple (x, y) indicating the goal cell.
      h_values: a dictionary storing the heuristic values for each cell.
    Returns:
      A list of cells (tuples) representing the path from start to goal, or None if no path is found.
      A set of cells (tuples) that were expanded during the search.
      A dictionary of came_from pointers for path reconstruction.
    """
    open_set = []
    start_h = h_values[start] if start in h_values else manhattan_distance(start, goal)
    heapq.heappush(open_set, (start_h, 0, start))
    
    came_from = {}
    g_score = {start: 0}
    closed_set = set()  # Track expanded states
    nodes_expanded = 0

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current), closed_set, came_from, nodes_expanded
        
        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + (h_values[neighbor] if neighbor in h_values else manhattan_distance(neighbor, goal))
                heapq.heappush(open_set, (f_score, -tentative_g, neighbor))
                came_from[neighbor] = current

    return None, closed_set, came_from, nodes_expanded

def adaptive_astar(grid, start, goal):
    """
    Adaptive A* algorithm.
    The agent searches from its current position to the goal and updates heuristic values after each search.
    """
    agent_position = start
    agent_knowledge = np.full_like(grid, UNBLOCKED)  # Initially, assume all cells are unblocked
    h_values = {}  # Store heuristic values for each cell
    returnedPath = []
    total_nodes_expanded = 0

    while agent_position != goal:
        # Run A* with adaptive heuristic values
        path, closed_set, came_from, nodes_expanded = astar_adaptive(agent_knowledge, agent_position, goal, h_values)
        total_nodes_expanded += nodes_expanded 
        if not path:
            print("No path found.")
            print(f"Total nodes expanded: {total_nodes_expanded}")
            return None, total_nodes_expanded
        
        # Update heuristic values for all expanded states
        goal_distance = len(path) - 1  # g(s_goal)
        for cell in closed_set:
            # Reconstruct the path to the cell to calculate g(s)
            cell_path = reconstruct_path(came_from, cell)
            g_cell = len(cell_path) - 1
            h_values[cell] = goal_distance - g_cell

        # Move the agent along the path
        for next_cell in path[1:]:  # Skip the first cell (current position)
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
            return returnedPath, total_nodes_expanded

    print("Reached the goal!")
    print(f"Total nodes expanded: {total_nodes_expanded}")
    return returnedPath, total_nodes_expanded

def find_nearest_unblocked(grid, position):
    """
    Finds the nearest unblocked cell to the given position.
    Uses a BFS search to find the closest unblocked cell.
    """
    from collections import deque

    rows, cols = grid.shape
    queue = deque([position])
    visited = set([position])

    while queue:
        x, y = queue.popleft()

        # If this cell is unblocked, return it
        if grid[x, y] == UNBLOCKED:
            return (x, y)

        # Check all four neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None  # No unblocked cells found (shouldn't happen in a valid grid)

# Example usage
if __name__ == "__main__":
    # Load a gridworld from a file
    grid = np.loadtxt("gridworlds/gridworld4.txt", dtype=int)

    # Define start and goal positions
    start = (0, 0)
    goal = (grid.shape[0] - 1, grid.shape[1] - 1)

    # Check if start or goal is blocked
    if grid[start] == BLOCKED:
        start = find_nearest_unblocked(grid, start)
        if start is None:
            exit()

    if grid[goal] == BLOCKED:
        goal = find_nearest_unblocked(grid, goal)
        if goal is None:
            exit()

    print(f"Using start: {start}, goal: {goal}")

    # Run Adaptive A*
    path, nodes_expanded = adaptive_astar(grid, start, goal)
    if path:
        print("Path found:")
        print(path)
    else:
        print("No path found.")