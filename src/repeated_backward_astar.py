import heapq
import numpy as np
from collections import deque

# Constants
UNBLOCKED = 0
BLOCKED = 1

def manhattan_distance(a, b):
    """
    Compute the Manhattan distance between two cells a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(cell, grid):
    """
    return all neighboring cells that are in bounds and unblocked.
    """
    neighbors = []
    x, y = cell
    #up, down, left, right
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

def astar(grid, start, goal, tie_breaking="larger_g"):
    """
    Perform an A* search on the grid from start to goal.
    
    returns:
        path: a list of tuples representing the path from start to goal, or None if no path is found.
    """
    open_set = []
    start_h = manhattan_distance(start, goal)
    heapq.heappush(open_set, (start_h, 0, start))
    
    came_from = {}       
    g_score = {start: 0} 
    nodes_expanded = 0 
    closed_set = set()

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        # Check if reached the goal.
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

    # If exit the loop without finding the goal, no path exists.
    return None, nodes_expanded

def repeated_backward_astar(grid, start, goal):
    """
    Repeated Backward A* algorithm.
    The agent searches from the goal to its current position and moves along the path.
    returns:
        path: a list of tuples representing the path from start to goal, or None if no path is found.
        total_nodes_expanded: The total number of nodes expanded during the search.
    """

    # Initialize the all needed variables
    agent_position = start
    agent_knowledge = np.full_like(grid, UNBLOCKED)  #assume all cells are unblocked
    returnedPath = []
    total_nodes_expanded = 0

    while agent_position != goal:
        path, nodes_expanded = astar(agent_knowledge, goal, agent_position)
        total_nodes_expanded += nodes_expanded 

        if not path:
            print("No path found.")
            print(f"Total nodes expanded: {total_nodes_expanded}")
            return None, total_nodes_expanded
        
        # Move the agent along the path (in reverse order)
        for next_cell in reversed(path):  
            if grid[next_cell[0], next_cell[1]] == BLOCKED:
                # Update the agent's knowledge: mark this cell as blocked
                agent_knowledge[next_cell[0], next_cell[1]] = BLOCKED
                # print("unBlocked cells so far:")
                # print(returnedPath)
                break 
            else:
                # Move to the next cell
                agent_position = next_cell
                returnedPath.append(agent_position)

        else:
            #agent_position == goal
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

    rows, cols = grid.shape
    queue = deque([position])
    visited = set([position])

    while queue:
        x, y = queue.popleft()

        if grid[x, y] == UNBLOCKED:
            return (x, y)

        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return None

# Example usage
if __name__ == "__main__":
    # Load a gridworld from a file
    grid = np.loadtxt("gridworlds/gridworld1.txt", dtype=int)

    # Define start and goa
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
    # Run Repeated Backward A*
    path, nodes_expanded = repeated_backward_astar(grid, start, goal)
    if path:
        print("Path found:")
        print(path)
    else:
        print("No path found.")