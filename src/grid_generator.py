import random
import numpy as np
import os

GRID_SIZE = 101 
NUM_GRIDWORLDS = 50  
BLOCKED_PROBABILITY = 0.3 
UNBLOCKED = 0 
BLOCKED = 1

def generate_gridworld():
    #initialize the grid with all unvisited cells
    grid = np.full((GRID_SIZE, GRID_SIZE), -1)
    #initialize the stack
    stack = []

    # Get random start point and append to the stack
    start_x, start_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    grid[start_x, start_y] = UNBLOCKED
    stack.append((start_x, start_y))

    # Perform DFS
    while True:
        #condition to check if the stack is empty but there are still unvisited cells
        if not stack:
            unvisited_cells = np.argwhere(grid == -1)
            if len(unvisited_cells) == 0:
                break #all cells are visited so break teh loop
            start_x, start_y = unvisited_cells[random.randint(0, len(unvisited_cells) - 1)]
            grid[start_x, start_y] = UNBLOCKED
            stack.append((start_x, start_y))

        x, y = stack[-1]
        neighbors = []

        #get neighbors and append if unvisited
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == -1:
                neighbors.append((nx, ny))

        if neighbors:
            #break tie randomly
            nx, ny = random.choice(neighbors)
            #Cell is blocked
            if random.random() < BLOCKED_PROBABILITY:
                grid[nx, ny] = BLOCKED
            #Cell is unblocked and append to the stack
            else:
                grid[nx, ny] = UNBLOCKED
                stack.append((nx, ny))
        else:
            #No unvisited neighbors so backtrack
            stack.pop()

    return grid

#function to save grids in text file
def save_gridworld(grid, filename):
    np.savetxt(filename, grid, fmt='%d')

def generate_and_save_gridworlds():
    if not os.path.exists("gridworlds"):
        os.makedirs("gridworlds")

    for i in range(NUM_GRIDWORLDS):
        grid = generate_gridworld()
        filename = f"gridworlds/gridworld{i+1}.txt"
        save_gridworld(grid, filename)

if __name__ == "__main__":
    generate_and_save_gridworlds()