import numpy as np
import matplotlib.pyplot as plt
import heapq

# Define the possible cell values
WALL = '#'
OPEN = ' '
START = 'S'
GOAL = 'G'
PATH = '.'

# Define possible movements (4-directional, no diagonal moves)
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def heuristic(node, goal):
    # Define a simple heuristic (Manhattan distance)
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def a_star(maze):
    start = None
    goal = None

    # Find the start and goal positions
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == START:
                start = (i, j)
            elif maze[i][j] == GOAL:
                goal = (i, j)

    if start is None or goal is None:
        raise ValueError("Start and/or goal not found in the maze")

    open_list = [(0, start)]
    closed_set = set()
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in g_score:
                path.insert(0, current)
                current = g_score[current]
            return path

        closed_set.add(current)

        for move in MOVES:
            neighbor = (current[0] + move[0], current[1] + move[1])

            if neighbor[0] < 0 or neighbor[0] >= len(maze) or neighbor[1] < 0 or neighbor[1] >= len(maze[0]) or maze[neighbor[0]][neighbor[1]] == WALL:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, 0):
                continue

            if tentative_g_score < g_score.get(neighbor, 0) or neighbor not in [node[1] for node in open_list]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Example 5x5 maze
maze = [
    "S### ",
    "  #  ",
    "  ###",
    "#   #",
    "### G"
]

path = a_star(maze)

if path:
    # Mark the path in the maze
    for i, j in path:
        maze[i] = maze[i][:j] + PATH + maze[i][j + 1:]

    # Create a numerical array to represent the maze colors
    rows, cols = len(maze), len(maze[0])
    color_map = {'#': 0, ' ': 1, 'S': 2, 'G': 3, '.': 4}
    numeric_grid = np.array([[color_map[cell] for cell in row] for row in maze])

    # Define colors for maze elements
    cmap = plt.matplotlib.colors.ListedColormap(['black', 'white', 'green', 'red', 'blue'])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the maze
    ax.matshow(numeric_grid, cmap=cmap)

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
else:
    print("No path found")
