import numpy as np
from Lab1.maze import generate_maze, fill_values
import matplotlib.pyplot as plt
from Lab1.drawer import draw_full_path
from random import randrange

xdim = 7
ydim = 8
dimensions = [7, 8]
walls = [(0, 2), (1, 2), (2, 2), (3, 2), (1, 5), (2, 5), (3, 5), (2, 6), (2, 7), (5, 1), (5, 1), (5, 2), (5, 3),
             (5, 4), (5, 5), (5, 6), (6, 4)]
actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
maze = generate_maze(dimensions, walls)
maze_mino = generate_maze(dimensions, [])
escape_pos = (6, 5)


def solve(Tmax, actions_mino):
    values = -np.ones((xdim, ydim, xdim, ydim, Tmax))
    actions_op = -np.ones((xdim, ydim, xdim, ydim, Tmax))

    # Fill in initial states
    for x_per in range(xdim):
        for y_per in range(ydim):
            for x_min in range(xdim):
                for y_min in range(ydim):
                    # Do not complete impossible settings i.e. person in the wall
                    if maze[x_per + 1, y_per + 1] and maze_mino[x_min + 1, y_min + 1]:
                        if ((x_per, y_per) == escape_pos) and ((x_min, y_min) != escape_pos):
                            values[x_per, y_per, x_min, y_min, Tmax - 1] = 1
                        else:
                            values[x_per, y_per, x_min, y_min, Tmax - 1] = 0

    for t in range(Tmax - 1):
        for x_per in range(xdim):
            for y_per in range(ydim):
                for x_min in range(xdim):
                    for y_min in range(ydim):
                        v, a = fill_values(x_per, y_per, x_min, y_min, t,
                                           Tmax, maze, maze_mino, values, escape_pos, actions, actions_mino)
                        values[x_per, y_per, x_min, y_min, Tmax - 2 - t] = v
                        actions_op[x_per, y_per, x_min, y_min, Tmax - 2 - t] = a
    return values, actions_op


def save_game(name, a, Tmax):
    person_path = [(0, 0)]
    min_path = [escape_pos]
    current = person_path[0]
    for t in range(Tmax):
        # Check if done.
        if person_path[t] == escape_pos:
            break

        # Calculate optimal path.
        action_index = int(a[person_path[t][0], person_path[t][1], min_path[t][0], min_path[t][1], t])
        new_point = (person_path[t][0] + actions[action_index][0], person_path[t][1] + actions[action_index][1])
        person_path.append(new_point)

        # Calculate random path.
        action = actions_mino[randrange(0, len(actions_mino), 1)]
        new_point = (min_path[t][0] + action[0], min_path[t][1] + action[1])
        while not maze_mino[new_point[0] + 1, new_point[1] + 1]:
            action = actions_mino[randrange(0, len(actions_mino), 1)]
            new_point = (min_path[t][0] + action[0], min_path[t][1] + action[1])

        min_path.append(new_point)
    images = draw_full_path(person_path, min_path, walls)

    for i, image in enumerate(images):
        image.save("images/" + name + str(i) + ".png")


if __name__ == '__main__':
    Tmax = 20

    # Version 1: Minotaur can not stay still
    actions_mino = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    v, a = solve(Tmax, actions_mino)
    initial = v[0, 0, 6, 5, :]
    plt.figure()
    plt.plot(initial, label="Probability of escape")
    plt.xlabel("T")
    plt.legend()
    plt.show()
    save_game("moving", a, Tmax)



    # Version 2: Minotaur can stay still
    actions_mino = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    v, a = solve(Tmax, actions_mino)
    initial = v[0, 0, 6, 5, :]
    plt.figure()
    plt.plot(initial, label="Probability of escape")
    plt.xlabel("T")
    plt.legend()
    plt.show()
    save_game("still", a, Tmax)
