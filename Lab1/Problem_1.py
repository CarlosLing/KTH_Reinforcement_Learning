import numpy as np
from Lab1.maze import generate_maze, fill_values


if __name__ == '__main__':

    xdim = 7
    ydim = 8
    Tmax = 20
    dimensions = [xdim, ydim]
    walls = [(0, 2), (1, 2), (2, 2), (3, 2), (1, 5), (2, 5), (3, 5), (2, 6), (2, 7), (5, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 4)]
    maze = generate_maze(dimensions, walls)
    maze_mino = generate_maze(dimensions, [])
    escape_pos = [6,5]

    values = -np.ones((xdim,ydim,xdim,ydim,Tmax))
    actions_op = -np.ones((xdim,ydim,xdim,ydim,Tmax))

    actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    actions_mino = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    # Fill in initial states
    for x_per in range(xdim):
        for y_per in range(ydim):
            for x_min in range(xdim):
                for y_min in range(ydim):
                    # Do not complete impossible settings i.e. person in the wall
                    if maze[x_per+1, y_per+1] and maze_mino[x_min+1,y_min+1]:
                        if ([x_per, y_per] == escape_pos) and ([x_min, y_min] != escape_pos):
                            values[x_per, y_per, x_min, y_min, Tmax-1] = 1
                        else:
                            values[x_per, y_per, x_min, y_min, Tmax - 1] = 0

    for t in range(Tmax-1):
        for x_per in range(xdim):
            for y_per in range(ydim):
                for x_min in range(xdim):
                    for y_min in range(ydim):
                        v, a = fill_values(x_per, y_per, x_min, y_min, t,
                                           Tmax, maze, maze_mino, values, escape_pos, actions, actions_mino)
                        values[x_per, y_per, x_min, y_min, Tmax - 2 - t] = v
                        actions_op[x_per, y_per, x_min, y_min, Tmax - 2 - t] = a

    # Vestion 2: Minotaur can stay still ################

    values_2 = -np.ones((xdim, ydim, xdim, ydim, Tmax))
    actions_op_2 = -np.ones((xdim, ydim, xdim, ydim, Tmax))

    actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    actions_mino = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]

    # Fill in initial states
    for x_per in range(xdim):
        for y_per in range(ydim):
            for x_min in range(xdim):
                for y_min in range(ydim):
                    # Do not complete imposible settings i.e. person in the wall
                    if maze[x_per + 1, y_per + 1] and maze_mino[x_min + 1, y_min + 1]:
                        if ([x_per, y_per] == escape_pos) and ([x_min, y_min] != escape_pos):
                            values_2[x_per, y_per, x_min, y_min, Tmax - 1] = 1
                        else:
                            values_2[x_per, y_per, x_min, y_min, Tmax - 1] = 0

    for t in range(Tmax - 1):
        for x_per in range(xdim):
            for y_per in range(ydim):
                for x_min in range(xdim):
                    for y_min in range(ydim):
                        v, a = fill_values(x_per, y_per, x_min, y_min, t,
                                           Tmax, maze, maze_mino, values_2, escape_pos, actions, actions_mino)
                        values_2[x_per, y_per, x_min, y_min, Tmax - 2 - t] = v
                        actions_op_2[x_per, y_per, x_min, y_min, Tmax - 2 - t] = a

    test1 = values[:, :, 6, 5, 4]


