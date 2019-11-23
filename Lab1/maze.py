import numpy as np


def generate_maze(shape, walls):
    """ The function generates a maze that will have dimension , shape, ans walls:  """
    maze = np.ones(shape)

    for i in walls:
        maze[i] = 0

    return np.pad(maze, 1, mode='constant')


def fill_values(x_per, y_per, x_min, y_min, t, Tmax, maze, maze_mino, values, escape_pos, actions, actions_mino):
    max_value = -1
    optimal_action = -1

    if maze[x_per + 1, y_per + 1] and maze_mino[x_min + 1, y_min + 1]:
        # if Escapes value = 1
        if ((x_per, y_per) == escape_pos) and ((x_min, y_min) != escape_pos):
            max_value = 1
        elif [x_per, y_per] == [x_min, y_min]:
            max_value = 0
        else:
            a = 0
            # iterate on actions for humans
            for a_per in actions:
                value = 0
                if maze[x_per + 1 + a_per[0], y_per + 1 + a_per[1]]:
                    sum_a_mino = 0
                    for a_mino in actions_mino:
                        if maze_mino[x_min + 1 + a_mino[0], y_min + 1 + a_mino[1]]:
                            value = value + values[
                                x_per + a_per[0], y_per + a_per[1], x_min + a_mino[0], y_min + a_mino[1], Tmax - 1 - t]
                            sum_a_mino = sum_a_mino + maze_mino[x_min + 1 + a_mino[0], y_min + 1 + a_mino[1]]
                    value = value / sum_a_mino
                else:
                    value = -10
                # We will choose the best action and save a code
                if max_value < value:
                    max_value = value
                    optimal_action = a

                a = a + 1

    return max_value, optimal_action


