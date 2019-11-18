import numpy as np


def generate_maze(shape, walls):
    """ The function generates a maze that will have dimension , shape, ans walls:  """
    maze = np.ones(shape)

    for i in walls:
        maze[i] = 0

    return np.pad(maze, 1, mode='constant')


if __name__ == '__main__':

    # Define the maze
    m_shape = (6, 7)
    m_walls = [(0, 2), (1, 2), (2, 2), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
    maze = generate_maze(m_shape, m_walls)

    # Define initial and final state
    initial_state = (1, 1)
    final_state = (6, 6)

    # Define reward, at each time if we want to
    reward = -1

    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    #

    line = 7

    # Check for a cycle
    repeated = False

    num = int(line)
    num_set = set({num})

    while ((not repeated) and num != 1):

        # I couldn't import numpy but if i could:
        # digits = np.array(list(str(num)), dtype = int)
        # num = sum(digits**2)

        num = 0
        for i in list(str(num)):
            num = num + int(i) ** 2

        if num in num_set:
            repeated = True

        num_set.add(num)

    if repeated:
        print(0, end="")
    else:
        print(1, end="")
