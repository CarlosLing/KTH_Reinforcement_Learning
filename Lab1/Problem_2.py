import numpy as np
import random
import matplotlib.pyplot as plt
from Lab1.maze import generate_maze
from Lab1.drawer import draw_full_path


xdim = 3
ydim = 6
start_pos = [0,0]
start_pos_pol = [1, 2]
banks_pos = [[0, 0], [2, 0], [0, 5], [2, 5]]
reward_bank = 10
reward_caught = -50
actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
actions_pol = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def check_compatibility(pos_rob, ini_pol, next_pol):
    dif_ini = [max(abs(pos_rob[0] - ini_pol[0]), 1), max(abs(pos_rob[1] - ini_pol[1]), 1)]
    dif_next = [abs(pos_rob[0] - next_pol[0]), abs(pos_rob[1] - next_pol[1])]
    return (dif_next[0] <= dif_ini[0]) and (dif_next[1] <= dif_ini[1])


def get_reward(p_rob, p_pol):
    if p_rob == p_pol:
        r = reward_caught
    elif p_rob in banks_pos:
        r = reward_bank
    else:
        r = 0

    return r


def move_robber(p_rob, action, maze):

    new_rob_maze = [p_rob[0] + action[0] + 1, p_rob[1] + action[1] + 1]
    # Is it possible to move where the action says?
    possible = int(maze[new_rob_maze[0], new_rob_maze[1]])
    # If yes move, else stay
    new_rob = [p_rob[0] + action[0] * possible, p_rob[1] + action[1] * possible]

    return new_rob


def get_position_policeman(p_rob, p_pol, maze):
    # Move policeman
    possible_actions_pol = []
    for i in actions_pol:
        pos_pol_maze = [p_pol[0] + i[0] + 1, p_pol[1] + i[1] + 1]
        if maze[pos_pol_maze[0], pos_pol_maze[1]]:
            if check_compatibility(p_rob, p_pol, [pos_pol_maze[0] - 1, pos_pol_maze[1] - 1]):
                possible_actions_pol.append(i)

    new_pol = []
    for action_pol in possible_actions_pol:
        new_pol.append([p_pol[0] + action_pol[0], p_pol[1] + action_pol[1]])

    return


def apply_action(p_rob, p_pol, action, maze):

    # Check for rewards
    if p_rob == p_pol:
        new_rob= start_pos
        new_pol = [start_pos_pol]
        reward = reward_caught
    else:
        if p_rob in banks_pos:
            reward = reward_bank
        else:
            reward = 0

        # Move robber
        new_rob_maze = [p_rob[0] + action[0] + 1, p_rob[1] + action[1] + 1]
        # Is it possible to move where the action says?
        possible = int(maze[new_rob_maze[0], new_rob_maze[1]])
        # If yes move, else stay
        new_rob = [p_rob[0] + action[0] * possible, p_rob[1] + action[1] * possible]

        # Move policeman
        possible_actions_pol = []
        for i in actions_pol:
            pos_pol_maze = [p_pol[0] + i[0] + 1, p_pol[1] + i[1] + 1]
            if maze[pos_pol_maze[0], pos_pol_maze[1]]:
                if check_compatibility(p_rob, p_pol, [pos_pol_maze[0] - 1, pos_pol_maze[1] - 1]):
                    possible_actions_pol.append(i)

        new_pol = []
        for action_pol in possible_actions_pol:
            new_pol.append([p_pol[0] + action_pol[0], p_pol[1] + action_pol[1]])

    return reward, new_rob, new_pol


def draw_game(name, V, Tmax):
    rob_path = [start_pos]
    pol_path = [start_pos_pol]
    for t in range(Tmax):
        if rob_path[t] == pol_path[t]:
            break

        # Calculate optimal path.
        value = -1000
        opt_new_pos = []
        for action in actions:
            new_pos = [rob_path[t][0] + action[0], rob_path[t][1] + action[1]]
            if maze[new_pos[0] + 1, new_pos[1] + 1]:

                available_pol_pos = []
                for pol_action in actions_pol:
                    new_pol_pos = [pol_path[t][0] + pol_action[0], pol_path[t][1] + pol_action[1]]
                    if check_compatibility(rob_path[t], pol_path[t], new_pol_pos):
                        if maze[new_pol_pos[0] + 1, new_pol_pos[1] + 1]:
                            available_pol_pos.append(new_pol_pos)

                val = 0
                for new_pol_pos in available_pol_pos:
                    val = val + V[new_pos[0], new_pos[1], new_pol_pos[0], new_pol_pos[1]]
                val = val/len(available_pol_pos)

                #print("Action:" + str(action) + " New Pos: " + str(new_pos) + " Value: " + str(val))
                if val > value:
                    value = val
                    opt_new_pos = new_pos

        rob_path.append(opt_new_pos)

        available_pol_pos = []
        for pol_action in actions_pol:
            new_pol_pos = [pol_path[t][0] + pol_action[0], pol_path[t][1] + pol_action[1]]
            if check_compatibility(rob_path[t], pol_path[t], new_pol_pos):
                if maze[new_pol_pos[0] + 1, new_pol_pos[1] + 1]:
                    available_pol_pos.append(new_pol_pos)

        pos = random.choice(available_pol_pos)
        pol_path.append(pos)

    images = draw_full_path(rob_path, pol_path, [], banks_pos)

    for i, image in enumerate(images):
        image.save("images/" + name + str(i) + ".png")

if __name__ == '__main__':

    df = np.arange(0, 10, 1)
    df = np.delete(df, 0)
    df = np.multiply(df, 0.1)
    df = np.insert(df, 8, 0.85)
    valu_ini = []
    for disc_factor in df:

        dimensions = [xdim, ydim]
        maze = generate_maze(dimensions, [])

        n_a = len(actions)
        # Initialize value function
        V = np.zeros((xdim, ydim, xdim, ydim))
        V1 = np.zeros((xdim, ydim, xdim, ydim))

        delta = 1000
        eps = 0.001
        limit = eps * (1-disc_factor) / disc_factor
        x = 1
        while delta > eps:
            x = x + 1
            for x_rob in range(xdim):
                for y_rob in range(ydim):
                    for x_pol in range(xdim):
                        for y_pol in range(ydim):
                            v = -1000
                            for action in range(n_a):
                                reward, new_rob, new_pol = apply_action([x_rob, y_rob], [x_pol, y_pol], actions[action], maze)
                                next_v = []
                                for pos_pol in new_pol:
                                    next_v.append(V[new_rob[0], new_rob[1], pos_pol[0], pos_pol[1]])
                                v = max(v, reward + disc_factor*(np.mean(next_v)))
                            V1[x_rob, y_rob, x_pol, y_pol] = v

            delta = np.linalg.norm(V-V1)

            V = np.ndarray.copy(V1)

        print(V[0,0,1,2])
        valu_ini.append(V[0,0,1,2])

        if disc_factor == 0.85:
            draw_game("bank_heist85-", V, 19)
        else:
            draw_game("bank_heist" + str(int(10*disc_factor)) + "-", V, 19)

    plt.figure()
    plt.plot(df, valu_ini)
    plt.xlabel("Discount Factor")
    plt.ylabel("Value")
    plt.show()



