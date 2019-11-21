import numpy as np
import random
from Lab1.maze import generate_maze


xdim = 3
ydim = 6
start_pos = [0,0]
start_pos_pol = [1, 2]
banks_pos = [[0, 0], [2, 0], [0, 5], [2, 5]]
reward_bank = 10
reward_caught = 50
actions_pol = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def check_compatibility(pos_rob, ini_pol, next_pol):
    dif_ini = [max(abs(pos_rob[0] - ini_pol[0]), 1), max(abs(pos_rob[1] - ini_pol[1]), 1)]
    dif_next = [abs(pos_rob[0] - next_pol[0]), abs(pos_rob[1] - next_pol[1])]
    return (dif_next[0] > dif_ini[0]) or (dif_next[1] > dif_ini[1])


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


if __name__ == '__main__':

    disc_factor = 0.8

    dimensions = [xdim, ydim]
    maze = generate_maze(dimensions, [])

    # Define robber actions
    actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    n_a = len(actions)
    # Initialize value function
    V = np.zeros((xdim, ydim, xdim, ydim))
    V1 = np.zeros((xdim, ydim, xdim, ydim))

    delta = 1000
    eps = 0.1
    limit = eps * (1-disc_factor) / disc_factor

    rewards = []
    rewards = []
    rewards = []

    for x_rob in range(xdim):
        for y_rob in range(ydim):
            for x_pol in range(xdim):
                for y_pol in range(ydim):
                    for action in range(n_a):
                        reward, new_rob, new_pol = apply_action([x_rob, y_rob], [x_pol, y_pol], action, maze)
                        rew
    while delta > eps:



        delta = np.linalg.norm(V-V1)

        V = V1





