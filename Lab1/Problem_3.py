import numpy as np
import random
import matplotlib.pyplot as plt
from Lab1.maze import generate_maze


xdim = 4
ydim = 4
start_pos = [0,0]
start_pos_pol = [xdim-1, ydim-1]
bank_pos = [1, 1]
reward_bank = 1
reward_caught = -10
actions_pol = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def choose_random_state(x_max, y_max):

    position_robber = [random.choice(range(x_max)), random.choice(range(y_max))]
    position_police = [random.choice(range(x_max)), random.choice(range(y_max))]

    return position_robber, position_police


def apply_action(p_rob, p_pol, action, maze):

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
            possible_actions_pol.append(i)

    action_pol = random.choice(possible_actions_pol)
    new_pol = [p_pol[0] + action_pol[0], p_pol[1] + action_pol[1]]

    # Check for rewards
    if new_pol == new_rob:
        reward = reward_caught
    elif new_rob == bank_pos:
        reward = reward_bank
    else:
        reward = 0

    return reward, new_rob, new_pol


if __name__ == '__main__':

    # Q-learning

    disc_factor = 0.8

    dimensions = [xdim, ydim]
    maze = generate_maze(dimensions, [])

    # Define robber actions
    actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
    n_a = len(actions)

    #Initialize Q and n functions
    Q = np.zeros((xdim, ydim, xdim, ydim, n_a))
    n_sa = np.zeros((xdim, ydim, xdim, ydim, n_a))

    N_iter = 10000000
    Q_plot = np.zeros((N_iter, n_a))

    for i in range(N_iter):
        # Choose random state
        p_rob, p_pol = choose_random_state(xdim, ydim)

        # Choose random action
        i_action = random.choice(range(n_a))
        action = actions[i_action]

        # Imput action to the system
        reward, new_rob, new_pol = apply_action(p_rob, p_pol, action, maze)

        # Update Q function
        n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] = n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] + 1
        stepsize = 1/(n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action]**(2/3))
        Q_prev = Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action]
        Q_next_state_max = max(Q[new_rob[0], new_rob[1], new_pol[0], new_pol[1], :])
        Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] = Q_prev + stepsize * (reward + disc_factor * Q_next_state_max - Q_prev)

        Q_plot[i, :] = Q[0, 0, 3, 3, :]

    plt.figure()
    plt.plot(Q_plot[0:1000000, :])
    plt.show()

    # SARSA

    # Initialize Q and n functions
    Q = np.zeros((xdim, ydim, xdim, ydim, n_a))
    n_sa = np.zeros((xdim, ydim, xdim, ydim, n_a))

    N_iter = 10000000
    Q_plots = np.zeros((N_iter, n_a))

    eps = 0.1

    for i in range(N_iter):
        # Choose random state
        p_rob, p_pol = choose_random_state(xdim, ydim)

        # The only difference is choosing the action
        if np.random.binomial(1, (1-eps)):
            # Choose the function with the best Q
            Q_vector = Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], :]
            i_action = np.argmax(Q_vector)
        else:
            # Choose random action
            i_action = random.choice(range(n_a))
        action = actions[i_action]

        # Imput action to the system
        reward, new_rob, new_pol = apply_action(p_rob, p_pol, action, maze)

        # For the next action
        if np.random.binomial(1, (1-eps)):
            # Choose the function with the best Q
            Q_vector = Q[new_rob[0], new_rob[1], new_pol[0], new_pol[1], :]
            i1_action = np.argmax(Q_vector)
        else:
            # Choose random action
            i1_action = random.choice(range(n_a))

        # Update Q function
        n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] = n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[
            1], i_action] + 1
        stepsize = 1 / (n_sa[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] ** (2 / 3))
        Q_prev = Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action]
        Q_state_max = Q[new_rob[0], new_rob[1], new_pol[0], new_pol[1], i1_action]
        Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] = Q_prev + stepsize * (
                    reward + disc_factor * Q_state_max - Q_prev)

        Q_plots[i, :] = Q[0, 0, 3, 3, :]

    plt.figure()
    plt.plot(Q_plots[0:10000000, :])
    plt.show()

