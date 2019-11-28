import numpy as np
import random
import matplotlib.pyplot as plt
from Lab1.drawer import draw_full_path
from Lab1.maze import generate_maze


xdim = 4
ydim = 4
start_pos = [0,0]
start_pos_pol = [xdim-1, ydim-1]
bank_pos = [1, 1]
reward_bank = 1
reward_caught = -10
actions_pol = [[0, 1], [0, -1], [1, 0], [-1, 0]]
actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
n_a = len(actions)

disc_factor = 0.8
dimensions = [xdim, ydim]
maze = generate_maze(dimensions, [])



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
    if p_pol == p_rob:
        reward = reward_caught
    elif p_rob == bank_pos:
        reward = reward_bank
    else:
        reward = 0

    return reward, new_rob, new_pol



def save_game(name, a, Tmax, SARSA = False, eps = 0.1):
    person_path = [(0, 0)]
    min_path = [(3,3)]
    current = person_path[0]
    for t in range(Tmax):

        # Calculate optimal path.
        if SARSA:
            if np.random.binomial(1, (1 - eps)):
                action_index = int(a[person_path[t][0], person_path[t][1], min_path[t][0], min_path[t][1]])
            else:
                action_index = random.choice(range(n_a))
        else:
            action_index = int(a[person_path[t][0], person_path[t][1], min_path[t][0], min_path[t][1]])

        new_point = (person_path[t][0] + actions[action_index][0], person_path[t][1] + actions[action_index][1])

        # Add Newpoints Robber
        person_path.append(new_point)

        # Calculate random path.
        action = actions_pol[random.randrange(0, len(actions_pol), 1)]
        new_point = (min_path[t][0] + action[0], min_path[t][1] + action[1])
        while not maze[new_point[0] + 1, new_point[1] + 1]:
            action = actions_pol[random.randrange(0, len(actions_pol), 1)]
            new_point = (min_path[t][0] + action[0], min_path[t][1] + action[1])

        # Add Newpoints Minotaur
        min_path.append(new_point)
    images = draw_full_path(person_path, min_path, [], [[1,1]])

    for i, image in enumerate(images):
        image.save("Lab1/images/" + name + str(i) + ".png")



if __name__ == '__main__':

    # Q-learning

    disc_factor = 0.8

    dimensions = [xdim, ydim]
    maze = generate_maze(dimensions, [])

    # Define robber actions

    #Initialize Q and n functions
    Q = np.zeros((xdim, ydim, xdim, ydim, n_a))
    n_sa = np.zeros((xdim, ydim, xdim, ydim, n_a))

    N_iter = 10000
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

    # Take optimal actions
    a = np.argmax(Q, axis=4)
    save_game("QLearning", a, 20)


    plt.figure()
    labels = ['Stay', 'Down', 'Up', 'Right', 'Left']
    for i in range(len(labels)):
        plt.plot(Q_plot[0:N_iter, i], label=labels[i])
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Q')
    plt.show()

    print("Q-Learning")
    print(Q_plot[N_iter-1,:])

    # SARSA
    eps_vector = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Initialize Q and n functions

    plt.figure()

    for eps in eps_vector:

        Q = np.zeros((xdim, ydim, xdim, ydim, n_a))
        n_sa = np.zeros((xdim, ydim, xdim, ydim, n_a))

        N_iter = 2000000
        Q_plots = np.zeros((N_iter, n_a))

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
            Q_state_next = Q[new_rob[0], new_rob[1], new_pol[0], new_pol[1], i1_action]
            Q[p_rob[0], p_rob[1], p_pol[0], p_pol[1], i_action] = Q_prev + stepsize * (
                    reward + disc_factor * Q_state_next - Q_prev)

            Q_plots[i, :] = Q[0, 0, 3, 3, :]

        #a = np.argmax(Q, axis=4)
        #save_game("SARSA", a, 20, SARSA=True)

        #plt.figure()
        #labels = ['Stay', 'Down', 'Up', 'Right', 'Left']
        #for i in range(len(labels)):
        #    plt.plot(Q_plots[0:N_iter, i], label=labels[i])
        #plt.legend()
        #plt.xlabel('Iteration')
        #plt.ylabel('Q')
        #plt.show()

        plt.plot(np.max(Q_plots[0:N_iter, :], axis=1), label="eps = " + str(eps))

        print("SARSA")
        print("Epsilon")
        print(eps)
        print("Q-Function Values")
        print(Q_plots[N_iter-1,:])

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Q')
    plt.show()




