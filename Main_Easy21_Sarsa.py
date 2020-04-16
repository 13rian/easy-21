import utils
import logging
import environment

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


# The logger
utils.init_logger(logging.DEBUG, fileName="log/app.log")
logger = logging.getLogger('Easy21')

# set the random seed
random.seed(a=None, version=2)


# constants
n0 = 100
lam_range = np.arange(0, 1.1, 0.1)
nIter = 10000      # number of episodes   

# define the indices of the different values
q_hit_index = 0        # q value for action hit
q_stick_index = 1      # q value for action stick
e_hit_index = 2        # eligibility trace for the hit action
e_stick_index = 3      # eligibility trace for the stick action
ns_index = 4           # number of times the state was visited
ns_hit_index = 5       # number of times the action hit was chosen in this state
ns_stick_index = 6     # number of times the action stick was chosen in this state


# initialize the value function
# x-index: dealer card -1
# y-index: player sum -1
# z        all properties for this state
state_info = np.zeros((10, 21, 7))


def sarsa_episode(lam):
    """
    executes one sarsa episode
    :param lam:     the lambda parameter
    :return:
    """
    # reset all eligibility traces
    state_info[:, :, e_hit_index] = 0
    state_info[:, :, e_stick_index] = 0
    
    # initialize the state S
    dealer_card = random.randint(1,10)
    player_card = random.randint(1,10)
    state = environment.State(dealer_card, player_card)
    
    # initialize the action A
    action = environment.Action.HIT
    if random.random() < 0.5:
        action = environment.Action.STICK
        
        
    # run one episode
    while not state.terminated:
        # define the starting state indices for the state matrix
        dealer_state_index = state.dealer_card - 1
        player_state_index = state.player_sum - 1
        
        # take the action A
        state_new = environment.step(state, action)
        reward = state_new.reward
        
        # define the indices of the new state
        dealer_state_index_new = state_new.dealer_card - 1
        player_state_index_new = state_new.player_sum - 1


        # pick the next action A' by using epsilon greedy
        if state_new.terminated:
            action_new = environment.Action.NONE
            
        else:
            epsilon = n0 / (n0 + state_info[dealer_state_index_new, player_state_index_new, ns_index])
            if random.random() < epsilon:
                # exploration, pick a random action
                if random.random() < 0.5:
                    action_new = environment.Action.HIT
                else:
                    action_new = environment.Action.STICK
        
            else:
                # pick the action greedily (largest action value)
                if state_info[dealer_state_index_new, player_state_index_new, q_hit_index] > state_info[dealer_state_index_new, player_state_index_new, q_stick_index]:
                    action_new = environment.Action.HIT
                else:
                    action_new = environment.Action.STICK
                
                
        # increment the counts
        state_info[dealer_state_index, player_state_index, ns_index] += 1
        
        if action == environment.Action.HIT:
            state_info[dealer_state_index, player_state_index, ns_hit_index] += 1
           
        if action == environment.Action.STICK:
            state_info[dealer_state_index, player_state_index, ns_stick_index] += 1

        
        # calculate delta
        if action == environment.Action.HIT:
            qValue = state_info[dealer_state_index, player_state_index, q_hit_index]
        else:
            qValue = state_info[dealer_state_index, player_state_index, q_stick_index]

        if state_new.terminated:
            q_value_new = 0
            
        else:
            if action_new == environment.Action.HIT:
                q_value_new = state_info[dealer_state_index_new, player_state_index_new, q_hit_index]
            else:
                q_value_new = state_info[dealer_state_index_new, player_state_index_new, q_stick_index]

        delta = reward + q_value_new - qValue

        
        # increment eligibility trace
        alpha = None
        if action == environment.Action.HIT:
            alpha = 1 / state_info[dealer_state_index, player_state_index, ns_hit_index]
            state_info[dealer_state_index, player_state_index, e_hit_index] += 1
        else:
            alpha = 1 / state_info[dealer_state_index, player_state_index, ns_stick_index]
            state_info[dealer_state_index, player_state_index, e_stick_index] += 1
            
        # update all values
        state_info[:, :, q_hit_index] += alpha * delta * state_info[:, :, e_hit_index]
        state_info[:, :, q_stick_index] += alpha * delta * state_info[:, :, e_stick_index]

        # update all eligibility traces
        state_info[:, :, e_hit_index] = lam * state_info[:, :, e_hit_index]
        state_info[:, :, e_stick_index] = lam * state_info[:, :, e_stick_index]


        # end this step
        state = state_new
        action = action_new





mc_q_values = np.load("action_values.npy")       # load the mc results
mse_error = np.zeros((len(lam_range), nIter))    # define the error array


def calc_error():
    """
    returns the mean squared error compared to the mc simulation
    :return:
    """
    sarsa_q_values = np.maximum(state_info[:, :, q_hit_index], state_info[:, :, q_stick_index])
    mse = np.mean(np.square(sarsa_q_values - mc_q_values))
    return mse
    

# start the sarsa control
for i in range(0, len(lam_range)):
    # initialize the state info
    state_info = np.zeros((10, 21, 7))
    
    for j in range(0, nIter):
        mse_error[i, j] = calc_error()
        sarsa_episode(lam_range[i])

 

# plot the action value function
fig = plt.figure(1)
ax = fig.gca(projection='3d')

X = np.arange(1, 22, 1)
Y = np.arange(1, 11, 1)
X, Y = np.meshgrid(X, Y)
Z = np.maximum(state_info[:, :, q_hit_index], state_info[:, :, q_stick_index])


# plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False)

# customize the z axis
# ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)



fig = plt.figure(2)
x = np.arange(nIter)
for i in range(0, mse_error.shape[0]):
    plt.plot(x, mse_error[i, :], label ="{:.1f}".format(lam_range[i]))
    
plt.legend(loc='best')
plt.xlabel("Episode")
plt.ylabel("MSE")
plt.show()
