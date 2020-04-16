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
n_iter = 1000000         # number of monte-carlo iterations

# define the indices of the different values
q_hit_index = 0        # q value for action hit
q_stick_index = 1      # q value for action stick
ns_index = 2           # number of times the state was visited
ns_hit_index = 3       # number of times the action hit was chosen in this state
ns_stick_index = 4     # number of times the action stick was chosen in this state


# initialize the value function
# x-index: dealer card -1
# y-index: player sum -1
# z        all properties for this state
state_info = np.zeros((10, 21, 5))


# function that executes one monte-carlo episode
def mc_episode():
    states = []     # holds all states of one episode
    actions = []    # holds all actions of one episode
    
    # create the initial state
    dealer_card = random.randint(1,10)
    player_card = random.randint(1,10)
    state = environment.State(dealer_card, player_card)


    # create the initial state
    while not state.terminated:
        states.append(state)
        
        # define the indices for the state matrix
        dealer_state_index = state.dealer_card - 1
        player_state_index = state.player_sum - 1
        
        # pick the action
        epsilon = n0 / (n0 + state_info[dealer_state_index, player_state_index, ns_index])
        if random.random() < epsilon:
            # exploration, pick a random action
            if random.random() < 0.5:
                action = environment.Action.HIT
            else:
                action = environment.Action.STICK
        
        else:
            # pick the action greedily (largest action value)
            if state_info[dealer_state_index, player_state_index, q_hit_index] > state_info[dealer_state_index, player_state_index, q_stick_index]:
                action = environment.Action.HIT
            else:
                action = environment.Action.STICK  
                
                
        # increment the counts
        state_info[dealer_state_index, player_state_index, ns_index] += 1
        
        if action == environment.Action.HIT:
            state_info[dealer_state_index, player_state_index, ns_hit_index] += 1
           
        if action == environment.Action.STICK:
            state_info[dealer_state_index, player_state_index, ns_stick_index] += 1
        
        # get a new state
        actions.append(action)
        state = environment.step(state, action)

    
    # update the action values
    for i in range(0, len(states)):
        s = states[i]
        a = actions[i]
        tot_reward = state.reward
    
        if not s.is_busted:
            dealer_state_index = s.dealer_card - 1
            player_state_index = s.player_sum - 1
            
            if a == environment.Action.HIT:
                alpha = 1 / state_info[dealer_state_index, player_state_index, ns_hit_index]
                value = state_info[dealer_state_index, player_state_index, q_hit_index]
                state_info[dealer_state_index, player_state_index, q_hit_index] += alpha * (tot_reward - value)
            else:
                alpha = 1 / state_info[dealer_state_index, player_state_index, ns_stick_index]
                value = state_info[dealer_state_index, player_state_index, q_stick_index]
                state_info[dealer_state_index, player_state_index, q_stick_index] += alpha * (tot_reward - value)


# start the monto-carlo control
for i in range(0, n_iter):
    mc_episode()
 

# plot the action value function
fig = plt.figure(1)
ax = fig.gca(projection='3d')

X = np.arange(1, 22, 1)
Y = np.arange(1, 11, 1)
X, Y = np.meshgrid(X, Y)
Z = np.maximum(state_info[:, :, q_hit_index], state_info[:, :, q_stick_index])

# save the matrix
np.save("action_values.npy", Z)


# plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False)

# customize the z axis
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
