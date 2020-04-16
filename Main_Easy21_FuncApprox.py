import logging
import environment
import utils

import numpy as np
import random
import matplotlib.pyplot as plt


# The logger
utils.init_logger(logging.DEBUG, fileName="log/app.log")
logger = logging.getLogger('Easy21')

# set the random seed
random.seed(a=None, version=2)


# constants
alpha = 0.01                        # step size
epsilon = 0.05                      # exploration
lam_range = np.arange(0, 1.1, 0.1)
n_iter = 10000                      # number of episodes

# define the indices of the different values
q_hit_index = 0        # q value for action hit
q_stick_index = 1      # q value for action stick
e_hit_index = 2        # eligibility trace for the hit action
e_stick_index = 3      # eligibility trace for the stick action


# initialize the value function approximation, the value is 1 if the state lies within the defined intervals
# x-index: dealer card approximation [1; 4] [4; 7] [7; 10]
# y-index: player sum approximation  [1; 6] [4; 9] [7; 12] [10; 15] [13; 18] [16; 21]
# z        all properties for this state
state_info = np.zeros((3, 6, 4))
# print(stateInfo)


# function that executes one sarsa episode
# lam:     the lambda parameter
def sarsa_episode(lam):
    # reset all eligibility traces
    state_info[:, :, e_hit_index] = 0
    state_info[:, :, e_stick_index] = 0
    
    # initialize the state S
    dealer_card = random.randint(1, 10)
    player_card = random.randint(1, 10)
    state = environment.State(dealer_card, player_card)
    features = state.get_features()
    
    # initialize the action A
    action = environment.Action.HIT
    if random.random() < 0.5:
        action = environment.Action.STICK
        
        
    # run one episode
    while not state.terminated:
        # take the action A
        state_new = environment.step(state, action)
        reward = state_new.reward
        features_new = state_new.get_features()
        
        
        # pick the next action A' by using epsilon greedy         
        action_new = None        
        if state_new.terminated:
            action_new = environment.Action.NONE
            
        else:
            if random.random() < epsilon:
                # exploration, pick a random action
                if random.random() < 0.5:
                    action_new = environment.Action.HIT
                else:
                    action_new = environment.Action.STICK
        
            else:
                # pick the action greedily (largest action value)
                v_hit = np.sum(np.multiply(features_new, state_info[:, :, q_hit_index]))
                v_stick = np.sum(np.multiply(features_new, state_info[:, :, q_stick_index]))
                if v_hit > v_stick:
                    action_new = environment.Action.HIT
                else:
                    action_new = environment.Action.STICK   

        
        # calculate delta
        if action == environment.Action.HIT:
            q_value = np.sum(np.multiply(features, state_info[:, :, q_hit_index]))
        else:
            q_value = np.sum(np.multiply(features, state_info[:, :, q_stick_index]))

        if state_new.terminated:
            q_value_new = 0
            
        else:
            if action_new == environment.Action.HIT:
                q_value_new = np.sum(np.multiply(features_new, state_info[:, :, q_hit_index]))
            else:
                q_value_new = np.sum(np.multiply(features_new, state_info[:, :, q_stick_index]))

        delta = reward + q_value_new - q_value


        # increment eligibility trace
        if action == environment.Action.HIT:
            state_info[:, :, e_hit_index] += features
        else:
            state_info[:, :, e_stick_index] += features
            
        # update all values
        state_info[:, :, q_hit_index] += alpha * delta * state_info[:, :, e_hit_index]
        state_info[:, :, q_stick_index] += alpha * delta * state_info[:, :, e_stick_index]

        # update all eligibility traces
        state_info[:, :, e_hit_index] = lam * state_info[:, :, e_hit_index]
        state_info[:, :, e_stick_index] = lam * state_info[:, :, e_stick_index]

        # end this step
        state = state_new
        action = action_new
        features = features_new





mc_q_values = np.load("action_values.npy")           # load the mc results
mse_error = np.zeros((len(lam_range), n_iter))       # define the error array


def calc_features_matrix():
    all_features = np.zeros((10, 21, 3, 6))          # dealer card - 1, players sum - 1, features 3x6
    for i in range(10):
        for j in range(21):
            state = environment.State(i+1, j+1)
            all_features[i, j, :, :] = state.get_features()
    
    return all_features


all_features = calc_features_matrix()


def calc_error():
    """
    returns the mean squared error compared to the mc simulation
    :return:
    """

    mse = 0
    for i in range(10):
        for j in range(21):
            # sarsa value
            q_hit = np.multiply(all_features[i, j], state_info[:, :, q_hit_index])
            q_stick = np.multiply(all_features[i, j], state_info[:, :, q_stick_index])
            q_sarsa = np.maximum(np.sum(q_hit), np.sum(q_stick))
            
            q_mc = mc_q_values[i, j]
            
            mse += np.square(q_sarsa - q_mc)
    
    mse = mse/210
    return mse
    

# start the sarsa control
for i in range(0, len(lam_range)):
    # initialize the state info
    logger.debug("lambda = " + str(lam_range[i]) + " start")
    state_info = np.zeros((3, 6, 4))
    
    # initialize the weights
    state_info[:, :, q_hit_index] = np.random.randn(3, 6)
    state_info[:, :, q_stick_index] = np.random.randn(3, 6)
    
    for j in range(0, n_iter):
        mse_error[i, j] = calc_error()
        sarsa_episode(lam_range[i])
        
    logger.debug("lambda = " + str(lam_range[i]) + " done")


fig = plt.figure(1)
x = np.arange(n_iter)
for i in range(0, mse_error.shape[0]):
    plt.plot(x, mse_error[i, :], label="{:.1f}".format(lam_range[i]))
    
plt.legend(loc='best')
plt.xlabel("Episode")
plt.ylabel("MSE")
plt.show()
