from enum import Enum
import random
import copy
import json
import numpy as np 


# initialize the random seed
random.seed(a=None, version=2)
  
prob_black = 2 / 3         # probability for a black card (value is added to the players sum)
dealer_threshold = 17      # dealer sticks if he has this value or a larger value


class Action(Enum):
    NONE = 1
    HIT = 2
    STICK = 3


class State:
    def __init__(self, dealer_card, player_sum):
        """
        defines the state of the game
        :param dealer_card:     card value of the dealer
        :param player_sum:      current value of the player
        """

        self.dealer_card = dealer_card
        self.player_sum = player_sum
        self.reward = 0                  # the reward
        self.is_busted = False            # true if the player lost
        self.terminated = False          # true if the game has finished
        
    def __str__(self):
        return json.dumps(self.__dict__)


    def get_features(self):
        """
        returns the matrix with the binary features for the function approximation
        :return:
        """
        features = np.zeros((3, 6))
        for j in range(features.shape[1]):
            j_lower = 3*j + 1
            j_upper = j_lower + 5
            for i in range(features.shape[0]):
                i_lower = 3*i + 1
                i_upper = i_lower + 3
                if self.dealer_card >= i_lower and self.dealer_card <= i_upper and self.player_sum >= j_lower and self.player_sum <= j_upper:
                    features[i, j] = 1
                    
        return features
                 

def step(state, action):
    """
    defines the environment of the game and executes one step
    :param state:       the state of the game
    :param action:      enum with hit or stick
    :return:
    """
    # calculate the new state by probing the environment
    new_state = copy.deepcopy(state)
        
    if action == Action.HIT:  
        new_state.player_sum = state.player_sum + hit()
        new_state.is_busted = is_busted(new_state.player_sum)
        if new_state.is_busted:
            new_state.terminated = True
            new_state.reward = -1
        
        return new_state
        
    if action == Action.STICK:
        new_state.terminated = True
        
        # play all dealer moves
        dealer_state = dealers_move(new_state.dealer_card)
        if dealer_state["is_busted"]:
            new_state.reward = 1
            return new_state
        
        # check if the player has a higher number of not
        if dealer_state["dealer_sum"] > new_state.player_sum:
            new_state.reward = -1
        elif dealer_state["dealer_sum"] == new_state.player_sum:
            new_state.reward = 0
        else:
            new_state.reward = 1
            
        return new_state
    
    if action == Action.NONE:
        new_state.terminated = True
        new_state.reward = 0
        return new_state
        


def dealers_move(dealer_card):
    """
    makes all moves of the dealer
    :param dealer_card:     the card of the dealer and return its value
    :return:                the game state of the dealer with dealer_sum, isBusted
    """
    dealer_sum = dealer_card
    terminated = False
    while not terminated:
        if dealer_sum < dealer_threshold:
            dealer_sum = dealer_sum + hit()
        else:
            terminated = True
            
        has_lost = is_busted(dealer_sum)
        if has_lost:
            terminated = True 
    
    dealer_state = {
        "dealer_sum": dealer_sum,
        "is_busted": has_lost
    }
           
    return dealer_state
    

def hit():
    """
    returns the card value if a player chooses hit
    :return:
    """
    sign = -1
    if random.random() < prob_black:
        sign = 1
    
    card_value = sign*random.randint(1, 10)
    return card_value


def is_busted(player_sum):
    """
    returns true if the passed value loses the game
    :param player_sum:  the summed card values of the player
    :return:
    """
    if player_sum < 1 or player_sum > 21:
        return True
    else:
        return False 
