import torch
import numpy as np
import random
import math
from Utils.utilsfunction import *
from Utils.statefunction import *

class QLearning:
    def __init__(self, NGT, hyperparams, earth, g, sat, qTable=None):
        '''
        Create a 6D PyTorch tensor to hold the current Q-values for each state and action pair: Q(s, a)
        The tensor contains 5 dimensions with the shape of the environment, as well as a 6th "action" dimension.
        The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in each state
        The value of each (state, action) pair is initialized randomly.
        '''
        satUp, satDown, satRight, satLeft = 3, 3, 3, 3
        linkedSats = getLinkedSats(sat, g, earth)
        self.linkedSats =  {'U': linkedSats['U'],
                            'D': linkedSats['D'],
                            'R': linkedSats['R'],
                            'L': linkedSats['L']}

        self.actions         = ('U', 'D', 'R', 'L')     # Up, Down, Left, Right
        self.Destinations    = NGT

        self.nStates    = satUp*satDown*satRight*satLeft*NGT
        self.nActions   = len(self.actions)
                
        if qTable is None:  # initialize it randomly if we are not going to import it
            self.qTable = torch.rand(satUp, satDown, satRight, satLeft, NGT, self.nActions)  # first 5 fields are states while 6th field is the action. 4050 values with 10 GTs
        else:
            self.qTable = torch.tensor(qTable)  # Convert numpy array to torch tensor

        self.alpha  = hyperparams.alpha
        self.gamma  = hyperparams.gamma
        self.epsilon= []
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1     = hyperparams.w1
        self.w2     = hyperparams.w2

        self.oldState  = (0,0,0,0,0)
        self.oldAction = 0

    def makeAction(self, block, sat, g, earth, prevSat=None):
        '''
        This function will:
        1. Check if the destination is the linked gateway. In that case it will just return 0 and the block will be sent there.
        2. Observation of the environment in order to determine state space and get the linked satellites.
        3. Chooses an action. Random one (Exploration) or the most valuable one (Exploitation). If the direction of that action has no linked satellite, the QValue will be -inf
        4. Receive reward/penalty
            Penalties: If the block visits again the same satellite. Reward = -1
                       Another one directly proportional to the length of the destination queue.
            Reward: So far, it will be higher if it gets physically closer to the satellite
        5. Updates Q-Table of the previous hop (Agent) with the following information:
            1. Reward      : Time waited at satB Queue && slant range reduction.
            2. maxNewQValue: Max Q Value of all possible actions at the new agent.
            3. Old state-action taken at satA in order to know where to update the Q-Table. 
            Everytime satB receives a dataBlock from satA satB will send the information required to update satA QTable.
        '''

        # There is no 'Done' state, it will simply continue until the time stops
        # simplemente se va a recibir una recompensa positiva si el satelite al que envias el paquete es el linkado al destino de este

        # 1. check if the destination is the linked gateway. The value of this action becomes 10. # ANCHOR plots route of delivered package Q-Learning
        if sat.linkedGT and block.destination.name == sat.linkedGT.name:
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = ArriveReward
            earth.rewards.append([ArriveReward, sat.env.now])
            if plotDeliver:
                if int(block.ID[len(block.ID)-1]) == 0: # Draws 1/10 arrivals
                    os.makedirs(earth.outputPath + '/pictures/', exist_ok=True) # drawing delivered
                    outputPath = earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    plotShortestPath(earth, block.QPath, outputPath, ID=block.ID, time = block.creationTime)
            
            return 0

        # 2. Observation of the environment
        newState = tuple(getState(block, sat, g, earth))
       
        # 3. Choose an action (the direction of the next hop)
        # randomly
        if explore and random.uniform(0, 1) < self.alignEpsilon(earth, sat):
            action = self.actions[random.randrange(len(self.actions))]
            while(self.linkedSats[action] == None): 
                action = self.actions[random.randrange(len(self.actions))]  # if that direction has no linked satellite
        
        # highest value
        else:
            qValues = self.qTable[newState]
            action  = self.actions[torch.argmax(qValues).item()]  # Most valuable action (The one that will give more reward) 
            while self.linkedSats[action] == None:
                self.qTable[newState][self.actions.index(action)] = -float('inf') # change qTable if that action is not available
                action = self.actions[torch.argmax(qValues).item()]

        destination = self.linkedSats[action]    # Action is the keyword of the chosen linked satellite, linkedSats is a dictionary with each satellite associated to its corresponding keyword

        # ACT -> [it is done outside, the next hop is added at sat.receiveBlock method to block.QPath]
        nextHop = [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)]

        # 4. Receive reward/penalty for the previous action
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if the next hop was already visited before the reward will be againPenalty
            if hop in block.QPath[:len(block.QPath)-2]:
                reward = againPenalty
            else:
                distanceReward = getDistanceReward(prevSat, sat, block.destination, self.w2)
                try:
                    queueReward    = getQueueReward(block.queueTime[len(block.queueTime)-1], self.w1)
                except IndexError:
                    queueReward = 0 # FIXME
                reward = distanceReward + queueReward
            
            earth.rewards.append([reward, sat.env.now])

        # 5. Updates Q-Table 
        # Update QTable of previous Node (Agent, satellite) if it was not a gateway     
            nextMax     = torch.max(self.qTable[newState]).item() # max value of next state given oldAction
            oldQValue   = prevSat.QLearning.qTable[block.oldState][block.oldAction].item()
            newQvalue   = (1-self.alpha) * oldQValue + self.alpha * (reward + self.gamma * nextMax) 
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = newQvalue
            
        else:
            # prev node was a gateway, no need to compute the reward
            reward = 0

        # this will be saved always, except when the next hop is the destination, where the process will have already returned
        block.oldState  = newState
        block.oldAction = self.actions.index(action)

        earth.step += 1

        return nextHop

    def alignEpsilon(self, earth, sat):
        global CurrentGTnumber
        epsilon = self.minEps + (self.maxEps - self.minEps) * math.exp(-LAMBDA * earth.step/(decayRate*(CurrentGTnumber**2)))
        earth.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def __repr__(self):
            return '\n Nº of destinations = {}\n Action Space = {}\n Nº of states = {}\n qTable: {}'.format(
            self.Destinations,
            self.actions,
            self.nStates,
            self.qTable)    