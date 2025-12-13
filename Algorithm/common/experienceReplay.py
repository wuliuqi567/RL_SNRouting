import random
import numpy as np
from collections import deque

class ExperienceReplay:
    def __init__(self, maxlen = 100):
        '''
        This is a buffer that holds information that are used during training process.

        Deque (Doubly Ended Queue). Deque is preferred over a list in the cases where we need quicker append and pop operations
        from both the ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared
        to a list that provides O(n) time complexity
        '''
        self.buffer = deque(maxlen=maxlen)

    def store(self, state, action, reward, nextState, terminated):
        '''
        appends a set of (state, action, reward, next state, terminated) to the experience replay buffer
        '''
        # if the buffer is full, it behave as a FIFO
        self.buffer.append((state, action, reward, nextState, terminated))

    def getBatch(self, batchSize):
        '''
        gets a random batch of samples from all the samples
        '''
        return random.sample(self.buffer, batchSize)

    def getArraysFromBatch(self, batch):
        '''
        gets the batch data divided into fields
        '''
        states  = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_st = np.array([x[3] for x in batch])
        dones   = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_st, dones

    @property
    def buffeSize(self):
        '''
        a pythonic way to use getters and setters in object-oriented programming
        this decorator is a built-in function that allows us to define methods that can be accessed like an attribute
        '''
        return len(self.buffer)