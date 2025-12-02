from configure import *

pathings    = ['hop', 'dataRate', 'dataRateOG', 'slant_range', 'Q-Learning', 'Deep Q-Learning', 'Policy Distillation', 'GNNPD']
pathing     = pathings[7]# dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate

GTs = [4]               # number of gateways to be tested
matching    = 'Positive_Grid'  # ['Markovian', 'Greedy', 'Positive_Grid']
diff        = True          # If up, the state space gives no coordinates about the neighbor and destination positions but the difference with respect to the current positions
diff_lastHop= True          # If up, this state is the same as diff, but it includes the last hop where the block was in order to avoid loops
reducedState= False         # if set to true the DNN will receive as input only the positional information, but not the queueing information
third_adj    = True          # If up, the state space includes the 3rd order neighbors information
n_order_adj = 4   

# =============================================================================
# 9. RL/DRL Hyperparameters
# =============================================================================
ddqn        = True      # Activates DDQN, where now there are two DNNs, a target-network and a q-network
alpha       = 0.0002      # learning rate for Q-Tables
alpha_dnn   = alpha      # learning rate for the deep neural networks
gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning
epsilon     = 0.1       # exploration factor for Q-Learning ONLY
tau         = 0.01       # rate of copying the weights from the Q-Network to the target network
learningRate= alpha     # Default learning rate for Adam optimizer
distillationLR = 0.00005 # Learning rate for the student optimizer in policy distillation
distillationLossFun = 'KL_v2' # Loss function for policy distillation. Options: 'MSE', 'Huber' and 'KL', 'KLv2'
nTrain      = 120         # The DNN will train every nTrain steps
noPingPong  = True      # when a neighbour is the destination satellite, send there directly without going through the dnn (Change policy)

# Deep Learning
MAX_EPSILON = 0.99      # Maximum value that the exploration parameter can have
MIN_EPSILON = 0.001     # Minimum value that the exploration parameter can have
LAMBDA      = 0.0005   # This value is used to decay the epsilon in the deep learning implementation
decayRate   = 30       # if 5s  set 90# sets the epsilon decay in the deep learning implementatio. If higher, the decay rate is slower. If lower, the decay is faster
Clipnorm    = 1         # Maximum value to the nom of the gradients. Prevents the gradients of the model parameters with respect to the loss function becoming too large
hardUpdate  = 1         # if up, the Q-network weights are copied inside the target network every updateF iterations. if down, this is done gradually
updateF     = 2800      # every updateF updates, the Q-Network will be copied inside the target Network. This is done if hardUpdate is up
batchSize   = 128        # batchSize samples are taken from bufferSize samples to train the network
bufferSize  = 100000        # bufferSize samples are used to train the network
train_epoch = 4