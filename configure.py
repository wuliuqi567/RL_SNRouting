# HOT PARAMS - This parameters should be revised before every simulation
pathings    = ['hop', 'dataRate', 'dataRateOG', 'slant_range', 'Q-Learning', 'Deep Q-Learning']
pathing     = pathings[5]# dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate

FL_Test     = True     # If True, it plots the model divergence the model divergence between agents
plotSatID   = True      # If True, plots the ID of each satellite
plotAllThro = True      # If True, it plots throughput plots for each single path between gateways. If False, it plots a single figure for overall Throughput
plotAllCon  = True      # If True, it plots congestion maps for each single path between gateways. If False, it plots a single figure for overall congestion

movementTime= 0.5       # Every movementTime seconds, the satellites positions are updated and the graph is built again
                        # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
ndeltas     = 5805.44/20 #1 Movement speedup factor. Every movementTime sats will move movementTime*ndeltas space. If bigger, will make the rotation distance bigger

Train       = True      # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
explore     = True      # If True, makes random actions eventually, if false only exploitation
importQVals = False     # imports either QTables or NN from a certain path
onlinePhase = False     # when set to true, each satellite becomes a different agent. Recommended using this with importQVals=True and explore=False
if onlinePhase:         # Just in case
    explore     = False
    importQVals = True
else:
    FL_Test = False

w1          = 20        # rewards the getting to empty queues
w2          = 20        # rewards getting closes phisycally   
w4          = 5         # Normalization for the distance reward, for the traveled distance factor 

gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning

GTs = [2]               # number of gateways to be tested
# Gateways are taken from https://www.ksat.no/ground-network-services/the-ksat-global-ground-station-network/ (Except for Malaga and Aalborg)
# GTs = [i for i in range(2,9)] # This is to make a sweep where scenarios with all the gateways in the range are considered

# Physical constants
rKM = 500               # radio in km of the coverage of each gateway
Re  = 6378e3            # Radius of the earth [m]
G   = 6.67259e-11       # Universal gravitational constant [m^3/kg s^2]
Me  = 5.9736e24         # Mass of the earth
Te  = 86164.28450576939 # Time required by Earth for 1 rotation
Vc  = 299792458         # Speed of light [m/s]
k   = 1.38e-23          # Boltzmann's constant
eff = 0.55              # Efficiency of the parabolic antenna

# Downlink parameters
f       = 20e9  # Carrier frequency GEO to ground (Hz)
B       = 500e6 # Maximum bandwidth
maxPtx  = 10    # Maximum transmission power in W
Adtx    = 0.26  # Transmitter antenna diameter in m
Adrx    = 0.26  #0.33 Receiver antenna diameter in m
pL      = 0.3   # Pointing loss in dB
Nf      = 2     #1.5 Noise figure in dB
Tn      = 290   #50 Noise temperature in K
min_rate= 10e3  # Minimum rate in kbps

# Uplink Parameters
balancedFlow= False         # if set to true all the generated traffic at each GT is equal
totalFlow   = 2*1000000000  # Total average flow per GT when the balanced traffc option is enabled. Malaga has 3*, LA has 3*, Nuuk/500
avUserLoad  = 8593 * 8      # average traffic usage per second in bits

# Block
BLOCK_SIZE   = 64800

# Movement and structure
# movementTime= 0.05      # Every movementTime seconds, the satellites positions are updated and the graph is built again
#                         # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
# ndeltas     = 5805.44/20#1 Movement speedup factor. This number will multiply deltaT. If bigger, will make the rotation distance bigger
saveISLs    = True     # save ISLs map
const_moved = False     # Movement flag. If up, it means it has moved
matching    = 'Greedy'  # ['Markovian', 'Greedy']
minElAngle  = 30        # For satellites. Value is taken from NGSO constellation design chapter.
mixLocs     = False     # If true, every time we make a new simulation the locations are going to change their order of selection
rotateFirst = False     # If True, the constellation starts rotated by 1 movement defined by ndeltas

# State pre-processing
coordGran   = 20            # Granularity of the coordinates that will be the input of the DNN: (Lat/coordGran, Lon/coordGran)
diff        = True          # If up, the state space gives no coordinates about the neighbor and destination positions but the difference with respect to the current positions
diff_lastHop= True          # If up, this state is the same as diff, but it includes the last hop where the block was in order to avoid loops
reducedState= False         # if set to true the DNN will receive as input only the positional information, but not the queueing information
notAvail    = 0             # this value is set in the state space when the satellite neighbour is not available

# Learning Hyperparameters
ddqn        = True      # Activates DDQN, where now there are two DNNs, a target-network and a q-network
# importQVals = False     # imports either QTables or NN from a certain path
plotPath    = False     # plots the map with the path after every decision
alpha       = 0.25      # learning rate for Q-Tables
alpha_dnn   = 0.01      # learning rate for the deep neural networks
# gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning
epsilon     = 0.1       # exploration factor for Q-Learning ONLY
tau         = 0.1       # rate of copying the weights from the Q-Network to the target network
learningRate= 0.001     # Default learning rate for Adam optimizer
plotDeliver = False     # create pictures of the path every 1/10 times a data block gets its destination
# plotSatID   = False     # If True, plots the ID of each satellite
GridSize    = 8         # Earth divided in GridSize rows for the grid. Used to be 15
winSize     = 20        # window size for the representation in the plots
markerSize  = 50        # Size of the markers in the plots
nTrain      = 2         # The DNN will train every nTrain steps
noPingPong  = True      # when a neighbour is the destination satellite, send there directly without going through the dnn (Change policy)

# Queues & State
infQueue    = 5000      # Upper boundary from where a queue is considered as infinite when obserbing the state
queueVals   = 10        # Values that the observed Queue can have, being 0 the best (Queue of 0) and max the worst (Huge queue or inexistent link).
latBias     = 90        # This value is added to the latitude of each position in the state space. This can be done to avoid negative numbers
lonBias     = 180       # Same but with longitude

# rewards
ArriveReward= 50        # Reward given to the system in case it sends the data block to the satellite linked to the destination gateway
# w1          = 20        # rewards the getting to empty queues
# w2          = 20        # rewards getting closes phisycally   
# w4          = 5         # Normalization for the distance reward, for the traveled distance factor  
againPenalty= -10       # Penalty if the satellite sends the block to a hop where it has already been
unavPenalty = -10       # Penalty if the satellite tries to send the block to a direction where there is no linked satellite
biggestDist = -1        # Normalization factor for the distance reward. This is updated in the creation of the graph.
firstMove   = True      # The biggest slant range is only computed the first time in order to avoid this value to be variable
distanceRew = 4          # 1: Distance reward normalized to total distance.
                         # 2: Distance reward normalized to average moving possibilities
                         # 3: Distance reward normalized to maximum close up
                         # 4: Distance reward normalized by max isl distance ~3.700 km for Kepler constellation. This is the one used in the papers.
                         # 5: Only negative rewards proportional to traveled distance normalized by 1.000 km

# Deep Learning
MAX_EPSILON = 0.99      # Maximum value that the exploration parameter can have
MIN_EPSILON = 0.001     # Minimum value that the exploration parameter can have
LAMBDA      = 0.0005    # This value is used to decay the epsilon in the deep learning implementation
decayRate   = 4         # sets the epsilon decay in the deep learning implementatio. If higher, the decay rate is slower. If lower, the decay is faster
Clipnorm    = 1         # Maximum value to the nom of the gradients. Prevents the gradients of the model parameters with respect to the loss function becoming too large
hardUpdate  = 1         # if up, the Q-network weights are copied inside the target network every updateF iterations. if down, this is done gradually
updateF     = 1000      # every updateF updates, the Q-Network will be copied inside the target Network. This is done if hardUpdate is up
batchSize   = 16        # batchSize samples are taken from bufferSize samples to train the network
bufferSize  = 50        # bufferSize samples are used to train the network

# Stop Loss
# Train       = True      # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
stopLoss    = False     # activates the stop loss function
nLosses     = 50        # Nº of loss samples used for the stop loss
lThreshold  = 0.5       # If the mean of the last nLosses are lower than lossThreshold, the mdoel stops training
TrainThis   = Train     # Local for a single scenario with a certain number of GTs. If the stop loss is activated, this will be set to False and the scenario will not train anymore. 
                        # When another scenario is about to run, TrainThis will be set to Train again

# Other
CurrentGTnumber = -1    # Number of active gateways. This number will be updated every time a gateway is added. In the simulation it will iterate the GTs list

###############################################################################
###############################      Paths      ###############################
###############################################################################

# nnpath      = './pre_trained_NNs/qNetwork_8GTs_6secs_nocon.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_8GTs_6secs_nocon.h5'
# nnpath      = './pre_trained_NNs/qNetwork_3GTs.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_3GTs.h5'
# nnpath      = './pre_trained_NNs/qNetwork_2GTs.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_2GTs.h5'
nnpath      = './pre_trained_NNs/qNetwork_2GTs_lastHop.h5'
nnpathTarget= './pre_trained_NNs/qTarget_2GTs_lastHop.h5'
tablesPath  = './pre_trained_NNs/qTablesExport_8GTs/'

FL_techs    = ['nothing', 'modelAnticipation', 'plane', 'full', 'combination']
FL_tech     = FL_techs[4]# dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate
if FL_tech == 'combination':
    global FL_counter
    FL_counter = 1

if pathing != 'Deep Q-Learning':
    FL_Test = False

if FL_Test:
    CKA_Values = []     # CKA matrix 
    num_samples = 10   # number of random samples to test the divergence between models
    print(f'Federated Learning ongoing: {FL_tech}. Number of random samples to test divergence: {num_samples}')

