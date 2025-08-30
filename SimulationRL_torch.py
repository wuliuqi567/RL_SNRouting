# SimulationRL.py
import os
import pickle
import numpy as np
import random
import simpy
import pandas as pd
import networkx as nx
from datetime import datetime
import time
import torch
import sys
from Algorithm.DDQNetwork import DDQNAgent
from Utils.logger import Logger
from Utils.utilsfunction import *
from configure import *
from globalvar import *
from Utils.plotfunction import *
import gc
import builtins
from Class.earth import Earth
from Class.auxiliaryClass import *


class hyperparam:
    def __init__(self, pathing):
        '''
        Hyperparameters of the Q-Learning model
        '''
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.ArriveR    = ArriveReward
        self.w1         = w1
        self.w2         = w2
        self.w4         = w4
        self.again      = againPenalty
        self.unav       = unavPenalty
        self.pathing    = pathing
        self.tau        = tau
        self.updateF    = updateF
        self.batchSize  = batchSize
        self.bufferSize = bufferSize
        self.hardUpdate = hardUpdate==1
        self.importQ    = importQVals
        self.MAX_EPSILON= MAX_EPSILON
        self.MIN_EPSILON= MIN_EPSILON
        self.LAMBDA     = LAMBDA
        self.plotPath  = plotPath
        self.coordGran  = coordGran
        self.ddqn       = ddqn
        self.latBias    = latBias
        self.lonBias    = lonBias
        self.diff       = diff
        self.explore    = explore
        self.reducedState= reducedState
        self.online     = onlinePhase
        self.diff_lastHop = diff_lastHop
 
    def __repr__(self):
        return 'Hyperparameters:\nalpha: {}\ngamma: {}\nepsilon: {}\nw1: {}\nw2: {}\n'.format(
        self.alpha,
        self.gamma,
        self.epsilon,
        self.w1,
        self.w2)


def saveHyperparams(outputPath, inputParams, hyperparams):
    print('Saving hyperparams at: ' + str(outputPath))
    hyperparams = ['Constellation: ' + str(inputParams['Constellation'][0]),
                'Import QTables: ' + str(hyperparams.importQ),
                'plotPath: ' + str(hyperparams.plotPath),
                'Test length: ' + str(inputParams['Test length'][0]),
                'Alphas: ' + str(hyperparams.alpha) + ', ' + str(alpha_dnn),
                'Gamma: ' + str(hyperparams.gamma),
                'Epsilon: ' + str(hyperparams.epsilon), 
                'Max epsilon: ' + str(hyperparams.MAX_EPSILON), 
                'Min epsilon: ' + str(hyperparams.MIN_EPSILON), 
                'Arrive Reward: ' + str(hyperparams.ArriveR), 
                'w1: ' + str(hyperparams.w1), 
                'w2: ' + str(hyperparams.w2),
                'w4: ' + str(hyperparams.w4),
                'againPenalty: ' + str(hyperparams.again),
                'unavPenalty: ' + str(hyperparams.unav),
                'Coords granularity: ' + str(hyperparams.coordGran),
                'Update freq: ' + str(hyperparams.updateF),
                'Batch Size: ' + str(hyperparams.batchSize),
                'Buffer Size: ' + str(hyperparams.bufferSize),
                'Hard Update: ' + str(hyperparams.hardUpdate),
                'Exploration: ' + str(hyperparams.explore),
                'DDQN: ' + str(hyperparams.ddqn),
                'Latitude bias: ' + str(hyperparams.latBias),
                'Longitude bias: ' + str(hyperparams.lonBias),
                'Diff: ' + str(hyperparams.diff),
                'Reduced State: ' + str(hyperparams.reducedState),
                'Online phase: ' + str(hyperparams.online)]

    # save hyperparams
    with open(outputPath + 'hyperparams.txt', 'w') as f:
        for param in hyperparams:
            f.write(param + '\n')


def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit/timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps/progress) - elapsedTime
        print("Simulation progress: {}% Estimated time remaining: {} seconds Current simulation time: {}".format(progress, int(estimatedTimeRemaining), env.now), end='\r')
        yield env.timeout(timeStepSize)
        progress += 1


def initialize(env, popMapLocation, GTLocation, distance, inputParams, movementTime, totalLocations, outputPath, matching='Greedy'):
    """
    Initializes an instance of the earth with cells from a population map and gateways from a csv file.
    During initialisation, several steps are performed to prepare for simulation:
        - GTs find the cells that within their ground coverage areas and "link" to them.
        - A certain LEO Constellation with a given architecture is created.
        - Satellites are distributed out to GTs so each GT connects to one satellite (if possible) and each satellite
        only has one connected GT.
        - A graph is created from all the GSLs and ISLs
        - Paths are created from each GT to all other GTs
        - Buffers and processes are created on all GTs and satellites used for sending the blocks throughout the network
    """
    print = builtins.print # Idk why but print breaks here so I had to rebuilt it
    # print(type(print))

    constellationType = inputParams['Constellation'][0]
    fraction = inputParams['Fraction'][0]
    testType = inputParams['Test type'][0]
    print(f'Fraction of traffic generated: {fraction}, test type: {testType}')
    # pathing  = inputParams['Pathing'][0]

    if testType == "Rates":
        getRates = True
    else:
        getRates = False

    # Load earth and gateways
    earth = Earth(env, popMapLocation, GTLocation, constellationType, inputParams, movementTime, totalLocations, getRates, outputPath=outputPath)

    print(earth)
    print()

    earth.linkCells2GTs(distance)
    earth.linkSats2GTs("Optimize")
    graph = createGraph(earth, matching=matching)
    earth.graph = graph

    for gt in earth.gateways:
        gt.graph = graph


    paths = []
    # make paths for all source destination pairs
    for GT in earth.gateways:
        for destination in earth.gateways:
            if GT != destination:
                if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                    path = getShortestPath(GT.name, destination.name, earth.pathParam, GT.graph)
                    GT.paths[destination.name] = path
                    paths.append(path)

    # add ISL references to all satellites and adjust data rate to GTs
    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)
            # Catalogues the inter-plane ISL as east or west (Right or left)
            sat.findInterNeighbours(earth)

    fiveNeighbors = ([0],[])
    pathNames = [name[0] for name in path]
    for plane in earth.LEO:
        for sat in plane.sats:
            if sat.linkedGT is not None:
                sat.adjustDownRate()
                # make a process for the GSL from sat to GT
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
            neighbors = list(nx.neighbors(graph, sat.ID))
            if len(neighbors) == 5:
                fiveNeighbors[0][0] += 1
                fiveNeighbors[1].append(neighbors)
            itt = 0
            for sat2 in sats:
                if sat2.ID in neighbors:
                    dataRate = nx.path_weight(graph,[sat2.ID, sat.ID], "dataRateOG")
                    distance = nx.path_weight(graph,[sat2.ID, sat.ID], "slant_range")
                    # check if satellite is inter- or intra-plane
                    if sat2.in_plane == sat.in_plane:
                        sat.intraSats.append((distance, sat2, dataRate))
                        # make a send buffer for intra ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsIntra.append(([sat.env.event()], [], sat2.ID))
                        # make a process for intra ISL
                        sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, True)))
                    else:
                        sat.interSats.append((distance, sat2, dataRate))
                        # make a send buffer for inter ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsInter.append(([sat.env.event()], [], sat2.ID))
                        # make a process for inter ISL
                        sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, False)))
                    itt += 1
                    if itt == len(neighbors):
                        break

    bottleneck2, minimum2 = findBottleneck(paths[1], earth, False)
    bottleneck1, minimum1 = findBottleneck(paths[0], earth, False, minimum2)

    print('Traffic generated per GT (totalAvgFlow per Milliard):')
    print('----------------------------------')
    for GT in earth.gateways:
        mins = []
        if GT.linkedSat[0] is not None:

            for pathKey in GT.paths:
                _, minimum = findBottleneck(GT.paths[pathKey], earth)
                mins.append(minimum)
            if GT.dataRate < GT.linkedSat[1].downRate:
                GT.getTotalFlow(1, "Step", 1, GT.dataRate, fraction)  # using data rate of the GSL uplink
            else:
                GT.getTotalFlow(1, "Step", 1, GT.linkedSat[1].downRate, fraction)  # using data rate of the GSL downlink
    print('----------------------------------')

    # In case we want to train the constellation we initialize the Q-Tables
    if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
        hyperparams = hyperparam(pathing)
    if pathing == 'Deep Q-Learning':
        if not onlinePhase:
            # Initialize global agent
            earth.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth)
        else:
            print('----------------------------------')
            print('Creating satellites agents...')
            if importQVals:
                print:(f'Importing the Neural networks from: \n{nnpath}\n{nnpathTarget}')
            for plane in earth.LEO:
                for sat in plane.sats:
                    sat.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth, sat.ID)
            print('----------------------------------')

    # save hyperparams
    if pathing == 'Q-Learning' or pathing == "Deep Q-Learning":
        saveHyperparams(earth.outputPath, inputParams, hyperparams)

    if pathing == 'Q-Learning':
        '''
        Q-Agents are initialized here
        '''
        earth.initializeQTables(len(earth.gateways), hyperparams, graph)

    return earth, graph, bottleneck1, bottleneck2


def RunSimulation(GTs, inputPath, outputPath, populationData, radioKM):
    start_time = datetime.now()
    '''
    this is required for the bar plot at the end of the simulation
    percentages = {'Queue time': [],
                'Propagation time': [],
                'Transmission time': [],
                'GTnumber' : []}
    '''
    inputParams = pd.read_csv(inputPath + "inputRL.csv")

    locations = inputParams['Locations'].copy()
    print('Nº of Gateways: ' + str(len(locations)))

    # pathing     = inputParams['Pathing'][0]
    testType    = inputParams['Test type'][0]
    testLength  = inputParams['Test length'][0]
    # numberOfMovements = 0

    print('Routing metric: ' + pathing)

    simulationTimelimit = testLength if testType != "Rates" else movementTime * testLength + 10

    firstGT = True
    for GTnumber in GTs:
        global CurrentGTnumber
        global Train
        global TrainThis
        global nnpath
        if FL_Test:
            global CKA_Values
        if ddqn:
            global nnpathTarget
        TrainThis       = Train
        CurrentGTnumber = GTnumber
        
        if firstGT:
            # nnpath  = f'./pre_trained_NNs/qNetwork_1GTs.h5'   # Already set
            firstGT = False
        else:
            nnpath  = f'{outputPath}/NNs/qNetwork_{GTnumber-1}GTs.h5'
            if ddqn:
                nnpathTarget = f'{outputPath}/NNs/qTarget_{GTnumber-1}GTs.h5'

        if len(GTs)>1:
            start_time_GT = datetime.now()

        env = simpy.Environment()

        if mixLocs: # changes the selected GTs every iteration
            firstLocs = locations[:max(GTs)]
            random.shuffle(firstLocs)
            locations[:max(GTs)] = firstLocs
            # random.shuffle(locations)
        inputParams['Locations'] = locations[:GTnumber] # only use the first GTnumber locations
        print('----------------------------------')
        print('Time:')
        print(datetime.now().strftime("%H:%M:%S"))
        print('Locations:')
        print(inputParams['Locations'][:GTnumber])
        print(f'Movement Time: {movementTime}')
        print(f'Rotation Factor: {ndeltas}')
        print(f'Minimum epsilon: {MIN_EPSILON}')
        print(f'Reward for deliver: {ArriveReward}')
        print(f'Stop Loss: {stopLoss}, number of samples considered: {nLosses}, threshold: {lThreshold}')
        print('----------------------------------')
        earth1, _, _, _ = initialize(env, populationData, inputPath + 'Gateways.csv', radioKM, inputParams, movementTime, locations, outputPath, matching=matching)
        earth1.outputPath = outputPath
        


# print ISL on Earth
        print('Saving ISLs map...')
        islpath = outputPath + '/ISL_maps/'
        os.makedirs(islpath, exist_ok=True) 
        earth1.plotMap(plotGT = True, plotSat = True, edges=True, save = True, outputPath=islpath, n=earth1.nMovs)
        plt.close()
        print('----------------------------------')



# run the simulation
        progress = env.process(simProgress(simulationTimelimit, env))
        startTime = time.time()
        env.run(simulationTimelimit)
        timeToSim = time.time() - startTime




        if testType == "Rates":
            plotRatesFigures()
        else:
            # save the congestion_test in getBlockTransmissionStats() function
            results, allLatencies, pathBlocks, blocks = getBlockTransmissionStats(timeToSim, inputParams['Locations'], inputParams['Constellation'][0], earth1, outputPath)
            print(f'DataBlocks lost: {earth1.lostBlocks}')
            


            # save & plot ftirst 2 GTs path latencies 
            # plot Latency with arrival time and block number of every block 
            # in /pngLatencies/
            plotSavePathLatencies(outputPath, GTnumber, pathBlocks)



            # Throughput figures
            # in /Throughput/
            print('Plotting Throughput...')
            plot_packet_latencies_and_uplink_downlink_throughput(blocks, outputPath, bins_num=30, save = True, plot_separately = plotAllThro)
            plot_throughput_cdf(blocks, outputPath, bins_num = 100, save = True, plot_separately = plotAllThro)
            


            
            if pathing == "Deep Q-Learning" or pathing == 'Q-Learning':

                # save & plot rewards in /Rewards/
                save_plot_rewards(outputPath, earth1.rewards, GTnumber)

                
                if not onlinePhase:
                    eps = earth1.DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                else:
                    eps = earth1.LEO[0].sats[0].DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                # save epsilons
                if Train:
                    epsDF = save_epsilons(outputPath, eps, GTnumber)
                    save_training_counts(outputPath, earth1.trains, GTnumber)
                else:
                    epsDF = None

                # save & plot all paths latencies
                print('Plotting latencies...')
                plotSaveAllLatencies(outputPath, GTnumber, allLatencies, epsDF)
            
            if pathing == "Deep Q-Learning":
                # save losses
                save_losses(outputPath, earth1, GTnumber)
                if FL_Test and const_moved:
                    print('Plotting CKA values...')
                    plot_cka_over_time(earth1.CKA, outputPath, GTnumber)
                
            else:
                print('Plotting latencies...')
                plotSaveAllLatencies(outputPath, GTnumber, allLatencies)

        plotShortestPath(earth1, pathBlocks[1][-1].path, outputPath)
        if not onlinePhase:
            plotQueues(earth1.queues, outputPath, GTnumber)

        print('Plotting link congestion figures...')
        plotCongestionMap(earth1, np.asarray(blocks), outputPath + '/Congestion_Test/', GTnumber, plot_separately=plotAllCon)

        print(f"number of gateways: {GTnumber}")
        print('Path:')
        print(pathBlocks[1][-1].path)
        print('Bottleneck:')
        print(findBottleneck(pathBlocks[1][-1].path, earth1))

        '''
        # add data for percentages bar plot
        # percentages['Queue time']       .append(results.meanQueueLatency)
        # percentages['Propagation time'] .append(results.meanPropLatency)
        # percentages['Transmission time'].append(results.meanTransLatency)
        # percentages['GTnumber']         .append(GTnumber)

        save congestion test data
        print('Saving congestion test data...')
        blocks = []
        for block in receivedDataBlocks:
            blocks.append(BlocksForPickle(block))
        blockPath = outputPath + f"./Results/Congestion_Test/{pathing} {float(pd.read_csv('inputRL.csv')['Test length'][0])}/"
        os.makedirs(blockPath, exist_ok=True)
        try:
            np.save("{}blocks_{}".format(blockPath, GTnumber), np.asarray(blocks),allow_pickle=True)
        except pickle.PicklingError:
            print('Error with pickle and profiling')
        '''

        # save learnt values
        if pathing == 'Q-Learning':
            saveQTables(outputPath, earth1)
        elif pathing == 'Deep Q-Learning':
            saveDeepNetworks(outputPath + '/NNs/', earth1)

        # percentages.clear()
        receivedDataBlocks  .clear()
        createdBlocks       .clear()
        pathBlocks          .clear()
        allLatencies        .clear()
        upGSLRates          .clear()
        downGSLRates        .clear()
        interRates          .clear()
        intraRate           .clear()
        del results
        del earth1
        del env
        del _
        gc.collect()

        if len(GTs)>1:
            print('----------------------------------')
            print('Time:')
            end_time_GT = datetime.now()
            print(end_time_GT.strftime("%H:%M:%S"))
            print('----------------------------------')
            elapsed_time_GT = end_time_GT - start_time_GT
            print(f"Elapsed time for {GTnumber} GTs: {elapsed_time_GT}")
            print('----------------------------------')

    # plotLatenciesBars(percentages, outputPath)

    print('----------------------------------')
    print('Time:')
    end_time = datetime.now()
    print(end_time.strftime("%H:%M:%S"))
    print('----------------------------------')
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
    print('----------------------------------')


if __name__ == '__main__':
    import os

    current_dir = os.getcwd()

    # nnpath          = f'./pre_trained_NNs/qNetwork_8GTs.h5'
    filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputPath      = current_dir + '/Results/{}_{}s_[{}]_Del_[{}]_w1_[{}]_w2_{}_GTs_onlinePhase_{}_{}/'.format(pathing, float(pd.read_csv("inputRL.csv")['Test length'][0]), ArriveReward, w1, w2, GTs, onlinePhase, filetime)
    populationMap   = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'
    os.makedirs(outputPath, exist_ok=True) 
    sys.stdout = Logger(outputPath + 'logfile.log')

    RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)
    # cProfile.run("RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)")
