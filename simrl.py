# SimulationRL.py
import os
import random
import logging
import simpy
import pandas as pd
import networkx as nx
from datetime import datetime
import time
import traceback
# from Algorithm.agent.mhgnn_agent import MHGNNAgent
from Utils.utilsfunction import *
from system_configure import *
import system_configure
from globalvar import *
from Utils.plotfunction import *
import gc
from Class.earth import Earth
from Class.auxiliaryClass import *

from Algorithm.agent import REGISTRY_Agents


logger = logging.getLogger(__name__)


def setup_logging(output_path=None):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handlers = [logging.StreamHandler()]

    if output_path is not None:
        log_file = os.path.join(output_path, 'logfile_logging.log')
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit/timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps/progress) - elapsedTime
        logger.info("Simulation progress: %s%% Estimated time remaining: %s seconds Current simulation time: %s", progress, int(estimatedTimeRemaining), env.now)
        yield env.timeout(timeStepSize)
        progress += 1


def initialize(env, agent_class, popMapLocation, allGateWayInfo, distance, movementTime, outputPath, matching='Greedy'):
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

    # constellationType = inputParams['Constellation'][0]
    # fraction = inputParams['Fraction'][0]
    # testType = inputParams['Test type'][0]

    logger.info('Fraction of traffic generated: %s, test type: %s', Fraction, Test_type)
    # pathing  = inputParams['Pathing'][0]

    if Test_type == "Rates":
        getRates = True
    else:
        getRates = False

    # Load earth and gateways
    earth = Earth(env, agent_class, popMapLocation, allGateWayInfo, Constellation, movementTime, getRates, outputPath=outputPath)
    
    # if algo == "global":
    # earth.agent = agent_class()
    # else:
    #     for plane in earth.LEO:
    #         for sat in plane.sats:
    #             sat.agent = MHGNNAgent(config)
    logger.info('%s', earth)
    logger.info('')

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
                    path = getShortestPath(GT.name, destination.name, GT.graph)
                    GT.paths[destination.name] = path
                    paths.append(path)

    # add ISL references to all satellites and adjust data rate to GTs
    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)
            # Catalogues the inter-plane ISL as east or west (Right or left)
            sat.findInterNeighbours(earth)
            sat.findIntraNeighbours(earth)

    # fiveNeighbors = ([0],[])
    # pathNames = [name[0] for name in path]
    for plane in earth.LEO:
        for sat in plane.sats:
            if sat.linkedGT is not None:
                sat.adjustDownRate()
                # make a process for the GSL from sat to GT
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
            neighbors = list(nx.neighbors(graph, sat.ID))
            # if len(neighbors) == 5:
            #     print(sat, '5 neighbors found')
                # fiveNeighbors[0][0] += 1
                # fiveNeighbors[1].append(neighbors)
            if len(neighbors) != 4 and sat.linkedGT is None:
                logger.info('%s %s neighbors found', sat, len(neighbors))

            itt = 0
            for sat2 in sats:
                if sat2.ID in neighbors:
                    dataRate = graph[sat2.ID][sat.ID]["dataRateOG"]
                    distance = graph[sat2.ID][sat.ID]["slant_range"]
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
    bottleneck1, _ = findBottleneck(paths[0], earth, False, minimum2)

    logger.info('Traffic generated per GT (totalAvgFlow per Milliard):')
    logger.info('----------------------------------')
    for GT in earth.gateways:
        mins = []
        if GT.linkedSat[0] is not None:

            for pathKey in GT.paths:
                _, minimum = findBottleneck(GT.paths[pathKey], earth)
                mins.append(minimum)
            if GT.dataRate < GT.linkedSat[1].downRate:
                GT.getTotalFlow(1, flowGenType, 1, GT.dataRate, Fraction)  # using data rate of the GSL uplink
            else:
                GT.getTotalFlow(1, flowGenType, 1, GT.linkedSat[1].downRate, Fraction)  # using data rate of the GSL downlink

    total_network_injected_flow = sum(
        gt.totalAvgFlow for gt in earth.gateways
        if getattr(gt, 'totalAvgFlow', None) is not None
    )
    logger.info('Total network injected flow: %s Gbps', total_network_injected_flow / 1000000000)
    logger.info('----------------------------------')



    return earth, graph, bottleneck1, bottleneck2


def RunSimulation(GTs, outputPath, agent_class, radioKM):
    start_time = datetime.now()
    '''
    this is required for the bar plot at the end of the simulation
    percentages = {'Queue time': [],
                'Propagation time': [],
                'Transmission time': [],
                'GTnumber' : []}
    '''
    allGateWayInfo = pd.read_csv("./Gateways.csv")
    populationDataDir = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'

    simulationTimelimit = Test_length if Test_type != "Rates" else movementTime * Test_length + 10

    for GTnumber in GTs:
        global CurrentGTnumber
        CurrentGTnumber = GTnumber
        system_configure.CurrentGTnumber = GTnumber
        
        if len(GTs)>1:
            start_time_GT = datetime.now()

        env = simpy.Environment()

        if mixLocs: # changes the selected GTs every iteration
            # Shuffle the first max(GTs) rows of the DataFrame
            # sample(frac=1) shuffles all rows, reset_index(drop=True) resets the index
            shuffled_rows = allGateWayInfo.iloc[:max(GTs)].sample(frac=1).reset_index(drop=True)
            
            # Assign the shuffled rows back to the original DataFrame
            # This ensures that Location, Latitude, and Longitude stay aligned for each row
            allGateWayInfo.iloc[:max(GTs)] = shuffled_rows.values
            # random.shuffle(locations)
        selectedGateWayLocations = {'Location': allGateWayInfo['Location'][:GTnumber]} # only use the first GTnumber locations
        selectedGateWayLocations.update({'Latitude': allGateWayInfo['Latitude'][:GTnumber]})
        selectedGateWayLocations.update({'Longitude': allGateWayInfo['Longitude'][:GTnumber]})
        logger.info('----------------------------------')
        logger.info('Time:')
        logger.info('%s', datetime.now().strftime("%H:%M:%S"))
        logger.info('Locations:')
        logger.info('%s', selectedGateWayLocations['Location'])
        logger.info('Movement Time: %s', movementTime)


        # earth1, _, _, _ = initialize(env, agent_class, populationDataDir, './Gateways.csv', radioKM, inputParams, movementTime, locations, outputPath, matching=matching)
        earth1, _, _, _ = initialize(env, agent_class, populationDataDir, allGateWayInfo, radioKM, movementTime, outputPath, matching=matching)
        
        earth1.outputPath = outputPath
        

        # run the simulation
        env.process(simProgress(simulationTimelimit, env))
        env.process(earth1.monitor_max_queue(interval=5))  # 每5ms统计一次
        startTime = time.time()
        try:
            env.run(simulationTimelimit)
        except Exception as e:
            # SimPy re-raises a *new* exception instance and stores the original (with traceback)
            # in e.__cause__. Print it so we can see the real origin of errors like KeyError('block').
            logger.error('Simulation crashed inside SimPy.')
            if getattr(e, "__cause__", None) is not None:
                logger.error('Original exception (SimPy __cause__):')
                traceback.print_exception(e.__cause__)
            else:
                traceback.print_exception(e)
            raise
        timeToSim = time.time() - startTime

        # save learnt values
        if earth1.agent is not None:
            earth1.agent.try_save_model()
        
        # for plane in earth1.LEO:
        #     for sat in plane.sats:
        #         qTable = sat.QLearning.qTable
        #         with open(path + sat.ID + '.npy', 'wb') as f:
        #             np.save(f, qTable)

        # plotting and saving results
        logger.info('Simulation time: %s seconds', round(timeToSim, 2))
        logger.info('----------------------------------')
        if Test_type == "Rates":
            plotRatesFigures()
        else:
            # save the congestion_test in getBlockTransmissionStats() function
            results, allLatencies, pathBlocks, blocks = getBlockTransmissionStats(timeToSim, selectedGateWayLocations['Location'], Constellation, earth1, outputPath)
            logger.info('DataBlocks lost: %s', earth1.lostBlocks)
        
            plotSavePathLatencies(outputPath, GTnumber, pathBlocks)

            logger.info('Plotting Throughput...')
            plot_packet_latencies_and_uplink_downlink_throughput(blocks, outputPath, bins_num=30, save = True, plot_separately = plotAllThro)
            # plot_throughput_cdf(blocks, outputPath, bins_num = 100, save = True, plot_separately = plotAllThro)
            
            logger.info('Plotting rewards...')
            save_plot_rewards(outputPath, earth1.rewards, GTnumber)
            # save & plot all paths latencies
            logger.info('Plotting latencies...')
            plotSaveAllLatencies(outputPath, GTnumber, allLatencies)
            logger.info('plot queun length')
            plotQueues(earth1.queues, outputPath, GTnumber)
            
        # 保存最大队列统计
        if earth1.max_queue_stats:
            df_max_queue = pd.DataFrame(
                earth1.max_queue_stats, 
                columns=['Time_ms', 'Max_Queue_Length', 'Node_ID']
            )
            df_max_queue.to_csv(outputPath + '/csv/' + f"Max_Queue_Stats_{GTnumber}_GTs.csv", index=False)
            
            # 绘制最大队列长度随时间变化
            plt.figure(figsize=(12, 6))
            plt.plot(df_max_queue['Time_ms'], df_max_queue['Max_Queue_Length'])
            plt.xlabel('Simulation Time (ms)')
            plt.ylabel('Maximum Queue Length')
            plt.title(f'Maximum Queue Length Over Time ({GTnumber} GTs)')
            plt.grid(True)
            plt.savefig(outputPath + '/pngQueues/' + f'Max_Queue_Over_Time_{GTnumber}_GTs.png', dpi=300)
            plt.close()


        # print('Plotting link congestion figures...')
        # plotCongestionMap(earth1, np.asarray(blocks), outputPath + '/Congestion_Test/', GTnumber, plot_separately=plotAllCon)

        

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
            logger.info('----------------------------------')
            logger.info('Time:')
            end_time_GT = datetime.now()
            logger.info('%s', end_time_GT.strftime("%H:%M:%S"))
            logger.info('----------------------------------')
            elapsed_time_GT = end_time_GT - start_time_GT
            logger.info('Elapsed time for %s GTs: %s', GTnumber, elapsed_time_GT)
            logger.info('----------------------------------')

    # plotLatenciesBars(percentages, outputPath)

    logger.info('----------------------------------')
    logger.info('Time:')
    end_time = datetime.now()
    logger.info('%s', end_time.strftime("%H:%M:%S"))
    logger.info('----------------------------------')
    elapsed_time = end_time - start_time
    logger.info('Elapsed time: %s', elapsed_time)
    logger.info('----------------------------------')


import yaml

def get_configs():
    """Get dict variable from a YAML file.
    Args:
        file_dir: the directory of the YAML file.

    Returns:
        config_dict: the keys and corresponding values in the YAML file.
    """
    base_config_file_dir = "./Algorithm/algo_config/base_config.yaml"
    with open(base_config_file_dir, "r") as f:
        try:
            base_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, base_config_file_dir + " error: {}".format(exc)
    return base_config_dict

if __name__ == '__main__':
    import os
    from ruamel.yaml import YAML

    yaml_ruamel = YAML()
    yaml_ruamel.preserve_quotes = True
    yaml_ruamel.indent(mapping=2, sequence=4, offset=2)
    
    config_file = "./Algorithm/algo_config/base_config.yaml"
    # test_outputPath = "SimResults/DDQN/2025-12-24/15-15-52_Starlink_3s_GTs_[2]_1adj/"

    with open(config_file, "r") as f:
        config_data = yaml_ruamel.load(f)
    
    use_rl_model = config_data.get('use_rl_model', True)
    agent_name = config_data['agent']
    run_mode_name = agent_name

    if use_rl_model:
        if agent_name in REGISTRY_Agents:
            agent_class = REGISTRY_Agents[agent_name]
        else:
            raise ValueError(f"Unknown RL agent '{agent_name}'. Available: {list(REGISTRY_Agents.keys())}")
    else:
        agent_class = None
        run_mode_name = "ShortestPath"
        logger.info('Running in shortest-path mode. RL model is disabled.')
    
    
    current_dir = os.getcwd()
    if not use_rl_model:
        filetime_ymd = datetime.now().strftime("%Y-%m-%d")
        filetime_hms = datetime.now().strftime("%H-%M-%S")
        outputPath = current_dir + f'/SimResults/{run_mode_name}/{filetime_ymd}/{filetime_hms}_{Constellation}_{Test_length}s_GTs_{GTs}' + f'_avUserLoad{avUserLoad}/'
        os.makedirs(outputPath, exist_ok=True)
    elif config_data["train_TA_model"]:
        filetime_ymd = datetime.now().strftime("%Y-%m-%d")
        filetime_hms = datetime.now().strftime("%H-%M-%S")
        # data_inputrl = pd.read_csv("inputRL.csv")
        # constellation = data_inputrl['Constellation'][0]
        # sim_time = data_inputrl['Test length'][0]
        outputPath      = current_dir + f'/SimResults/{run_mode_name}/{filetime_ymd}/{filetime_hms}_{Constellation}_{Test_length}s_GTs_{GTs}/train/'
        os.makedirs(outputPath, exist_ok=True)
    else:
        mode_load_dir = config_data.get('mode_load_dir', None)
        if config_data['use_student_network']:
            outputPath = os.path.join(current_dir, mode_load_dir + f'test_student_network_avUserLoad{avUserLoad}/')
        else:
            outputPath = os.path.join(current_dir, mode_load_dir + f'test_teacher_network_avUserLoad{avUserLoad}/')
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    setup_logging(outputPath)

    config_data['outputPath'] = outputPath
    config_data['CurrentGTnumber'] = GTs[0]

    with open(config_file, "w") as f:
        yaml_ruamel.dump(config_data, f)
            
    save_system_config_to_json(os.path.join(outputPath, "system_config_dump.json"))
    RunSimulation(GTs, outputPath, agent_class, radioKM=rKM)
    # cProfile.run("RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)")
