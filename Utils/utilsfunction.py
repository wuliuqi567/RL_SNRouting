import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from configure import *
from Class.auxiliaryClass import *
from Class.orbitalPlane import OrbitalPlane
from Class.edge import edge
import numba
import pickle

def getBlockTransmissionStats(timeToSim, GTs, constellationType, earth, outputPath):
    '''
    General Block transmission stats
    '''
    allTransmissionTimes = []    # list of all the transmission times of the blocks
    largestTransmissionTime = (0, None) # (Transmission time, Block) of the largest transmission time
    mostHops = (0, None) # (Number of hops, Block) of the block with most hops
    queueLat = [] # list of all the queue latencies of the blocks
    txLat = [] # lisr of 
    propLat = [] 
    # latencies = [queueLat, txLat, propLat]
    blocks = [] # 
    allLatencies= []
    pathBlocks  = [[],[]]
    first       = earth.gateways[0]
    second      = earth.gateways[1]

    earth.pathParam

    for block in receivedDataBlocks: # 
        time = block.getTotalTransmissionTime()
        hops = len(block.checkPoints)
        blocks.append(BlocksForPickle(block))

        if largestTransmissionTime[0] < time:
            largestTransmissionTime = (time, block)

        if mostHops[0] < hops:
            mostHops = (hops, block)

        allTransmissionTimes.append(time)

        queueLat.append(block.getQueueTime()[0])
        txLat.append(block.txLatency)
        propLat.append(block.propLatency)
        
        # [creation time, total latency, arrival time, source, destination, block ID, queue time, transmission latency, prop latency]
        allLatencies.append([block.creationTime, block.totLatency, block.creationTime+block.totLatency, block.source.name, block.destination.name, block.ID, block.getQueueTime()[0], block.txLatency, block.propLatency])
        # pre-process the received data blocks. create the rows that will be saved in csv
        if block.source == first and block.destination == second:
            pathBlocks[0].append([block.totLatency, block.creationTime+block.totLatency])
            pathBlocks[1].append(block)
        
    # save congestion test data
    # blockPath = f"./Results/Congestion_Test/{pathing} {float(pd.read_csv('inputRL.csv')['Test length'][0])}/"
    print('Saving congestion test data...\n')
    blockPath = outputPath + '/Congestion_Test/'     
    os.makedirs(blockPath, exist_ok=True)
    try:
        global CurrentGTnumber
        np.save("{}blocks_{}".format(blockPath, CurrentGTnumber), np.asarray(blocks),allow_pickle=True)
    except pickle.PicklingError:
        print('Error with pickle and profiling')

    avgTime = np.mean(allTransmissionTimes)
    totalTime = sum(allTransmissionTimes)

    print("\n########## Results #########\n")
    print(f"The simulation took {timeToSim} seconds to run")
    print(f"A total of {len(createdBlocks)} data blocks were created")
    print(f"A total of {len(receivedDataBlocks)} data blocks were transmitted")
    print(f"A total of {len(createdBlocks) - len(receivedDataBlocks)} data blocks were stuck")
    print(f"Average transmission time for all blocks were {avgTime}")
    print('Total latecies:\nQueue time: {}%\nTransmission time: {}%\nPropagation time: {}%'.format(
        '%.4f' % float(sum(queueLat)/totalTime*100),
        '%.4f' % float(sum(txLat)/totalTime*100),
        '%.4f' % float(sum(propLat)/totalTime*100)))

    results = Results(finishedBlocks=blocks,
                      constellation=constellationType,
                      GTs=GTs,
                      meanTotalLatency=avgTime,
                      meanQueueLatency=np.mean(queueLat),
                      meanPropLatency=np.mean(propLat),
                      meanTransLatency=np.mean(txLat),
                      perQueueLatency = sum(queueLat)/totalTime*100,
                      perPropLatency = sum(propLat)/totalTime*100,
                      perTransLatency = sum(txLat)/totalTime*100)

    return results, allLatencies, pathBlocks, blocks

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


def get_direction(Satellites):
    '''
    Gets the direction of the satellites so each transceiver antenna can be set to one direction.
    '''
    N = len(Satellites)
    direction = np.zeros((N,N), dtype=np.int8)
    for i in range(N):
        epsilon = -Satellites[i].inclination    # orbital plane inclination
        for j in range(N):
            direction[i,j] = np.sign(Satellites[i].y*math.sin(epsilon)+
                                    Satellites[i].z*math.cos(epsilon)-Satellites[j].y*math.sin(epsilon)-
                                    Satellites[j].z*math.cos(epsilon))
    return direction


def get_pos_vectors_omni(Satellites):
    '''
    Given a list of satellites returns a list with x, y, z coordinates and the plane where they are (meta)
    '''
    N = len(Satellites)
    Positions = np.zeros((N,3))
    meta = np.zeros(N, dtype=np.int_)
    for n in range(N):
        Positions[n,:] = [Satellites[n].x, Satellites[n].y, Satellites[n].z]
        meta[n] = Satellites[n].in_plane

    return Positions, meta


def get_slant_range(edge):
        return(edge.slant_range)


# @numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def get_slant_range_optimized(Positions, N):
    '''
    returns a matrix with the all the distances between the satellites (optimized)
    '''
    slant_range = np.zeros((N,N))
    for i in range(N):
        slant_range[i,i] = math.inf
        for j in range(i+1,N):
            slant_range[i,j] = np.linalg.norm(Positions[i,:] - Positions[j,:])
    slant_range += np.transpose(slant_range)
    return slant_range


@numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def los_slant_range(_slant_range, _meta, _max, _Positions):
    '''
    line of sight slant range
    '''
    _slant_range_new = np.copy(_slant_range)
    _N = len(_slant_range)
    for i in range(_N):
        for j in range(_N):
            if _slant_range_new[i,j] > _max[_meta[i], _meta[j]]:
                _slant_range_new[i,j] = math.inf
    return _slant_range_new


def get_data_rate(_slant_range_los, interISL):
    """
    Given a matrix of slant ranges returns a matrix with all the shannon dataRates possibles between all the satellites.
    """
    speff_thresholds = np.array(
        [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
         1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
         2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
         3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
         5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
    lin_thresholds = np.array(
        [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
         1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
         3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
         8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
         16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
         45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])

    pathLoss = 10*np.log10((4 * math.pi * _slant_range_los * interISL.f / Vc)**2)   # Free-space pathloss in dB
    snr = 10**((interISL.maxPtx_db + interISL.G - pathLoss - interISL.No)/10)       # SNR in times
    shannonRate = interISL.B*np.log2(1+snr)                                         # data rates matrix in bits per second

    speffs = np.zeros((len(_slant_range_los),len(_slant_range_los)))

    for n in range(len(_slant_range_los)):
        for m in range(len(_slant_range_los)):
            feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr[n,m])]
            if feasible_speffs.size == 0:
                speffs[n, m] = 0
            else:
                speffs[n,m] = interISL.B * feasible_speffs[-1]

    return speffs


def markovianMatchingTwo(earth):
    '''
    Returns a list of edge class elements. Each edge stands for a connection between two satellites. On that class
    the slant range and the data rate between both satellites are stored as attributes.
    This function is for satellites with two transceivers antennas that will enable two inter-plane ISL each one
    in a different direction.
    Intra-plane ISL are also computed and returned in _A_Markovian list

    It is not the optimal solution, but it is from 10 to 1000x faster.
    Minimizes the total cost of the constellation matching problem.
    '''

    _A_Markovian    = []    # list with all the
    Satellites      = []    # list with all the satellites
    W_M             = []    # list with the distances of every possible link between sats
    covered         = set() # Set with the connections already covered

    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    interISL = RFlink(
        frequency=26e9,
        bandwidth=500e6,
        maxPtx=10,
        aDiameterTx=0.26,
        aDiameterRx=0.26,
        pointingLoss=0.3,
        noiseFigure=2,
        noiseTemperature=290,
        min_rate=10e3
    )

    # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)              # Number of planes in LEO
    Max_slnt_rng = np.zeros((M,M))  # All ISL slant ranges must me lowe than 'Max_slnt_rng[i, j]'

    Orb_heights  = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i,_j] = (np.sqrt( (Orb_heights[_i] + Re)**2 - Re**2 ) +
                                np.sqrt( (Orb_heights[_j] + Re)**2 - Re**2 ) )


    # Get data rate old method
    ###########################################################
    direction       = get_direction(Satellites)             # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)      # position and plane of all the satellites
    slant_range     = get_slant_range_optimized(Positions, N)                       # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)   # distance matrix but if d>dMax, d=infinite
    shannonRate     = get_data_rate(slant_range_los, interISL)                      # max dataRate

    '''
    Compute all possible edges between different plane satellites whose transceiver antennas are free.
    if slant range > max slant range then that edge is not added
    '''
    ###########################################################
    for i in range(N):
        for j in range(i+1,N):
            if Satellites[i].in_plane != Satellites[j].in_plane and ((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered):
                if slant_range_los[i,j] < 6000e3: # math.inf:
                    W_M.append(edge(Satellites[i].ID,Satellites[j].ID,slant_range_los[i,j],direction[i,j], direction[j,i], shannonRate[i,j]))

    W_sorted=sorted(W_M,key=get_slant_range) # NOTE we could choose shannonRate instead

    # from all the possible links adds only the uncovered with the best weight possible
    ###########################################################
    while W_sorted:
        if  ((W_sorted[0].i,W_sorted[0].dji) not in covered) and ((W_sorted[0].j,W_sorted[0].dij) not in covered):
            _A_Markovian.append(W_sorted[0])
            covered.add((W_sorted[0].i,W_sorted[0].dji))
            covered.add((W_sorted[0].j,W_sorted[0].dij))
        W_sorted.pop(0)

    # add intra-ISL edges
    ###########################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane        *nPerPlane    +sat.i_in_plane
            j = sat.upper.in_plane  *nPerPlane    +sat.upper.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.upper.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane  *nPerPlane    +sat.lower.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.lower.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

    return _A_Markovian

def greedyMatching(earth):
    '''
    Returns a list of edge class elements based on a greedy algorithm.
    Each satellite is connected to its upper and lower satellite in the same orbital plane (intra-plane),
    and the nearest satellites to the east and west in different planes (inter-plane).
    The slant range and the data rate between satellites are stored as attributes in the edge class.
    '''

    _A_Greedy = []  # list to store edges
    Satellites = []  # list of all satellites

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    # inter-plane ISL 
    ##############################################################
    # link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

   # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)              # Number of planes in LEO
    Max_slnt_rng = np.zeros((M,M))  # All ISL slant ranges must be lowe than 'Max_slnt_rng[i, j]'

    Orb_heights  = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i,_j] = (np.sqrt( (Orb_heights[_i] + Re)**2 - Re**2 ) +
                                np.sqrt( (Orb_heights[_j] + Re)**2 - Re**2 ) )
            
    # Compute positions and slant ranges
    ##############################################################
    direction       = get_direction(Satellites)             # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)      # position and plane of all the satellites
    slant_range     = get_slant_range_optimized(Positions, N)                       # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)   # distance matrix but if d>dMax, d=infinite
    shannonRate     = get_data_rate(slant_range_los, interISL)                      # max dataRate

    # Create edges for inter-plane links (closest east and west satellites)
    for i, sat in enumerate(Satellites):
        closest_east, closest_west = None, None
        min_east_distance, min_west_distance = float('inf'), float('inf')

        for j, other_sat in enumerate(Satellites):
            if sat.in_plane != other_sat.in_plane:
                if slant_range_los[i, j] < min_east_distance and Positions[j, 0] > Positions[i, 0]:  # East satellite
                    closest_east, min_east_distance = other_sat, slant_range_los[i, j]
                elif slant_range_los[i, j] < min_west_distance and Positions[j, 0] < Positions[i, 0]:  # West satellite
                    closest_west, min_west_distance = other_sat, slant_range_los[i, j]

        # Add edges for closest east and west satellites
        if closest_east:
            _A_Greedy.append(edge(sat.ID, closest_east.ID, min_east_distance, None, None, shannonRate[i, Satellites.index(closest_east)]))
        if closest_west:
            _A_Greedy.append(edge(sat.ID, closest_west.ID, min_west_distance, None, None, shannonRate[i, Satellites.index(closest_west)]))
        
    # intra-plane ISL links (upper and lower neighbors)
    ##############################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane        *nPerPlane    +sat.i_in_plane
            j = sat.upper.in_plane  *nPerPlane    +sat.upper.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.upper.ID,     # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            None, None,                                     # directions
            shannonRate[i,j]))                              # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane  *nPerPlane    +sat.lower.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.lower.ID,     # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            None, None,                                     # directions
            shannonRate[i,j]))                              # Max dataRate

    return _A_Greedy


def deleteDuplicatedLinks(satA, g, earth):
    '''
    Given a satellite, searches for its east and west neighbour. If the east or west link is duplicated,
    it will remove the link with a higher latitude difference, keeping the horizontal links
    '''

    def getMostHorizontal(currentSat, satA, satB):
        '''
        Chooses the dat with the closest latitude to currentSat
        '''
        return (satA, satB) if abs(satA.latitude-currentSat.latitude)<abs(satB.latitude-currentSat.latitude) else (satB, satA)

    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if(dir == 3):                                         # Found Satellite at East
                if linkedSats['R'] is not None:
                    # print(f"{satA.ID} east satellite duplicated: {linkedSats['R'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['R'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['R']  = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['R']  = satB

            elif(dir == 4):                                         # Found Satellite at West
                if linkedSats['L'] is not None:
                    # print(f"{satA.ID} West satellite duplicated: {linkedSats['L'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['L'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['L']  = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['L']  = satB

def establishRemainingISLs(earth, g):
    Satellites = []

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    # Gather positions and other parameters
    Positions, meta = get_pos_vectors_omni(Satellites)
    direction = get_direction(Satellites)
    slant_range = get_slant_range_optimized(Positions, len(Satellites))

    # Prepare link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

    # Calculate maximum slant range
    Max_slnt_rng = np.zeros((len(earth.LEO), len(earth.LEO)))
    Orb_heights = [plane.h for plane in earth.LEO]
    for i in range(len(earth.LEO)):
        for j in range(len(earth.LEO)):
            Max_slnt_rng[i, j] = (np.sqrt((Orb_heights[i] + Re)**2 - Re**2) +
                                  np.sqrt((Orb_heights[j] + Re)**2 - Re**2))

    # Define slant range and data rate matrices
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)
    shannonRate = get_data_rate(slant_range_los, interISL)

    # Identify satellites with specific missing neighbors
    satellites_with_no_right = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.right is None}
    satellites_with_no_left = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.left is None}

    # Calculate potential matches sorted by horizontal alignment
    potential_links = []
    for sat_r in satellites_with_no_right:
        for sat_l in satellites_with_no_left:
            if sat_r.in_plane != sat_l.in_plane:
                idx_r = Satellites.index(sat_r)
                idx_l = Satellites.index(sat_l)
                if slant_range_los[idx_r, idx_l] < math.inf:
                    # Handle longitude wrapping correctly
                    longitude_difference = (satellites_with_no_left[sat_l][0] - satellites_with_no_right[sat_r][0] + 360) % 360
                    if longitude_difference > 0 and longitude_difference < 180:
                        # lat_diff = abs(satellites_with_no_right[sat_r][1] - satellites_with_no_left[sat_l][1])
                        lat_diff = abs(sat_r.latitude-sat_l.latitude)
                        potential_links.append((lat_diff, sat_r, sat_l, slant_range_los[idx_r, idx_l]))

    # Sort by latitude difference to prioritize horizontal links
    # potential_links.sort()
    potential_links.sort(key=lambda x: x[0])  # Uses latitude difference as sort key


    # Establish links from closest to farthest in terms of horizontal alignment
    for lat_diff, sat_r, sat_l, distance in potential_links:
        if sat_r.right is None and sat_l.left is None:
            g.add_edge(sat_r.ID, sat_l.ID, slant_range=distance,
                       dataRate=1/shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)],
                       dataRateOG=shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)], hop=1)
            sat_r.right = sat_l
            sat_l.left = sat_r
            # print(f"Established horizontal link between {sat_r.ID} (right) and {sat_l.ID} (left) with latitude difference {lat_diff:.2f} deg and distance: {distance/1000:.2F} km.")

    return g


def createGraph(earth, matching='Greedy'):
    '''
    Each satellite has two transceiver antennas that are connected to the closest satellite in east and west direction to a satellite
    from another plane (inter-ISL). Each satellite also has anoteher two transceiver antennas connected to the previous and to the
    following satellite at their orbital plane (intra-ISL).
    A graph is created where each satellite is a node and each connection is an edge with a specific weight based either on the
    inverse of the maximum data rate achievable, total distance or number of hops.
    '''
    g = nx.Graph()

    # add LEO constellation
    ###############################
    for plane in earth.LEO:
        for sat in plane.sats:
            g.add_node(sat.ID, sat=sat)

    # add gateways and GSL edges
    ###############################
    for GT in earth.gateways:
        if GT.linkedSat[1]:
            g.add_node(GT.name, GT = GT)            # add GT as node
            g.add_edge(GT.name, GT.linkedSat[1].ID, # add GT linked sat as edge
            slant_range = GT.linkedSat[0],          # slant range
            invDataRate = 1/GT.dataRate,            # Inverse of dataRate
            dataRateOG = GT.dataRate,               # original shannon dataRate
            hop = 1)                                # in case we just want to count hops

    # add inter-ISL and intra-ISL edges
    ###############################
    if matching=='Markovian':
        markovEdges = markovianMatchingTwo(earth)
    elif matching=='Greedy':
        markovEdges = greedyMatching(earth)
    print(f'Matching: {matching}')
    # print('----------------------------------')

    global biggestDist
    global firstMove
    # biggestDist = -1
    for markovEdge in markovEdges:
        g.add_edge(markovEdge.i, markovEdge.j,  # source and destination IDs
        slant_range = markovEdge.slant_range,   # slant range
        dataRate = 1/markovEdge.shannonRate,    # Inverse of dataRate # FIXME sometimes markovEdge.shannonRate is 0
        dataRateOG = markovEdge.shannonRate,    # Original shannon datRate
        hop = 1,                                # in case we just want to count hops
        dij = markovEdge.dij,
        dji = markovEdge.dji)
        if firstMove and markovEdge.slant_range > biggestDist:  # keep the biggest possible distance for the normalization of the rewards
            biggestDist = markovEdge.slant_range

    # remove duplicated links and keep the most horizontal ones
    print('Removing duplicated links...')
    for plane in earth.LEO:
        for sat in plane.sats:
            deleteDuplicatedLinks(sat, g, earth)
        
    earth.graph = g
    
    # update the neighbors
    for plane in earth.LEO:
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)
            sat.findInterNeighbours(earth)

    print('Establishing remaining edges...')
    g = establishRemainingISLs(earth, g)


    if firstMove:
        print(f'Biggest slant range between satellites: {biggestDist/1000:.2f} km')
        firstMove = False
    print('----------------------------------')

    return g

def findByID(earth, satID):
    '''
    given the ID of a satellite, this function will return the corresponding satellite object
    '''
    for plane in earth.LEO:
        for sat in plane.sats:
            if (sat.ID == satID):
                return sat
            
def getDirection(satA, satB):
    '''
    Returns the direction from satA to satB, considering the Earth's wrap-around for longitude.
    '''

    def normalize_longitude(lon):
        # Normalize longitude to the range [-math.pi, math.pi]
        return ((lon + math.pi) % (2 * math.pi)) - math.pi

    planei = int(satA.in_plane)
    planej = int(satB.in_plane)

    if planei == planej:
        if satA.latitude < satB.latitude:
            return 1  # Go Upper
        else:
            return 2  # Go Lower

    # Normalize the longitudes
    norm_lonA = normalize_longitude(satA.longitude)
    norm_lonB = normalize_longitude(satB.longitude)

    # Calculate the normalized longitude difference
    lon_diff = normalize_longitude(norm_lonB - norm_lonA)

    # Decide direction based on normalized difference
    if lon_diff > 0:
        return 3  # Go Right
    else:
        return 4  # Go Left
    
def plotShortestPath(earth, path, outputPath, ID=None, time=None):
    earth.plotMap(True, True, path=path, ID=ID,time=time)
    plt.savefig(outputPath + 'popMap_' + path[0][0] + '_to_' + path[len(path)-1][0] + '.png', dpi = 500)
    # plt.show()
    plt.close()


# @profile
def create_Constellation(specific_constellation, env, earth):

    if specific_constellation == "small":               # Small Walker star constellation for tests.
        print("Using small walker Star constellation")
        P = 4					# Number of orbital planes
        N_p = 8 				# Number of satellites per orbital plane
        N = N_p*P				# Total number of satellites
        height = 1000e3			# Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 53	# Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True		# Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30

    elif specific_constellation =="Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        N = N_p*P
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        N = N_p*P
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648
        N_p = int(N/P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Starlink":			# Phase 1 550 km altitude orbit shell
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N/P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30                     # Number of orbital planes
        N = 1200                   # Total number of satellites
        N_p = int(N/P)             # Number of satellites per orbital plane
        height = 600e3             # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 86.4   # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True         # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30
    else:
        print("Not valid Constellation Name")
        P = np.NaN
        N_p = np.NaN
        N = np.NaN
        height = np.NaN
        inclination_angle = np.NaN
        Walker_star = False
        exit()

    distribution_angle = 2*math.pi  # Angle in which the orbital planes are distributed in

    if Walker_star:
        distribution_angle /= 2
    orbital_planes = []

    # Add orbital planes and satellites
    # Orbital_planes.append(orbital_plane(0, height, 0, math.radians(inclination_angle), N_p, min_elevation_angle, 0))
    for i in range(0, P):
        orbital_planes.append(OrbitalPlane(str(i), height, i*distribution_angle/P, math.radians(inclination_angle), N_p,
                                           min_elevation_angle, str(i) + '_', env, earth))

    return orbital_planes



def getShortestPath(source, destination, weight, g):
    '''
    Gives you the shortest path between a source and a destination and plots it if desired.
    Uses the 'dijkstra' algorithm to compute the sortest path, where the total weight of the path can be either the sum of inverse
    of the maximumm dataRate achevable, the total slant range or the number of hops taken between source and destination.

    returns a list where each element is a sublist with the name of the node, its longitude and its latitude.
    '''

    path = []
    try:
        shortest = nx.shortest_path(g, source, destination, weight = weight)    # computes the shortest path [dataRate, slant_range, hops]
        for hop in shortest:                                                    # pre process the data so it can be used in the future
            key = list(g.nodes[hop])[0]
            if shortest.index(hop) == 0 or shortest.index(hop) == len(shortest)-1:
                path.append([hop, g.nodes[hop][key].longitude, g.nodes[hop][key].latitude])
            else:
                path.append([hop, math.degrees(g.nodes[hop][key].longitude), math.degrees(g.nodes[hop][key].latitude)])
    except Exception as e:
        print(f"getShortestPath Caught an exception: {e}")
        print('No path between ' + source + ' and ' + destination + ', check the graph to see more details.')
        return -1
    return path



def computeOutliers(g):
    '''
    Given a graph, will return the throughput and slant range thresholds that will be used to find the outliers
    (Devices with bad conditions)
    '''
    # define outliers
    slantRanges = []
    dataRates   = []

    for edge in list(g.edges()):
        slantRanges.append(g.edges[edge]['slant_range'])
        dataRates  .append(g.edges[edge]['dataRateOG'])

    # Slant Range Outliers
    slantRanges = pd.Series(slantRanges)
    Q3 = slantRanges.describe()['75%']
    Q1 = slantRanges.describe()['25%']
    IQR = Q3 - Q1
    upperFence = Q3 + (1.5*IQR)

    # Data Rate Outliers
    dataRates = pd.Series(dataRates)
    Q3 = dataRates.describe()['75%']
    Q1 = dataRates.describe()['25%']
    IQR = Q3 - Q1
    lowerFence = Q1 - (1.5*IQR)

    return lowerFence, upperFence


def getQueues(sat, threshold=None, DDQN = False):
    '''
    When !DDQN, this function will return True if one of the satellite queues has a length over a limit or they are
    missing one link

    Each satellite has a queue for each link which includes both ISL and GSL (sat 2 GT). The Queues are implemented as
    tuples that contain a list of simpy events, a list of the data blocks, and the ID of the satellite for the link
    (there is no ID for the GT queues). The structure is tuple[list[Simpy.event], list[DataBlock], ID].
    The list of events will always have at least one event present which will be non-triggered when there are no blocks
    in the queue. When blocks are present, there will be as many triggered events as there are blocks.

    On the GTs, there is one queue which has the same structure as the queues for the GSLs on the satellites:
    tuple[list[Simpy.event], list[DataBlock]]

    ISLs Queues: sendBufferSats where each entry is a separate queue.
    GSLs Queues: sendBufferGT. While there will never be more than one queue in this list.
    GTs  Queues: sendBuffer which is just the tuple itself

    In our case we will just choose the highest queue of all the ISLs and compare it to a threshold

    The try excepts are for those cases where the linked satellite does not have the 4 linked satllites queues.
    IF THE SATELLITE DOES NOT HAVE 4 LINEKD SATELLITES IT WILL BE CONSIDERED AS HIGH QUEUE
    '''
    queuesLen = []
    infQueue  = False
    queuesDic = {'U': np.inf,
                 'D': np.inf,
                 'R': np.inf,
                 'L': np.inf}
    try:
       queuesLen.append(len(sat.sendBufferSatsIntra[0][1]))
       queuesDic['U'] = len(sat.sendBufferSatsIntra[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
       queuesLen.append(len(sat.sendBufferSatsIntra[1][1]))
       queuesDic['D'] = len(sat.sendBufferSatsIntra[1][1])

    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[0][1]))
        queuesDic['R'] = len(sat.sendBufferSatsInter[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[1][1]))
        queuesDic['L'] = len(sat.sendBufferSatsInter[1][1])
    except (IndexError, AttributeError):
        infQueue = True

    if not DDQN:
        return max(queuesLen) > threshold or infQueue
    else:
        return queuesDic
    
def hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
    '''
    This function will return true if the satellites distance between them > trheshold or if their throughpuyt < trheshold
    They are far away or the link is weak
    '''
    slantRange     = g.edges[satA.ID, satB.ID]['slant_range']
    throughputSats = g.edges[satA.ID, satB.ID]['dataRateOG']

    return (slantRange > thresholdSL or throughputSats < thresholdTHR)



def getSatScore(satA, satB, g):
    '''
    This function will compute the score of sending the package from satA to satB
    0: (Low  slant range || high throughput) && low queue
    1:  High slant range && low  throughput  && low queue
    2:  High queue

    Queue threshold:
    As high queue threshold we have set 125 packets, which is the 92 percentile of all the queues when we have 13 GTs
    (The moment when we start having congestion with slant range policy). The waiting time of a queue with 125 blocks
    is 9 msg (Each packet in the queue lasts ~0.072ms)
    '''
    thresholdQueue = 125
    thresholdTHR, thresholdSL = computeOutliers(g)

    if satB is None or getQueues(satB, thresholdQueue):
        return 2
    elif hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
        return 1
    else:
        return 0

def getState(Block, satA, g, earth):
    '''
    Given a dataBlock and the current satellite this function will return a list with the 
    values of the 5 fields of the state space.
    Destination: linked satellite to the destination gateway index.

    we initialize the score of the satellites in 2 (worst case) because we do not know if they 
    will actually have a linked satellite in that direction.
    If they have it the satellite score will replace the initialization score (2) but if they dont 
    have it, as we need a score in order to set the state space we will give the worst score and
    send a None in the destinations dict. That action will be initialized with -infinite in the QTable
    '''
    destination  = getDestination(Block, g)
    state        = [2, 2, 2, 2, destination]   

    state[0] = getSatScore(satA, satA.QLearning.linkedSats['U'], g)
    state[1] = getSatScore(satA, satA.QLearning.linkedSats['D'], g)
    state[2] = getSatScore(satA, satA.QLearning.linkedSats['R'], g)
    state[3] = getSatScore(satA, satA.QLearning.linkedSats['L'], g)

    return state

def getDestination(Block, g, sat = None):
    '''
    Returns:
    blockDestination: Position of the satellite linked to the block destination Gateway among a list of all the
                      satellites linked to Gateways
    linkedGateway:    If the satellite provided is linked to a gateway, it will return the position of the satellite in
                      the mentioned list. Otherwise it will return -1.
    '''
    destination = list(g.edges(Block.destination.name))[0][1]    # ID of the Satellite linked to the block destination GT
    blockDestination = (linkedSatsList(g)[1] == destination).argmax()

    if sat is None:
        return blockDestination
    else:
        pass
        # satDest = Block.destination.linkedSat[1]
        # return getGridPosition(GridSize, [tuple([math.degrees(satDest.latitude), math.degrees(satDest.longitude), satDest.ID])], False, False)[0]

def linkedSatsList(g):
    '''
    This funtion retunrs a dictionary (Gateway: linekdSatellite)
    '''
    linkedSats = []
    for node in g.nodes:
        if not node[0].isdigit():
            linkedSats.append(list(g.edges(node))[0])
    return pd.DataFrame(linkedSats)


def getSlantRange(satA, satB):
    '''
    given 2 satellites, it will return the slant range between them (With the method used at 'get_slant_range_optimized')
    '''
    return np.linalg.norm(np.array((satA.x, satA.y, satA.z)) - np.array((satB.x, satB.y, satB.z)))  # posA - posB


def getDistanceReward(satA, satB, destination, w2):
    '''
    This function will return the instant reward regarding to the slant range reduction from actual node to destination
    just after the agent takes an action (destination is the satellite linked to the destination Gateway)

    TSLa: Total slant range from sat A to destination
    TSLb: Total slant range from sat B to destination
    SLR : Slant Range reduction after taking the action (Going from satA to satB)

    Formula: w*(SLR + TSLa)/TSLa = w*(TSLa - TSLb + TSLa)/TSLa = w*(2*TSLa - TSLb)/TSLa
    '''
    balance   = -1      # centralizes the result in 0

    TSLa = getSlantRange(satA, destination)
    TSLb = getSlantRange(satB, destination)
    return w2*((2*TSLa-TSLb)/TSLa + balance)

def getQueueReward(queueTime, w1):
    '''
    Given the queue time in seconds, this function will return the queue reward.
    With 125 packets, 9ms Queue (The thershold that we take to consider a queue as high) the reward will be -0.04 (with w1 = 2)
    '''
    return w1*(1-10**queueTime)

def getBiasedLatitude(sat):
    try:
        return (int(math.degrees(sat.latitude))+latBias)/coordGran
    except AttributeError as e:
        # print(f"getBiasedLatitude Caught an exception: {e}")
        return notAvail


def getBiasedLongitude(sat):
    try:
        return (int(math.degrees(sat.longitude))+lonBias)/coordGran
    except AttributeError as e:
        # print(f"getBiasedLongitude Caught an exception: {e}")
        return notAvail

def getDeepSatScore(queueLength):
    # return 1 if queueLength > infQueue else (int(np.floor(queueVals*np.log10(queueLength + 1)/np.log10(infQueue))))/queueVals
    return queueVals if queueLength > infQueue else int(np.floor(queueVals*np.log10(queueLength + 1)/np.log10(infQueue)))
