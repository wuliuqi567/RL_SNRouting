from Utils.utilsfunction import *


# for QLearningAgent
def getLinkedSats(satA, g, earth):
    '''
    Given a satellite the function will return a list with the linked satellite at each direction.
    If that direction has no linked satellite, it will be None
    At the graph each edge is a satA, satB pair with properties like dirij or dirji, i will always
    be the satellite of the lowest plane and 1 will be righ direction (East).

    SAT UP:      northest linked satellite
    SAT DOWN:    southest linked satellite
    SAT LEFT:    linked satellite with lower  plane ID
    SAT RIGHT:   linked satellite with higher plane ID
    '''
    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if(dir == 1 and linkedSats['U'] is None):               # Found a satellite at north
                linkedSats['U']  = satB
            elif(dir == 1):                                         # Found second North, this sat is on South Pole
                if satB.latitude > linkedSats['U'].latitude:
                    # the satellite seen is more at north than Up one, so is set as new Up
                    linkedSats['D'] = linkedSats['U']
                    linkedSats['U'] = satB
                else:
                    # the satellite seen is less at north than Up one, so is set as Down
                    linkedSats['D'] = satB

            elif(dir == 2 and linkedSats['D'] is None):             # Found satellite at South
                linkedSats['D']  = satB   
            elif(dir == 2):                                         # Found second Down, this sat is on North Pole
                if satB.latitude < linkedSats['D'].latitude:        
                    linkedSats['U'] = linkedSats['D']
                    linkedSats['D'] = satB
                else:
                    linkedSats['U'] = satB

            elif(dir == 3):                                         # Found Satellite at East
                # if linkedSats['R'] is not None:
                #     print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
                linkedSats['R']  = satB

            elif(dir == 4):                                         # Found Satellite at West
                # if linkedSats['L'] is not None:
                #     print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
                linkedSats['L']  = satB

        else:
            pass
    return linkedSats

def get_2nd_order_neighbors(sat, g, earth):
    """
    Get the 2nd order neighbors of a given satellite.
    二阶邻居：通过一个中介节点可达的节点（不包括一阶邻居和自身）
    理论上应该有8个节点（在完整mesh网络中）
    """
    first_order_neighbors = set()
    second_order_neighbors = set()
    
    # 获取一阶邻居
    linked_sats = getLinkedSats(sat, g, earth)
    for direction, neighbor in linked_sats.items():
        if neighbor is not None:
            first_order_neighbors.add(neighbor)
            
            # 获取每个一阶邻居的邻居作为二阶邻居候选
            neighbor_linked_sats = getLinkedSats(neighbor, g, earth)
            for _, second_neighbor in neighbor_linked_sats.items():
                if (second_neighbor is not None and 
                    second_neighbor != sat and  # 排除自身
                    second_neighbor not in first_order_neighbors):  # 排除一阶邻居
                    second_order_neighbors.add(second_neighbor)
        else:
            print(f"Warning: Satellite {sat.ID} missing {direction} neighbor")

    result = list(second_order_neighbors)
    expected_count = 8  # 在完整mesh网络中的期望数量
    
    if len(result) != expected_count:
        print(f"Warning: Satellite {sat.ID} has {len(result)} second-order neighbors instead of {expected_count}.")
        print(f"  First-order neighbors: {len(first_order_neighbors)}")
        print(f"  This may be normal for edge satellites or incomplete topology.")
    
    return result

def get_3rd_order_neighbors(sat, g, earth):
    """
    Get the 3rd order neighbors of a given satellite.
    三阶邻居：通过二个中介节点可达的节点（不包括自身、一阶和二阶邻居）
    理论上应该有12个节点（在完整mesh网络中）
    """
    # 获取一阶邻居
    first_order_neighbors = set()
    linked_sats = getLinkedSats(sat, g, earth)
    for neighbor in linked_sats.values():
        if neighbor is not None:
            first_order_neighbors.add(neighbor)
    
    # 获取二阶邻居
    second_order_neighbors = set(get_2nd_order_neighbors(sat, g, earth))
    
    # 获取三阶邻居
    third_order_neighbors = set()
    for second_neighbor in second_order_neighbors:
        # 获取二阶邻居的所有邻居
        second_neighbor_linked_sats = getLinkedSats(second_neighbor, g, earth)
        for _, third_neighbor in second_neighbor_linked_sats.items():
            if (third_neighbor is not None and
                third_neighbor != sat and  # 排除自身
                third_neighbor not in first_order_neighbors and  # 排除一阶邻居
                third_neighbor not in second_order_neighbors):   # 排除二阶邻居
                third_order_neighbors.add(third_neighbor)
            elif third_neighbor is None:
                print(f"Warning: Second-order neighbor {second_neighbor.ID} has None neighbor")

    # result = list(third_order_neighbors)
    expected_count = 12  # 在完整mesh网络中的期望数量
    
    # print(f"Info: Satellite {sat.ID} topology analysis:")
    # print(f"  - 1st order neighbors: {len(first_order_neighbors)}")
    # print(f"  - 2nd order neighbors: {len(second_order_neighbors)}")
    # print(f"  - 3rd order neighbors: {len(result)} (expected: {expected_count})")
    
    if len(third_order_neighbors) != expected_count:
        print(f"  Note: Different count may be normal for edge satellites or sparse topology.")
    
    return list(first_order_neighbors), list(second_order_neighbors), list(third_order_neighbors)


# for DDQNAgent
def getDeepLinkedSats(satA, g, earth):
    '''
    Given a satellite, this function will return a dictionary with the linked satellite
    at each direction based on the new definition of upper and lower satellites.
    Satellite at the right and left are determined based on inter-plane links.
    '''
    linkedSats = {'U':None, 'D':None, 'R':None, 'L':None}

    # Use the provided logic to find intra-plane neighbours (upper and lower)
    # satA.findIntraNeighbours(earth)
    linkedSats['U'] = satA.upper
    linkedSats['D'] = satA.lower
    linkedSats['R'] = satA.right
    linkedSats['L'] = satA.left

    # # Find inter-plane neighbours (right and left)
    # for edge in list(g.edges(satA.ID)):
    #     if edge[1][0].isdigit():
    #         satB = findByID(earth, edge[1])
    #         dir = getDirection(satA, satB)
    #         if(dir == 3):                                         # Found Satellite at East
    #             if linkedSats['R'] is not None:
    #                 print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
    #             linkedSats['R']  = satB

    #         elif(dir == 4):                                       # Found Satellite at West
    #             if linkedSats['L'] is not None:
    #                 print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
    #             linkedSats['L']  = satB
    #     else:
    #         pass

    return linkedSats

def getDeepStateReduced(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None
    return np.array([getBiasedLatitude(linkedSats['U']),                        # Up link Positions
                    getBiasedLongitude(linkedSats['U']),
                    getBiasedLatitude(linkedSats['D']),                         # Down link Positions
                    getBiasedLongitude(linkedSats['D']),
                    getBiasedLatitude(linkedSats['R']),                         # Right link Positions
                    getBiasedLongitude(linkedSats['R']),
                    getBiasedLatitude(linkedSats['L']),                         # Left link Positions
                    getBiasedLongitude(linkedSats['L']),
                    getBiasedLatitude(sat),                                     # Actual Latitude
                    getBiasedLongitude(sat),                                    # Actual Longitude
                    getBiasedLatitude(satDest),                                 # Destination Latitude
                    getBiasedLongitude(satDest)]).reshape(1,-1) 


def normalize_angle_diff(angle_diff):
    # Ensure the angle difference is within [-180, 180]
    return (angle_diff + 180) % 360 - 180

def get_relative_position(neighbor_sat, current_coord, is_lat=True):
    # Convert and calculate relative position, considering the 180-degree discontinuity
    try:
        neighbor_coord = math.degrees(neighbor_sat.latitude if is_lat else neighbor_sat.longitude)
        current_coord = math.degrees(current_coord)
        diff = normalize_angle_diff(neighbor_coord - current_coord)
        return diff / coordGran
    except AttributeError:
        return notAvail
    
def get_absolute_position(coord, bias, gran):
    # Convert absolute position to a normalized value within the specified range
    return (math.degrees(coord) + bias) / gran

def get_last_satellite(block, sat): # REVIEW if index here are the same as decision index
    '''This will return information about the last block hop in relation to the current satellite:
    -1: Constellation moved and the last block's satellite is not connected to current satellite
    0: Upper neighbour
    1: Lower neighbour
    2: Right Neighbour
    3: Left  Neighbour'''
    actIndex = -1
    try:
        if len(block.QPath) > 2:
            if sat.upper and sat.upper.ID == block.QPath[-2][0]:
                actIndex = 0
            elif sat.lower and sat.lower.ID == block.QPath[-2][0]:
                actIndex = 1
            elif sat.right and sat.right.ID == block.QPath[-2][0]:
                actIndex = 2
            elif sat.left and sat.left.ID == block.QPath[-2][0]:
                actIndex = 3
        return actIndex
    except AttributeError as e:
        print(f'An error occurred when checking if {block.QPath[-2][0]} is a neighbour satellite of {sat.ID}')
        return actIndex

def getDeepStateDiffLastHop(block, sat, linkedSats):

    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude

    # Queues
    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)

    state = [
        # Previous satellite information
        get_last_satellite(block, sat),
        # Up link scores and positions
        getDeepSatScore(queuesU['U']),
        getDeepSatScore(queuesU['D']),
        getDeepSatScore(queuesU['R']),
        getDeepSatScore(queuesU['L']),
        get_relative_position(linkedSats['U'], current_lat, is_lat=True),
        get_relative_position(linkedSats['U'], current_lon, is_lat=False),

        # Down link scores and positions
        getDeepSatScore(queuesD['U']),
        getDeepSatScore(queuesD['D']),
        getDeepSatScore(queuesD['R']),
        getDeepSatScore(queuesD['L']),
        get_relative_position(linkedSats['D'], current_lat, is_lat=True),
        get_relative_position(linkedSats['D'], current_lon, is_lat=False),

        # Right link scores and positions
        getDeepSatScore(queuesR['U']),
        getDeepSatScore(queuesR['D']),
        getDeepSatScore(queuesR['R']),
        getDeepSatScore(queuesR['L']),
        get_relative_position(linkedSats['R'], current_lat, is_lat=True),
        get_relative_position(linkedSats['R'], current_lon, is_lat=False),

        # Left link scores and positions
        getDeepSatScore(queuesL['U']),
        getDeepSatScore(queuesL['D']),
        getDeepSatScore(queuesL['R']),
        getDeepSatScore(queuesL['L']),
        get_relative_position(linkedSats['L'], current_lat, is_lat=True),
        get_relative_position(linkedSats['L'], current_lon, is_lat=False),

        # Absolute current satellite's coordinates
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),

        # Destination's differential coordinates
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ]

    return np.array(state).reshape(1, -1)


def getDeepState(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    queuesU = getQueues(linkedSats['U'], DDQN = True)
    queuesD = getQueues(linkedSats['D'], DDQN = True)
    queuesR = getQueues(linkedSats['R'], DDQN = True)
    queuesL = getQueues(linkedSats['L'], DDQN = True)
    return np.array([getDeepSatScore(queuesU['U']),                             # Up link scores
                    getDeepSatScore(queuesU['D']),
                    getDeepSatScore(queuesU['R']),
                    getDeepSatScore(queuesU['L']),
                    getBiasedLatitude(linkedSats['U']),                         # Up link Positions
                    getBiasedLongitude(linkedSats['U']),
                    getDeepSatScore(queuesD['U']),                              # Down link scores
                    getDeepSatScore(queuesD['D']),
                    getDeepSatScore(queuesD['R']),
                    getDeepSatScore(queuesD['L']),
                    getBiasedLatitude(linkedSats['D']),                         # Down link Positions
                    getBiasedLongitude(linkedSats['D']),
                    getDeepSatScore(queuesR['U']),                              # Right link scores
                    getDeepSatScore(queuesR['D']),
                    getDeepSatScore(queuesR['R']),
                    getDeepSatScore(queuesR['L']),
                    getBiasedLatitude(linkedSats['R']),                         # Right link Positions
                    getBiasedLongitude(linkedSats['R']),
                    getDeepSatScore(queuesL['U']),                              # Left link scores
                    getDeepSatScore(queuesL['D']),
                    getDeepSatScore(queuesL['R']),
                    getDeepSatScore(queuesL['L']),
                    getBiasedLatitude(linkedSats['L']),                         # Left link Positions
                    getBiasedLongitude(linkedSats['L']),

                    # int(math.degrees(sat.latitude))+latBias,                    # Actual Latitude
                    # int(math.degrees(sat.longitude))+lonBias,                   # Actual Longitude
                    # int(math.degrees(satDest.latitude))+latBias,                # Destination Latitude
                    # int(math.degrees(satDest.longitude))+lonBias]).reshape(1,-1)# Destination Longitude

                    getBiasedLatitude(sat),                                     # Actual Latitude
                    getBiasedLongitude(sat),                                    # Actual Longitude
                    getBiasedLatitude(satDest),                                 # Destination Latitude
                    getBiasedLongitude(satDest)]).reshape(1,-1)                 # Destination Longitude


def get_sat_info(sat, center_sat):
    """
    Obtain information about the satellite's queue lengths and positions.
    """
    queues = getQueues(sat, DDQN=True)
    state = [
        getDeepSatScore(queues['U']),
        getDeepSatScore(queues['D']),
        getDeepSatScore(queues['R']),
        getDeepSatScore(queues['L']),
        get_relative_position(sat, center_sat.latitude, is_lat=True),
        get_relative_position(sat, center_sat.longitude, is_lat=False),
    ]
    return np.array(state).reshape(1, -1)

def obtain_3rd_order_neighbor_info(block, sat, g, earth):
    """
    Obtain 3rd order neighbor information for the given satellite links.
    以当前节点为中心的三阶邻居信息，
    拼接方式；上一个跳的信息，up link的上下左右队列信息，down link的上下左右队列信息，right link的上下左右队列信息，left link的上下左右队列信息
    """
    
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None
    # 1 dimension
    last_satellite_info = np.array(get_last_satellite(block, sat)).reshape(1, -1)
    
    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude
    # 2 dimension
    cur_pos = np.array([
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),
    ])
    # 3 dimension
    # Destination's differential coordinates
    dest_pos = np.array([
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ])
    # each sat has 6 dimension info, total 12 * 6 + 8 * 6 + 4 * 6 = 144 dimension
    first_order_neighbors, second_order_neighbors, third_order_neighbors = get_3rd_order_neighbors(sat, g, earth)
    adjacent_info = np.concatenate([get_sat_info(neighbor, sat) for neighbor in first_order_neighbors], axis=1)
    # second order neighbors
    adjacent_info = np.concatenate([adjacent_info] + [get_sat_info(neighbor, sat) for neighbor in second_order_neighbors], axis=1)
    # third order neighbors
    adjacent_info = np.concatenate([adjacent_info] + [get_sat_info(neighbor, sat) for neighbor in third_order_neighbors], axis=1)
    # total 1 + 144 + 2 + 2 = 149 dimension, 第一阶的邻居信息共 1+2+2+4*6 =29维，二三阶的邻居信息共 8*6 + 12*6 = 120维
    return np.concatenate((last_satellite_info, cur_pos.reshape(1, -1), dest_pos.reshape(1, -1), adjacent_info), axis=1)
