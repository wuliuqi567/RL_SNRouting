from Utils.utilsfunction import *
from dgl import from_networkx
import torch
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
def getDeepLinkedSats(satA, g=None, earth=None):
    '''
    Given a satellite, this function will return a dictionary with the linked satellite
    at each direction based on the new definition of upper and lower satellites.
    Satellite at the right and left are determined based on inter-plane links.
    '''
    # IMPORTANT: If a graph is provided, we must derive neighbors from the graph edges.
    # Otherwise, the agent may select a "neighbor" (e.g., satA.upper) that is not actually
    # connected in the current topology, which later causes failures when enqueuing to send buffers.
    if g is not None and earth is not None:
        return getLinkedSats(satA, g, earth)

    linkedSats = {'U': None, 'D': None, 'R': None, 'L': None}

    # Use the provided logic to find intra-plane neighbours (upper and lower)
    # satA.findIntraNeighbours(earth)
    linkedSats['U'] = getattr(satA, 'upper', None)
    linkedSats['D'] = getattr(satA, 'lower', None)
    linkedSats['R'] = getattr(satA, 'right', None)
    linkedSats['L'] = getattr(satA, 'left', None)

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

def getLinkDataRate(sat):
    # Returns the Shannon Rate of the link in the specified direction
    linkedSats = getDeepLinkedSats(sat, None, None)
    
    ISLDataRate = {"U": None, "D": None, "R": None, "L": None}
    ISLDistance = {"U": None, "D": None, "R": None, "L": None}
    # sat.intraSats is a list: [(distance, sat2, dataRate), ...]
    for edge in sat.intraSats:
        distance = edge[0]
        neighbor_sat = edge[1]
        dataRate = edge[2]
        for direction, linked_sat in linkedSats.items():
            if neighbor_sat.ID == linked_sat.ID:
                ISLDataRate[direction] = dataRate
                ISLDistance[direction] = distance

    for edge in sat.interSats:
        distance = edge[0]
        neighbor_sat = edge[1]
        dataRate = edge[2]
        for direction, linked_sat in linkedSats.items():
            if neighbor_sat.ID == linked_sat.ID:
                ISLDataRate[direction] = dataRate
                ISLDistance[direction] = distance
    
    return ISLDataRate, ISLDistance
            

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

def get_sat_info_v2(cur_sat, cen_sat, dest_sat):
    """
    Obtain information about the satellite's queue lengths and positions.
    """
    pos_states = getPositionState(cur_sat, cen_sat, dest_sat)
    queues_states = getQueuesStates(cur_sat)
    return np.concatenate((pos_states, queues_states), axis=1)

def getPositionState(cur_sat, cen_sat, dest_sat):
    """
    Obtain absolute position state for the given satellite.
    """
    if cur_sat.latitude is None or cur_sat.longitude is None:
        return np.array([[notAvail, notAvail, notAvail, notAvail, notAvail, notAvail]])
    # 1. 绝对坐标
    abs_pos = np.array([
        get_absolute_position(cur_sat.latitude, latBias, coordGran),
        get_absolute_position(cur_sat.longitude, lonBias, coordGran),
    ]).reshape(1, -1)

    # 2. 当前卫星相对于中心节点的位移
    cur_pos = np.array([
        get_relative_position(cen_sat, cur_sat.latitude, is_lat=True),
        get_relative_position(cen_sat, cur_sat.longitude, is_lat=False)
    ]).reshape(1, -1)

    # 3. 相对于目的地面站连接的卫星的位移
    dest_pos = np.array([
        get_relative_position(dest_sat, cur_sat.latitude, is_lat=True),
        get_relative_position(dest_sat, cur_sat.longitude, is_lat=False)
    ]).reshape(1, -1)
    return np.concatenate((abs_pos, cur_pos, dest_pos), axis=1)

def getQueuesStates(sat):
    """
    Obtain queue states for the given satellite.
    """
    queues = getQueues(sat, DDQN=True)
    state = [
        getDeepSatScore(queues['U']),
        getDeepSatScore(queues['D']),
        getDeepSatScore(queues['R']),
        getDeepSatScore(queues['L']),
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

def obtain_1rd_neighbor_info(block, sat, g, earth):
    """
    Obtain 1rd order neighbor information for the given satellite links.
    以当前节点为中心的一阶邻居信息，
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
    first_order_neighbors = set()
    linked_sats = getLinkedSats(sat, g, earth)
    for neighbor in linked_sats.values():
        if neighbor is not None:
            first_order_neighbors.add(neighbor)
    adjacent_info = np.concatenate([get_sat_info(neighbor, sat) for neighbor in first_order_neighbors], axis=1)
    zero_padding = np.zeros((1, 120))  # 修正为2D数组以匹配concatenate
    # total 1 + 24 + 2 + 2 + 120 = 149 dimension, 第一阶的邻居信息共 4*6 =24维，补充120维零填充
    return np.concatenate((last_satellite_info, cur_pos.reshape(1, -1), dest_pos.reshape(1, -1), adjacent_info, zero_padding), axis=1)

# 基于当前卫星在图中的位置，从nx图中获取子图satsub，然后构建一个dgl图dglsatsub，
# 
# 然后获取每个节点的队列位置属性添加到图dglsatsub中，最后再获取节点和边的状态。

def get_subgraph_state(block, sat, g, earth, n_order=4):
    """
    Generates the subgraph state for the current satellite using an n-order ego graph.

    This function constructs a local subgraph centered at the current satellite, converts it
    to a DGL graph, and populates it with node and edge features. It handles the removal
    of ground station nodes (source and destination) from the graph to focus on satellite
    topology.

    Args:
        block: The data block being routed, containing source and destination information.
        sat: The current satellite object where the decision is being made.
        g (networkx.Graph): The global network topology graph.
        earth: The simulation environment object containing all network entities.

    Returns:
        dgl.DGLGraph: A DGL graph representing the local subgraph.
            - ndata['feat']: Node features tensor of shape (N, feature_dim).
            - edata['weight']: Edge features tensor of shape (E, 2), containing normalized
              slant range and data rate.
    """
    # # 获取n阶子图
    # n_order = 4
    
    # 手动实现 ego_graph 的逻辑以避免 deepcopy
    
    # 1. 获取无向图视图，避免 deepcopy
    if g.is_directed():
        g_undirected = g.to_undirected(as_view=True)
    else:
        g_undirected = g
        
    # 2. 找到 n 阶邻居节点
    nodes = list(nx.single_source_shortest_path_length(g_undirected, sat.ID, cutoff=n_order).keys())
    
    # 3. 创建子图的浅拷贝 (nx.Graph(subgraph) 默认是浅拷贝属性)
    n_order_graph = nx.Graph(g_undirected.subgraph(nodes))

    satDest = block.destination.linkedSat[1]

    # Remove all gateways from the subgraph to ensure only satellites remain
    for GT in earth.gateways:
        if GT.name in n_order_graph.nodes():
            n_order_graph.remove_node(GT.name)

    # --- FIX: 强制节点映射顺序一致性 ---
    # 将节点标签转换为整数，并按排序顺序排列，确保 DGL 和 NetworkX 使用相同的 ID 映射
    # label_attribute='orig_id' 保留原始节点 ID (如 '17_20') 以备查
    n_order_graph_int = nx.convert_node_labels_to_integers(n_order_graph, ordering='sorted', label_attribute='orig_id')
    
    # 获取排序后的原始节点名称列表，用于查找 center_node_id
    sorted_node_names = sorted(list(n_order_graph.nodes()))
    
    # 找到中心节点在排序列表中的索引，这就是它在 n_order_graph_int 中的新整数 ID
    center_node_id = sorted_node_names.index(sat.ID)
    
    dgl_g = from_networkx(n_order_graph_int) 
    
    # 获取一阶邻居在DGL图中的ID
    # 在DGL图中，邻居可以通过 successors 或 predecessors 获取（无向图两者相同）
    first_order_neighbor_ids = dgl_g.successors(center_node_id).tolist()
    
    # 将中心节点ID和一阶邻居ID存储在图的属性中，方便后续使用
    dgl_g.ndata['is_center'] = torch.zeros(dgl_g.num_nodes(), dtype=torch.bool)
    dgl_g.ndata['is_center'][center_node_id] = True
    
    dgl_g.ndata['is_first_order'] = torch.zeros(dgl_g.num_nodes(), dtype=torch.bool)
    dgl_g.ndata['is_first_order'][first_order_neighbor_ids] = True

    src, dst = dgl_g.edges()
    
    weights = []

    for src_idx, dst_idx in zip(src, dst):
        # 直接使用整数索引访问转换后的图，避免映射错误
        u_int = src_idx.item()
        v_int = dst_idx.item()
        
        try:
            slant_range = n_order_graph_int.edges[u_int, v_int]['slant_range']
            dataRate = n_order_graph_int.edges[u_int, v_int]['dataRateOG']
            weights.append([slant_range, dataRate])
        except KeyError as e:
            # 获取原始 ID 用于报错信息
            u_orig = n_order_graph_int.nodes[u_int]['orig_id']
            v_orig = n_order_graph_int.nodes[v_int]['orig_id']
            print(f"KeyError accessing edge: {u_orig} ({u_int}) -> {v_orig} ({v_int})")
            raise e
        
    # 将weights特征对应归一化
    weights = np.array(weights)
    if weights.size > 0:
        slant_ranges = weights[:, 0]
        data_rates = weights[:, 1]

        # 归一化处理
        if slant_ranges.max() != slant_ranges.min():
            slant_ranges_norm = (slant_ranges - slant_ranges.min()) / (slant_ranges.max() - slant_ranges.min())
        else:
            slant_ranges_norm = np.zeros_like(slant_ranges)  # 如果所有值相同，归一化为0

        if data_rates.max() != data_rates.min():
            data_rates_norm = (data_rates - data_rates.min()) / (data_rates.max() - data_rates.min())
        else:
            data_rates_norm = np.zeros_like(data_rates)  # 如果所有值相同，归一化为0

        # 更新边特征
        dgl_g.edata['weight'] = torch.tensor(np.column_stack((slant_ranges_norm, data_rates_norm)), dtype=torch.float32)
    else:
        dgl_g.edata['weight'] = torch.zeros((0, 2), dtype=torch.float32)
    
    # 收集所有节点的特征
    features = []
    # DGL节点ID是连续的整数 0, 1, ... N-1
    for nodeId in range(dgl_g.num_nodes()):
        # origin_nodeid = node_list[nodeId] # 映射回原始ID
        # 使用 n_order_graph_int 中的 orig_id 属性获取原始 ID
        origin_nodeid = n_order_graph_int.nodes[nodeId]['orig_id']
        sat_node = findByID(earth, origin_nodeid)
        node_data = get_sat_info_v2(sat_node, sat, satDest)
        features.append(node_data)

    # 一次性赋值特征
    if features:
        # np.vstack 将列表堆叠成 (N, D)
        dgl_g.ndata['feat'] = torch.tensor(np.vstack(features), dtype=torch.float32)
        dgl_g.ndata['1st_order_feat'] = dgl_g.ndata['feat'] * (dgl_g.ndata['is_center'] | dgl_g.ndata['is_first_order']).unsqueeze(1).float()

    return dgl_g

# def get_1st_order_neighbors(dgl_graph):
#     """
#     Get the 1st order neighbors of the center satellite in the DGL graph.
#     # 将dgl_graph图中除了is_center和is_first_order节点特征外的其他特征设置为0
#     """
#     first_order_g = dgl_graph.local_var()
#     keep_mask = first_order_g.ndata['is_center'] | first_order_g.ndata['is_first_order']
#     # 扩展 mask 维度以匹配 feat (N, D)
#     mask_expanded = keep_mask.unsqueeze(1).float()
    
#     first_order_g.ndata['pdfeat'] = first_order_g.ndata['feat'] * mask_expanded

#     return first_order_g