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



def getDeepStateDiffLastHop(block, sat, linkedSats):
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
    