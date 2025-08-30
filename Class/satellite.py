import math
import os
import numpy as np
import simpy
import networkx as nx
from configure import *
from globalvar import *
from Class.auxiliaryClass import *

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


class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, orbitalPlane, quota = 500, power = 10):
        self.ID = ID                    # A unique ID given to every satellite
        self.orbPlane = orbitalPlane    # Pointer to the orbital plane which the sat belongs to
        self.in_plane = in_plane        # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane    # Index in orbital plane
        self.quota = quota              # Quota of the satellite
        self.h = h                      # Altitude of deployment
        self.power = power              # Transmission power
        self.minElevationAngle = minElAngle # Value is taken from NGSO constellation design chapter

        # Spherical Coordinates before inclination (r,theta,phi)
        self.r = Re+self.h
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude

        # Inclination of the orbital plane
        self.inclination = inclination

        # Cartesian coordinates  (x,y,z)
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)

        self.polar_angle = self.theta               # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z/self.r)   # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat

        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0

        # simpy
        self.env = env
        self.sendBufferGT = ([env.event()], [])  # ([self.env.event()], [DataBlock(0, 0, "0", 0)])
        self.sendBlocksGT = []  # env.process(self.sendBlock())  # simpy processes which send the data blocks
        self.sats = []
        self.linkedGT = None
        self.GTDist = None
        # list of data blocks waiting on their propagation delay.
        self.tempBlocks = []  # This list is used to so the block can have their paths changed when the constellation is moved

        self.intraSats = []
        self.interSats = []
        self.sendBufferSatsIntra = []
        self.sendBufferSatsInter = []
        self.sendBlocksSatsIntra = []
        self.sendBlocksSatsInter = []
        self.newBuffer  = [False]

        self.QLearning  = None  # Q-learning table that will be updated in case the pathing is 'Q-Learning'
        self.DDQNA      = None  # DDQN agent for each satellite. Only used in the online phase
        self.maxSlantRange = self.GetmaxSlantRange()

    def GetmaxSlantRange(self):
        """
        Maximum distance from satellite to edge of coverage area is calculated using the following formula:
        D_max(minElevationAngle, h) = sqrt(Re**2*sin**2(minElevationAngle) + 2*Re*h + h**2) - Re*sin(minElevationAngle)
        This formula is based on the NGSO constellation design chapter page 16.
        """
        eps = math.radians(self.minElevationAngle)

        distance = math.sqrt((Re+self.h)**2-(Re*math.cos(eps))**2) - Re*math.sin(eps)

        return distance

    def __repr__(self):
        return '\nID = {}\n orbital plane= {}, index in plane= {}, h={}\n pos r = {}, pos theta = {},' \
               ' pos phi = {},\n pos x= {}, pos y= {}, pos z= {}\n inclination = {}\n polar angle = {}' \
               '\n latitude = {}\n longitude = {}'.format(
                self.ID,
                self.in_plane,
                self.i_in_plane,
                '%.2f' % self.h,
                '%.2f' % self.r,
                '%.2f' % self.theta,
                '%.2f' % self.phi,
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % math.degrees(self.inclination),
                '%.2f' % math.degrees(self.polar_angle),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % math.degrees(self.longitude))

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        """
        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay and adding the block to the send-buffer afterwards. Since there are multiple buffers,
        this function looks at the next step in the blocks path and adds the block to the correct send-buffer.
        When Q-Learning or Deep learning is used, this function is where the next step in the block's path is found.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.

        Using this structure, if there are to be implemented limits on the sizes of the "receive-buffer" it could be
        handled by either limiting the amount of these processes that can occur at the same time, or limiting the size
        of the send-buffer.
        """
        # wait for block to fully propagate
        self.tempBlocks.append(block)

        # print(f'{self.env.now}: {self.ID} received block {block.ID} with path {block.path} and propTime {propTime}')

        yield self.env.timeout(propTime)

        if block.path == -1:
            return

        # KPI: propLatency receive block from sat
        block.propLatency += propTime

        for i, tempBlock in enumerate(self.tempBlocks):
            if block.ID == tempBlock.ID:
                self.tempBlocks.pop(i)
                break

        try: # ANCHOR Save Queue time csv
            block.queueTime.append((block.checkPointsSend[len(block.checkPointsSend)-1]- block.checkPoints[len(block.checkPoints)-1]))
        except IndexError:  # Either it is the first satellite for the datablock or the datablock has no checkpoints appendeds
            # print('Index error')
            pass

        block.checkPoints.append(self.env.now)

        # if QLearning or Deep Q-Learning we:
        # Compute the next hop in the path and add it to the second last position (Last is the destination gateway)
        # we let the (Deep) Q-model choose the next hop and it will be added to the block.QPath as mentioned
        # if the next hop is the linked gateway it will simply not add anything and will let the model work normally
        if ((self.QLearning) or (self.orbPlane.earth.DDQNA is not None) or (self.DDQNA is not None)):
            if len(block.QPath) > 3: # the block does not come from a gateway
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth, prevSat = (findByID(self.orbPlane.earth, block.QPath[len(block.QPath)-3][0])))
            else:
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph, self.orbPlane.earth)

            if nextHop != 0:
                block.QPath.insert(len(block.QPath)-1 ,nextHop)
                pathPlot = block.QPath.copy()
                pathPlot.pop()
            else:
                pathPlot = block.QPath.copy()
            
            # If plotPath plots an image for every action taken. Plots 1/10 of blocks. # ANCHOR plot action satellite
            #################################################################
            if self.orbPlane.earth.plotPaths:
                if int(block.ID[len(block.ID)-1]) == 0:
                    os.makedirs(self.orbPlane.earth.outputPath + '/pictures/', exist_ok=True) # create output path
                    outputPath = self.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    # plotShortestPath(self.orbPlane.earth, pathPlot, outputPath)
                    plotShortestPath(self.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time = block.creationTime)
            #################################################################

            path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
        else:
            path = block.path   # if there is no Q-Learning we will work with the path as normally

        # get this satellites index in the blocks path
        index = None
        for i, step in enumerate(path):
            if self.ID == step[0]:
                index = i

        if not index:
            print(path)

        # check if next step in path is GT (last step in path)
        if index == len(path) - 2:
            # add block to GT send-buffer
            if not self.sendBufferGT[0][0].triggered:
                self.sendBufferGT[0][0].succeed()
                self.sendBufferGT[1].append(block)
            else:
                newEvent = self.env.event().succeed()
                self.sendBufferGT[0].append(newEvent)
                self.sendBufferGT[1].append(block)

        else:
            ID = None
            isIntra = False
            # get ID of next sat
            for sat in self.intraSats:
                id = sat[1].ID
                if id == path[index + 1][0]:
                    ID = sat[1].ID
                    isIntra = True
            for sat in self.interSats:
                id = sat[1].ID
                if id == path[index + 1][0]:
                    ID = sat[1].ID

            if ID is not None:
                sendBuffer = None
                # find send-buffer for the satellite
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                # ANCHOR save the queue length that the block found at its next hop
                self.orbPlane.earth.queues.append(len(sendBuffer[1]))
                block.queue.append(len(sendBuffer[1]))

                # add block to buffer
                if not sendBuffer[0][0].triggered:
                    sendBuffer[0][0].succeed()
                    sendBuffer[1].append(block)
                else:
                    newEvent = self.env.event().succeed()
                    sendBuffer[0].append(newEvent)
                    sendBuffer[1].append(block)

            else:
                print(
                    "ERROR! Sat {} tried to send block to {} but did not have it in its linked satellite list".format(
                        self.ID, path[index + 1][0]))

    def sendBlock(self, destination, isSat, isIntra = None):
        """
        Simpy process function:

        Sends data blocks that are filled and added to one of the send-buffers, a buffer which consists of a list of
        events and data blocks. Since there are multiple send-buffers, the function finds the correct buffer given
        information regarding the desired destination satellite or GT. The function monitors the send-buffer, and when
        the buffer contains one or more triggered events, the function will calculate the time it will take to send the
        block and trigger an event which notifies a separate process that a block has been sent.

        A process is running this method for each ISL and for the downLink GSL the satellite has. This will usually be
        4 ISL processes and 1 GSL process.
        """

        if isIntra is not None:
            sendBuffer = None
            if isSat:
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
        else:
            sendBuffer = self.sendBufferGT

        while True:
            try:
                yield sendBuffer[0][0]

                # print(f'Satellite Sending block {sendBuffer[1][0].ID} from satellite {self.ID} to {destination[1].ID} at time {self.env.now}')

                # ANCHOR KPI: queueLatency at sat
                sendBuffer[1][0].checkPointsSend.append(self.env.now)

                if isSat:
                    timeToSend = sendBuffer[1][0].size / destination[2]

                    propTime = self.timeToSend(destination)
                    yield self.env.timeout(timeToSend)

                    receiver = destination[1]

                else:
                    propTime = self.timeToSend(self.linkedGT.linkedSat)
                    timeToSend = sendBuffer[1][0].size / self.downRate
                    yield self.env.timeout(timeToSend)

                    receiver = self.linkedGT

                # When the constellations move, the only case where this process can simply continue, is when the
                # receiver is the same, and there is a block already ready to be sent. The only place where the process
                # can continue from, is as a result right here. Furthermore, the only processes this can happen for are
                # the inter-ISL processes.
                # Due to having to remake buffers when the satellites move, it is necessary for the process to "find"
                # the correct buffer again - the process uses a reference to the buffer: "sendBuffer".
                # To avoid remaking the reference every time a block is sent, the list of boolean values: self.newBuffer
                # is used to indicate when the constellation is moved,

                if True in self.newBuffer and not isIntra and isSat: # remake reference to buffer
                    if isIntra is not None:
                        sendBuffer = None
                        if isSat:
                            if isIntra:
                                for buffer in self.sendBufferSatsIntra:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                            else:
                                for buffer in self.sendBufferSatsInter:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                    else:
                        sendBuffer = self.sendBufferGT

                    for index, val in enumerate(self.newBuffer):
                        if val: # each process will one by one remake their reference, and change one value to True.
                                # After all processes has done this, all values are back to False
                            self.newBuffer[index] = False
                            break

                # ANCHOR KPI: txLatency ISL
                sendBuffer[1][0].txLatency += timeToSend
                receiver.createReceiveBlockProcess(sendBuffer[1][0], propTime)

                # remove from own buffer
                if len(sendBuffer[0]) == 1:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
                    sendBuffer[0].append(self.env.event())

                else:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
            except simpy.Interrupt:
                # print(f'Simpy interrupt at sending block at satellite {self.ID} to {destination[1].ID}') # FIXME Are they really lost blocks?
                # self.orbPlane.earth.lostBlocks+=1
                break

    def adjustDownRate(self):

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
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedGT.linkedSat[0]*self.ngeo2gt.f/Vc)**2)
        snr = 10**((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No)/10)
        shannonRate = self.ngeo2gt.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]

        self.downRate = speff

    def timeToSend(self, linkedSat):
        """
        Calculates the propagation time of a block going from satellite to satellite.
        """
        distance = linkedSat[0]
        pTime = distance/Vc
        return pTime

    def findIntraNeighbours(self, earth):
        '''
        Finds intra-plane neighbours
        '''
        self.linked = None                                                      # Closest sat linked
        self.upper  = earth.LEO[self.in_plane].sats[self.i_in_plane-1]          # Previous sat in the same plane
        if self.i_in_plane < self.n_sat-1:
            self.lower = earth.LEO[self.in_plane].sats[self.i_in_plane+1]       # Following sat in the same plane
        else:
            self.lower = earth.LEO[self.in_plane].sats[0]                       # last satellite of the plane

    def findInterNeighbours(self, earth):
        '''
        Sets the inter plane neighbors for each satellite that will be used for DRL
        '''
        g = earth.graph
        self.right = None
        self.left  = None
        # Find inter-plane neighbours (right and left)
        for edge in list(g.edges(self.ID)):
            if edge[1][0].isdigit():
                satB = findByID(earth, edge[1])
                dir = getDirection(self, satB)
                if(dir == 3):                                         # Found Satellite at East
                    # if self.right is not None:
                    #     print(f"{self.ID} east satellite duplicated! Replacing {self.right.ID} with {satB.ID}.")
                    self.right  = satB

                elif(dir == 4):                                       # Found Satellite at West
                    # if self.left is not None:
                    #     print(f"{self.ID} west satellite duplicated! Replacing {self.left.ID} with {satB.ID}.")
                    self.left  = satB
                elif(dir==1 or dir==2):
                    pass
                else:
                    print(f'Sat: {satB.ID} direction not found with respect to {self.ID}')
            else:   # it is a GT
                pass
        
    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
        # Updating spherical coordinates upon rotation (these are phi, theta before inclination)
        self.phi = longitude
        self.theta = self.theta + 2*math.pi*delta_t/period
        self.theta = self.theta % (2*math.pi)

        # Calculating x,y,z coordinates with inclination
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        self.polar_angle = self.theta  # Angle within orbital plane [radians]
        # updating latitude and longitude after rotation [degrees]
        self.latitude = math.asin(self.z/self.r)  # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0