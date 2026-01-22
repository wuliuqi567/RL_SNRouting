import math
import numpy as np
import networkx as nx
from system_configure import *
import system_configure
from globalvar import *
import geopy.distance
from PIL import Image
import pandas as pd
import time
from Class.auxiliaryClass import *
from Class.gateWay import Gateway
from Utils.utilsfunction import *
from Utils.flfunction import *
from Utils.statefunction import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize

class Earth:
    def __init__(self, env, agent_class, img_path, allGateWayInfo, constellation, deltaT, getRates = False, window=None, outputPath='/'):
        # Input the population count data
        # img_path = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'
        self.outputPath = outputPath
        # self.plotPaths = plotPath
        self.lostBlocks = 0
        self.queues = []
        self.loss   = []
        self.lossAv = []
        self.agent = agent_class()
        self.step   = 0
        self.nMovs  = 0     # number of total movements done by the constellation
        self.epsilon= []    # set of epsilon values
        self.rewards= []    # set of rewards
        self.trains = []    # Set of times when a fit to any dnn has happened
        self.graph  = None
        # self.CKA    = []

        pop_count_data = Image.open(img_path)

        pop_count = np.array(pop_count_data)
        pop_count[pop_count < 0] = 0  # ensure there are no negative values

        # total image sizes
        [self.total_x, self.total_y] = pop_count_data.size

        self.total_cells = self.total_x * self.total_y

        # List of all cells stored in a 2d array as per the order in dataset
        self.cells = []
        for i in range(self.total_x):
            self.cells.append([])
            for j in range(self.total_y):
                self.cells[i].append(Cell(self.total_x, self.total_y, i, j, pop_count[j][i]))

        # window is a list with the coordinate bounds of our window of interest
        # format for window = [western longitude, eastern longitude, southern latitude, northern latitude]
        if window is not None:  # if window provided
            # latitude, longitude bounds:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            # dataset pixel bounds:
            self.windowx = (
            (int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
            self.windowy = (
            (int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))
        else:  # set window size as entire world if no window provided
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        # import gateways from .csv
        self.gateways = []

        select_gateways_loc = allGateWayInfo['Location'].tolist()[:system_configure.CurrentGTnumber]
        totalLocations = allGateWayInfo['Location'].tolist()

        for i in range(len(select_gateways_loc)):
            name = allGateWayInfo['Location'][i]
            gtLati = allGateWayInfo['Latitude'][i]
            gtLongi = allGateWayInfo['Longitude'][i]
            self.gateways.append(Gateway(name, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                        len(allGateWayInfo['Latitude']), env, totalLocations, self))

        # create data Blocks on all GTs.
        if not getRates:
            for gt in self.gateways:
                gt.makeFillBlockProcesses(self.gateways)

        # create constellation of satellites
        self.LEO = create_Constellation(constellation, env, self)

        if rotateFirst:
            print('Rotating constellation...')
            for constellation in self.LEO:
                constellation.rotate(ndeltas*deltaT)

        # Simpy process for handling moving the constellation and the satellites within the constellation
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))

    def linkCells2GTs(self, distance):
        """
        Finds the cells that are within the coverage areas of all GTs and links them ensuring that a cell only links to
        a single GT.
        """
        start = time.time()

        # Find cells that are within range of all GTs
        for i, gt in enumerate(self.gateways):
            print("Finding cells within coverage area of GT {} of {}".format(i+1, len(self.gateways)), end='\r')
            gt.findCellsWithinRange(self, distance)
        print('\r')
        print("Time taken to find cells that are within range of all GTs: {} seconds".format(time.time() - start))

        start = time.time()

        # Add reference for cells to the GT they are closest to
        for cells in self.cells:
            for cell in cells:
                if cell.gateway is not None:
                    cell.gateway[0].addCell([(math.degrees(cell.latitude),
                                                     math.degrees(cell.longitude)),
                                                    cell.users,
                                                    cell.gateway[1]])

        print("Time taken to add cell information to all GTs: {} seconds".format(time.time() - start))
        print()

    def linkSats2GTs(self, method):
        """
        Links GTs to satellites. One satellite is only allowed to link to one GT.
        """
        sats = []
        for orbit in self.LEO:
            for sat in orbit.sats:
                sat.linkedGT = None
                sat.GTDist = None
                sats.append(sat)

        if method == "Greedy":
            for GT in self.gateways:
                GT.orderSatsByDist(self.LEO)
                GT.addRefOnSat()

            for orbit in self.LEO:
                for sat in orbit.sats:
                    if sat.linkedGT is not None:
                        sat.linkedGT.link2Sat(sat.GTDist, sat)
        elif method == "Optimize":
            # make cost matrix
            SxGT = np.array([[99999 for _ in range(len(sats))] for _ in range(len(self.gateways))])
            for i, GT in enumerate(self.gateways):
                GT.orderSatsByDist(self.LEO)
                for val, entry in enumerate(GT.satsOrdered):
                    SxGT[i][entry[2][0]] = val

            # find assignment of GSL which minimizes the cost from the cost matrix
            rowInd, colInd = linear_sum_assignment(SxGT)

            # link satellites and GTs
            for i, GT in enumerate(self.gateways):
                if SxGT[rowInd[i]][colInd[i]] < len(GT.satsOrdered):
                    sat = GT.satsOrdered[SxGT[rowInd[i]][colInd[i]]]
                    GT.link2Sat(sat[0], sat[1])
                else:
                    GT.linkedSat = (None, None)
                    print("no satellite for GT {}".format(GT.name))

    def getCellUsers(self):
        """
        Used for plotting the population map.
        """
        temp = []
        for i, cellList in enumerate(self.cells):
            temp.append([])
            for cell in cellList:
                temp[i].append(cell.users)
        return temp


    def getGSLDataRates(self):
        upDataRates = []
        downDataRates = []
        for GT in self.gateways:
            if GT.linkedSat[0] is not None:
                upDataRates.append(GT.dataRate)

        for orbit in self.LEO:
            for satellite in orbit.sats:
                if satellite.linkedGT is not None:
                    downDataRates.append(satellite.downRate)

        return upDataRates, downDataRates

    def getISLDataRates(self):
        interDataRates = []
        highRates = 0
        for orbit in self.LEO:
            for satellite in orbit.sats:
                for satData in satellite.interSats:
                    if satData[2] > 3e9:
                        highRates += 1
                    interDataRates.append(satData[2])
        return interDataRates

    def moveConstellation(self, env, deltaT=3600, getRates = False):
        """
        Simpy process function:

        Moves the constellations in terms of the Earth's rotation and moves the satellites within the constellations.
        The movement is based on the time that has passed since last constellation movement and is defined by the
        "deltaT" variable.

        After the satellites have been moved a process of re-linking all links, both GSLs and ISLs, is conducted where
        the paths for all blocks are re-made, the blocks are moved (if necessary) to the correct buffers, and all
        processes managing the send-buffers are checked to ensure they will still work correctly.
        """

        # Get the data rate for a intra plane ISL - used for testing
        if getRates:
            intraRate.append(self.LEO[0].sats[0].intraSats[0][2])

        while True:
            print('Creating/Moving constellation: Updating satellites position and links.')
            if getRates:
                # get data rates for all inter plane ISLs and all GSLs (up and down) - used for testing
                upDataRates, downDataRates = self.getGSLDataRates()
                inter = self.getISLDataRates()

                for val in upDataRates:
                    upGSLRates.append(val)

                for val in downDataRates:
                    downGSLRates.append(val)

                for val in inter:
                    interRates.append(val)

            yield env.timeout(deltaT)

            # clear satellite references on all GTs
            for GT in self.gateways:
                GT.satsOrdered = []
                GT.linkedSat = (None, None)

            # rotate constellation and satellites
            for plane in self.LEO:
                plane.rotate(ndeltas*deltaT)

            # relink satellites and GTs
            self.linkSats2GTs("Optimize")

            # create new graph and add references to all GTs for every rotation
            # prevGraph = self.graph
            graph = createGraph(self, matching=matching)
            self.graph = graph
            for GT in self.gateways:
                GT.graph = graph

            if self.agent is not None:
                self.updateSatelliteProcessesRL(graph)
                # pass
            else:
                # self.updateSatelliteProcessesCorrect(graph)
                print('No agent defined, skipping satellite process updates.')

            self.updateGTPaths()
            self.nMovs += 1
            # if saveISLs:
            #     print('Constellation moved! Saving ISLs map...')
            #     islpath = self.outputPath + '/ISL_maps/'
            #     os.makedirs(islpath, exist_ok=True) 
            #     self.plotMap(plotGT = True, plotSat = True, edges=True, save = True, outputPath=islpath, n=self.nMovs)
            #     plt.close()
            print('***********Constellation moved! All satellite processes updated. \n'   )


    def _process_buffer_rl(self, buffer, sat, buf_pos_flag=False):
        """Helper function to process a buffer for RL path updates.
        
        buf_pos_flag: True: 发送到地面站的缓冲区， False: 发送到卫星的缓冲区

        """
        if buf_pos_flag:
            # 如果是发送到地面站的缓冲区，假设数据包直接发往地面站，不进行RL决策
            # 这里我们不需要做任何事情，因为数据包已经在正确的缓冲区中，
            # 且后续逻辑会处理GSL的更新（如果连接的GT变了，updateSatelliteProcessesRL后续代码会处理）
            return
        index = 0
        while index < len(buffer[1]):
            block = buffer[1][index]
            nextHop = None

            def _step_id(step):
                return step[0] if isinstance(step, (list, tuple)) else step

            # Calculate next hop
            if len(block.QPath) > 3:  # the block does not come from a gateway
                if self.agent is not None:
                    # If we couldn't determine prev_id but path is long, we might have an issue
                    # But we'll try to proceed, maybe makeDeepAction can handle None or we can find it differently
                    # prev_obj = findByID(self, prev_id) if prev_id is not None else None
                    
                    # UPDATE: Use current sat as prevSat to model the "wait" step, and remove 'recalculate' to append reward instead of overwrite
                    nextHop = self.agent.makeDeepAction(block, sat, sat.orbPlane.earth.gateways[0].graph, self, sat)
            else:
                # Block is near start of path or short path
                if self.agent is not None:
                    print('len QPath <=3')
                    # UPDATE: Use current sat as prevSat
                    nextHop = self.agent.makeDeepAction(block, sat, sat.orbPlane.earth.gateways[0].graph, self, sat)

            # Handle the result
            if nextHop == -1:
                # exceed max hops, drop the block
                print('Exceed max hops, dropping block', block.ID, 'at satellite', sat.ID)
                # Remove block from buffer and add to dropBlocks
                buffer[1].pop(index)
                buffer[0].pop(index)
                dropBlocks.append(block)
                # Don't increment index since we removed an element
                continue

            elif nextHop == -2:
                # no available action, drop the block
                print("No available action, dropping block", block.ID, "at satellite", sat.ID)
                # Remove block from buffer and add to dropBlocks
                buffer[1].pop(index)
                buffer[0].pop(index)
                dropBlocks.append(block)
                # Don't increment index since we removed an element
                continue

            elif nextHop == 0:
                # arrive at destination
                block.QPath.pop(-2)
                if sat.ID != block.QPath[-2]:
                    print("Error: block should arrive at destination but nextHop is 0", block.QPath)
                
            elif isinstance(nextHop, list): #['18_20', -133.12012370027912, 42.210392220392556] dest.ID, longitude, latitude
                # Keep QPath structure consistent with Class/satellite.py::receiveBlock
                # - If block already had a next-hop chosen at this sat (buffered-to-send), QPath ends with: [..., sat, nextHop, dest]
                #   -> update nextHop by replacing [-2]
                # - If block has not yet chosen next-hop at this sat (shouldn't normally be in a send-buffer, but can happen
                #   during movement/redistribution), QPath ends with: [..., sat, dest]
                #   -> insert nextHop right before dest (same as receiveBlock)
                if len(block.QPath) >= 2 and _step_id(block.QPath[-2]) == sat.ID:
                    block.QPath.insert(len(block.QPath) - 1, nextHop)
                elif len(block.QPath) >= 3 and _step_id(block.QPath[-3]) == sat.ID:
                    block.QPath[-2] = nextHop
                else:
                    # Fallback: locate sat inside QPath and update/insert relative to that position.
                    sat_index = None
                    for i, step in enumerate(block.QPath):
                        if _step_id(step) == sat.ID:
                            sat_index = i
                            break
                    if sat_index is None:
                        print(f"QPath error: sat {sat.ID} not in QPath for block {block.ID}: {block.QPath}")
                    elif sat_index == len(block.QPath) - 2:
                        block.QPath.insert(len(block.QPath) - 1, nextHop)
                    elif sat_index + 1 < len(block.QPath) - 1:
                        block.QPath[sat_index + 1] = nextHop
                    else:
                        print(f"QPath error: unexpected sat index for block {block.ID} on sat {sat.ID}: {block.QPath}")
            else:
                print("Error unknown nextHop type in receiveBlock:", type(nextHop))
                # Should probably remove block or skip

            index += 1

    def updateSatelliteProcessesRL(self, graph):
        """
        Function which ensures all processes on all satellites are updated after constellation movement.
        
        This function takes into account that the paths are not complete and the next step may not have been chosen yet.

        Steps:
            1. Update ISL neighbors for all satellites (both intra-plane and inter-plane).
            2. For all buffers, use RL agent to recalculate next hops for blocks.
            3. Handle inter-plane ISLs:
                - If link still exists: keep first block in buffer, redistribute others
                - If link broken: redistribute all blocks, stop process
            4. Handle intra-plane ISLs:
                - Similar to inter-plane, but links should not change
            5. Handle GSL:
                - If satellite has linked GT: restart send process
                - If GSL link lost: mark blocks in buffer as lost (added to dropBlocks)
            6. Redistribute blocks to correct buffers based on their updated paths.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        
        Note: When GSL link is lost, blocks in GT buffer are marked as lost (conservative approach).
        """
        # update linked sats
        sats = []
        for plane in self.LEO:
            for sat in plane.sats:
                sats.append(sat)
                # update ISL. Intra-plane should not change
                sat.findIntraNeighbours(self)
                sat.findInterNeighbours(self)

        print('Remaking paths for all blocks in constellation...')
        for plane in self.LEO:
            for sat in plane.sats:
                # get next step for all blocks
                # doing this here assumes that the constellation movement will have a limited effect on the links
                # and that the queue sizes will not change significantly.

                # intra satellite buffers
                for buffer in sat.sendBufferSatsIntra:
                    self._process_buffer_rl(buffer, sat)

                # inter satellite buffers
                for buffer in sat.sendBufferSatsInter:
                    self._process_buffer_rl(buffer, sat)
                
                # 假设在移动时会将发送到地面站的数据发完，也就是这些数据不在进行决策，直接发到地面站，但是链路已经变了，怎么处理呢
                self._process_buffer_rl(sat.sendBufferGT, sat, buf_pos_flag=True)
                # print('Remade paths for all blocks in satellite', sat.ID)
                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].QPath):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if index is not None and index + 1 < len(buffer[1][0].QPath) and buffer[1][0].QPath[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = ([sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]], buffer[2])
                                    
                                    # Restart process to update neighbor info (distance, dataRate)
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else: # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        print('Inter-plane ISL broken, redistributing blocks from satellite', sat.ID, 'to satellite', buffer[2])
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].QPath):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if index is not None and index + 1 < len(buffer[1][0].QPath) and buffer[1][0].QPath[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            # Keep only the first element, safely handle length mismatch
                            if len(sat.sendBufferSatsIntra[bufferIndex][1]) > 1:
                                sat.sendBufferSatsIntra[bufferIndex][1][:] = [sat.sendBufferSatsIntra[bufferIndex][1][0]]
                            if len(sat.sendBufferSatsIntra[bufferIndex][0]) > 1:
                                sat.sendBufferSatsIntra[bufferIndex][0][:] = [sat.sendBufferSatsIntra[bufferIndex][0][0]]

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                    # make new send process for new linked GT
                    # We assume that even if the link changed, the blocks in the buffer should be sent to the current linked GT
                    # (User assumption: packets are sent to ground station before switching / directly sent)
                    sat.sendBlocksGT.append(
                        sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []
                    
                    # GSL link lost - blocks in buffer cannot be sent directly
                    # Option 1: Mark as lost (conservative approach)
                    # Option 2: Redistribute to other paths (if possible)
                    if sat.sendBufferGT[1]:
                        for block in sat.sendBufferGT[1]:
                            print(f'---------Block {block.ID} from Sat {sat.ID} lost due to GSL link lost')
                            # Add to lost blocks instead of assuming delivered
                            dropBlocks.append(block)
                            self.lostBlocks += 1
                        
                        # Clear buffer
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                try:
                    blocksToDistribute.sort(key=lambda x: x[0])
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    print(f'Something wrong with: \n{blocksToDistribute}')
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].QPath):
                        # Handle case where step is not a list/tuple (e.g. int ID)
                        if isinstance(step, (list, tuple)):
                            step_id = step[0]
                        else:
                            step_id = step

                        if sat.ID == step_id:
                            index = i
                            break

                    # check if next step in path is GT (last step in path)
                    if index is None:
                        print(f'Satellite {sat.ID} not found in the QPath: {block[1].QPath}') # FIXME This should not happen. Debugging I realized when this happens the previous satellite is twice in last positions of QPath, instead of prevSat and currentSat. The current sat was the linked to the gateways bu after the movement it is not anymore.
                        self.lostBlocks += 1
                        # Add block to dropBlocks for proper tracking
                        dropBlocks.append(block[1])
                    elif index == len(block[1].QPath) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                                        break
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                                        break

                            if sendBuffer is not None:
                                # add block to buffer
                                if not sendBuffer[0][0].triggered:
                                    sendBuffer[0][0].succeed()
                                    sendBuffer[1].append(block[1])
                                else:
                                    newEvent = sat.env.event().succeed()
                                    sendBuffer[0].append(newEvent)
                                    sendBuffer[1].append(block[1])
                            else:
                                print(f"Buffer for next satellite {ID} in path could not be found on sat {sat.ID}")
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateGTPaths(self):
        """
        Updates all paths for all GTs going to all other GTs and ensures that all blocks waiting to be sent has the
        correct path.
        """
        # make new paths for all GTs
        for GT in self.gateways:
            for destination in self.gateways:
                if GT != destination:
                    if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                        path = getShortestPath(GT.name, destination.name, GT.graph)
                        GT.paths.update({destination.name: path})

                    else:
                        GT.paths.update({destination.name: []})
                        print("no path from gateway!!")

            # update paths for all blocks in send-buffer
            for block in GT.sendBuffer[1]:
                block.path = GT.paths[block.destination.name]
                block.isNewPath = True
                block.QPath = [block.path[0], block.path[1], block.path[len(block.path) - 1]]
                # We add a Qpath field for the Q-Learning case. Only source and destination will be added
                # after that, every hop will be added at the second last position.


    def plotMap(self, plotGT = True, plotSat = True, path = None, bottleneck = None, save = False, ID=None, time=None, edges=False, arrow_gap=0.008, outputPath='', paths=None, fileName="map.png", n = None):
        if paths is None:
            plt.figure()
        else:
            plt.figure(figsize=(6, 3))

        legend_properties = {'size': 10, 'weight': 'bold'}
        markerscale = 1.5
        usage_threshold = 10   # In percentage

        # Compute the link usage
        def calculate_link_usage(paths):
            link_usage = {}
            for path in paths:
                for i in range(len(path) - 1):
                    start_node, end_node = path[i], path[i+1]
                    link_str = '{}_{}'.format(start_node[0], end_node[0])

                    # Coordinates for plotting
                    coordinates = [(start_node[1], start_node[2]), (end_node[1], end_node[2])]

                    if link_str in link_usage:
                        link_usage[link_str]['count'] += 1
                    else:
                        link_usage[link_str] = {'count': 1, 'coordinates': coordinates}
            return link_usage

        # Function to adjust arrow start and end points
        def adjust_arrow_points(start, end, gap_value):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist == 0:  # To avoid division by zero
                return start, end
            gap_scaled = gap_value * 1440  # Adjusting arrow_gap to coordinate system
            new_start = (start[0] + gap_scaled * dx / dist, start[1] + gap_scaled * dy / dist)
            new_end = (end[0] - gap_scaled * dx / dist, end[1] - gap_scaled * dy / dist)
            return new_start, new_end

        # Code for plotting edges with arrow gap
        if edges:
            if n is not None:
                fileName = outputPath + f"ISLs_map_{n}.png"
            else:
                fileName = outputPath + "ISLs_map.png"
            for plane in self.LEO:
                for sat in plane.sats:
                    orig_start_x = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    orig_start_y = int((0.5 - math.degrees(sat.latitude) / 180) * 720)

                    for connected_sat in sat.intraSats + sat.interSats:
                        orig_end_x = int((0.5 + math.degrees(connected_sat[1].longitude) / 360) * 1440)
                        orig_end_y = int((0.5 - math.degrees(connected_sat[1].latitude) / 180) * 720)

                        # Adjust arrow start and end points
                        adj_start, adj_end = adjust_arrow_points((orig_start_x, orig_start_y), (orig_end_x, orig_end_y), arrow_gap)

                        plt.arrow(adj_start[0], adj_start[1], adj_end[0] - adj_start[0], adj_end[1] - adj_start[1], 
                                shape='full', lw=0.5, length_includes_head=True, head_width=5)

            # Plot edges between gateways and satellites
            for GT in self.gateways:
                    if GT.linkedSat[1]:  # Check if there's a linked satellite
                        gt_x = GT.gridLocationX  # Use gridLocationX for gateway X coordinate
                        gt_y = GT.gridLocationY  # Use gridLocationY for gateway Y coordinate
                        sat_x = int((0.5 + math.degrees(GT.linkedSat[1].longitude) / 360) * 1440)  # Satellite longitude
                        sat_y = int((0.5 - math.degrees(GT.linkedSat[1].latitude) / 180) * 720)    # Satellite latitude

                        # Adjust only the endpoint for the arrow
                        _, adj_end = adjust_arrow_points((gt_x, gt_y), (sat_x, sat_y), arrow_gap)
                        
                        plt.arrow(gt_x, gt_y, adj_end[0] - gt_x, adj_end[1] - gt_y,
                                shape='full', lw=0.5, length_includes_head=True, head_width=5)
                        
        if plotSat:
            colors = cm.rainbow(np.linspace(0, 1, len(self.LEO)))

            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    gridSatX = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    gridSatY = int((0.5 - math.degrees(sat.latitude) / 180) * 720) #GT.totalY)
                    scat2 = plt.scatter(gridSatX, gridSatY, marker='o', s=18, linewidth=0.5, edgecolors='black', color=c, label=sat.ID)
                    # if plotSatID:
                    #     plt.text(gridSatX + 10, gridSatY - 10, sat.ID, fontsize=6, ha='left', va='center')    # ANCHOR plots the text of the ID of the satellites

        if plotGT:
            for GT in self.gateways:
                scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY, marker='x', c='r', s=28, linewidth=1.5, label = GT.name)

        # Print path if given
        if path:
            if bottleneck:
                xValues = [[], [], []]
                yValues = [[], [], []]
                minimum = np.amin(bottleneck[1])
                length = len(path)
                index = 0
                arr = 0
                minFound = False

                while index < length:
                    xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                    yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                    if not minFound:
                        if bottleneck[1][index] == minimum:
                            arr+=1
                            xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                            xValues[arr].append(int((0.5 + path[index+1][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index+1][2] / 180) * 720))  # latitude
                            arr+=1
                            minFound = True
                    index += 1

                scat3 = plt.plot(xValues[0], yValues[0], 'b')
                scat3 = plt.plot(xValues[1], yValues[1], 'r')
                scat3 = plt.plot(xValues[2], yValues[2], 'b')
            else:
                xValues = []
                yValues = []
                for hop in path:
                    xValues.append(int((0.5 + hop[1] / 360) * 1440))     # longitude
                    yValues.append(int((0.5 - hop[2] / 180) * 720))      # latitude
                scat3 = plt.plot(xValues, yValues)  # , marker='.', c='b', linewidth=0.5, label = hop[0])

        # Plot the map with the usage of all the links
        if paths is not None:
            link_usage = calculate_link_usage([block.QPath for block in paths]) if self.agent is not None else calculate_link_usage([block.path for block in paths])

            # After calculating max_usage in the plotting section
            try:
                max_usage = max(info['count'] for info in link_usage.values())
                min_usage = max_usage * 0.1  # Set minimum usage to 10% of the maximum
            except ValueError:
                print("Error: No data available for plotting congestion map.")
                print('Link usage values:\n')
                print(link_usage.values())  # FIXME why does this break when few values?
                return  -1 # If this is within a function, it will exit. If not, remove or adjust this line.

            # Find the most used link
            most_used_link = max(link_usage.items(), key=lambda x: x[1]['count'])
            print(f"Most used link: {most_used_link[0]}, Packets: {most_used_link[1]['count']}")

            norm = Normalize(vmin=usage_threshold, vmax=100)
            # cmap = cm.get_cmap('RdYlGn_r')  # Use a red-yellow-green reversed colormap
            # cmap = cm.get_cmap('inferno_r')  # Use a darker colormap
            cmap = cm.get_cmap('cool')  # Use a darker colormap

            for link_str, info in link_usage.items():
                usage = info['count']
                # Convert usage to a percentage of the maximum, with a floor of usage_threshold%
                usage_percentage = max(usage_threshold, (usage / max_usage) * 100)  # Ensure minimum of usage_threshold%
                # Adjust width based on usage_percentage instead of raw usage
                width = 0.5 + (usage_percentage / 100) * 2  # Use usage_percentage for scaling
                
                # Use usage_percentage for color scaling
                color = cmap(norm(usage_percentage))  # This line should use `usage_percentage` for color scaling

                coordinates = info['coordinates']

                # Get original start and end points for adjusting
                orig_start_x, orig_start_y = (0.5 + coordinates[0][0] / 360) * 1440, (0.5 - coordinates[0][1] / 180) * 720
                orig_end_x, orig_end_y = (0.5 + coordinates[1][0] / 360) * 1440, (0.5 - coordinates[1][1] / 180) * 720

                # Adjust start and end points using adjust_arrow_points
                (start_x, start_y), (end_x, end_y) = adjust_arrow_points((orig_start_x, orig_start_y), (orig_end_x, orig_end_y), arrow_gap)

                # Calculate control points for a slight curve, adjusted for the new start and end points
                mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                ctrl_x, ctrl_y = mid_x + (end_y - start_y) / 10, mid_y - (end_x - start_x) / 5  # Adjust divisor for curve tightness

                # Create a Bezier curve for the directed link with adjusted start and end points
                verts = [(start_x, start_y), (ctrl_x, ctrl_y), (end_x, end_y)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)

                # Ensure this color variable is used for the FancyArrowPatch
                patch = FancyArrowPatch(path=path, arrowstyle='-|>', color=color, linewidth=width, mutation_scale=5, zorder=0.5)
                plt.gca().add_patch(patch)

            # Add legend for congestion color coding
            ax = plt.gca()
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ticks = [10] + list(np.linspace(10, 100, num=5))  # Ticks from 10% to 100%
            plt.colorbar(sm, ax=ax, orientation='vertical', label='Relative Traffic Load (%)', fraction=0.02, pad=0.04, ticks=[int(tick) for tick in ticks]) 
            # plt.colorbar(sm, orientation='vertical', fraction=0.02, pad=0.04, ticks=[int(tick) for tick in ticks]) 
            # plt.colorbar(sm, orientation='vertical', label='Number of packets', fraction=0.02, pad=0.04)

            plt.xticks([])
            plt.yticks([])
            # outPath = outputPath + "/CongestionMapFigures/"
            # fileName = outPath + "/CongestionMap.png"
            # os.makedirs(outPath, exist_ok=True)


        if plotSat and plotGT:
            plt.legend([scat1, scat2], ['Gateways', 'Satellites'], loc=3, prop=legend_properties, markerscale=markerscale)
        elif plotSat:
            plt.legend([scat2], ['Satellites'], loc=3, prop=legend_properties, markerscale=markerscale)
        elif plotGT:
            plt.legend([scat1], ['Gateways'], loc=3, prop=legend_properties, markerscale=markerscale)

        plt.xticks([])
        plt.yticks([])

        if paths is None:
            cell_users = np.array(self.getCellUsers()).transpose()
            plt.imshow(cell_users, norm=LogNorm(), cmap='viridis')
        else:
            plt.gca().invert_yaxis()

        # plt.show()
        # plt.imshow(np.log10(np.array(self.getCellUsers()).transpose() + 1), )

        # Add title
        if time is not None and ID is not None:
            plt.title(f"Creation time: {time*1000:.0f}ms, block ID: {ID}")

        if save:
            plt.tight_layout()
            plt.savefig(fileName, dpi=1000, bbox_inches='tight', pad_inches=0)   
  

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []
        xG = []
        yG = []
        zG = []
        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')
        for GT in self.gateways:
            xG.append(GT.x)
            yG.append(GT.y)
            zG.append(GT.z)
        ax.scatter(xG, yG, zG, marker='^')
        plt.show()

    def __repr__(self):
        return 'total divisions in x = {}\n total divisions in y = {}\n total cells = {}\n window of operation ' \
               '(longitudes) = {}\n window of operation (latitudes) = {}'.format(
                self.total_x,
                self.total_y,
                self.total_cells,
                self.windowx,
                self.windowy)
