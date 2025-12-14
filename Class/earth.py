import math
import os
import numpy as np
import simpy
import networkx as nx
from system_configure import *
from globalvar import *
import geopy.distance
from PIL import Image
import pandas as pd
import time
from Class.auxiliaryClass import *
from Class.gateWay import Gateway
from Algorithm.QLearning import QLearning
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
    def __init__(self, env, agent_class, img_path, gt_path, constellation, inputParams, deltaT, totalLocations, getRates = False, window=None, outputPath='/'):
        # Input the population count data
        # img_path = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'
        self.outputPath = outputPath
        # self.plotPaths = plotPath
        self.lostBlocks = 0
        self.queues = []
        self.loss   = []
        self.lossAv = []
        # self.DDQNA  = None
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

        gateways = pd.read_csv(gt_path)

        length = 0
        for i, location in enumerate(gateways['Location']):
            for name in inputParams['Locations']:
                if name in location.split(","):
                    length += 1

        if inputParams['Locations'][0] != 'All':
            for i, location in enumerate(gateways['Location']):
                for name in inputParams['Locations']:
                    if name in location.split(","):
                        lName = gateways['Location'][i]
                        gtLati = gateways['Latitude'][i]
                        gtLongi = gateways['Longitude'][i]
                        self.gateways.append(Gateway(lName, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                                   length, env, totalLocations, self))
                        break
        else:
            for i in range(len(gateways['Latitude'])):
                name = gateways['Location'][i]
                gtLati = gateways['Latitude'][i]
                gtLongi = gateways['Longitude'][i]
                self.gateways.append(Gateway(name, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                           len(gateways['Latitude']), env, totalLocations, self))


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
        # self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

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
                    if plotSatID:
                        plt.text(gridSatX + 10, gridSatY - 10, sat.ID, fontsize=6, ha='left', va='center')    # ANCHOR plots the text of the ID of the satellites

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
            link_usage = calculate_link_usage([block.QPath for block in paths]) if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning' else calculate_link_usage([block.path for block in paths])

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
  
    def initializeQTables(self, NGT, hyperparams, g):
        '''
        QTables initialization at each satellite
        '''
        print('----------------------------------')

        # path = './Results/Q-Learning/qTablesImport/qTablesExport/' + str(NGT) + 'GTs/'
        path = tablesPath

        if importQVals:
            print('Importing Q-Tables from: ' + path)
        else:
            print('Initializing Q-tables...')
        
        i = 0
        for plane in self.LEO:
            for sat in plane.sats:
                i += 1
                if importQVals:
                    with open(path + sat.ID + '.npy', 'rb') as f:
                        qTable = np.load(f)
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat, qTable=qTable)
                else:
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat)

        if importQVals:
            print(str(i) + ' Q-Tables imported!')
        else:
            print(str(i) + ' Q-Tables created!')
        print('----------------------------------')

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
