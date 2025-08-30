from configure import *

class DataBlock:
    """
    Class for outgoing block of data from the gateways.
    Instead of simulating the individual data packets from each user, data is gathered at the GTs in blocks - one for
    each destination GT. Once a block is filled with data it is sent as one unit to the destination GT.
    """

    def __init__(self, source, destination, ID, creationTime):
        self.size = BLOCK_SIZE  # size in bits
        self.destination = destination
        self.source = source
        self.ID = ID            # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = None  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = None  # the simulation time at which the block left the GT.
        self.checkPoints = []   # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = []   # list of times after the block was sent at each node
        self.path = []
        self.queueLatency = (None, None) # total time acumulated in the queues
        self.txLatency = 0      # total transmission time
        self.propLatency = 0    # total propagation latency
        self.totLatency = 0     # total latency
        self.isNewPath = False
        self.oldPath = []
        self.newPath = []
        self.QPath   = []
        self.queue   = []
        self.queueTime= []
        self.oldState  = None
        self.oldAction = None
        # self.oldReward = None

    def getQueueTime(self):
        '''
        The queue latency is computed in two steps:
        First one: time when the block is sent for the first time - time when the the block is created
        Rest of the steps: sum(checkpoint (Arrival time at node) - checkpointsSend (send time at previous node))
        '''
        queueLatency = [0, []]
        queueLatency[0] += self.timeAtFirstTransmission - self.creationTime        # ANCHOR queue first step
        queueLatency[1].append(self.timeAtFirstTransmission - self.creationTime)
        for arrived, sendReady in zip(self.checkPoints, self.checkPointsSend):  # rest of the steps
            queueLatency[0] += sendReady - arrived
            queueLatency[1].append(sendReady - arrived)

        self.queueLatency = queueLatency
        return queueLatency

    def getTotalTransmissionTime(self):
        totalTime = 0
        if len(self.checkPoints) == 1:
            return self.checkPoints[0] - self.timeAtFirstTransmission

        lastTime = self.creationTime
        for time in self.checkPoints:
            totalTime += time - lastTime
            lastTime = time
        # ANCHOR KPI: totLatency
        self.totLatency = totalTime
        return totalTime

    def __repr__(self):
        return'ID = {}\n Source:\n {}\n Destination:\n {}\nTotal latency: {}'.format(
            self.ID,
            self.source,
            self.destination,
            self.totLatency
        )
