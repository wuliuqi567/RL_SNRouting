class edge:
    def  __init__(self, sati, satj, slant_range, dji, dij, shannonRate):
        '''
        dji && dij are deprecated. We do not use them anymore to decide which neighbour is at the right or left direction. We are using their coordinates.
        It is used in the markovian matching only
        '''
        self.i = sati   # sati ID
        self.j = satj   # satj ID
        self.slant_range = slant_range  # distance between both sats
        self.dji = dji  # direction from sati to satj
        self.dij = dij  # direction from sati to satj
        self.shannonRate = shannonRate  # max dataRate between sat1 and satj

    def  __repr__(self):
        return '\n node i: {}, node j: {}, slant_range: {}, shannonRate: {}'.format(
    self.i,
    self.j,
    self.slant_range,
    self.shannonRate)

    def __cmp__(self, other):
        if hasattr(other, 'slant_range'):    # returns true if has 'weight' attribute
            return self.slant_range.__cmp__(other.slant_range)
