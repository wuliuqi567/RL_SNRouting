# =============================================================================
# 1. Simulation & Pathing Configuration
# =============================================================================

GTs = [2]               # number of gateways to be tested
# Gateways are taken from https://www.ksat.no/ground-network-services/the-ksat-global-ground-station-network/ (Except for Malaga and Aalborg)
# GTs = [i for i in range(2,9)] # This is to make a sweep where scenarios with all the gateways in the range are considered

CurrentGTnumber = -1    # Number of active gateways. This number will be updated every time a gateway is added. In the simulation it will iterate the GTs list


# =============================================================================
# 2. Physical Constants
# =============================================================================
rKM = 500               # radio in km of the coverage of each gateway
Re  = 6378e3            # Radius of the earth [m]
G   = 6.67259e-11       # Universal gravitational constant [m^3/kg s^2]
Me  = 5.9736e24         # Mass of the earth
Te  = 86164.28450576939 # Time required by Earth for 1 rotation
Vc  = 299792458         # Speed of light [m/s]
k   = 1.38e-23          # Boltzmann's constant
eff = 0.55              # Efficiency of the parabolic antenna

# =============================================================================
# 3. Link & Traffic Parameters
# =============================================================================
# Downlink parameters
f       = 20e9  # Carrier frequency GEO to ground (Hz)
B       = 500e6 # Maximum bandwidth
maxPtx  = 10    # Maximum transmission power in W
Adtx    = 0.26  # Transmitter antenna diameter in m
Adrx    = 0.26  #0.33 Receiver antenna diameter in m
pL      = 0.3   # Pointing loss in dB
Nf      = 2     #1.5 Noise figure in dB
Tn      = 290   #50 Noise temperature in K
min_rate= 10e3  # Minimum rate in kbps

# Uplink Parameters
balancedFlow= False         # if set to true all the generated traffic at each GT is equal
totalFlow   = 2*1000000000  # Total average flow per GT when the balanced traffc option is enabled. Malaga has 3*, LA has 3*, Nuuk/500
avUserLoad  = 15593 * 8      # average traffic usage per second in bits

# Block
BLOCK_SIZE   = 64800

# =============================================================================
# 4. Movement & Constellation
# =============================================================================
movementTime= 0.5       # Every movementTime seconds, the satellites positions are updated and the graph is built again
                        # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
ndeltas     = 5805.44/20 #1 Movement speedup factor. Every movementTime sats will move movementTime*ndeltas space. If bigger, will make the rotation distance bigger

saveISLs    = True     # save ISLs map
const_moved = False     # Movement flag. If up, it means it has moved
matching    = 'Positive_Grid'  # ['Markovian', 'Greedy', 'Positive_Grid']
minElAngle  = 30        # For satellites. Value is taken from NGSO constellation design chapter.
mixLocs     = False     # If true, every time we make a new simulation the locations are going to change their order of selection
rotateFirst = False     # If True, the constellation starts rotated by 1 movement defined by ndeltas
GridSize    = 8         # Earth divided in GridSize rows for the grid. Used to be 15

# =============================================================================
# 5. State Representation
# =============================================================================
coordGran   = 20            # Granularity of the coordinates that will be the input of the DNN: (Lat/coordGran, Lon/coordGran)
diff        = True          # If up, the state space gives no coordinates about the neighbor and destination positions but the difference with respect to the current positions
diff_lastHop= True          # If up, this state is the same as diff, but it includes the last hop where the block was in order to avoid loops
reducedState= False         # if set to true the DNN will receive as input only the positional information, but not the queueing information
third_adj    = True          # If up, the state space includes the 3rd order neighbors information
n_order_adj = 4             # Order of the neighbors to be included in the subgraph state representation
notAvail    = 0             # this value is set in the state space when the satellite neighbour is not available
infQueue    = 5000      # Upper boundary from where a queue is considered as infinite when obserbing the state
queueVals   = 10        # Values that the observed Queue can have, being 0 the best (Queue of 0) and max the worst (Huge queue or inexistent link).
latBias     = 90        # This value is added to the latitude of each position in the state space. This can be done to avoid negative numbers
lonBias     = 180       # Same but with longitude

import json

def save_system_config_to_json(file_path="system_config_dump.json"):
    """
    将当前文件中的所有全局变量保存到 JSON 文件中。
    """
    config_data = {}
    # 获取当前的全局变量字典
    current_globals = globals().copy()
    
    for key, value in current_globals.items():
        # 过滤掉内置变量（以__开头）、函数、模块以及导入的库
        if key.startswith("__") or callable(value) or isinstance(value, type(json)):
            continue
        
        # 尝试将变量添加到字典中
        # 这里可以根据需要添加更多的类型检查或转换
        try:
            # 检查是否可以被 JSON 序列化，如果不行则转为字符串
            json.dumps(value)
            config_data[key] = value
        except (TypeError, OverflowError):
            config_data[key] = str(value)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print(f"Configuration successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving configuration: {e}")
    
    return config_data

if __name__ == "__main__":
    save_system_config_to_json()



