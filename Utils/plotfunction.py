import matplotlib.pyplot as plt
import numpy as np
import logging
from system_configure import *
from globalvar import *
from Class.auxiliaryClass import Results
import os
import pickle
import pandas as pd
from collections import defaultdict
import seaborn as sns
import torch


logger = logging.getLogger(__name__)

def plotLatencies(percentages, pathing, savePath):
    '''
    Bar plot where each bar is a scenario with a different nº of gateways and where each color represents one of the three latencies.
    '''
    # plot percent stacked barplot
    barWidth= 0.85
    r       = percentages['GTnumber']
    numbers = percentages['GTnumber']
    GTnumber= len(r)

    plt.bar(r, percentages['Propagation time'], color='#b5ffb9', edgecolor='white', width=barWidth, label="Propagation time")   # Propagation time
    plt.bar(r, percentages['Queue time'], bottom=percentages['Propagation time'], color='#f9bc86',                              # Queue time
             edgecolor='white', width=barWidth, label="Queue time")
    plt.bar(r, percentages['Transmission time'], bottom=[i+j for i,j in zip(percentages['Propagation time'],                    # Tx time
            percentages['Queue time'])], color='#a3acff', edgecolor='white', width=barWidth, label="Transmission time")

    # Custom x axis
    plt.xticks(numbers)
    plt.xlabel("Nº of gateways")
    plt.ylabel('Latency')

    # Add a legend
    plt.legend(loc='lower left')
    
    # Show and save graphic
    plt.savefig(
        savePath + '{}_gatewaysTotal.png'.format(GTnumber))

    data = {"numb gateways": r, "prop delay": percentages['Propagation time'], "Queue delay": percentages['Queue time'], "transmission delay": percentages['Transmission time']}
    d = pd.DataFrame(data=data)
    d.to_csv(savePath + "delayFractions.csv")
    # try:
    #     plt.savefig('Latency3/{}/Percentages_{}_gateways.png'.format(pathing, GTnumber))
    # except:
    #     plt.savefig('./Code/Latency3/{}/Percentages_{}_gateways.png'.format(pathing, GTnumber))
    # plt.show()

def extract_block_index(block_id):
    return int(block_id.split('_')[-1])

def plotRatesFigures():
    values = [upGSLRates, downGSLRates, interRates, intraRate]

    plt.figure()
    plt.hist(np.asarray(interRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Inter plane ISL data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(upGSLRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Uplink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(downGSLRates)/1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Downlink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

def getBlockTransmissionStats(timeToSim, GTs, constellationType, earth, outputPath):
    '''
    General Block transmission stats
    '''
    totalTime = 0.0
    queue_sum = 0.0
    tx_sum = 0.0
    prop_sum = 0.0
    block_count = 0

    allLatencies = []
    pathBlocks = [[], []]
    first       = earth.gateways[0]
    second      = earth.gateways[1]

    # Reuse existing received block objects instead of building a duplicated wrapper list.
    # This avoids an additional O(n) memory footprint inside this function.
    blocks = receivedDataBlocks

    # earth.pathParam

    for block in receivedDataBlocks:
        time = block.getTotalTransmissionTime()
        queue_time = block.getQueueTime()[0]

        totalTime += time
        queue_sum += queue_time
        tx_sum += block.txLatency
        prop_sum += block.propLatency
        block_count += 1
        
        # [creation time, total latency, arrival time, source, destination, block ID, queue time, transmission latency, prop latency]
        allLatencies.append([
            block.creationTime,
            block.totLatency,
            block.creationTime + block.totLatency,
            block.source.name,
            block.destination.name,
            block.ID,
            queue_time,
            block.txLatency,
            block.propLatency,
        ])
        # pre-process the received data blocks. create the rows that will be saved in csv
        if block.source == first and block.destination == second:
            pathBlocks[0].append([block.totLatency, block.creationTime+block.totLatency])
            pathBlocks[1].append(block)
        
    # save congestion test data
    # blockPath = f"./Results/Congestion_Test/{pathing} {float(pd.read_csv('inputRL.csv')['Test length'][0])}/"
    # print('Saving congestion test data...\n')
    # blockPath = outputPath + '/Congestion_Test/'     
    # os.makedirs(blockPath, exist_ok=True)
    # try:
    #     global CurrentGTnumber
    #     np.save("{}blocks_{}".format(blockPath, CurrentGTnumber), np.asarray(blocks),allow_pickle=True)
    # except pickle.PicklingError:
    #     print('Error with pickle and profiling')

    avgTime = float(totalTime / block_count) if block_count > 0 else 0.0
    created_blocks = len(createdBlocks)
    received_blocks = len(receivedDataBlocks)
    stuck_blocks = created_blocks - received_blocks - len(dropBlocks)

    if totalTime > 0:
        queue_pct = float(queue_sum / totalTime * 100)
        tx_pct = float(tx_sum / totalTime * 100)
        prop_pct = float(prop_sum / totalTime * 100)
    else:
        queue_pct = 0.0
        tx_pct = 0.0
        prop_pct = 0.0

    logger.info('########## Results #########')
    logger.info('The simulation took %s seconds to run', timeToSim)
    logger.info('A total of %s data blocks were created', created_blocks)
    logger.info('A total of %s data blocks were transmitted', received_blocks)
    logger.info('A total of %s data blocks were lost', len(dropBlocks))
    logger.info('A total of %s data blocks were stuck', stuck_blocks)
    logger.info('Average transmission time for all blocks were %s', avgTime)
    logger.info(
        'Total latecies: Queue time: %.4f%%, Transmission time: %.4f%%, Propagation time: %.4f%%',
        queue_pct,
        tx_pct,
        prop_pct
    )

    os.makedirs(outputPath, exist_ok=True)
    block_info_file = os.path.join(outputPath, 'blockInfo.csv')
    block_info_columns = [
        'createdBlocks',
        'receivedDataBlocks',
        'stuckBlocks',
        'avgTime',
        'Queue time',
        'Transmission time',
        'Propagation time',
    ]
    block_info_row = pd.DataFrame([
        {
            'createdBlocks': created_blocks,
            'receivedDataBlocks': received_blocks,
            'stuckBlocks': stuck_blocks,
            'avgTime': avgTime,
            'Queue time': queue_pct,
            'Transmission time': tx_pct,
            'Propagation time': prop_pct,
        }
    ], columns=block_info_columns)

    if not os.path.exists(block_info_file):
        pd.DataFrame(columns=block_info_columns).to_csv(block_info_file, index=False)
    block_info_row.to_csv(block_info_file, mode='a', index=False, header=False)

    results = Results(finishedBlocks=blocks,
                      constellation=constellationType,
                      GTs=GTs,
                      meanTotalLatency=avgTime,
                      meanQueueLatency=(queue_sum / block_count if block_count > 0 else 0.0),
                      meanPropLatency=(prop_sum / block_count if block_count > 0 else 0.0),
                      meanTransLatency=(tx_sum / block_count if block_count > 0 else 0.0),
                      perQueueLatency = queue_pct,
                      perPropLatency = prop_pct,
                      perTransLatency = tx_pct)

    return results, allLatencies, pathBlocks, blocks


def plotSavePathLatencies(outputPath, GTnumber, pathBlocks):
    # figure of latencies between two first gateways
    latency = []
    arrival = []
    for item in pathBlocks[0]:
        latency.append(item[0])
        arrival.append(item[1])
    plt.scatter(arrival, latency, c='r')
    plt.xlabel("Time")
    plt.ylabel("Latency")               
    os.makedirs(outputPath + '/pngLatencies/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/pngLatencies/' + '{}_gatewaysTime.png'.format(GTnumber))
    plt.close()  

    # x axis is the number of the arrival, not the time
    xs = [l for l in range(len(latency))]
    plt.figure()
    plt.scatter(xs,latency, c='r')
    plt.xlabel("Arrival index")
    plt.ylabel('Latency')
    plt.savefig(outputPath + '/pngLatencies/' + '{}_gateways.png'.format(GTnumber))
    plt.close()

    # Save latencies
    os.makedirs(outputPath + '/csv/', exist_ok=True) # create output path
    data = {'Latency': [l for l in latency], 'Arrival Time': [t for t in arrival]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "pathLatencies_{}_gateways.csv".format(GTnumber), index=False)
    # os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path


def plot_packet_latencies_and_uplink_downlink_throughput(data, outputPath, bins_num=30, save=False, plot_separately=True):
    """
     生成“时延散点 + 吞吐量曲线”的联合图。

     图的构成：
     1) 主 y 轴（ax1）：包到达时延散点图
         - x: 包到达时刻（arrival_time）
         - y: 端到端时延（totLatency）
     2) 次 y 轴（ax2）：上/下行吞吐量折线
         - 上行吞吐量：按 creation_time 统计每个时间分箱内的包数
         - 下行吞吐量：按 arrival_time 统计每个时间分箱内的包数
         - 吞吐量计算：throughput = (counts * BLOCK_SIZE / 1e3) / bin_width

     参数：
     - data: 可迭代 block 对象，要求至少包含
        - block.path（用于提取源/宿）
        - block.creationTime
        - block.totLatency
     - outputPath: 输出目录
     - bins_num: 时间分箱数量（越大曲线越细，越容易抖动）
     - save: True 时保存到 outputPath/Throughput；False 时直接显示
     - plot_separately: True 按 (src, dst) 分路由作图；False 合并所有路由

     注意：
     - 当前实现保持历史口径，吞吐量单位标注为 Mbps，且使用 /1e3 的换算。
     - 若你希望严格按 bit/s->Mbit/s，可改为 /1e6（会改变数值尺度）。
    """

    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # 按 (源网关, 目的网关) 分组，便于按路径分别绘图
    paths_data = defaultdict(list)
    for block in data:
        src = block.path[0][0]        # Source
        dst = block.path[-1][0]       # Destination
        paths_data[(src, dst)].append(block)

    # Function to plot data for a single path or combined
    def plot_path_data(blocks, src=None, dst=None):
        fig, ax1 = plt.subplots(figsize=(8, 4))
        
        # 先按创建时间排序，保证时间轴递增
        blocks = sorted(blocks, key=lambda b: b.creationTime)
        
        # 提取时序数据并统一转换为 ms
        creation_times = np.array([block.creationTime for block in blocks]) * 1000  # ms
        arrival_times = np.array([block.creationTime + block.totLatency for block in blocks]) * 1000  # ms
        latencies = np.array([block.totLatency * 1000 for block in blocks])  # ms

        # Scatter plot for packet arrival times vs latency
        arrival_scatter = ax1.scatter(arrival_times, latencies, color='#1E90FF', label='Packet Delivery', alpha=0.6, s=10)
        
        # Configure primary y-axis for latency
        ax1.set_xlabel('Time [ms]', fontsize=16)
        ax1.set_ylabel('Average E2E Latency [ms]', fontsize=16)
        
        # 在副轴绘制吞吐量曲线（与时延共享 x 轴）
        ax2 = ax1.twinx()

        # 在 [最早创建时刻, 最晚到达时刻] 范围内等距划分时间箱
        # 每个箱子对应一个“局部吞吐量估计”
        time_bins = np.linspace(min(creation_times), max(arrival_times), num=bins_num)
        
        # 上行吞吐量：统计 creation_time 在每个 bin 的包数
        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        # 下行吞吐量：统计 arrival_time 在每个 bin 的包数
        # 公式核心：throughput = (包数 * 包大小) / 时间窗宽
        # np.diff(time_bins) 是每个时间窗宽度（ms）
        uplink_throughput = (uplink_counts * BLOCK_SIZE / 1e3) / np.diff(time_bins)  # Mbps
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)
        downlink_throughput = (downlink_counts * BLOCK_SIZE / 1e3) / np.diff(time_bins)  # Mbps

        # x 轴使用每个 bin 的左边界（time_bins[:-1]）来画吞吐量曲线
        uplink_line, = ax2.plot(time_bins[:-1], uplink_throughput, color='#00008B', lw=2, label='Uplink Throughput')
        downlink_line, = ax2.plot(time_bins[:-1], downlink_throughput, color='#1E90FF', lw=2, label='Downlink Throughput')
        
        # Configure secondary y-axis for throughput
        ax2.set_ylabel('Throughput [Mbps]', fontsize=16)
        
        # Combine legends
        handles = [arrival_scatter, uplink_line, downlink_line]
        labels = [handle.get_label() for handle in handles]
        ax1.legend(handles, labels, loc='upper center', fontsize=12)

        # Display grid and layout adjustments
        ax1.grid(True)
        ax2.grid(True, linestyle=':', linewidth=0.5)
        plt.tight_layout()
        
        # Save or show plot
        if save:
            filename = f'{src}_{dst}_path_latency_throughput.png' if src and dst else 'combined_path_latency_throughput.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        else:
            plt.show()
        plt.close()

    # 根据配置选择：按路径分别绘图，或合并全部路径绘图
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_path_data(blocks, src, dst)
    else:
        all_blocks = [block for blocks in paths_data.values() for block in blocks]
        plot_path_data(all_blocks)


def plot_throughput_cdf(data, outputPath, bins_num=100, save=False, plot_separately=True):
    """
    Generate and save a CDF plot of the throughput. Either plot each route separately or
    combine all routes into a single plot based on the `plot_separately` flag.
    """
    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # Group blocks by (source, destination) paths
    paths_data = defaultdict(list)
    for block in data:
        src = block.path[0][0]  # Source
        dst = block.path[-1][0]  # Destination
        paths_data[(src, dst)].append(block)

    # Helper function to plot CDF for a given set of blocks
    def plot_cdf_for_path(blocks, src=None, dst=None):
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort blocks by creation time
        blocks = sorted(blocks, key=lambda b: b.creationTime)
        
        # Extract creation times and arrival times
        creation_times = np.array([block.creationTime for block in blocks])
        arrival_times = np.array([block.creationTime + block.totLatency for block in blocks])
        
        # Define time bins and calculate throughput
        time_bins = np.linspace(min(creation_times), max(arrival_times), num=bins_num)
        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        uplink_throughput = (uplink_counts * BLOCK_SIZE / 1e6) / np.diff(time_bins)  # Mbps
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)
        downlink_throughput = (downlink_counts * BLOCK_SIZE / 1e6) / np.diff(time_bins)  # Mbps
        
        # Sort and calculate CDF
        uplink_throughput_sorted = np.sort(uplink_throughput)
        downlink_throughput_sorted = np.sort(downlink_throughput)
        uplink_cdf = np.arange(1, len(uplink_throughput_sorted) + 1) / len(uplink_throughput_sorted)
        downlink_cdf = np.arange(1, len(downlink_throughput_sorted) + 1) / len(downlink_throughput_sorted)
        
        # Plot CDFs
        ax.plot(uplink_throughput_sorted, uplink_cdf, label='Uplink Throughput', color='#00008B', lw=2)
        ax.plot(downlink_throughput_sorted, downlink_cdf, label='Downlink Throughput', color='#1E90FF', lw=2)
        
        # Configure plot
        ax.set_xlabel('Throughput [Mbps]', fontsize=16)
        ax.set_ylabel('CDF', fontsize=16)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Adjust layout, save plot, and close
        plt.tight_layout()
        if save:
            filename = f'Throughput_CDF_{src}_to_{dst}.png' if src and dst else 'Throughput_CDF_All_Paths.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        else:
            plt.show()
        plt.close()

    # Plot each path separately or all paths combined based on flag
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_cdf_for_path(blocks, src, dst)
    else:
        all_blocks = [block for blocks in paths_data.values() for block in blocks]
        plot_cdf_for_path(all_blocks)


def save_plot_rewards(outputPath, reward, GTnumber, window_size=200):
    rewards = [x[0] for x in reward]
    times   = [x[1] for x in reward]
    data    = pd.DataFrame({'Rewards': rewards, 'Time': times})

    # Smoothed Rewards
    data['Smoothed Rewards'] = data['Rewards'].rolling(window=window_size).mean()

    # Top 10% and Bottom 10% Rewards
    # data['Top 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(lambda x: np.mean(np.partition(x, -int(len(x)*0.1))[-int(len(x)*0.1):]), raw=True)
    # data['Bottom 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(lambda x: np.mean(np.partition(x, int(len(x)*0.1))[:int(len(x)*0.1)]), raw=True)

    # Plotting
    plt.figure(figsize=(8, 4))
    # line1, = plt.plot(data['Time'], data['Top 10% Avg Rewards'], color='skyblue', linewidth=2, label='Top 10% reward')
    line2, = plt.plot(data['Time'], data['Smoothed Rewards'], color='blue', linewidth=2, label='Average reward')
    # line3, = plt.plot(data['Time'], data['Bottom 10% Avg Rewards'], color='navy', linewidth=2, label='Bottom 10% reward')

    # plt.legend(handles=[line1, line2, line3], fontsize=15, loc='upper right')
    plt.legend(handles=[line2], fontsize=15, loc='upper right')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Time [ms]", fontsize=15)
    plt.ylabel("Average rewards", fontsize=15)
    plt.grid(True)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Save plot
    rewards_dir = os.path.join(outputPath, 'Rewards')
    plt.tight_layout()
    os.makedirs(rewards_dir, exist_ok=True)  # create output path
    plt.savefig(os.path.join(rewards_dir, "rewards_{}_gateways.png".format(GTnumber)))#, bbox_inches='tight')
    plt.close()

    # Save CSV
    csv_dir = os.path.join(outputPath, 'csv')
    os.makedirs(csv_dir, exist_ok=True)  # create output path
    data.to_csv(os.path.join(csv_dir, "rewards_{}_gateways.csv".format(GTnumber)), index=False)

    return data

def save_epsilons(outputPath, eps, GTnumber):
    epsilons = [x[0] for x in eps]
    times    = [x[1] for x in eps]
    plt.plot(times, epsilons)
    plt.title("Epsilon over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Epsilon")
    os.makedirs(outputPath + '/epsilons/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/epsilons/' + "epsilon_{}_gateways.png".format(GTnumber))
    plt.close()

    data = {'epsilon': [e for e in epsilons], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    os.makedirs(outputPath + '/csv/' , exist_ok=True) # create output path
    df.to_csv(outputPath + '/csv/' + "epsilons_{}_gateways.csv".format(GTnumber), index=False)

    return df


def save_training_counts(outputPath, train_times, GTnumber):
    # Extract times
    times = [x[0]*1000 for x in train_times]

    # Calculate cumulative trainings over time
    training_counts = list(range(1, len(times) + 1))

    # Plotting the cumulative number of trainings
    plt.plot(times, training_counts)
    plt.title("Cumulative trainings over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of Trainings")

    # Create output path and save the figure
    os.makedirs(outputPath + '/trainings/', exist_ok=True)
    plt.savefig(outputPath + '/trainings/' + "trainings_{}_gateways.png".format(GTnumber))
    plt.close()

    # Prepare data for saving
    data = {'time': times, 'trainings': training_counts}
    df = pd.DataFrame(data)

    # Create CSV output path and save data
    os.makedirs(outputPath + '/csv/', exist_ok=True)
    df.to_csv(outputPath + '/csv/' + "trainings_{}_gateways.csv".format(GTnumber), index=False)



def plotSaveAllLatencies(outputPath, GTnumber, allLatencies, epsDF=None, annotate_min_latency=True):  
    # preprocess and setup
    GTnumber_Max = 4 # max number of gts for displaying the legend. If the number of GTs is bigger than this, then no legend is displayed
    sns.set(font_scale=1.5)
    window_size = winSize
    marker_size = markerSize
    df = pd.DataFrame(allLatencies, columns=['Creation Time', 'Latency', 'Arrival Time', 'Source', 
                                             'Destination', 'Block ID', 'QueueTime', 'TxTime', 'PropTime'])
    df['Block Index'] = df['Block ID'].apply(extract_block_index)
    df = df.sort_values(by=['Source', 'Destination', 'Block Index'])
    df.to_csv(outputPath + '/csv/' + "allLatencies_{}_gateways.csv".format(GTnumber))

    # Convert time values to milliseconds
    df['Creation Time'] *= 1000
    df['Arrival Time']  *= 1000
    df['Latency']       *= 1000
    if epsDF is not None:
        epsDF['time']   *= 1000

    # Calculate the rolling average for each unique path
    df['Path'] = df['Source'].astype(str) + ' -> ' + df['Destination'].astype(str)
    df['Latency_Rolling_Avg'] = df.groupby('Path')['Latency'].transform(lambda x: x.rolling(window=window_size).mean())
    
    # Metrics for x-axis
    metrics = ['Arrival Time', 'Creation Time']

    # Create subplots
    fig, axes = plt.subplots(len(metrics), 2, figsize=(18, 18))

    for i, metric in enumerate(metrics):
        # Line Plots on the left (column index 0)
        lineplot = sns.lineplot(x=metric, y='Latency_Rolling_Avg', hue='Path', ax=axes[i, 0], data=df)
        axes[i, 0].set_title(f'Latency Trends Over {metric} (Window Size = {window_size})')
        axes[i, 0].set_xlabel(metric + ' (ms)')
        axes[i, 0].set_ylabel('Average Latency (ms)')

        # Annotate minimum latency for Creation Time only
        if annotate_min_latency and metric == 'Creation Time':
            unique_paths = df['Path'].unique()
            for path in unique_paths:
                df_path = df[df['Path'] == path]
                min_latency = df_path['Latency_Rolling_Avg'].min()
                try:
                    min_pos = df_path[metric][df_path['Latency_Rolling_Avg'].idxmin()]
                    axes[i, 0].annotate(f'{min_latency:.0f} ms', xy=(min_pos, min_latency), 
                                        xytext=(-50, 30), textcoords='offset points', 
                                        arrowprops=dict(arrowstyle='->', color='black'))
                except KeyError as e:
                    print(f"Error annotating minimum latency for the path path {path}: {e}")


        # Scatter Plots on the right (column index 1)
        scatterplot = sns.scatterplot(x=metric, y='Latency', hue='Path', ax=axes[i, 1], data=df, marker='o', s=marker_size)
        axes[i, 1].set_title(f'Individual Latency Points Over {metric}')
        axes[i, 1].set_xlabel(metric)
        axes[i, 1].set_ylabel('Latency')

        # Create a twin y-axis for epsilon data if epsDF is not None
        if epsDF is not None:
            ax2 = axes[i, 0].twinx()
            line3 = sns.lineplot(x='time', y='epsilon', data=epsDF, color='purple', label='Epsilon', ax=ax2)

            # Merge legends
            handles1, labels1 = axes[i, 0].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            axes[i, 0].legend(handles1 + handles2, labels1 + labels2, loc='upper right')
            # Check if ax2 has a legend before trying to remove it
            if ax2.get_legend():
                ax2.get_legend().remove()
        else:
            # Handle legend for the case when epsDF is None
            handles, labels = axes[i, 0].get_legend_handles_labels()
            axes[i, 0].legend(handles, labels, loc='upper right')

        if GTnumber > GTnumber_Max:
            # Disable legends on both subplots
            if axes[i, 0].get_legend():
                axes[i, 0].get_legend().set_visible(False)
            if axes[i, 1].get_legend():
                axes[i, 1].get_legend().set_visible(False)

        
    # Adjust the layout
    plt.tight_layout()
    os.makedirs(outputPath + '/pngAllLatencies/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/pngAllLatencies/' + '{}_gateways_All_Latencies_subplots.png'.format(GTnumber), dpi = 300)
    plt.close()
    sns.set()

def save_losses(outputPath, earth1, GTnumber):
    losses = [x[0] for x in earth1.loss]
    times  = [x[1] for x in earth1.loss]
    plt.plot(times, losses)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysTime.png".format(GTnumber))
    plt.close()

    data = {'loss': [l for l in losses], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "loss_{}_gateways.csv".format(GTnumber), index=False)
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path

    xs = [l for l in range(len(losses))]
    plt.plot(xs, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Steps")
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysSteps.png".format(GTnumber))
    plt.close()

    # save losses average
    plt.plot(range(len(earth1.lossAv)), earth1.lossAv)
    plt.xlabel("Steps")
    plt.ylabel("Loss average")
    plt.title("Loss average over Steps")
    os.makedirs(outputPath + '/loss/', exist_ok=True) # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysAverage.png".format(GTnumber))
    plt.close()


def plot_cka_over_time(cka_data, outputPath, nGTs):
    """
    Plots each CKA value over time in milliseconds, connecting 'before' and 'after' points with a dashed line
    and using different colors for each type of dot, with quartile ranges represented by error bars.
    
    Parameters:
    - cka_data: List of [CKA_before, CKA_after, timestamp] entries.
    """
    path = outputPath + 'FL/'
    os.makedirs(path, exist_ok=True)  # create output path

    # Extract times and calculate CKA values for before and after
    times = [entry[2] * 1000 for entry in cka_data]  # Convert time to milliseconds
    cka_before_values = [np.mean(entry[0]) for entry in cka_data]
    cka_after_values = [np.mean(entry[1]) for entry in cka_data]

    # Calculate quartile ranges for before and after values
    cka_before_quartiles = [np.percentile(entry[0], [25, 75]) for entry in cka_data]
    cka_after_quartiles = [np.percentile(entry[1], [25, 75]) for entry in cka_data]
    cka_before_25th, cka_before_75th = zip(*cka_before_quartiles)
    cka_after_25th, cka_after_75th = zip(*cka_after_quartiles)

    # Construct the sequence for line plot: interleave before and after mean values
    line_times = [time for time in times for _ in (0, 1)]
    line_values = [val for pair in zip(cka_before_values, cka_after_values) for val in pair]

    # Set y-axis limits with margin to avoid cutting T-caps and ensure the max is exactly 1
    y_min = min(min(cka_before_25th), min(cka_after_25th)) * 0.95
    y_max = 1

    # Plotting
    plt.figure(figsize=(10, 6))

    # Line connecting mean CKA values
    plt.plot(line_times, line_values, label='CKA Value Sequence', color='gray', linestyle='-.', alpha=0.7)

    # Error bars for 'CKA Before FL' and 'CKA After FL' with T-caps
    cka_before_yerr = [np.abs(np.subtract(cka_before_values, cka_before_25th)), 
                       np.abs(np.subtract(cka_before_75th, cka_before_values))]
    cka_after_yerr = [np.abs(np.subtract(cka_after_values, cka_after_25th)), 
                      np.abs(np.subtract(cka_after_75th, cka_after_values))]

    plt.errorbar(times, cka_before_values, yerr=cka_before_yerr, fmt='s', color='blue', 
                 ecolor='blue', capsize=8, capthick=2, label='CKA Before FL Quartiles')
    plt.errorbar(times, cka_after_values, yerr=cka_after_yerr, fmt='s', color='green', 
                 ecolor='green', capsize=8, capthick=2, label='CKA After FL Quartiles')

    # Set x-axis and y-axis limits with a dynamic y-axis minimum
    plt.xlim(min(times) - 20, max(times) + 20)
    # plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation for y-axis

    # Labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('CKA Value')
    plt.title('CKA Values Over Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'CKA_over_time_{str(nGTs)}_GTs.png'), dpi=300, bbox_inches='tight')

    # Save mean CKA values over time
    mean_cka_values = np.column_stack((times, cka_before_values, cka_after_values))
    np.savetxt(os.path.join(path, 'mean_cka_values.csv'), mean_cka_values, delimiter=',', 
               header="Time_ms,CKA_Before,CKA_After", comments='')

    # Save individual CKA matrices before and after FL for each timestamp
    for i, entry in enumerate(cka_data):
        np.savetxt(os.path.join(path, f'cka_matrix_before_{i}.csv'), entry[0], delimiter=',')
        np.savetxt(os.path.join(path, f'cka_matrix_after_{i}.csv'), entry[1], delimiter=',')


def plotShortestPath(earth, path, outputPath, ID=None, time=None):
    earth.plotMap(True, True, path=path, ID=ID,time=time)
    plt.savefig(outputPath + 'popMap_' + path[0][0] + '_to_' + path[len(path)-1][0] + '.png', dpi = 500)
    # plt.show()
    plt.close()


def plotQueues(queues, outputPath, GTnumber):
    '''
    Will plot the cumulative distribution function (CDF) and probability density function (PDF) of all the queues that each package has faced.
    ''' 
    os.makedirs(outputPath + '/pngQueues/', exist_ok=True) # create output path
    plt.hist(queues, bins=max(queues), cumulative=True, density = True, label='CDF DATA', histtype='step', alpha=0.55, color='blue')
    plt.xlabel('Queue length')
    plt.legend(loc = 'lower left')
    plt.savefig(outputPath + '/pngQueues/' + 'Queues_{}_Gateways.png'.format(GTnumber))
    plt.close()
    d = pd.DataFrame(queues)
    d.to_csv(outputPath + '/csv/' + "Queues_{}_Gateways.csv".format(GTnumber), index = False)


def plotCongestionMap(self, paths, outPath, GTnumber, plot_separately=True):
    def extract_gateways(path):
    # Assuming QPath's first and last elements contain gateway identifiers
        if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
            return path.QPath[0][0], path.QPath[-1][0]
        else:
            return path.path[0][0], path.path[-1][0]
        
    os.makedirs(outPath, exist_ok=True)

    # Identify unique routes and filter by packet threshold (100 packets)
    unique_routes = {}
    for block in paths:
        p = block.QPath if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning' else block.path
        if p:  # Ensure QPath or path is not empty
            gateways = extract_gateways(block)
            if gateways in unique_routes:
                unique_routes[gateways] += 1
            else:
                unique_routes[gateways] = 1

    filtered_routes = {route: count for route, count in unique_routes.items() if count > 100} # REVIEW Packet threshold for path visualization 500

    # Plot for all routes combined
    if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
        all_routes_paths = [block for block in paths if block.QPath and extract_gateways(block) in filtered_routes]
    else:
        all_routes_paths = [block for block in paths if block.path and extract_gateways(block) in filtered_routes]

    done = self.plotMap(plotGT=True, plotSat=True, edges=False, save=True, paths=np.asarray(all_routes_paths),
                 fileName=os.path.join(outPath, f"all_routes_CongestionMap_{GTnumber}GTs.png"))
    plt.close()
    if done == -1:
        print('Congestion map for all routes not available')

    # Plot for each unique route above the threshold
    if plot_separately:
        for route, count in filtered_routes.items():
            if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
                route_paths = [block for block in paths if extract_gateways(block) == route and block.QPath]
            else:
                route_paths = [block for block in paths if extract_gateways(block) == route and block.path]

            done = self.plotMap(plotGT=True, plotSat=True, edges=False, save=True, paths=np.asarray(route_paths),
                        fileName=os.path.join(outPath, f"CongestionMap_{route[0]}_to_{route[1]}_{GTnumber}GTs.png"))
            plt.close()
            if done == -1:
                print(f'Congestion map for {route} not available')


def findBottleneck(path, earth, plot=False, minimum=None):
    """
    找出通信路径中的瓶颈链路。
    
    该函数遍历给定路径中的所有链路（网关到卫星、卫星到卫星、卫星回到网关），
    收集每条链路的信息，包括链路端点、数据速率、地理位置等信息，
    最后返回所有链路的瓶颈信息（最小数据速率的链路）。
    
    参数：
        path (list): 通信路径，形式为 [(源ID, ...), (中间节点ID, ...), ..., (目的地ID, ...)]
                    其中path[0][0]是源节点ID，path[-1][0]是目的地节点ID
        earth (Earth): 地球环境对象，包含所有网关(gateways)和LEO星座(LEO)信息
        plot (bool): 是否绘制路径及瓶颈链路的地图，默认为False
        minimum (float): 可选的最小数据速率阈值，用于计算链路时延（最小速率/链路速率）
    
    返回：
        tuple: (bottleneck, minimum)
            - bottleneck (list): 包含4个子列表的列表
                - bottleneck[0]: 链路信息列表，每个元素为"源节点ID,目的地节点ID"的字符串
                - bottleneck[1]: 数据速率列表（Gbps或相应单位），对应各条链路的传输速率
                - bottleneck[2]: 纬度列表，记录每条链路的源节点纬度坐标
                - bottleneck[3]: 时延列表（仅当输入minimum>0时），计算为minimum/链路速率
            - minimum (float): 路径中所有链路的最小数据速率值
    
    函数流程：
        1. 处理起始网关段：遍历所有网关，找到路径源点对应的网关，
           记录从该网关到下一相邻节点(path[1][0])的链路信息
        2. 处理中间链路段：遍历路径中的所有卫星节点，对于每个卫星，
           查找其星座间(interSats)和星座内(intraSats)的链路，
           匹配路径中的相邻节点对，记录链路参数
        3. 处理终结网关段：找到路径目的地对应的网关，
           记录从前一相邻节点(path[-2][0])到该网关的下行链路(downRate)信息
        4. 可视化：如果plot=True，在地球地图上绘制完整路径及瓶颈链路
        5. 统计：计算所有链路中的最小数据速率作为该路径的瓶颈
    """
    # 初始化瓶颈信息容器：[链路信息, 数据速率, 纬度, 时延]
    bottleneck = [[], [], [], []]
    
    # ===== 第一阶段：收集起始网关到第一个卫星的链路信息 =====
    for GT in earth.gateways:
        if GT.name == path[0][0]:
            # 记录网关上行链路：源网关 -> 路径中的下一个节点
            bottleneck[0].append(str(path[0][0].split(",")[0]) + "," + str(path[1][0]))
            # 记录该网关的上行数据速率
            bottleneck[1].append(GT.dataRate)
            # 记录该网关的纬度
            bottleneck[2].append(GT.latitude)
            # 如果提供了最小速率参数，计算该链路的时延（最小速率/链路速率）
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    # ===== 第二阶段：收集路径中间的卫星间链路信息 =====
    for i, step in enumerate(path[1:], 1):  # 从path[1]开始遍历，i从1开始
        # 遍历所有LEO轨道平面
        for orbit in earth.LEO:
            # 遍历该轨道平面中的所有卫星
            for satellite in orbit.sats:
                # 找到路径中当前节点对应的卫星
                if satellite.ID == step[0]:
                    
                    # 检查该卫星的星座间链路(inter-plane ISL)
                    for sat in satellite.interSats:
                        # sat[1]是链接目标卫星，sat[2]是链路数据速率
                        if sat[1].ID == path[i + 1][0]:  # 检查目标是否为路径中的下一节点
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])  # 记录星座间链路速率
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
                    
                    # 检查该卫星的星座内链路(intra-plane ISL)
                    for sat in satellite.intraSats:
                        if sat[1].ID == path[i + 1][0]:  # 检查目标是否为路径中的下一节点
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])  # 记录星座内链路速率
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])

    # ===== 第三阶段：收集最后一个卫星到目地网关的链路信息 =====
    for GT in earth.gateways:
        if GT.name == path[-1][0]:  # 找到路径目的地对应的网关
            # 记录网关下行链路：路径中的前一个节点 -> 目地网关
            bottleneck[0].append(str(path[-2][0]) + "," + str(path[-1][0].split(",")[0]))
            # 记录该网关的下行数据速率
            bottleneck[1].append(GT.linkedSat[1].downRate)
            # 记录该网关的纬度
            bottleneck[2].append(GT.latitude)
            # 如果提供了最小速率参数，计算该链路的时延
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    # ===== 第四阶段：可视化（可选） =====
    if plot:
        # 在地球地图上绘制完整路径及瓶颈链路信息
        earth.plotMap(True, True, path, bottleneck)
        plt.show()
        plt.close()

    # ===== 第五阶段：统计 =====
    # 计算所有链路中的最小数据速率，作为该路径的瓶颈
    minimum = np.amin(bottleneck[1])
    
    # 返回瓶颈信息和最小数据速率
    return bottleneck, minimum


def saveQTables(outputPath, earth):
    print('Saving Q-Tables at: ' + outputPath)
    # create output path if it does not exist
    path = outputPath + 'qTablesExport_' + str(len(earth.gateways)) + 'GTs/'
    os.makedirs(path, exist_ok=True) 

    # save Q-Tables
    for plane in earth.LEO:
        for sat in plane.sats:
            qTable = sat.QLearning.qTable
            with open(path + sat.ID + '.npy', 'wb') as f:
                np.save(f, qTable)


def saveDeepNetworks(outputPath, earth):
    print('Saving Deep Neural networks at: ' + outputPath)
    os.makedirs(outputPath, exist_ok=True) 
    if not onlinePhase:
        torch.save(earth.DDQNA.qNetwork.state_dict(), outputPath + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
        if ddqn:
            torch.save(earth.DDQNA.qTarget.state_dict(), outputPath + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
    else:
        for plane in earth.LEO:
            for sat in plane.sats:
                torch.save(sat.DDQNA.qNetwork.state_dict(), outputPath + sat.ID + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
                if ddqn:
                    torch.save(sat.DDQNA.qTarget.state_dict(), outputPath + sat.ID + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.pth')


def saveNNModel(outputPath, earth):
    print('Saving Deep Neural networks at: ' + outputPath)
    os.makedirs(outputPath, exist_ok=True) 
    if not onlinePhase:
        torch.save(earth.DDQNA.qNetwork.state_dict(), outputPath + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
        if ddqn:
            torch.save(earth.DDQNA.qTarget.state_dict(), outputPath + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
            torch.save(earth.DDQNA.sNetwork.state_dict(), outputPath + 'sNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
    else:
        for plane in earth.LEO:
            for sat in plane.sats:
                torch.save(sat.DDQNA.qNetwork.state_dict(), outputPath + sat.ID + 'qNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
                if ddqn:
                    torch.save(sat.DDQNA.qTarget.state_dict(), outputPath + sat.ID + 'qTarget_'+ str(len(earth.gateways)) + 'GTs' + '.pth')
                    torch.save(sat.DDQNA.sNetwork.state_dict(), outputPath + sat.ID + 'sNetwork_'+ str(len(earth.gateways)) + 'GTs' + '.pth')