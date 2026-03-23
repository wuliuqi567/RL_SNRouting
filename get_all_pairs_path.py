"""
get_all_pairs_path.py
=====================
Build a **static** satellite-network topology (no simulation / no traffic),
compute the shortest path between every ordered pair of active gateways,
and save the results to CSV + JSON files.

Usage:
    python get_all_pairs_path.py              # use defaults from system_configure.py
    python get_all_pairs_path.py --gts 4      # override the number of active gateways
    python get_all_pairs_path.py --weight slant_range   # override shortest-path weight

Output files (written to ./AllPairsPaths/):
    all_pairs_paths.csv   - one row per (source, destination) pair
    all_pairs_paths.json  - full path detail including every hop's coordinates
"""

import os
import sys
import math
import json
import argparse
import logging
import simpy
import pandas as pd
import networkx as nx
from datetime import datetime

from system_configure import *
import system_configure
from globalvar import *
from Utils.utilsfunction import createGraph, getShortestPath
from Class.earth import Earth

logger = logging.getLogger(__name__)


def setup_logging():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_static_network(gt_number: int, radio_km: float):
    """
    Build the Earth, constellation and graph **without** starting any SimPy
    traffic processes.  Returns (earth, graph).
    """
    allGateWayInfo = pd.read_csv("./Gateways.csv")
    populationDataDir = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'

    system_configure.CurrentGTnumber = gt_number

    env = simpy.Environment()

    # getRates=True => Earth.__init__ will NOT create fillBlock / sendBlock processes
    earth = Earth(env, None, populationDataDir, allGateWayInfo,
                  Constellation, movementTime, getRates=True, outputPath='./')

    # link cells & satellites
    earth.linkCells2GTs(radio_km)
    earth.linkSats2GTs("Optimize")

    # build the static graph (ISL + GSL)
    graph = createGraph(earth, matching=matching)
    earth.graph = graph

    for gt in earth.gateways:
        gt.graph = graph

    return earth, graph


def compute_all_pairs_paths(earth, graph, weight: str):
    """
    Compute the shortest path for every ordered (source, destination) gateway
    pair.  Returns a list of dicts with summary + full hop info.
    """
    results = []
    computed_pairs = set()

    for src_gt in earth.gateways:
        for dst_gt in earth.gateways:
            if src_gt is dst_gt:
                continue
            # avoid duplicate: only compute each unordered pair once
            pair_key = frozenset((src_gt.name, dst_gt.name))
            if pair_key in computed_pairs:
                continue
            computed_pairs.add(pair_key)

            # skip if either GT has no satellite link
            if src_gt.linkedSat[0] is None or dst_gt.linkedSat[0] is None:
                results.append({
                    'source': src_gt.name,
                    'destination': dst_gt.name,
                    'reachable': False,
                    'hops': None,
                    'total_slant_range_km': None,
                    'total_propagation_delay_ms': None,
                    'min_dataRate_bps': None,
                    'path_node_ids': None,
                    'path_detail': None,
                })
                continue

            try:
                nx_path = nx.shortest_path(graph, src_gt.name, dst_gt.name, weight=weight)
            except nx.NetworkXNoPath:
                results.append({
                    'source': src_gt.name,
                    'destination': dst_gt.name,
                    'reachable': False,
                    'hops': None,
                    'total_slant_range_km': None,
                    'total_propagation_delay_ms': None,
                    'min_dataRate_bps': None,
                    'path_node_ids': None,
                    'path_detail': None,
                })
                continue

            # gather per-hop metrics
            total_slant_range = 0.0
            min_rate = float('inf')
            hops_detail = []

            for i, node_id in enumerate(nx_path):
                nd = graph.nodes[node_id]
                lat = nd.get('latitude', None)
                lon = nd.get('longitude', None)
                # gateways store degrees directly; satellites store radians
                if lat is not None and i != 0 and i != len(nx_path) - 1:
                    lat = math.degrees(lat)
                    lon = math.degrees(lon)
                hops_detail.append({
                    'hop': i,
                    'node_id': str(node_id),
                    'latitude': round(lat, 4) if lat is not None else None,
                    'longitude': round(lon, 4) if lon is not None else None,
                })

                if i > 0:
                    edge_data = graph[nx_path[i - 1]][nx_path[i]]
                    sr = edge_data.get('slant_range', 0)
                    total_slant_range += sr
                    rate = edge_data.get('dataRateOG', float('inf'))
                    if rate < min_rate:
                        min_rate = rate

            prop_delay_ms = (total_slant_range / Vc) * 1000  # metres / (m/s) -> s -> ms

            results.append({
                'source': src_gt.name,
                'destination': dst_gt.name,
                'reachable': True,
                'hops': len(nx_path) - 1,
                'total_slant_range_km': round(total_slant_range / 1000, 2),
                'total_propagation_delay_ms': round(prop_delay_ms, 4),
                'min_dataRate_bps': min_rate if min_rate != float('inf') else None,
                'path_node_ids': ' -> '.join(str(n) for n in nx_path),
                'path_detail': hops_detail,
            })

    return results


def save_results(results, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ---- CSV (flat, one row per pair) ----
    csv_rows = []
    for r in results:
        csv_rows.append({
            'source': r['source'],
            'destination': r['destination'],
            'reachable': r['reachable'],
            'hops': r['hops'],
            'total_slant_range_km': r['total_slant_range_km'],
            'total_propagation_delay_ms': r['total_propagation_delay_ms'],
            'min_dataRate_Mbps': round(r['min_dataRate_bps'] / 1e6, 2) if r['min_dataRate_bps'] else None,
            'path_node_ids': r['path_node_ids'],
        })
    csv_path = os.path.join(output_dir, 'all_pairs_paths.csv')
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    logger.info('CSV saved to %s  (%d rows)', csv_path, len(df))

    # ---- JSON (includes full hop detail) ----
    json_path = os.path.join(output_dir, 'all_pairs_paths.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info('JSON saved to %s', json_path)

    return csv_path, json_path


def print_summary(results):
    reachable = [r for r in results if r['reachable']]
    unreachable = [r for r in results if not r['reachable']]

    logger.info('====================================')
    logger.info('All-Pairs Path Summary')
    logger.info('====================================')
    logger.info('Total pairs       : %d', len(results))
    logger.info('Reachable pairs   : %d', len(reachable))
    logger.info('Unreachable pairs : %d', len(unreachable))

    if reachable:
        hops_list = [r['hops'] for r in reachable]
        delay_list = [r['total_propagation_delay_ms'] for r in reachable]
        dist_list = [r['total_slant_range_km'] for r in reachable]

        logger.info('------------------------------------')
        logger.info('Hops   - min: %d,  max: %d,  avg: %.1f',
                     min(hops_list), max(hops_list), sum(hops_list) / len(hops_list))
        logger.info('Prop delay (ms) - min: %.2f,  max: %.2f,  avg: %.2f',
                     min(delay_list), max(delay_list), sum(delay_list) / len(delay_list))
        logger.info('Distance (km)   - min: %.1f,  max: %.1f,  avg: %.1f',
                     min(dist_list), max(dist_list), sum(dist_list) / len(dist_list))
        logger.info('------------------------------------')

        # show each pair
        for r in reachable:
            logger.info('  %s  ->  %s  |  %d hops  |  %.2f ms  |  %.1f km',
                         r['source'], r['destination'],
                         r['hops'], r['total_propagation_delay_ms'],
                         r['total_slant_range_km'])

    if unreachable:
        logger.info('Unreachable pairs:')
        for r in unreachable:
            logger.info('  %s  ->  %s', r['source'], r['destination'])

    logger.info('====================================')


# =====================================================================
# main
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute all-pairs shortest paths in a static LEO satellite network.')
    parser.add_argument('--gts', type=int, default=None,
                        help='Number of active gateways (default: first value in GTs list)')
    parser.add_argument('--weight', type=str, default=None,
                        choices=['slant_range', 'hop', 'dataRate'],
                        help='Edge weight for shortest-path (default: from system_configure)')
    parser.add_argument('--output', type=str, default='./AllPairsPaths',
                        help='Output directory (default: ./AllPairsPaths)')
    parser.add_argument('--radio_km', type=float, default=None,
                        help='Ground-coverage radius in km (default: rKM from system_configure)')
    args = parser.parse_args()

    setup_logging()

    gt_number = args.gts if args.gts is not None else GTs[0]
    weight = args.weight if args.weight is not None else shortest_path_weight
    radio_km = args.radio_km if args.radio_km is not None else rKM

    logger.info('Building static network with %d gateways, constellation=%s, weight=%s ...',
                gt_number, Constellation, weight)

    earth, graph = build_static_network(gt_number, radio_km)

    logger.info('Graph: %d nodes, %d edges', graph.number_of_nodes(), graph.number_of_edges())
    logger.info('Active gateways: %s', [gt.name for gt in earth.gateways])

    results = compute_all_pairs_paths(earth, graph, weight)

    print_summary(results)

    csv_path, json_path = save_results(results, args.output)

    logger.info('Done. Files written to %s', args.output)
