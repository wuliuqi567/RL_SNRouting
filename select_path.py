"""
路径分析脚本：分析 all_pairs_path.csv 中的路径
功能1: 找两条源目节点都不同的路径，它们经过 n (1,2,3,...) 个相同的中间节点
功能2: 找三条源目节点都不同的路径，它们有共同经过的中间节点
"""

import csv
import os
from itertools import combinations
from collections import defaultdict


def load_paths(csv_path):
    """加载 CSV 文件中的所有可达路径，提取中间节点集合"""
    paths = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['reachable'].strip() != 'True':
                continue
            source = row['source'].strip()
            destination = row['destination'].strip()
            path_str = row['path_node_ids'].strip()
            # 解析路径节点: "A -> B -> C -> D" => [A, B, C, D]
            nodes = [n.strip() for n in path_str.split('->')]
            # 中间节点 = 去掉首尾（源和目的）
            intermediate = set(nodes[1:-1]) if len(nodes) > 2 else set()
            paths.append({
                'source': source,
                'destination': destination,
                'intermediate': intermediate,
                'all_nodes': nodes,
                'hops': int(row['hops'])
            })
    return paths


def find_two_paths_with_n_common_nodes(paths, n_values=(1, 2, 3)):
    """
    找两条路径，要求：
    - 两条路径的源节点和目的节点都不同（4个端点互不相同）
    - 两条路径共享恰好 n 个相同的中间节点
    对每个 n 值，输出若干示例。
    """
    results = {n: [] for n in n_values}
    max_examples = 5  # 每个 n 值最多展示的示例数

    for i, j in combinations(range(len(paths)), 2):
        p1, p2 = paths[i], paths[j]
        # 4个端点互不相同
        endpoints = {p1['source'], p1['destination'], p2['source'], p2['destination']}
        if len(endpoints) < 4:
            continue
        # 计算共同中间节点
        common = p1['intermediate'] & p2['intermediate']
        num_common = len(common)
        if num_common in results and len(results[num_common]) < max_examples:
            results[num_common].append({
                'path1': f"{p1['source']} -> {p1['destination']}",
                'path2': f"{p2['source']} -> {p2['destination']}",
                'common_nodes': sorted(common),
                'num_common': num_common
            })

        # 提前退出：所有 n 都收集够了
        if all(len(v) >= max_examples for v in results.values()):
            break

    return results


def find_two_paths_statistics(paths):
    """
    统计两条路径（源目都不同）共享中间节点数量的分布
    """
    distribution = defaultdict(int)
    total_pairs = 0

    for i, j in combinations(range(len(paths)), 2):
        p1, p2 = paths[i], paths[j]
        endpoints = {p1['source'], p1['destination'], p2['source'], p2['destination']}
        if len(endpoints) < 4:
            continue
        common = p1['intermediate'] & p2['intermediate']
        distribution[len(common)] += 1
        total_pairs += 1

    return distribution, total_pairs


def find_three_paths_with_common_nodes(paths, min_common=1, max_examples=5):
    """
    找三条路径，要求：
    - 三条路径的 6 个端点（源+目的）互不相同
    - 三条路径有 >= min_common 个共同的中间节点
    """
    results = []

    # 为了效率，先过滤中间节点较多的路径，建立中间节点到路径的倒排索引
    node_to_paths = defaultdict(list)
    for idx, p in enumerate(paths):
        for node in p['intermediate']:
            node_to_paths[node].append(idx)

    # 按共享节点分组寻找候选三元组
    visited_triples = set()

    for node, path_indices in node_to_paths.items():
        if len(path_indices) < 3:
            continue
        # 从共享该节点的路径中取三元组
        for triple in combinations(path_indices, 3):
            if triple in visited_triples:
                continue
            visited_triples.add(triple)

            i, j, k = triple
            p1, p2, p3 = paths[i], paths[j], paths[k]

            # 6个端点互不相同
            endpoints = {
                p1['source'], p1['destination'],
                p2['source'], p2['destination'],
                p3['source'], p3['destination']
            }
            if len(endpoints) < 6:
                continue

            # 三条路径共同的中间节点
            common = p1['intermediate'] & p2['intermediate'] & p3['intermediate']
            if len(common) >= min_common:
                results.append({
                    'path1': f"{p1['source']} -> {p1['destination']}",
                    'path2': f"{p2['source']} -> {p2['destination']}",
                    'path3': f"{p3['source']} -> {p3['destination']}",
                    'common_nodes': sorted(common),
                    'num_common': len(common)
                })
                if len(results) >= max_examples:
                    return results

    return results


def print_separator(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'AllPairsPaths/all_pairs_paths.csv')
    print(f"正在加载路径数据: {csv_path}")
    paths = load_paths(csv_path)
    print(f"共加载 {len(paths)} 条可达路径\n")

    # ========== 功能1: 两条路径共享中间节点的统计 ==========
    print_separator("功能1: 两条路径（源目都不同）共享中间节点数量的统计分布")
    distribution, total_pairs = find_two_paths_statistics(paths)
    print(f"满足源目都不同的路径对总数: {total_pairs}")
    print(f"\n{'共享中间节点数':>15} | {'路径对数量':>10} | {'占比':>8}")
    print("-" * 45)
    for n in sorted(distribution.keys()):
        count = distribution[n]
        pct = count / total_pairs * 100 if total_pairs > 0 else 0
        print(f"{n:>15} | {count:>10} | {pct:>7.2f}%")

    # ========== 功能1: 具体示例 ==========
    print_separator("功能1: 两条路径共享恰好 n 个中间节点的示例 (n=1,2,3)")
    two_path_results = find_two_paths_with_n_common_nodes(paths, n_values=(2, 3, 4, 5))

    for n in (2, 3, 4, 5):
        print(f"\n--- 共享恰好 {n} 个中间节点的路径对 ---")
        examples = two_path_results[n]
        if not examples:
            print("  未找到符合条件的路径对")
            continue
        for idx, ex in enumerate(examples, 1):
            print(f"\n  示例 {idx}:")
            print(f"    路径1: {ex['path1']}")
            print(f"    路径2: {ex['path2']}")
            print(f"    共享中间节点 ({ex['num_common']}个): {ex['common_nodes']}")

    # ========== 功能2: 三条路径共享中间节点 ==========
    print_separator("功能2: 三条路径（源目都不同）共享中间节点的示例")

    for min_common in (2, 3, 4, 5):
        print(f"\n--- 三条路径共享至少 {min_common} 个中间节点 ---")
        three_path_results = find_three_paths_with_common_nodes(
            paths, min_common=min_common, max_examples=3
        )
        if not three_path_results:
            print("  未找到符合条件的路径组合")
            continue
        for idx, ex in enumerate(three_path_results, 1):
            print(f"\n  示例 {idx}:")
            print(f"    路径1: {ex['path1']}")
            print(f"    路径2: {ex['path2']}")
            print(f"    路径3: {ex['path3']}")
            print(f"    共享中间节点 ({ex['num_common']}个): {ex['common_nodes']}")


if __name__ == '__main__':
    main()
