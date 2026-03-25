from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Pair = Tuple[str, str]
RatePair = Tuple[str, str, float]


def load_gateway_names(gateway_csv_path: str | Path) -> List[str]:
	"""Load gateway names from Gateways.csv."""
	csv_path = Path(gateway_csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(f"Gateway file not found: {csv_path}")

	names: List[str] = []
	with csv_path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = str(row.get("Location", "")).strip()
			if name:
				names.append(name)
	return names


def _pair_key(src: str, dst: str) -> Pair:
	return src.strip(), dst.strip()


def _expand_excluded_pairs(excluded_pairs: Iterable[Pair] | None, exclude_reverse: bool) -> set[Pair]:
	blocked: set[Pair] = set()
	if not excluded_pairs:
		return blocked

	for src, dst in excluded_pairs:
		p = _pair_key(src, dst)
		blocked.add(p)
		if exclude_reverse:
			blocked.add((p[1], p[0]))
	return blocked


def _make_candidate_pairs(nodes: Sequence[str], blocked_pairs: set[Pair]) -> List[Pair]:
	candidates: List[Pair] = []
	for src in nodes:
		for dst in nodes:
			if src == dst:
				continue
			pair = (src, dst)
			if pair in blocked_pairs:
				continue
			candidates.append(pair)
	return candidates


def append_generated_flows(
	base_pairs: Sequence[RatePair],
	gateway_names: Sequence[str],
	generated_pair_count: int = 10,
	generated_total_flow_mbps: float = 50.0,
	selection_mode: str = "fixed",
	excluded_pairs: Iterable[Pair] | None = None,
	exclude_reverse_pairs: bool = True,
	exclude_existing_pairs: bool = True,
	exclude_existing_nodes: bool = True,
	seed: int = 42,
) -> List[RatePair]:
	"""Append generated OD pairs to existing fixed pairs.

	Default behavior:
	- append 10 OD pairs
	- generated total flow = 50 Mbps (split equally across generated pairs)
	- do not use existing selected nodes from base_pairs
	"""
	base_pairs_list: List[RatePair] = [(s, d, float(r)) for s, d, r in base_pairs]

	if generated_pair_count <= 0 or generated_total_flow_mbps <= 0:
		return base_pairs_list

	all_nodes = [str(name).strip() for name in gateway_names if str(name).strip()]
	all_nodes = list(dict.fromkeys(all_nodes))

	used_nodes = set()
	if exclude_existing_nodes:
		for src, dst, _ in base_pairs_list:
			used_nodes.add(src)
			used_nodes.add(dst)

	if exclude_existing_nodes:
		candidate_nodes = [node for node in all_nodes if node not in used_nodes]
		if len(candidate_nodes) < 2:
			candidate_nodes = all_nodes
	else:
		candidate_nodes = all_nodes

	blocked_pairs = _expand_excluded_pairs(excluded_pairs, exclude_reverse_pairs)
	if exclude_existing_pairs:
		existing = [(src, dst) for src, dst, _ in base_pairs_list]
		blocked_pairs |= _expand_excluded_pairs(existing, exclude_reverse_pairs)

	candidates = _make_candidate_pairs(candidate_nodes, blocked_pairs)
	if not candidates:
		return base_pairs_list

	if selection_mode not in ("fixed", "random"):
		raise ValueError("selection_mode must be 'fixed' or 'random'.")

	if selection_mode == "random":
		rng = random.Random(seed)
		rng.shuffle(candidates)
	else:
		candidates = sorted(candidates)

	selected_count = min(generated_pair_count, len(candidates))
	selected_pairs = candidates[:selected_count]
	if selected_count == 0:
		return base_pairs_list

	per_pair_rate_bps = generated_total_flow_mbps * 1e6 / selected_count
	generated_pairs: List[RatePair] = [
		(src, dst, per_pair_rate_bps) for src, dst in selected_pairs
	]

	return base_pairs_list + generated_pairs


def build_traffic_pairs(
	base_pairs: Sequence[RatePair],
	gateway_csv_path: str | Path,
	generated_pair_count: int = 10,
	generated_total_flow_mbps: float = 50.0,
	selection_mode: str = "fixed",
	excluded_pairs: Iterable[Pair] | None = None,
	exclude_reverse_pairs: bool = True,
	exclude_existing_pairs: bool = True,
	exclude_existing_nodes: bool = True,
	seed: int = 42,
) -> List[RatePair]:
	"""Convenience wrapper: load gateways and append generated flows."""
	gateway_names = load_gateway_names(gateway_csv_path)
	return append_generated_flows(
		base_pairs=base_pairs,
		gateway_names=gateway_names,
		generated_pair_count=generated_pair_count,
		generated_total_flow_mbps=generated_total_flow_mbps,
		selection_mode=selection_mode,
		excluded_pairs=excluded_pairs,
		exclude_reverse_pairs=exclude_reverse_pairs,
		exclude_existing_pairs=exclude_existing_pairs,
		exclude_existing_nodes=exclude_existing_nodes,
		seed=seed,
	)

