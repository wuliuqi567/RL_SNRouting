import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_flow_expr(flow_text: str) -> float:
	expr = flow_text.strip().lower().replace("_", "")
	if expr.endswith("e6"):
		return float(expr[:-2]) * 1e6
	if expr.endswith("e9"):
		return float(expr[:-2]) * 1e9
	return float(expr)


def format_flow_expr(flow_bps: float) -> str:
	value_mbps = flow_bps / 1e6
	if abs(value_mbps - round(value_mbps)) < 1e-9:
		return f"{int(round(value_mbps))}e6"
	return f"{value_mbps}e6"


def find_traffic_pairs_block(config_text: str) -> tuple[int, int, str]:
	assignment_match = re.search(r"(?m)^(?!\s*#)\s*trafficPairs\s*=\s*\[", config_text)
	if assignment_match is None:
		raise ValueError("Cannot find active 'trafficPairs = [' in system_configure.py")

	start_idx = assignment_match.start()
	open_bracket_idx = config_text.find("[", assignment_match.start())
	if open_bracket_idx == -1:
		raise ValueError("Cannot find opening '[' for trafficPairs block")

	depth = 0
	end_idx = -1
	for i in range(open_bracket_idx, len(config_text)):
		ch = config_text[i]
		if ch == "[":
			depth += 1
		elif ch == "]":
			depth -= 1
			if depth == 0:
				end_idx = i
				break

	if end_idx == -1:
		raise ValueError("Cannot find closing ']' for trafficPairs block")

	return start_idx, end_idx, config_text[start_idx:end_idx + 1]


def count_active_traffic_pairs(config_text: str) -> int:
	_, _, block = find_traffic_pairs_block(config_text)
	tuple_line_pattern = re.compile(
		r"^(\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*)([^,\)]+)(\s*\)\s*,?\s*(#.*)?)$"
	)

	count = 0
	for line in block.splitlines():
		stripped = line.strip()
		if not stripped or stripped.startswith("#"):
			continue
		if tuple_line_pattern.match(line):
			count += 1
	return count


def parse_target_indices(indices_expr: str, pair_count: int) -> set[int]:
	selected: set[int] = set()
	if not indices_expr.strip():
		return selected

	for token in indices_expr.split(","):
		part = token.strip()
		if not part:
			continue

		if "-" in part:
			bounds = [x.strip() for x in part.split("-", maxsplit=1)]
			if len(bounds) != 2 or not bounds[0] or not bounds[1]:
				raise ValueError(f"Invalid range in --target-indices: '{part}'")
			start = int(bounds[0])
			end = int(bounds[1])
			if start <= 0 or end <= 0:
				raise ValueError("--target-indices must use positive 1-based positions")
			if end < start:
				raise ValueError(f"Invalid range in --target-indices: '{part}'")
			for idx_1based in range(start, end + 1):
				if idx_1based > pair_count:
					raise ValueError(
						f"--target-indices position {idx_1based} out of range (trafficPairs size={pair_count})"
					)
				selected.add(idx_1based - 1)
		else:
			idx_1based = int(part)
			if idx_1based <= 0:
				raise ValueError("--target-indices must use positive 1-based positions")
			if idx_1based > pair_count:
				raise ValueError(
					f"--target-indices position {idx_1based} out of range (trafficPairs size={pair_count})"
				)
			selected.add(idx_1based - 1)

	return selected


def resolve_target_indices(
	pair_count: int,
	target_first_n: int | None,
	target_indices_expr: str | None,
) -> set[int]:
	if pair_count <= 0:
		raise ValueError("No active traffic pair tuple found")

	if target_first_n is not None and target_first_n < 0:
		raise ValueError("--target-first-n must be >= 0")

	selected: set[int] = set()
	if target_first_n is not None:
		if target_first_n > pair_count:
			raise ValueError(
				f"--target-first-n ({target_first_n}) exceeds trafficPairs size ({pair_count})"
			)
		selected.update(range(target_first_n))

	if target_indices_expr:
		selected.update(parse_target_indices(target_indices_expr, pair_count))

	if target_first_n is None and not target_indices_expr:
		return set(range(pair_count))

	if not selected:
		raise ValueError("No target flow selected. Check --target-first-n/--target-indices")

	return selected


def update_traffic_pairs_rates(config_text: str, flow_expr: str, target_indices: set[int]) -> str:
	start_idx, end_idx, block = find_traffic_pairs_block(config_text)

	tuple_line_pattern = re.compile(
		r"^(\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*)([^,\)]+)(\s*\)\s*,?\s*(#.*)?)$"
	)

	updated_lines = []
	replaced_count = 0
	tuple_idx = 0
	for line in block.splitlines():
		stripped = line.strip()
		if not stripped or stripped.startswith("#"):
			updated_lines.append(line)
			continue

		match = tuple_line_pattern.match(line)
		if match:
			if tuple_idx in target_indices:
				updated_lines.append(f"{match.group(1)}{flow_expr}{match.group(3)}")
				replaced_count += 1
			else:
				updated_lines.append(line)
			tuple_idx += 1
		else:
			updated_lines.append(line)

	if replaced_count == 0:
		raise ValueError("No selected traffic pair tuple found to update")

	updated_block = "\n".join(updated_lines)
	return config_text[:start_idx] + updated_block + config_text[end_idx + 1:]


def build_flow_list(start_bps: float, end_bps: float, step_bps: float) -> list[float]:
	if step_bps <= 0:
		raise ValueError("step flow must be positive")
	if end_bps < start_bps:
		raise ValueError("end flow must be >= start flow")

	flows = []
	current = start_bps
	while current <= end_bps + 1e-6:
		flows.append(current)
		current += step_bps
	return flows


def run_flow_sweep(
	start_flow: str,
	end_flow: str,
	step_flow: str,
	restore_config: bool,
	target_first_n: int | None,
	target_indices_expr: str | None,
) -> None:
	repo_root = Path(__file__).resolve().parents[1]
	config_path = repo_root / "system_configure.py"
	simrl_path = repo_root / "simrl.py"

	if not config_path.exists():
		raise FileNotFoundError(f"Missing config file: {config_path}")
	if not simrl_path.exists():
		raise FileNotFoundError(f"Missing simrl entry: {simrl_path}")

	start_bps = parse_flow_expr(start_flow)
	end_bps = parse_flow_expr(end_flow)
	step_bps = parse_flow_expr(step_flow)
	flows = build_flow_list(start_bps, end_bps, step_bps)

	print(f"[SP-test] Total runs: {len(flows)}")
	print(f"[SP-test] Flow sweep: {format_flow_expr(start_bps)} -> {format_flow_expr(end_bps)}, step={format_flow_expr(step_bps)}")

	original_text = config_path.read_text(encoding="utf-8")
	pair_count = count_active_traffic_pairs(original_text)
	target_indices = resolve_target_indices(pair_count, target_first_n, target_indices_expr)
	selected_1based = [idx + 1 for idx in sorted(target_indices)]
	print(f"[SP-test] trafficPairs size={pair_count}, selected={selected_1based}")

	try:
		for run_idx, flow_bps in enumerate(flows, start=1):
			flow_expr = format_flow_expr(flow_bps)
			updated_text = update_traffic_pairs_rates(original_text, flow_expr, target_indices)
			config_path.write_text(updated_text, encoding="utf-8")

			print(f"\n[SP-test] ({run_idx}/{len(flows)}) Running simrl.py with flow={flow_expr}")
			result = subprocess.run(
				[sys.executable, str(simrl_path)],
				cwd=str(repo_root),
				check=False,
			)

			if result.returncode != 0:
				raise RuntimeError(f"simrl.py failed at flow={flow_expr}, return code={result.returncode}")

	finally:
		if restore_config:
			config_path.write_text(original_text, encoding="utf-8")
			print("\n[SP-test] system_configure.py restored to original content.")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Sweep fixed-pair traffic flow and run simrl.py multiple times."
	)
	parser.add_argument("--start-flow", default="600e6", help="Start flow in bps expression, e.g. 500e6")
	parser.add_argument("--end-flow", default="1000e6", help="End flow in bps expression, e.g. 800e6")
	parser.add_argument("--step-flow", default="25e6", help="Step flow in bps expression, e.g. 25e6")
	parser.add_argument(
		"--target-first-n",
		type=int,
		default=None,
		help="Override first N traffic pairs (1-based order in trafficPairs)",
	)
	parser.add_argument(
		"--target-indices",
		default=None,
		help="Override selected 1-based indices, e.g. '2,4-6'",
	)
	parser.add_argument(
		"--no-restore-config",
		action="store_true",
		help="Do not restore system_configure.py after sweep",
	)
	args = parser.parse_args()

	run_flow_sweep(
		start_flow=args.start_flow,
		end_flow=args.end_flow,
		step_flow=args.step_flow,
		restore_config=not args.no_restore_config,
		target_first_n=args.target_first_n,
		target_indices_expr=args.target_indices,
	)


if __name__ == "__main__":
	main()
