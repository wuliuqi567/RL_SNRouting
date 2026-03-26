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


def update_traffic_pairs_rates(config_text: str, flow_expr: str) -> str:
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

	block = config_text[start_idx:end_idx + 1]
	tuple_line_pattern = re.compile(
		r"^(\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*)([^,\)]+)(\s*\)\s*,?\s*(#.*)?)$"
	)

	updated_lines = []
	replaced_count = 0
	for line in block.splitlines():
		stripped = line.strip()
		if not stripped or stripped.startswith("#"):
			updated_lines.append(line)
			continue

		match = tuple_line_pattern.match(line)
		if match:
			updated_lines.append(f"{match.group(1)}{flow_expr}{match.group(3)}")
			replaced_count += 1
		else:
			updated_lines.append(line)

	if replaced_count == 0:
		raise ValueError("No active traffic pair tuple found to update")

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


def run_flow_sweep(start_flow: str, end_flow: str, step_flow: str, restore_config: bool) -> None:
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

	try:
		for run_idx, flow_bps in enumerate(flows, start=1):
			flow_expr = format_flow_expr(flow_bps)
			updated_text = update_traffic_pairs_rates(original_text, flow_expr)
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
	)


if __name__ == "__main__":
	main()
