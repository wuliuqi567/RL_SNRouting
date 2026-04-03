import argparse
import concurrent.futures
import os
import re
import shutil
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
    return f"{value_mbps:.3f}e6"


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
        if stripped and not stripped.startswith("#") and tuple_line_pattern.match(line):
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
        default_first_n = min(2, pair_count)
        return set(range(default_first_n))

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
        raise ValueError("No active traffic pair tuple found to update")

    updated_block = "\n".join(updated_lines)
    return config_text[:start_idx] + updated_block + config_text[end_idx + 1:]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_workspace(repo_root: Path, workspace_root: Path, flow_expr: str) -> Path:
    workspace_root.mkdir(parents=True, exist_ok=True)

    link_targets = [
        "Algorithm",
        "Class",
        "Utils",
        "Population Map",
        "AllPairsPaths",
        "Gateways.csv",
        "globalvar.py",
    ]
    for rel in link_targets:
        link_or_copy(repo_root / rel, workspace_root / rel)

    # Keep all test outputs centralized to main repo results directory.
    link_or_copy(repo_root / "SimResults_new", workspace_root / "SimResults_new")

    simrl_src = repo_root / "simrl.py"
    shutil.copy2(simrl_src, workspace_root / "simrl.py")

    config_src = repo_root / "system_configure.py"
    original_text = config_src.read_text(encoding="utf-8")
    pair_count = count_active_traffic_pairs(original_text)
    target_indices = resolve_target_indices(
        pair_count=pair_count,
        target_first_n=prepare_workspace.target_first_n,
        target_indices_expr=prepare_workspace.target_indices_expr,
    )
    updated_text = update_traffic_pairs_rates(original_text, flow_expr, target_indices)
    (workspace_root / "system_configure.py").write_text(updated_text, encoding="utf-8")

    return workspace_root


prepare_workspace.target_first_n = None
prepare_workspace.target_indices_expr = None


def run_single_flow(repo_root: Path, runtime_root: Path, flow_bps: float, python_exec: str) -> tuple[float, int, Path]:
    flow_expr = format_flow_expr(flow_bps)
    flow_tag = f"flow_{int(round(flow_bps / 1e6)):04d}M"
    workspace = runtime_root / flow_tag
    prepare_workspace(repo_root, workspace, flow_expr)

    cmd = [python_exec, "simrl.py"]
    result = subprocess.run(cmd, cwd=str(workspace), check=False)
    return flow_bps, result.returncode, workspace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MHGNN integration test: sweep fixed-pair flow and run simrl.py in parallel isolated workspaces."
    )
    parser.add_argument("--start-flow", default="600e6", help="Start flow in bps expression")
    parser.add_argument("--end-flow", default="1000e6", help="End flow in bps expression")
    parser.add_argument("--step-flow", default="25e6", help="Step flow in bps expression")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel runs")
    parser.add_argument(
        "--target-first-n",
        type=int,
        default=None,
        help="Override first N traffic pairs (default: first 2 pairs)",
    )
    parser.add_argument(
        "--target-indices",
        default=None,
        help="Override selected 1-based indices, e.g. '2,4-6'",
    )
    parser.add_argument(
        "--runtime-dir",
        default="intergrated_test/.mhgnn_parallel_runs",
        help="Directory for isolated runtime workspaces",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to run simrl.py (default: current interpreter)",
    )
    parser.add_argument(
        "--clean-runtime-dir",
        action="store_true",
        help="Delete runtime workspaces before launching new runs",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = (repo_root / args.runtime_dir).resolve()

    if args.clean_runtime_dir and runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    start_bps = parse_flow_expr(args.start_flow)
    end_bps = parse_flow_expr(args.end_flow)
    step_bps = parse_flow_expr(args.step_flow)
    flows = build_flow_list(start_bps, end_bps, step_bps)

    config_path = repo_root / "system_configure.py"
    original_text = config_path.read_text(encoding="utf-8")
    pair_count = count_active_traffic_pairs(original_text)
    target_indices = resolve_target_indices(pair_count, args.target_first_n, args.target_indices)
    selected_1based = [idx + 1 for idx in sorted(target_indices)]
    prepare_workspace.target_first_n = args.target_first_n
    prepare_workspace.target_indices_expr = args.target_indices

    print(f"[MHGNN-test] Total runs: {len(flows)}")
    print(
        f"[MHGNN-test] Flow sweep: {format_flow_expr(start_bps)} -> {format_flow_expr(end_bps)}, "
        f"step={format_flow_expr(step_bps)}, workers={args.workers}"
    )
    print(f"[MHGNN-test] trafficPairs size={pair_count}, selected={selected_1based}")
    print(f"[MHGNN-test] Runtime dir: {runtime_root}")

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_single_flow, repo_root, runtime_root, flow_bps, args.python_exec): flow_bps
            for flow_bps in flows
        }
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            flow_bps = futures[future]
            flow_expr = format_flow_expr(flow_bps)
            try:
                _flow_bps, return_code, workspace = future.result()
            except Exception as exc:
                failed.append((flow_expr, f"exception: {type(exc).__name__}: {exc}"))
                print(f"[MHGNN-test] ({idx}/{len(flows)}) flow={flow_expr} -> FAILED ({type(exc).__name__})")
                continue

            if return_code != 0:
                failed.append((flow_expr, f"return_code={return_code}"))
                print(f"[MHGNN-test] ({idx}/{len(flows)}) flow={flow_expr} -> FAILED (code={return_code})")
            else:
                print(f"[MHGNN-test] ({idx}/{len(flows)}) flow={flow_expr} -> OK ({workspace.name})")

    if failed:
        print("\n[MHGNN-test] Some runs failed:")
        for flow_expr, reason in failed:
            print(f"  - {flow_expr}: {reason}")
        raise SystemExit(1)

    print("\n[MHGNN-test] All runs finished successfully.")


if __name__ == "__main__":
    main()
