import argparse
import csv
import re
from pathlib import Path


def parse_flow_info_from_run_name(run_name: str) -> dict:
	total_flow_gbps = None
	total_match = re.search(r"_t(?P<gbps>\d+(?:\.\d+)?)G", run_name)
	if total_match:
		total_flow_gbps = float(total_match.group("gbps"))

	pair_flow_matches = re.findall(r"_[A-Z]+2[A-Z]+(?P<mbps>\d+(?:\.\d+)?)M\b", run_name)
	pair_flows_mbps = [float(item) for item in pair_flow_matches] if pair_flow_matches else []

	return {
		"flow_total_gbps": total_flow_gbps,
		"flow_min_mbps": min(pair_flows_mbps) if pair_flows_mbps else None,
		"flow_max_mbps": max(pair_flows_mbps) if pair_flows_mbps else None,
		"flow_avg_mbps": (sum(pair_flows_mbps) / len(pair_flows_mbps)) if pair_flows_mbps else None,
		"flow_values_mbps": ",".join(str(int(v) if float(v).is_integer() else v) for v in pair_flows_mbps),
	}


def collect_target_files(base_dir: Path, csv_name: str) -> list[Path]:
	pattern = f"**/{csv_name}"
	files = sorted(base_dir.glob(pattern))
	filtered = [
		path for path in files
		if path.name == csv_name
		and path.parent != base_dir
		and path.parent.name != "csv"
	]
	return filtered


def sort_merged_rows(rows: list[dict], sort_by: str, descending: bool) -> list[dict]:
	if sort_by == "none":
		return rows

	numeric_sort_fields = {"flow_total_gbps", "flow_min_mbps", "flow_max_mbps", "flow_avg_mbps"}

	def key_func(row: dict):
		value = row.get(sort_by)
		if sort_by in numeric_sort_fields:
			if value is None or value == "":
				return float("-inf") if descending else float("inf")
			try:
				return float(value)
			except (TypeError, ValueError):
				return float("-inf") if descending else float("inf")
		return str(value)

	return sorted(rows, key=key_func, reverse=descending)


def merge_block_info(base_dir: Path, csv_name: str, output_name: str, sort_by: str, descending: bool) -> tuple[Path, int, int]:
	if not base_dir.exists():
		raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

	target_files = collect_target_files(base_dir, csv_name)
	if not target_files:
		raise FileNotFoundError(f"No '{csv_name}' found under: {base_dir}")

	merged_rows = []
	all_input_columns = set()
	for csv_path in target_files:
		run_dir = csv_path.parent
		flow_info = parse_flow_info_from_run_name(run_dir.name)
		rows_added_for_file = 0
		with csv_path.open("r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			if reader.fieldnames:
				all_input_columns.update(
					name for name in reader.fieldnames
					if isinstance(name, str) and name.strip()
				)

			for row in reader:
				row["source_csv"] = str(csv_path.relative_to(base_dir))
				row["run_dir"] = run_dir.name
				row["flow_total_gbps"] = flow_info["flow_total_gbps"]
				row["flow_min_mbps"] = flow_info["flow_min_mbps"]
				row["flow_max_mbps"] = flow_info["flow_max_mbps"]
				row["flow_avg_mbps"] = flow_info["flow_avg_mbps"]
				row["flow_values_mbps"] = flow_info["flow_values_mbps"]
				merged_rows.append(row)
				rows_added_for_file += 1

		if rows_added_for_file == 0:
			merged_rows.append(
				{
					"source_csv": str(csv_path.relative_to(base_dir)),
					"run_dir": run_dir.name,
					"flow_total_gbps": flow_info["flow_total_gbps"],
					"flow_min_mbps": flow_info["flow_min_mbps"],
					"flow_max_mbps": flow_info["flow_max_mbps"],
					"flow_avg_mbps": flow_info["flow_avg_mbps"],
					"flow_values_mbps": flow_info["flow_values_mbps"],
				}
			)

	if not merged_rows:
		raise ValueError(f"No rows found in '{csv_name}' files under: {base_dir}")

	merged_rows = sort_merged_rows(merged_rows, sort_by=sort_by, descending=descending)

	meta_columns = [
		# "source_csv",
		# "run_dir",
		"flow_total_gbps",
		"flow_min_mbps",
		"flow_max_mbps",
		"flow_avg_mbps",
		"flow_values_mbps",
	]
	fieldnames = meta_columns + sorted(col for col in all_input_columns if col not in meta_columns)

	output_path = base_dir / output_name
	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in merged_rows:
			writer.writerow({name: row.get(name, "") for name in fieldnames})

	return output_path, len(target_files), len(merged_rows)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Collect blockInfo CSVs under a date directory and enrich with flow info from folder names."
	)
	parser.add_argument(
		"--base-dir",
		default="SimResults_new/ShortestPath/2026-03-26",
		help="Date directory to scan, e.g. SimResults_new/ShortestPath/2026-03-26",
	)
	parser.add_argument(
		"--csv-name",
		default="blockInfo.csv",
		help="CSV file name to collect from each run subfolder (default: blockInfo.csv)",
	)
	parser.add_argument(
		"--output-name",
		default="blockInfo_merged.csv",
		help="Output merged CSV name under base directory",
	)
	parser.add_argument(
		"--sort-by",
		default="flow_total_gbps",
		choices=["none", "flow_total_gbps", "flow_min_mbps", "flow_max_mbps", "flow_avg_mbps", "run_dir"],
		help="Sort merged rows by the selected key",
	)
	parser.add_argument(
		"--desc",
		action="store_true",
		help="Sort in descending order (default is ascending)",
	)
	args = parser.parse_args()

	base_dir = Path(args.base_dir).resolve()
	output_path, file_count, row_count = merge_block_info(
		base_dir,
		args.csv_name,
		args.output_name,
		sort_by=args.sort_by,
		descending=args.desc,
	)
	print(f"Merged file written: {output_path}")
	print(f"Collected files: {file_count}")
	print(f"Merged rows: {row_count}")


if __name__ == "__main__":
	main()
