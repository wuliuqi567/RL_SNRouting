import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in: {csv_path}")
    return rows


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return 0.0
    return float(value)


def prepare_series(rows: list[dict]) -> dict:
    records = []
    for row in rows:
        flow = to_float(row, "flow_total_gbps")
        created = to_float(row, "createdBlocks")
        received = to_float(row, "receivedDataBlocks")
        stuck = to_float(row, "stuckBlocks")

        if created > 0:
            delivery_ratio = received / created
            loss_ratio = stuck / created
        else:
            delivery_ratio = 0.0
            loss_ratio = 0.0

        queue = to_float(row, "Queue time")
        tx = to_float(row, "Transmission time")
        prop = to_float(row, "Propagation time")
        total_component = queue + tx + prop
        if total_component > 0:
            queue_share = queue / total_component
            tx_share = tx / total_component
            prop_share = prop / total_component
        else:
            queue_share = tx_share = prop_share = 0.0

        records.append(
            {
                "flow_total_gbps": flow,
                "avg_time": to_float(row, "avgTime"),
                "queue_time": queue,
                "tx_time": tx,
                "prop_time": prop,
                "queue_share": queue_share,
                "tx_share": tx_share,
                "prop_share": prop_share,
                "created_blocks": created,
                "received_blocks": received,
                "stuck_blocks": stuck,
                "delivery_ratio": delivery_ratio,
                "loss_ratio": loss_ratio,
            }
        )

    records.sort(key=lambda item: item["flow_total_gbps"])

    return {
        "flow": [item["flow_total_gbps"] for item in records],
        "avg_time": [item["avg_time"] for item in records],
        "queue_time": [item["queue_time"] for item in records],
        "tx_time": [item["tx_time"] for item in records],
        "prop_time": [item["prop_time"] for item in records],
        "queue_share": [item["queue_share"] for item in records],
        "tx_share": [item["tx_share"] for item in records],
        "prop_share": [item["prop_share"] for item in records],
        "created_blocks": [item["created_blocks"] for item in records],
        "received_blocks": [item["received_blocks"] for item in records],
        "stuck_blocks": [item["stuck_blocks"] for item in records],
        "delivery_ratio": [item["delivery_ratio"] for item in records],
        "loss_ratio": [item["loss_ratio"] for item in records],
    }


def apply_plot_style() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
        }
    )


def plot_latency_trends(data: dict, out_dir: Path) -> Path:
    flow = data["flow"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)

    plots = [
        ("Average latency", data["avg_time"], "#1f77b4", "o"),
        ("Queue time", data["queue_time"], "#d62728", "s"),
        ("Transmission time", data["tx_time"], "#2ca02c", "^"),
        ("Propagation time", data["prop_time"], "#9467bd", "d"),
    ]

    for ax, (title, series, color, marker) in zip(axes.flat, plots):
        ax.plot(flow, series, color=color, marker=marker)
        ax.set_title(title)
        ax.set_ylabel("Time")
        ax.grid(True, alpha=0.35)

    for ax in axes[1, :]:
        ax.set_xlabel("Total offered flow (Gbps)")

    fig.suptitle("Latency Components vs Offered Flow", y=0.98)
    fig.tight_layout()

    out_path = out_dir / "fig_latency_subplots_vs_flow.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_latency_composition(data: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    flow = data["flow"]

    ax.stackplot(
        flow,
        data["queue_share"],
        data["tx_share"],
        data["prop_share"],
        labels=["Queue share", "Transmission share", "Propagation share"],
        alpha=0.85,
    )

    ax.set_xlabel("Total offered flow (Gbps)")
    ax.set_ylabel("Normalized contribution")
    ax.set_title("Latency Component Composition vs Offered Flow")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = out_dir / "fig2_latency_composition_vs_flow.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_reliability(data: dict, out_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    flow = data["flow"]

    line1 = ax1.plot(flow, data["delivery_ratio"], color="#1f77b4", marker="o", label="Delivery ratio")
    line2 = ax1.plot(flow, data["loss_ratio"], color="#d62728", marker="s", label="Loss ratio")
    ax1.set_xlabel("Total offered flow (Gbps)")
    ax1.set_ylabel("Ratio")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    bars = ax2.bar(flow, data["stuck_blocks"], width=0.018, alpha=0.22, color="#9467bd", label="Stuck blocks")
    ax2.set_ylabel("Stuck blocks")

    handles = line1 + line2 + [bars]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True)

    ax1.set_title("Reliability and Congestion Indicators vs Offered Flow")
    fig.tight_layout()

    out_path = out_dir / "fig3_reliability_vs_flow.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_block_volume(data: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    flow = data["flow"]

    ax.plot(flow, data["created_blocks"], marker="o", label="Created blocks")
    ax.plot(flow, data["received_blocks"], marker="s", label="Received blocks")
    ax.plot(flow, data["stuck_blocks"], marker="^", label="Stuck blocks")

    ax.set_xlabel("Total offered flow (Gbps)")
    ax.set_ylabel("Block count")
    ax.set_title("Block Volume vs Offered Flow")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out_path = out_dir / "fig4_block_volume_vs_flow.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SP blockInfo summary and generate publication-style figures."
    )
    parser.add_argument(
        "--input-csv",
        default="intergrated_test/SP-blockInfo_merged.csv",
        help="Input merged CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="intergrated_test/SP_analysis_figures",
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    data = prepare_series(rows)

    apply_plot_style()
    outputs = [plot_latency_trends(data, output_dir)]

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    for path in outputs:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
