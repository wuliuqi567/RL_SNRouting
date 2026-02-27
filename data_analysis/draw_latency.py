"""
路径时延曲线绘制工具

该模块用于从CSV文件中读取路径时延数据，并绘制多个算法的时延对比图。
主要功能：
- 自动发现和加载时延CSV文件
- 支持数据平滑处理（EMA或移动平均）
- 生成高质量的对比图表
- 支持保存为PDF和PNG格式
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def _load_latency_series(csv_path: PathLike) -> np.ndarray:
	"""
	从CSV文件中加载路径时延数据序列

	参数:
		csv_path: CSV文件路径

	返回:
		np.ndarray: 时延数据的numpy数组

	异常:
		ValueError: 如果时延数据为空
	"""
	# 读取CSV文件（时延文件有Latency和Arrival Time两列）
	df = pd.read_csv(csv_path)

	# 如果DataFrame为空，抛出异常
	if df.empty:
		raise ValueError(f"Latency data is empty in: {csv_path}")

	# 优先使用Latency列
	if "Latency" in df.columns:
		latency_col = "Latency"
	elif "latency" in df.columns:
		latency_col = "latency"
	else:
		# 如果没有找到Latency列，使用第一个数值列
		for col in df.columns:
			if pd.api.types.is_numeric_dtype(df[col]):
				latency_col = col
				break
		else:
			raise ValueError(f"No valid latency column found in: {csv_path}")

	# 转换为数值类型，无效值转为NaN，然后删除NaN值
	latency = pd.to_numeric(df[latency_col], errors="coerce").dropna().to_numpy()

	if latency.size == 0:
		raise ValueError(f"Latency data is empty in: {csv_path}")

	return latency


def _smooth_series(values: np.ndarray, window: int = 500, method: str = "ema") -> np.ndarray:
	"""
	对数据序列进行平滑处理

	参数:
		values: 原始数据序列
		window: 平滑窗口大小，默认500
		method: 平滑方法，可选"ema"（指数移动平均）或"ma"（简单移动平均）

	返回:
		np.ndarray: 平滑后的数据序列

	异常:
		ValueError: 如果method参数不是"ema"或"ma"
	"""
	series = pd.Series(values)

	if method.lower() == "ema":
		# 指数移动平均（EMA）：对近期数据赋予更高权重
		smooth = series.ewm(span=max(2, window), adjust=False).mean()
	elif method.lower() in {"ma", "moving", "moving_average"}:
		# 简单移动平均（MA）：窗口内所有数据权重相同
		smooth = series.rolling(window=max(2, window), min_periods=1).mean()
	else:
		raise ValueError("method must be one of: 'ema', 'ma'")

	return smooth.to_numpy()


def discover_latency_csvs(
	root_dir: PathLike,
	algorithms: Sequence[str],
	split: str = "train",
	gateways: Optional[int] = None,
) -> Dict[str, Path]:
	"""
	自动发现指定算法的时延CSV文件

	该函数会在root_dir下搜索每个算法的时延CSV文件。
	搜索策略：
	1. 如果指定了gateways，优先搜索精确匹配的文件（如pathLatencies_4_gateways.csv）
	2. 如果精确匹配失败，则搜索通配符模式（如pathLatencies*gateways.csv）
	3. 如果找到多个匹配文件，选择最新修改的文件

	参数:
		root_dir: 根目录路径
		algorithms: 算法名称列表
		split: 数据集划分，默认"train"（训练集）
		gateways: 网关数量，如果指定则优先搜索对应数量的文件

	返回:
		Dict[str, Path]: 算法名称到CSV文件路径的映射

	异常:
		FileNotFoundError: 如果根目录不存在
	"""
	root = Path(root_dir)
	if not root.exists():
		raise FileNotFoundError(f"Directory does not exist: {root}")

	discovered: Dict[str, Path] = {}

	# 遍历每个算法
	for algo in algorithms:
		algo_dir = root / algo
		if not algo_dir.exists():
			continue

		# 构建搜索模式列表
		patterns = []
		if gateways is not None:
			# 如果指定了网关数量，优先搜索精确匹配的文件
			patterns.append(f"**/{split}/csv/pathLatencies_{gateways}_gateways.csv")
		# 添加通配符模式作为备选
		patterns.append(f"**/{split}/csv/pathLatencies*gateways.csv")

		# 按优先级尝试每个模式
		matches: list[Path] = []
		for pattern in patterns:
			matches.extend(algo_dir.glob(pattern))
			if matches:
				break  # 找到匹配就停止搜索

		if not matches:
			continue

		# 如果找到多个文件，选择最新修改的
		matches = sorted(matches, key=lambda p: p.stat().st_mtime)
		discovered[algo] = matches[-1]

	return discovered


def plot_latency_convergence(
	csv_files: Mapping[str, PathLike],
	save_path: Optional[PathLike] = None,
	title: str = "Path Latency Convergence",
	smooth_window: int = 500,
	smooth_method: str = "ema",
	max_points: Optional[int] = 12000,
	show_raw: bool = True,
	dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
	"""
	绘制多个算法的路径时延对比图

	参数:
		csv_files: 算法名称到CSV文件路径的映射
		save_path: 保存路径（如果提供，会同时保存PDF和PNG格式）
		title: 图表标题
		smooth_window: 平滑窗口大小
		smooth_method: 平滑方法（"ema"或"ma"）
		max_points: 最大数据点数，超过则降采样
		show_raw: 是否显示原始数据（半透明）
		dpi: 图像分辨率

	返回:
		Tuple[plt.Figure, plt.Axes]: matplotlib的图形和坐标轴对象

	异常:
		ValueError: 如果csv_files为空
	"""
	if not csv_files:
		raise ValueError("csv_files is empty. Provide at least one csv path.")

	# 使用seaborn风格的白色网格背景
	plt.style.use("seaborn-whitegrid")
	# 创建图形和坐标轴，使用constrained_layout自动调整布局
	fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)

	# 获取tab10配色方案的颜色列表
	colors = plt.get_cmap("tab10").colors

	# 遍历每个算法的CSV文件
	for idx, (label, csv_path) in enumerate(csv_files.items()):
		try:
			# 加载时延数据
			latency = _load_latency_series(csv_path)
		except ValueError as e:
			# 如果数据为空，打印警告并跳过
			print(f"[WARN] Skipping {label}: {e}")
			continue

		# 如果数据点过多，进行降采样以提高绘图性能
		if max_points is not None and latency.size > max_points:
			step = int(np.ceil(latency.size / max_points))
			latency = latency[::step]

		# 对数据进行平滑处理
		smooth_latency = _smooth_series(latency, window=smooth_window, method=smooth_method)
		# 生成x轴坐标（数据包编号，从1开始）
		x = np.arange(1, len(latency) + 1)
		# 为每个算法分配颜色（循环使用配色方案）
		color = colors[idx % len(colors)]

		# 如果需要，绘制原始数据（半透明，细线）
		if show_raw:
			ax.plot(x, latency, color=color, alpha=0.16, linewidth=0.8)

		# 绘制平滑后的数据（不透明，粗线，带标签）
		ax.plot(x, smooth_latency, color=color, linewidth=2.2, label=label)

	# 设置图表标题和坐标轴标签
	ax.set_title(title, fontsize=13, fontweight="bold")
	ax.set_xlabel("Packet Index", fontsize=11)
	ax.set_ylabel("Latency (s)", fontsize=11)
	ax.tick_params(axis="both", labelsize=10)
	# 添加图例（无边框，自动选择最佳位置）
	ax.legend(frameon=False, fontsize=10, ncol=1, loc="best")

	# 隐藏顶部和右侧的边框
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	# 添加网格线
	ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	# 如果提供了保存路径，同时保存PDF和PNG格式
	if save_path is not None:
		output = Path(save_path)
		output.parent.mkdir(parents=True, exist_ok=True)

		# 保存为原始指定格式（通常是PDF）
		fig.savefig(output, dpi=dpi, bbox_inches="tight")

		# 同时保存为PNG格式
		png_path = output.with_suffix(".png")
		fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
		print(f"[INFO] Figure saved to: {output}")
		print(f"[INFO] PNG version saved to: {png_path}")

	return fig, ax


def plot_from_simresults(
	simresults_root: PathLike,
	algorithms: Sequence[str] = ("DDQN", "GAT", "MHGNN", "MPNN"),
	split: str = "train",
	gateways: Optional[int] = 4,
	save_path: Optional[PathLike] = None,
	title: str = "Path Latency Comparison",
	smooth_window: int = 500,
	smooth_method: str = "ema",
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Path]]:
	"""
	从SimResults目录自动发现CSV文件并绘制对比图

	这是一个便捷函数，整合了CSV文件发现和绘图功能。

	参数:
		simresults_root: SimResults根目录路径
		algorithms: 要对比的算法列表
		split: 数据集划分（"train"或"test"）
		gateways: 网关数量
		save_path: 保存路径（会同时保存PDF和PNG格式）
		title: 图表标题
		smooth_window: 平滑窗口大小
		smooth_method: 平滑方法

	返回:
		Tuple[plt.Figure, plt.Axes, Dict[str, Path]]:
			- matplotlib图形对象
			- matplotlib坐标轴对象
			- 实际使用的CSV文件映射

	异常:
		FileNotFoundError: 如果没有找到任何CSV文件
	"""
	# 自动发现各算法的CSV文件
	csv_map = discover_latency_csvs(
		root_dir=simresults_root,
		algorithms=algorithms,
		split=split,
		gateways=gateways,
	)

	# 检查是否有算法的CSV文件未找到
	missing = [algo for algo in algorithms if algo not in csv_map]
	if missing:
		print(f"[WARN] Latency csv not found for: {', '.join(missing)}")

	# 如果没有找到任何CSV文件，抛出异常
	if not csv_map:
		raise FileNotFoundError(
			f"No latency csv found under '{simresults_root}'. Check directory and pattern."
		)

	# 绘制对比图
	fig, ax = plot_latency_convergence(
		csv_files=csv_map,
		save_path=save_path,
		title=title,
		smooth_window=smooth_window,
		smooth_method=smooth_method,
	)
	return fig, ax, csv_map


if __name__ == "__main__":
	# ========== 主程序入口 ==========
	# 获取项目根目录（当前文件的上一级目录）
	repo_root = Path(__file__).resolve().parents[1]

	# 候选的结果目录列表
	candidate_roots = [
		# repo_root / "SimResults",  # 可选：通用结果目录
		repo_root / "best_simresult" / "4GTs",  # 4个网关的最佳结果
	]

	# 查找第一个存在的目录
	selected_root = None
	for root in candidate_roots:
		if root.exists():
			selected_root = root
			break

	if selected_root is None:
		raise FileNotFoundError("Cannot find SimResults or best_simresult/4GTs directory.")

	# 设置输出图片路径（PDF格式）
	figure_path = repo_root / "data_analysis" / "path_latency_comparison.pdf"

	# 绘制并保存对比图（会同时生成PDF和PNG）
	fig, _, used_csvs = plot_from_simresults(
		simresults_root=selected_root,
		algorithms=("DDQN", "GAT", "MHGNN", "MPNN"),  # 要对比的算法
		# split="train",  # 使用训练集数据
		split="test_teacher_network",  # 使用测试集数据
		gateways=4,  # 4个网关
		save_path=figure_path,
		title="Path Latency Comparison (4 Gateways)",
		smooth_window=500,  # 平滑窗口大小
		smooth_method="ema",  # 使用指数移动平均
	)

	# 打印使用的CSV文件信息
	print("[INFO] Loaded csv files:")
	for algo, path in used_csvs.items():
		print(f"  - {algo}: {path}")

	# 显示图形窗口
	plt.show()
