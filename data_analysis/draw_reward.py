"""
奖励收敛曲线绘制工具

该模块用于从CSV文件中读取训练奖励数据，并绘制多个算法的奖励收敛对比图。
主要功能：
- 自动发现和加载奖励CSV文件
- 支持数据平滑处理（EMA或移动平均）
- 生成高质量的对比图表
- 支持保存为PDF和PNG格式
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def _pick_reward_column(df: pd.DataFrame) -> str:
	"""
	从DataFrame中自动选择奖励列

	优先级：
	1. 首先尝试常见的奖励列名称（如"Smoothed Rewards", "Rewards"等）
	2. 如果没有找到，则选择第一个包含有效数值的数值型列

	参数:
		df: 包含奖励数据的DataFrame

	返回:
		str: 选中的奖励列名称

	异常:
		ValueError: 如果没有找到有效的数值型奖励列
	"""
	# 候选的奖励列名称列表，按优先级排序
	candidates = [
		"Smoothed Rewards",
		"Rewards",
		"reward",
		"rewards",
		"Reward",
	]

	# 首先尝试从候选列表中查找
	for column in candidates:
		if column in df.columns and df[column].notna().any():
			return column

	# 如果候选列表中没有找到，则查找第一个有效的数值型列
	for column in df.columns:
		if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().any():
			return column

	raise ValueError("No valid numeric reward column was found in csv file.")


def _pick_time_column(df: pd.DataFrame) -> Optional[str]:
	"""
	从DataFrame中自动选择时间列

	优先查找常见的时间列名（如"Time"、"time"等），
	如果未找到则返回None。
	"""
	for column in ["Time", "time", "timestamp", "Timestamp"]:
		if column in df.columns:
			return column
	return None


def _load_time_reward_series(csv_path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
	"""
	从CSV文件中加载时间与奖励数据序列（按有效奖励对齐）

	参数:
		csv_path: CSV文件路径

	返回:
		Tuple[np.ndarray, np.ndarray]: (时间序列, 奖励序列)

	异常:
		ValueError: 如果奖励数据为空
	"""
	# 读取CSV文件
	df = pd.read_csv(csv_path)
	# 自动选择奖励列
	reward_col = _pick_reward_column(df)
	time_col = _pick_time_column(df)

	# 转换为数值类型
	reward_num = pd.to_numeric(df[reward_col], errors="coerce")

	# 以reward有效性为基准对齐时间与奖励
	valid_mask = reward_num.notna()
	reward = reward_num[valid_mask].to_numpy()

	if reward.size == 0:
		raise ValueError(f"Reward data is empty in: {csv_path}")

	if time_col is not None:
		time_num = pd.to_numeric(df[time_col], errors="coerce")
		time_num = time_num[valid_mask]
		# 如果时间列存在但全部无效，则回退为索引
		if time_num.notna().any():
			time_values = time_num.fillna(method="ffill").fillna(method="bfill").to_numpy()
		else:
			time_values = np.arange(1, reward.size + 1)
	else:
		# 未找到时间列时回退为索引（兼容旧CSV）
		time_values = np.arange(1, reward.size + 1)

	return time_values, reward


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


def discover_reward_csvs(
	root_dir: PathLike,
	algorithms: Sequence[str],
	split: str = "train",
	gateways: Optional[int] = None,
) -> Dict[str, Path]:
	"""
	自动发现指定算法的奖励CSV文件

	该函数会在root_dir下搜索每个算法的奖励CSV文件。
	搜索策略：
	1. 如果指定了gateways，优先搜索精确匹配的文件（如rewards_4_gateways.csv）
	2. 如果精确匹配失败，则搜索通配符模式（如rewards*_gateways.csv）
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
			patterns.append(f"**/{split}/csv/rewards_{gateways}_gateways.csv")
		# 添加通配符模式作为备选
		patterns.append(f"**/{split}/csv/rewards*_gateways.csv")

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


def plot_reward_convergence(
	csv_files: Mapping[str, PathLike],
	save_path: Optional[PathLike] = None,
	title: str = "Reward Convergence",
	smooth_window: int = 250,
	smooth_method: str = "ema",
	max_points: Optional[int] = 12000,
	show_raw: bool = True,
	dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
	"""
	绘制多个算法的奖励收敛对比图

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
			# 加载时间与奖励数据
			x, reward = _load_time_reward_series(csv_path)
		except ValueError as e:
			# 如果数据为空，打印警告并跳过
			print(f"[WARN] Skipping {label}: {e}")
			continue

		# 如果数据点过多，进行降采样以提高绘图性能
		if max_points is not None and reward.size > max_points:
			step = int(np.ceil(reward.size / max_points))
			x = x[::step]
			reward = reward[::step]

		# 对数据进行平滑处理
		smooth_reward = _smooth_series(reward, window=smooth_window, method=smooth_method)
		# 为每个算法分配颜色（循环使用配色方案）
		color = colors[idx % len(colors)]

		# 如果需要，绘制原始数据（半透明，细线）
		if show_raw:
			ax.plot(x, reward, color=color, alpha=0.16, linewidth=0.8)

		# 绘制平滑后的数据（不透明，粗线，带标签）
		ax.plot(x, smooth_reward, color=color, linewidth=2.2, label=label)

	# 设置图表标题和坐标轴标签
	ax.set_title(title, fontsize=13, fontweight="bold")
	ax.set_xlabel("Episode", fontsize=11)
	ax.set_ylabel("Reward", fontsize=11)
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
	title: str = "Reward Convergence Comparison",
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
	csv_map = discover_reward_csvs(
		root_dir=simresults_root,
		algorithms=algorithms,
		split=split,
		gateways=gateways,
	)

	# 检查是否有算法的CSV文件未找到
	missing = [algo for algo in algorithms if algo not in csv_map]
	if missing:
		print(f"[WARN] Reward csv not found for: {', '.join(missing)}")

	# 如果没有找到任何CSV文件，抛出异常
	if not csv_map:
		raise FileNotFoundError(
			f"No reward csv found under '{simresults_root}'. Check directory and pattern."
		)

	# 绘制对比图
	fig, ax = plot_reward_convergence(
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
		repo_root / "SimBestModel" / "4GTs",  # 4个网关的最佳结果
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
	figure_path = repo_root / "data_analysis" / "reward_convergence_comparison.pdf"

	# 绘制并保存对比图（会同时生成PDF和PNG）
	fig, _, used_csvs = plot_from_simresults(
		simresults_root=selected_root,
		algorithms=("DDQN", "GAT", "MHGNN", "MPNN"),  # 要对比的算法
		split="train",  # 使用训练集数据
		gateways=4,  # 4个网关
		save_path=figure_path,
		title="Reward Convergence (4 Gateways)",
		smooth_window=500,  # 平滑窗口大小
		smooth_method="ema",  # 使用指数移动平均
	)

	# 打印使用的CSV文件信息
	print("[INFO] Loaded csv files:")
	for algo, path in used_csvs.items():
		print(f"  - {algo}: {path}")

	# 显示图形窗口
	plt.show()
