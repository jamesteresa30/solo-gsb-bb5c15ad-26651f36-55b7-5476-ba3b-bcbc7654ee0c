"""跨展柜环境传播分析（propagation）。

把两个展柜（source / target）在**重叠时段**内的同一维度读数，按
固定步长 ``step_seconds`` 重采样为规则网格，并在枚举的时滞
``lag ∈ [-max_lag_seconds, +max_lag_seconds]``（步长的整数倍）上
计算**皮尔逊相关系数**，推断环境变化由哪个展柜先发生（传播方向）。

网格插值
--------

- 网格起点取两柜读数时间范围交集的起点（ceil 对齐到步长），网格点
  为该起点起每 ``step_seconds`` 一个、不超出交集终点的时刻；
- 网格点恰好落在读数时刻上时取该读数；落在两读数之间时按时间
  **线性插值**，不做外推（交点之外的网格点无效）；
- 被插值的两个相邻读数间隔**大于** ``max_gap_seconds`` 时，该网格
  点所在区间曲线断开、该点无效（恰好等于 ``max_gap_seconds`` 仍视为
  连续，与越界剂量接口的断档规则一致）；读数时刻本身不受两侧间隔
  影响，始终有效。

时滞与皮尔逊相关
----------------

- 时滞单位为秒，枚举范围为 ``[-max_lag_seconds, +max_lag_seconds]``
  内 ``step_seconds`` 的整数倍（含端点）；
- **正时滞表示 source 领先 target**：配对为
  ``source(t)`` 与 ``target(t + lag)``——target 的响应滞后 lag 秒；
- 每个时滞只用两侧**都有效**的网格点配对计算；配对数少于 2，或任
  一序列为常量（标准差为 0）时相关系数不可用（``None``），绝不产出
  ``NaN``。

择优规则（依次）：相关绝对值更大 → 有效样本数更多 → 时滞绝对值更
小 → 时滞值本身更小（负时滞优先，保证严格确定）。
"""

from __future__ import annotations

import datetime as dt
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"

DIRECTION_SOURCE_TO_TARGET = "source_to_target"
DIRECTION_TARGET_TO_SOURCE = "target_to_source"
DIRECTION_SYNCHRONOUS = "synchronous"


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class LagCorrelation:
    """单个时滞下的皮尔逊相关结果。"""

    lag_seconds: float
    sample_count: int = 0
    correlation: float | None = None
    available: bool = False


@dataclass(frozen=True)
class PropagationResult:
    """全部时滞的相关明细与择优结论。"""

    source_sn: str
    target_sn: str
    dimension: str
    overlap_start: dt.datetime | None
    overlap_end: dt.datetime | None
    grid_start: dt.datetime | None
    grid_end: dt.datetime | None
    step_seconds: float
    max_lag_seconds: float
    max_gap_seconds: float
    grid_point_count: int
    source_valid_count: int
    target_valid_count: int
    lags: tuple[LagCorrelation, ...]
    best_lag_seconds: float | None
    direction: str | None
    best_correlation: float | None
    available: bool


def _dimension_value(sample: _SampleLike, dimension: str) -> float:
    if dimension == TEMPERATURE:
        return sample.temperature
    if dimension == HUMIDITY:
        return sample.humidity
    raise ValueError(f"未知维度: {dimension}")  # pragma: no cover - 接口层已校验


def interpolate_grid(
    samples: list[_SampleLike],
    dimension: str,
    grid: list[dt.datetime],
    max_gap_seconds: float,
) -> list[float | None]:
    """把单个展柜的读数重采样到规则网格（不做外推）。

    samples 无需预先排序；返回与 ``grid`` 等长的序列，无法取值（交点
    之外）或跨断档（相邻读数间隔大于 ``max_gap_seconds``）的位置为
    ``None``。
    """
    ordered = sorted(samples, key=lambda s: s.ts)
    times = [s.ts for s in ordered]
    values = [_dimension_value(s, dimension) for s in ordered]
    n = len(times)
    out: list[float | None] = []

    for ts in grid:
        idx = bisect_left(times, ts)
        if idx < n and times[idx] == ts:
            out.append(values[idx])  # 恰好落在读数时刻：始终有效
            continue
        if idx == 0 or idx == n:
            out.append(None)  # 交点之外：不外推
            continue
        t0, t1 = times[idx - 1], times[idx]
        if (t1 - t0).total_seconds() > max_gap_seconds:
            out.append(None)  # 跨越断档：该点未知
            continue
        frac = (ts - t0).total_seconds() / (t1 - t0).total_seconds()
        out.append(values[idx - 1] + (values[idx] - values[idx - 1]) * frac)
    return out


def build_grid(
    start: dt.datetime,
    end: dt.datetime,
    step_seconds: float,
) -> list[dt.datetime]:
    """``[start, end]`` 上以 ``start`` 为起点、``step_seconds`` 为步长的
    规则网格；``end < start`` 时为空。"""
    span = (end - start).total_seconds()
    if span < 0.0:
        return []
    # 浮点兜底：span 恰为步长整数倍时不丢末点
    count = int(math.floor(span / step_seconds + 1e-9))
    return [
        start + dt.timedelta(seconds=step_seconds * k)
        for k in range(count + 1)
    ]


def overlap_grid_window(
    source_times: list[dt.datetime],
    target_times: list[dt.datetime],
    start: dt.datetime | None,
    end: dt.datetime | None,
    step_seconds: float,
) -> tuple[dt.datetime | None, dt.datetime | None, list[dt.datetime]]:
    """两柜读数时间范围（再与查询窗取交集）上的重叠网格。

    网格起点把交集起点 **ceil 对齐**到步长（起点本身在网格上时保持不
    变），保证首个网格点不早于交集起点。交集为空时网格为空。
    """
    lo = max(source_times[0], target_times[0])
    hi = min(source_times[-1], target_times[-1])
    if start is not None:
        lo = max(lo, start)
    if end is not None:
        hi = min(hi, end)
    if lo > hi:
        return None, None, []

    origin = source_times[0]
    offset = (lo - origin).total_seconds()
    aligned = math.ceil(offset / step_seconds - 1e-9) * step_seconds
    grid_start = origin + dt.timedelta(seconds=aligned)
    if grid_start < lo:  # 浮点兜底
        grid_start += dt.timedelta(seconds=step_seconds)
    if grid_start > hi:
        return lo, hi, []
    return lo, hi, build_grid(grid_start, hi, step_seconds)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """皮尔逊相关系数；样本不足或常量序列返回 ``None``（绝不返回 NaN）。"""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None  # 常量序列（或数值退化）：相关无定义
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy)
    if not math.isfinite(r):
        return None  # 防御：任何非有限结果一律标为不可用
    return min(1.0, max(-1.0, r))


def _lag_rank_key(item: LagCorrelation) -> tuple:
    """择优排序键（越大越优）：|r| → 样本数 → −|lag| → −lag。"""
    assert item.correlation is not None and item.available
    return (
        abs(item.correlation),
        item.sample_count,
        -abs(item.lag_seconds),
        -item.lag_seconds,
    )


def lag_direction(lag_seconds: float) -> str:
    if lag_seconds > 0:
        return DIRECTION_SOURCE_TO_TARGET
    if lag_seconds < 0:
        return DIRECTION_TARGET_TO_SOURCE
    return DIRECTION_SYNCHRONOUS


def analyze_propagation(
    source_samples: list[_SampleLike],
    target_samples: list[_SampleLike],
    *,
    source_sn: str,
    target_sn: str,
    dimension: str,
    start: dt.datetime | None,
    end: dt.datetime | None,
    step_seconds: float,
    max_lag_seconds: float,
    max_gap_seconds: float,
) -> PropagationResult:
    """两柜同维读数在重叠时段上的时滞相关分析（纯计算，确定性）。"""
    source_ordered = sorted(source_samples, key=lambda s: s.ts)
    target_ordered = sorted(target_samples, key=lambda s: s.ts)

    overlap_start = overlap_end = grid_start = grid_end = None
    lags: list[LagCorrelation] = []
    if source_ordered and target_ordered:
        overlap_start, overlap_end, grid = overlap_grid_window(
            [s.ts for s in source_ordered],
            [s.ts for s in target_ordered],
            start,
            end,
            step_seconds,
        )
        source_grid = interpolate_grid(
            source_ordered, dimension, grid, max_gap_seconds
        )
        target_grid = interpolate_grid(
            target_ordered, dimension, grid, max_gap_seconds
        )
        if grid:
            grid_start, grid_end = grid[0], grid[-1]
    else:  # pragma: no cover - 服务层已保证两柜都有读数
        grid = source_grid = target_grid = []

    source_valid = [v is not None for v in source_grid]
    target_valid = [v is not None for v in target_grid]

    max_steps = int(math.floor(max_lag_seconds / step_seconds + 1e-9))
    n = len(grid)
    for k in range(-max_steps, max_steps + 1):
        lag_seconds = step_seconds * k
        # 正时滞 source 领先：source[i] 对 target[i+k]
        lo = max(0, -k)
        hi = min(n, n - k)
        xs: list[float] = []
        ys: list[float] = []
        for i in range(lo, hi):
            if source_valid[i] and target_valid[i + k]:
                xs.append(source_grid[i])  # type: ignore[arg-type]
                ys.append(target_grid[i + k])  # type: ignore[arg-type]
        r = pearson_correlation(xs, ys)
        lags.append(
            LagCorrelation(
                lag_seconds=lag_seconds,
                sample_count=len(xs),
                correlation=r,
                available=r is not None,
            )
        )

    usable = [item for item in lags if item.available]
    if usable:
        best = max(usable, key=_lag_rank_key)
        best_lag = best.lag_seconds
        best_correlation = best.correlation
        direction = lag_direction(best_lag)
    else:
        best_lag = best_correlation = direction = None

    return PropagationResult(
        source_sn=source_sn,
        target_sn=target_sn,
        dimension=dimension,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        grid_start=grid_start,
        grid_end=grid_end,
        step_seconds=step_seconds,
        max_lag_seconds=max_lag_seconds,
        max_gap_seconds=max_gap_seconds,
        grid_point_count=n,
        source_valid_count=sum(source_valid),
        target_valid_count=sum(target_valid),
        lags=tuple(lags),
        best_lag_seconds=best_lag,
        direction=direction,
        best_correlation=best_correlation,
        available=bool(usable),
    )
