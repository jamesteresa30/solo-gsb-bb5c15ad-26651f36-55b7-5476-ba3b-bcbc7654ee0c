"""请求与响应模型（OpenAPI 由此生成）。"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Confidence = Literal["baseline", "high", "medium", "low"]


def _to_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


class ReadingIn(BaseModel):
    device_sn: str = Field(..., min_length=1, max_length=64, examples=["CASE-007"])
    ts: dt.datetime = Field(..., description="ISO8601 时间戳；无时区按 UTC 处理", examples=["2026-09-01T08:00:00+08:00"])
    temperature: float = Field(..., ge=-50.0, le=100.0, examples=[20.4])
    humidity: float = Field(..., ge=0.0, le=100.0, examples=[55.2])

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: dt.datetime) -> dt.datetime:
        return _to_utc(v)


class BatchIngest(BaseModel):
    readings: list[ReadingIn] = Field(..., min_length=1, max_length=10000)
    penalty: float | None = Field(None, ge=0.0, description="本次重放使用的惩罚项；缺省沿用展柜已保存值或默认值")
    replay: bool = Field(True, description="入库后是否立即重放受影响展柜的时间线")


class AnchorIn(BaseModel):
    device_sn: str = Field(..., min_length=1, max_length=64)
    ts: dt.datetime
    label: str | None = Field(None, max_length=128)
    ref_temperature: float | None = Field(None, ge=-50.0, le=100.0)
    ref_humidity: float | None = Field(None, ge=0.0, le=100.0)

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: dt.datetime) -> dt.datetime:
        return _to_utc(v)


class SegmentOut(BaseModel):
    index: int
    start_ts: dt.datetime
    end_ts: dt.datetime
    sample_count: int
    mean_temperature: float
    mean_humidity: float
    temperature_std: float
    humidity_std: float
    delta_temperature: float
    delta_humidity: float
    magnitude: float = Field(..., description="缩放加权后的欧氏变化幅度")
    confidence: Confidence
    start_anchored: bool
    end_anchored: bool
    cost: float


class AnchorOut(BaseModel):
    device_sn: str
    ts: dt.datetime
    label: str | None
    ref_temperature: float | None
    ref_humidity: float | None
    aligned: bool
    aligned_ts: dt.datetime | None = Field(None, description="实际对齐到的读数时间戳")


class SegmentationOut(BaseModel):
    device_sn: str
    version: int
    penalty: float
    sample_count: int
    anchor_count: int
    total_cost: float
    replayed: bool
    replayed_at: dt.datetime | None
    unmatched_anchors: list[dt.datetime]
    segments: list[SegmentOut]


class IngestSummary(BaseModel):
    device_sn: str
    inserted: int
    merged: int
    affected: bool = Field(..., description="该展柜是否被重放（存在新读数或合并）")
    before_version: int
    after_version: int
    replay: "SegmentationOut | None" = None


class IngestResponse(BaseModel):
    accepted: int
    inserted: int
    merged: int
    replayed_devices: list[str]
    summaries: list[IngestSummary]


class CaseOut(BaseModel):
    device_sn: str
    version: int
    penalty: float
    reading_count: int
    anchor_count: int
    replayed_at: dt.datetime | None


class ReplayPreviewResponse(BaseModel):
    device_sn: str
    current_version: int
    would_bump_version: bool
    changed: bool = Field(..., description="预览分段与当前已存储分段是否不同")
    penalty: float
    added_readings: int
    added_anchors: int
    affected_from_ts: dt.datetime | None
    affected_to_ts: dt.datetime | None
    preview: SegmentationOut
    current: "SegmentationOut | None" = None


TimelineReason = Literal[
    "range_start",
    "range_end",
    "temp_min",
    "temp_max",
    "humidity_min",
    "humidity_max",
    "aligned_anchor",
    "segment_boundary",
    "lttb",
]


class TimelinePointOut(BaseModel):
    ts: dt.datetime
    temperature: float
    humidity: float
    reasons: list[TimelineReason] = Field(
        ..., description="该点被保留的全部原因，按固定顺序排列（可多个）"
    )


class TimelineOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_points: int
    original_count: int = Field(..., description="过滤后、降采样前的读数数量")
    returned_count: int
    points: list[TimelinePointOut]


class CalibratedPointOut(BaseModel):
    ts: dt.datetime
    temperature: float = Field(..., description="原始温度读数（°C）")
    humidity: float = Field(..., description="原始湿度读数（%RH）")
    temperature_offset: float = Field(
        ..., description="该时刻温度偏移（参考值曲线插值结果，°C）"
    )
    humidity_offset: float = Field(
        ..., description="该时刻湿度偏移（参考值曲线插值结果，%RH）"
    )
    calibrated_temperature: float = Field(..., description="校正温度 = 原值 + 温度偏移")
    calibrated_humidity: float = Field(..., description="校正湿度 = 原值 + 湿度偏移")


class CalibratedTimelineOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    sample_count: int = Field(..., description="过滤后返回的读数数量")
    aligned_anchor_count: int = Field(
        ..., description="已对齐到读数的锚点数量（未必都参与某一维曲线）"
    )
    points: list[CalibratedPointOut]


ExcursionDimension = Literal["temperature", "humidity"]


class ExcursionEventOut(BaseModel):
    start_ts: dt.datetime
    end_ts: dt.datetime
    dimensions: list[ExcursionDimension] = Field(
        ..., description="触发该事件的维度（temperature 在前，可多个）"
    )
    duration_seconds: float = Field(..., description="事件持续秒数")
    temp_dose: float = Field(..., description="事件内温度越界剂量（°C·秒）")
    humidity_dose: float = Field(..., description="事件内湿度越界剂量（%RH·秒）")
    peak_temperature: float | None = Field(
        None, description="温度峰值：越界幅度最大处的温度（未触发温度时为 null）"
    )
    peak_temperature_ts: dt.datetime | None = None
    peak_humidity: float | None = Field(
        None, description="湿度峰值：越界幅度最大处的湿度（未触发湿度时为 null）"
    )
    peak_humidity_ts: dt.datetime | None = None


class ExcursionsOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_gap_seconds: float = Field(..., description="生效的最大采样间隔（秒）")
    temp_min: float | None = None
    temp_max: float | None = None
    humidity_min: float | None = None
    humidity_max: float | None = None
    sample_count: int = Field(..., description="过滤后参与计算的读数数量")
    event_count: int
    events: list[ExcursionEventOut]


class ExposureBudgetDimensionOut(BaseModel):
    dimension: ExcursionDimension
    budget: float = Field(
        ..., description="该维越界剂量预算（温度 °C·秒 / 湿度 %RH·秒），非负"
    )
    cumulative_dose: float = Field(
        ..., description="查询窗末累计越界剂量（断档区间不计入）"
    )
    remaining: float = Field(
        ..., description="剩余额度（budget − cumulative_dose；超出预算时为负）"
    )
    budget_exhausted: bool = Field(
        ..., description="窗内累计剂量是否已达到预算"
    )
    exhausted_ts: dt.datetime | None = Field(
        None,
        description=(
            "首次达到预算的精确时刻（在线段内按越界幅度线性变化反解）；"
            "零预算在该维首次越界起点耗尽；窗末仍未达到预算为 null"
        ),
    )


class ExposureBudgetOut(BaseModel):
    device_sn: str
    start: dt.datetime | None = Field(None, description="实际生效的起始过滤（UTC），未指定为 null")
    end: dt.datetime | None = Field(None, description="实际生效的结束过滤（UTC），未指定为 null")
    max_gap_seconds: float = Field(..., description="生效的最大采样间隔（秒）")
    temp_min: float | None = None
    temp_max: float | None = None
    humidity_min: float | None = None
    humidity_max: float | None = None
    temp_budget: float | None = Field(None, description="温度剂量预算（未配置该维为 null）")
    humidity_budget: float | None = Field(None, description="湿度剂量预算（未配置该维为 null）")
    sample_count: int = Field(..., description="过滤后参与计算的读数数量")
    dimensions: list[ExposureBudgetDimensionOut] = Field(
        ..., description="各维累计剂量与预算状态（temperature 在前）"
    )


IncidentDirection = Literal["up", "down", "flat"]


PropagationDirection = Literal[
    "source_to_target", "target_to_source", "synchronous"
]


class PropagationLagOut(BaseModel):
    lag_seconds: float = Field(
        ...,
        description=(
            "时滞（秒）；正值表示 source 领先 target，即 source(t) 与 "
            "target(t+lag) 配对"
        ),
    )
    sample_count: int = Field(
        ..., description="该时滞下两侧都有效的成对数（仅这些点参与相关）"
    )
    available: bool = Field(
        ..., description="相关系数是否可用（样本不足或常量序列时为 false）"
    )
    correlation: float | None = Field(
        None,
        description="皮尔逊相关系数 [−1, 1]；不可用时为 null（不产生 NaN）",
    )


class PropagationOut(BaseModel):
    source_sn: str
    target_sn: str
    dimension: ExcursionDimension = Field(
        ..., description="分析维度：temperature 或 humidity"
    )
    start: dt.datetime | None = Field(
        None, description="实际生效的起始过滤（UTC），未指定为 null"
    )
    end: dt.datetime | None = Field(
        None, description="实际生效的结束过滤（UTC），未指定为 null"
    )
    step_seconds: float = Field(..., description="规则网格步长（秒）")
    max_lag_seconds: float = Field(..., description="枚举时滞的最大绝对值（秒，含端点）")
    max_gap_seconds: float = Field(
        ..., description="相邻读数间隔超过该值（秒）时插值点无效"
    )
    overlap_start: dt.datetime | None = Field(
        None, description="两柜读数时间范围（与查询窗交集）的重叠起点"
    )
    overlap_end: dt.datetime | None = Field(
        None, description="两柜读数时间范围（与查询窗交集）的重叠终点"
    )
    grid_start: dt.datetime | None = Field(
        None, description="首个网格点时刻（重叠起点 ceil 对齐到步长）；无网格为 null"
    )
    grid_end: dt.datetime | None = Field(None, description="末个网格点时刻；无网格为 null")
    grid_point_count: int = Field(..., description="重叠时段上的网格点总数")
    source_valid_count: int = Field(..., description="source 侧有效的网格点数")
    target_valid_count: int = Field(..., description="target 侧有效的网格点数")
    available: bool = Field(
        ..., description="是否至少有一个时滞得到可用相关系数"
    )
    best_lag_seconds: float | None = Field(
        None,
        description=(
            "最佳时滞（秒）；按 |相关| → 有效样本数 → 时滞绝对值 → 时滞值"
            " 依次择优；全部不可用时为 null"
        ),
    )
    direction: PropagationDirection | None = Field(
        None,
        description=(
            "传播方向：正时滞 source_to_target（source 领先），"
            "负时滞 target_to_source，零时滞 synchronous；不可用时为 null"
        ),
    )
    best_correlation: float | None = Field(
        None, description="最佳时滞对应的皮尔逊相关系数；不可用时为 null"
    )
    lags: list[PropagationLagOut] = Field(
        ..., description="各时滞明细，按时滞值升序（负 → 正）排列"
    )


class IncidentBoundaryOut(BaseModel):
    device_sn: str
    ts: dt.datetime = Field(..., description="该展柜非基线段段首（突变发生时刻）")
    direction: IncidentDirection = Field(
        ..., description="边界方向：以占主导的温/湿跳变分量符号判定"
    )
    magnitude: float = Field(..., description="该边界的缩放加权变化幅度")
    delta_temperature: float
    delta_humidity: float


class IncidentOut(BaseModel):
    start_ts: dt.datetime = Field(..., description="事件区间起点（成员中最早的边界）")
    end_ts: dt.datetime = Field(..., description="事件区间终点（成员中最晚的边界）")
    duration_seconds: float = Field(..., description="事件跨度（end_ts - start_ts，秒）")
    device_count: int = Field(..., description="参与该事件的 distinct 展柜数")
    total_magnitude: float = Field(..., description="各展柜边界幅度之和")
    boundaries: list[IncidentBoundaryOut] = Field(
        ..., description="各展柜保留的边界（每个展柜一条），按 (ts, device_sn) 排序"
    )


class IncidentsOut(BaseModel):
    start: dt.datetime = Field(..., description="实际生效的起始过滤（UTC）")
    end: dt.datetime = Field(..., description="实际生效的结束过滤（UTC）")
    tolerance_seconds: float = Field(..., description="候选窗允许的最大边界跨度（秒）")
    min_devices: int = Field(..., description="合法候选要求的最少 distinct 展柜数")
    candidate_count: int = Field(
        ..., description="枚举得到的全部合法连续候选窗数量（去重后）"
    )
    boundary_count: int = Field(..., description="过滤后参与检测的非基线段首数量")
    incident_count: int
    incidents: list[IncidentOut]


class HealthOut(BaseModel):
    status: str
    db: str


IngestSummary.model_rebuild()
ReplayPreviewResponse.model_rebuild()
