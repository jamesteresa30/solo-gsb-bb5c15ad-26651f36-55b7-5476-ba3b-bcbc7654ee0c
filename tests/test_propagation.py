"""跨展柜环境传播分析：重叠时段固定步长插值、断档点无效、时滞皮尔逊
相关与择优规则。

纯算法测试覆盖：正时滞（source 领先）/负时滞检出、恰好 max_gap 连续、
跨断档插值点无效、常量/样本不足不可用且无 NaN、择优并列规则、ceil
对齐网格与查询窗子集；接口集成测试覆盖 422 参数冲突组合、404、
无重叠空结果与响应结构。
"""

from __future__ import annotations

import datetime as dt
import math
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine, init_db
from app.main import app
from app.propagation import (
    DIRECTION_SOURCE_TO_TARGET,
    DIRECTION_SYNCHRONOUS,
    DIRECTION_TARGET_TO_SOURCE,
    analyze_propagation,
    interpolate_grid,
    pearson_correlation,
)
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _sample(minute: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minute), t, h)


def _reading(sn, minute, t, h=50.0):
    return {
        "device_sn": sn,
        "ts": (BASE + dt.timedelta(minutes=minute)).isoformat(),
        "temperature": t,
        "humidity": h,
    }


# 非周期、确定性的伪信号（i 与 2.3·i 的交错，避免周期信号在多个时滞上
# 产生完全相关）
VALUES = [
    0.0, 1.0, 2.3, 2.0, 4.6, 4.0, 6.9, 6.0, 9.2, 8.0,
    11.5, 10.0, 13.8, 12.0, 16.1, 14.0, 18.4, 16.0, 20.7, 18.0,
]


def _series(values, *, minute0=0.0, step_min=1.0, t0=20.0):
    return [
        Sample(
            BASE + dt.timedelta(minutes=minute0 + step_min * i),
            t0 + v,
            50.0,
        )
        for i, v in enumerate(values)
    ]


def _analyze(src, tgt, **kw):
    kw.setdefault("step_seconds", 60.0)
    kw.setdefault("max_lag_seconds", 300.0)
    kw.setdefault("max_gap_seconds", 600.0)
    return analyze_propagation(
        src,
        tgt,
        source_sn="A",
        target_sn="B",
        dimension="temperature",
        start=None,
        end=None,
        **kw,
    )


# ---------- 纯算法：插值与断档 ----------


def test_interpolate_grid_linear_within_brackets():
    samples = [_sample(0, 20.0), _sample(2, 24.0)]  # 2 分钟间隔
    grid = [BASE + dt.timedelta(seconds=30 * i) for i in range(5)]  # 0..2min
    out = interpolate_grid(samples, "temperature", grid, 600.0)
    assert out[0] == pytest.approx(20.0)
    assert out[1] == pytest.approx(21.0)
    assert out[2] == pytest.approx(22.0)
    assert out[3] == pytest.approx(23.0)
    assert out[4] == pytest.approx(24.0)


def test_interpolate_no_extrapolation():
    samples = [_sample(1, 20.0), _sample(2, 22.0)]
    grid = [BASE + dt.timedelta(seconds=30 * i) for i in range(7)]  # 0..3min
    out = interpolate_grid(samples, "temperature", grid, 600.0)
    assert out[0] is None  # 首读数之前（0s）
    assert out[1] is None  # 30s 仍在首读数（60s）之前
    assert out[2] == 20.0  # 恰好首读数
    assert out[4] == 22.0  # 恰好末读数
    assert out[5] is None  # 末读数之后
    assert out[6] is None  # 末读数之后


def test_interpolate_gap_greater_than_max_gap_invalidates_interior():
    # 间隔 11 分钟 > max_gap=600s：中间插值点全部无效，读数时刻仍有效
    samples = [_sample(0, 20.0), _sample(11, 30.0)]
    grid = [BASE + dt.timedelta(seconds=60 * i) for i in range(12)]
    out = interpolate_grid(samples, "temperature", grid, 600.0)
    assert out[0] == 20.0
    assert all(v is None for v in out[1:-1])
    assert out[-1] == 30.0


def test_interpolate_gap_equal_max_gap_still_continuous():
    # 恰好 600s：仍连续（含等于规则）
    samples = [_sample(0, 20.0), _sample(10, 30.0)]
    grid = [BASE + dt.timedelta(seconds=60 * i) for i in range(11)]
    out = interpolate_grid(samples, "temperature", grid, 600.0)
    assert all(v is not None for v in out)
    assert out[5] == pytest.approx(25.0)


def test_interpolate_exact_knot_always_valid_even_next_to_gap():
    # 读数时刻不受两侧间隔影响
    samples = [
        _sample(0, 20.0),
        _sample(1, 21.0),
        _sample(20, 22.0),  # 与前一点间隔 19 分钟
    ]
    grid = [BASE + dt.timedelta(seconds=60)]
    out = interpolate_grid(samples, "temperature", grid, 600.0)
    assert out == [21.0]


# ---------- 纯算法：皮尔逊相关 ----------


def test_pearson_basic():
    assert pearson_correlation([0, 1, 2], [0, 2, 4]) == pytest.approx(1.0)
    assert pearson_correlation([0, 1, 2], [4, 2, 0]) == pytest.approx(-1.0)


def test_pearson_constant_and_short_series_is_none():
    assert pearson_correlation([1, 1, 1], [0, 1, 2]) is None
    assert pearson_correlation([0, 1, 2], [3, 3, 3]) is None
    assert pearson_correlation([1], [2]) is None
    assert pearson_correlation([], []) is None
    # 绝不为 NaN
    r = pearson_correlation([1, 1, 1], [1, 1, 1])
    assert r is None or math.isfinite(r)


# ---------- 纯算法：时滞检出 ----------


def test_positive_lag_source_leads_target():
    src = _series(VALUES)
    # target(t) = source(t-2min)：target 落后 2 分钟；边缘用首值填充
    tgt = _series([VALUES[0], VALUES[0]] + VALUES[:-2])
    r = _analyze(src, tgt)
    assert r.best_lag_seconds == pytest.approx(120.0)
    assert r.direction == DIRECTION_SOURCE_TO_TARGET
    assert r.best_correlation == pytest.approx(1.0)
    # 明细按时滞升序、范围含端点
    lags = [l.lag_seconds for l in r.lags]
    assert lags == sorted(lags)
    assert lags[0] == -300.0 and lags[-1] == 300.0
    assert all(l.available for l in r.lags)
    best_detail = next(l for l in r.lags if l.lag_seconds == 120.0)
    assert best_detail.sample_count == 18


def test_negative_lag_target_leads_source():
    # target(t) = source(t+2min)：target 领先 2 分钟 → 时滞为负
    src = _series([VALUES[0], VALUES[0]] + VALUES[:-2])
    tgt = _series(VALUES)
    r = _analyze(src, tgt)
    assert r.best_lag_seconds == pytest.approx(-120.0)
    assert r.direction == DIRECTION_TARGET_TO_SOURCE
    assert r.best_correlation == pytest.approx(1.0)


def test_synchronous_zero_lag():
    r = _analyze(_series(VALUES), _series(VALUES))
    assert r.best_lag_seconds == 0.0
    assert r.direction == DIRECTION_SYNCHRONOUS
    assert r.best_correlation == pytest.approx(1.0)


def test_constant_target_all_lags_unavailable_no_nan():
    src = _series(VALUES)
    tgt = _series([0.0] * len(VALUES))
    r = _analyze(src, tgt)
    assert r.available is False
    assert r.best_lag_seconds is None
    assert r.direction is None
    assert r.best_correlation is None
    assert all(not l.available for l in r.lags)
    assert all(l.correlation is None for l in r.lags)
    assert all(l.sample_count >= 0 for l in r.lags)


def test_short_overlap_fewer_than_two_pairs_unavailable():
    # 两柜只在 1 分钟处相切：每个时滞最多 1 个配对
    src = _series(VALUES[:5], minute0=0.0)
    tgt = _series(VALUES[:5], minute0=4.0)
    r = _analyze(src, tgt, step_seconds=60.0, max_lag_seconds=120.0)
    assert r.grid_point_count == 1
    assert r.available is False
    assert all(l.correlation is None for l in r.lags)
    # 零时滞恰好有 1 个配对（仍不足）
    zero = next(l for l in r.lags if l.lag_seconds == 0.0)
    assert zero.sample_count == 1


def test_gap_invalidates_interior_grid_points():
    src = _series(VALUES[:8]) + _series(VALUES[19:20], minute0=19.0)
    tgt = _series(VALUES)
    r = _analyze(src, tgt, max_lag_seconds=60.0)
    # source 有效点：0..7 与 19 共 9 个读数时刻；8..18 为跨断档插值点
    assert r.source_valid_count == 9
    assert r.target_valid_count == 20
    zero = next(l for l in r.lags if l.lag_seconds == 0.0)
    assert zero.sample_count == 9
    # 只用成对有效点：相关有限且不为 NaN
    assert zero.correlation is not None and math.isfinite(zero.correlation)


def test_pairing_drops_points_invalid_on_either_side():
    src = _series(VALUES[:8]) + _series(VALUES[19:20], minute0=19.0)
    tgt = _series(VALUES)
    r = _analyze(src, tgt, max_lag_seconds=60.0)
    # source 有效网格点：i ∈ {0..7, 19}
    # 时滞 +60（k=1）：i ∈ [0..18] 且源有效 → 仅 i=0..7，共 8 对
    plus = next(l for l in r.lags if l.lag_seconds == 60.0)
    assert plus.sample_count == 8
    # 时滞 −60（k=−1）：source[i] 对 target[i−1]，i ∈ [1..19] 且源有效
    # → i ∈ {1..7, 19}，共 8 对
    minus = next(l for l in r.lags if l.lag_seconds == -60.0)
    assert minus.sample_count == 8


def test_tie_prefers_smaller_abs_lag_then_negative_value():
    from app.propagation import LagCorrelation, _lag_rank_key

    # 完全相同序列：零时滞 |r|=1 且 |lag| 最小
    r = _analyze(_series(VALUES), _series(VALUES))
    assert r.best_lag_seconds == 0.0

    # 直接验证择优排序键：|r| → 样本数 → −|lag| → −lag
    neg = LagCorrelation(lag_seconds=-60.0, sample_count=10, correlation=0.8, available=True)
    pos = LagCorrelation(lag_seconds=60.0, sample_count=10, correlation=0.8, available=True)
    assert _lag_rank_key(neg) > _lag_rank_key(pos)  # ±lag 并列：负时滞优先

    small = LagCorrelation(lag_seconds=120.0, sample_count=10, correlation=0.8, available=True)
    assert _lag_rank_key(pos) > _lag_rank_key(small)  # 更小时滞绝对值优先

    more = LagCorrelation(lag_seconds=120.0, sample_count=12, correlation=0.8, available=True)
    assert _lag_rank_key(more) > _lag_rank_key(pos)  # 更多有效样本优先

    stronger = LagCorrelation(lag_seconds=240.0, sample_count=10, correlation=0.9, available=True)
    assert _lag_rank_key(stronger) > _lag_rank_key(pos)  # 更大 |相关| 优先


def test_grid_ceil_aligned_to_step():
    src = _series(VALUES)
    # target 读数整体偏移 30 秒：重叠起点 00:00:30，ceil 对齐到 1 分钟网格
    tgt = [
        Sample(BASE + dt.timedelta(seconds=60 * i + 30), 20.0 + VALUES[i], 50.0)
        for i in range(20)
    ]
    r = _analyze(src, tgt, max_lag_seconds=60.0)
    assert r.overlap_start == BASE + dt.timedelta(seconds=30)
    assert r.grid_start == BASE + dt.timedelta(seconds=60)
    assert r.grid_end == BASE + dt.timedelta(minutes=19)
    assert r.grid_point_count == 19


def test_query_window_subsets_grid():
    r = analyze_propagation(
        _series(VALUES),
        _series(VALUES),
        source_sn="A",
        target_sn="B",
        dimension="temperature",
        start=BASE + dt.timedelta(minutes=2),
        end=BASE + dt.timedelta(minutes=10),
        step_seconds=60.0,
        max_lag_seconds=120.0,
        max_gap_seconds=600.0,
    )
    assert r.grid_point_count == 9  # 分钟 2..10
    assert r.grid_start == BASE + dt.timedelta(minutes=2)


def test_no_overlap_empty_grid_all_unavailable():
    src = _series(VALUES[:5], minute0=0.0)
    tgt = _series(VALUES[:5], minute0=60.0)
    r = _analyze(src, tgt)
    assert r.grid_point_count == 0
    assert r.overlap_start is None and r.overlap_end is None
    assert r.available is False
    assert r.best_lag_seconds is None
    assert r.direction is None
    assert len(r.lags) == 11  # -300..300 步长 60
    assert all(not l.available and l.correlation is None for l in r.lags)


def test_humidity_dimension_uses_humidity_values():
    src = [Sample(BASE + dt.timedelta(seconds=60 * i), 20.0, 50.0 + VALUES[i]) for i in range(20)]
    tgt = [
        Sample(
            BASE + dt.timedelta(seconds=60 * i),
            20.0,
            50.0 + ([VALUES[0], VALUES[0]] + VALUES[:-2])[i],
        )
        for i in range(20)
    ]
    r = analyze_propagation(
        src,
        tgt,
        source_sn="A",
        target_sn="B",
        dimension="humidity",
        start=None,
        end=None,
        step_seconds=60.0,
        max_lag_seconds=300.0,
        max_gap_seconds=600.0,
    )
    assert r.dimension == "humidity"
    assert r.best_lag_seconds == pytest.approx(120.0)


def test_deterministic_regardless_of_input_order():
    src = _series(VALUES)
    tgt = _series([VALUES[0], VALUES[0]] + VALUES[:-2])
    r1 = _analyze(src, tgt)
    r2 = _analyze(list(reversed(src)), list(reversed(tgt)))
    key = lambda r: [(l.lag_seconds, l.sample_count, l.correlation) for l in r.lags]
    assert key(r1) == key(r2)
    assert r1.best_lag_seconds == r2.best_lag_seconds


# ---------- 接口集成 ----------


def _post_case(client, sn, values, *, minute0=0.0):
    payload = [
        _reading(sn, minute0 + i, 20.0 + v) for i, v in enumerate(values)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200, r.text


BASE_PARAMS = {
    "source_sn": "CASE-A",
    "target_sn": "CASE-B",
    "dimension": "temperature",
    "step_seconds": 60,
    "max_lag_seconds": 300,
    "max_gap_seconds": 600,
}


def test_propagation_api_positive_lag(client):
    _post_case(client, "CASE-A", VALUES)
    _post_case(client, "CASE-B", [VALUES[0], VALUES[0]] + VALUES[:-2])

    r = client.get("/api/v1/propagation", params=BASE_PARAMS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_sn"] == "CASE-A"
    assert body["target_sn"] == "CASE-B"
    assert body["best_lag_seconds"] == 120.0
    assert body["direction"] == "source_to_target"
    assert body["best_correlation"] == pytest.approx(1.0)
    assert body["available"] is True
    assert body["grid_point_count"] == 20
    assert body["source_valid_count"] == 20
    assert len(body["lags"]) == 11
    assert [l["lag_seconds"] for l in body["lags"]] == [
        -300.0, -240.0, -180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0, 240.0, 300.0
    ]
    # 响应 JSON 中不得出现 NaN（标准 JSON 也无法编码 NaN）
    assert "NaN" not in r.text


def test_propagation_api_constant_series_unavailable(client):
    _post_case(client, "CASE-A", VALUES)
    _post_case(client, "CASE-B", [0.0] * len(VALUES))

    r = client.get("/api/v1/propagation", params=BASE_PARAMS)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["best_lag_seconds"] is None
    assert body["direction"] is None
    assert body["best_correlation"] is None
    assert all(l["correlation"] is None and l["available"] is False for l in body["lags"])


def test_propagation_api_no_overlap_empty(client):
    _post_case(client, "CASE-A", VALUES[:5])
    # B 的读数在 60 分钟之后，与 A（0..4 分钟）完全不重叠
    payload = [
        _reading("CASE-B", 60 + i, 20.0 + VALUES[i]) for i in range(5)
    ]
    r = client.post("/api/v1/readings/batch", json={"readings": payload})
    assert r.status_code == 200

    r = client.get("/api/v1/propagation", params=BASE_PARAMS)
    assert r.status_code == 200
    body = r.json()
    assert body["grid_point_count"] == 0
    assert body["available"] is False
    assert body["best_lag_seconds"] is None


def test_propagation_api_404_missing_devices(client):
    _post_case(client, "CASE-A", VALUES[:5])

    # source 不存在（先判 source）
    p1 = dict(BASE_PARAMS, source_sn="NOPE", target_sn="CASE-A")
    r = client.get("/api/v1/propagation", params=p1)
    assert r.status_code == 404
    assert "NOPE" in r.json()["detail"]

    # target 不存在
    p2 = dict(BASE_PARAMS, source_sn="CASE-A", target_sn="NOPE")
    r = client.get("/api/v1/propagation", params=p2)
    assert r.status_code == 404
    assert "NOPE" in r.json()["detail"]

    # 两者都不存在
    p3 = dict(BASE_PARAMS, source_sn="NOPE1", target_sn="NOPE2")
    r = client.get("/api/v1/propagation", params=p3)
    assert r.status_code == 404
    assert "NOPE1" in r.json()["detail"]


def test_propagation_api_422_conflicts(client):
    _post_case(client, "CASE-A", VALUES[:5])
    _post_case(client, "CASE-B", VALUES[:5])

    get = lambda **over: client.get(
        "/api/v1/propagation", params=dict(BASE_PARAMS, **over)
    )

    # start 晚于 end
    r = get(
        start=(BASE + dt.timedelta(minutes=10)).isoformat(),
        end=(BASE + dt.timedelta(minutes=1)).isoformat(),
    )
    assert r.status_code == 422

    # source 与 target 相同
    assert get(target_sn="CASE-A").status_code == 422

    # step 非正 / max_lag 非正 / max_gap 非正
    assert get(step_seconds=0).status_code == 422
    assert get(step_seconds=-10).status_code == 422
    assert get(max_lag_seconds=0).status_code == 422
    assert get(max_gap_seconds=-1).status_code == 422

    # max_lag < step
    assert get(step_seconds=600, max_lag_seconds=300).status_code == 422

    # max_lag 不是 step 的整数倍
    assert get(step_seconds=60, max_lag_seconds=250).status_code == 422
    # 整数倍合法
    assert get(step_seconds=60, max_lag_seconds=300).status_code == 200

    # 非法维度
    assert get(dimension="pressure").status_code == 422

    # 缺必填参数
    for key in ("source_sn", "target_sn", "dimension", "step_seconds",
                "max_lag_seconds", "max_gap_seconds"):
        params = {k: v for k, v in BASE_PARAMS.items() if k != key}
        assert client.get("/api/v1/propagation", params=params).status_code == 422

    # 时间格式错误
    assert get(start="not-a-time").status_code == 422


def test_propagation_api_readonly(client, db):
    """传播分析为只读：调用前后读数数量与版本号不变。"""
    from app.database import CaseState

    _post_case(client, "CASE-A", VALUES)
    _post_case(client, "CASE-B", VALUES)
    before = {
        sn: (db.get(CaseState, sn).version, db.get(CaseState, sn).reading_count)
        for sn in ("CASE-A", "CASE-B")
    }
    r = client.get("/api/v1/propagation", params=BASE_PARAMS)
    assert r.status_code == 200
    db.expire_all()
    after = {
        sn: (db.get(CaseState, sn).version, db.get(CaseState, sn).reading_count)
        for sn in ("CASE-A", "CASE-B")
    }
    assert before == after
