"""
筹码筛选指标：一个条件对应一个函数，每个函数同时给出看多 / 看空判定。

用法示例:
    from chip_new import load_stock, _stock_chip_pack
    from Metric import screen_stock, list_metrics

    stock = load_stock("data/sz_002384_daily.csv", "sz.002384", start_date="2025-01-01")
    results, _, df = _stock_chip_pack(stock)
    report = screen_stock(stock, idx=-1)
    for m in report:
        print(m.name, m.bull.triggered, m.bear.triggered)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import pandas as pd

from chip_new import _period_cost

Side = Literal["bull", "bear", "neutral"]

DEFAULT_PERIODS = (5, 10, 20, 30)


# ---------------------------------------------------------------------------
# 权重评分配置
# ---------------------------------------------------------------------------
# 结构: (weight, bull_mult)
#   weight   — 该指标的权重，各组之和见下方注释
#   bull_mult — 看多侧方向系数：
#     1.0 = 真看多信号（方向性判断）
#     0.3 = 排除型信号（"没过热"≠看多，只是排除看空）
#     0.7 = 次要利好信号（收敛是温和利好）
#
# 公式: score_i = (bull_strength * bull_mult - bear_strength) * weight_i
#        total_score = Σ score_i
#
# 组权重上限:
#   组A (#1#4#5 成本相关) = 0.25  (0.10+0.09+0.06)
#   组B (#2#7  获利/高位)  = 0.20  (0.12+0.08)
#   组C (#3#9#10 集中度)  = 0.25  (0.10+0.07+0.08)
#   组D (#6#8  独立趋势)  = 0.30  (0.15+0.15)
#   合计 = 1.00
#
# 阈值（初版，需回测校准）:
#   total > +0.25 → 适合买入
#   total < -0.20 → 适合卖出
#   中间           → 观望
WEIGHT_CONFIG: dict[str, tuple[float, float]] = {
    # 组A — 成本位置（共享 close / avg_cost / 周期成本）
    "成本突破":       (0.10, 1.0),   # ① 最核心：close vs avg_cost
    "周期成本排列":    (0.09, 1.0),   # ④ 阶梯结构，比 #5 独立
    "股价vs周期成本":  (0.06, 1.0),   # ⑤ 和 #1 几乎同一参考系，最低权
    # 组B — 获利 / 高位（共享 profit_ratio, premium）
    "获利盘":         (0.12, 1.0),   # ② 筹码分析基石
    "高位风险":       (0.08, 0.3),   # ⑦ bull_mult=0.3：未过热是排除型
    # 组C — 集中度（共享 p90_concentration, p90_width）
    "筹码集中度":     (0.10, 1.0),   # ③ 纯集中度判断
    "筹码发散":       (0.07, 0.7),   # ⑨ bull_mult=0.7：收敛是次要利好
    "低位密集启动":   (0.08, 1.0),   # ⑩ 触发苛刻但含金量高
    # 组D — 独立趋势信号
    "平均成本趋势":   (0.15, 1.0),   # ⑥ 趋势确认，择时核心
    "回踩支撑":       (0.15, 1.0),   # ⑧ 经典买点，与 #6 互补
}

BUY_THRESHOLD = 0.25   # total > 此值 → 适合买入
SELL_THRESHOLD = -0.20  # total < 此值 → 适合卖出


@dataclass
class MetricSignal:
    """单侧（看多或看空）判定结果。"""

    triggered: bool
    strength: float = 0.0  # 0~1，越高表示越符合该侧条件
    message: str = ""

    def __bool__(self) -> bool:
        return self.triggered


@dataclass
class MetricPair:
    """一种筛选条件的看多 / 看空结果。"""

    name: str
    bull: MetricSignal
    bear: MetricSignal
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreDetail:
    """单个指标的加权得分明细。"""

    name: str
    weight: float
    bull_mult: float
    bull_strength: float
    bear_strength: float
    raw_score: float       # (bull_s * bull_mult - bear_s)
    weighted_score: float  # raw_score * weight


@dataclass
class ScoreResult:
    """加权打分总结果。"""

    total_score: float
    details: list[ScoreDetail]
    action: str            # "买入" / "卖出" / "观望"
    buy_threshold: float
    sell_threshold: float


@dataclass
class ScreenContext:
    """单日筛选上下文。"""

    result: Mapping[str, Any]
    df: pd.DataFrame
    idx: int
    prev_result: Optional[Mapping[str, Any]] = None
    history: Sequence[Mapping[str, Any]] = ()
    period_costs: dict[int, float] = field(default_factory=dict)


def _sig(triggered: bool, strength: float = 0.0, message: str = "") -> MetricSignal:
    strength = max(0.0, min(1.0, float(strength)))
    return MetricSignal(triggered=bool(triggered), strength=strength, message=message)


def _pair(name: str, bull: MetricSignal, bear: MetricSignal, **meta) -> MetricPair:
    return MetricPair(name=name, bull=bull, bear=bear, meta=meta)

#作用是从一个结果字典里取字段值，并尽量转换成浮点数
def _f(result: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    v = result.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

#判断数据是否有效
def _valid(*values: float) -> bool:
    return all(pd.notna(v) for v in values)


def build_context(
    stock: Mapping[str, Any],
    idx: int = -1,
    *,
    use_decay: Optional[bool] = None,
    periods: Sequence[int] = DEFAULT_PERIODS,
) -> ScreenContext:
    """
    从 load_stock 返回的 stock 字典构造筛选上下文。

    use_decay: None 时跟随 stock['decay_on']；True/False 强制用修正/原始筹码结果。
    """
    if use_decay is None:
        use_decay = bool(stock.get("decay_on") and stock.get("has_decay_toggle"))
    if use_decay:
        results, _, df = stock["results_adj"], stock["checkpoints_adj"], stock["df"]
    else:
        results, _, df = stock["results_raw"], stock["checkpoints_raw"], stock["df_raw"]

    n = len(results)
    if n == 0:
        raise ValueError("results 为空")
    idx = idx if idx >= 0 else n + idx
    idx = max(0, min(idx, n - 1))

    hist_start = max(0, idx - 29)
    history = results[hist_start:idx]
    prev = results[idx - 1] if idx > 0 else None
    pc = {p: _period_cost(df, idx, p) for p in periods}

    return ScreenContext(
        result=results[idx],
        df=df,
        idx=idx,
        prev_result=prev,
        history=history,
        period_costs=pc,
    )


# ---------------------------------------------------------------------------
# 1. 收盘价突破 / 跌破平均成本
# ---------------------------------------------------------------------------
def metric_cost_breakout(
    ctx: ScreenContext,
    *,
    min_premium_pct: float = 0.0,
) -> MetricPair:
    """收盘价相对 avg_cost：站上成本偏多，跌破偏空。"""
    close = _f(ctx.result, "close")
    avg = _f(ctx.result, "avg_cost")
    if not _valid(close, avg):
        return _pair("成本突破", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    ratio = close / avg if avg > 0 else float("nan")
    premium = ratio - 1.0
    bull_ok = premium > min_premium_pct
    bear_ok = premium < -min_premium_pct
    bull_strength = min(1.0, max(0.0, premium / 0.05)) if bull_ok else 0.0
    bear_strength = min(1.0, max(0.0, -premium / 0.05)) if bear_ok else 0.0

    return _pair(
        "成本突破",
        _sig(bull_ok, bull_strength, f"收盘 {close:.2f} > 平均成本 {avg:.2f}（+{premium:.2%}）"),
        _sig(bear_ok, bear_strength, f"收盘 {close:.2f} < 平均成本 {avg:.2f}（{premium:.2%}）"),
        close=close,
        avg_cost=avg,
        premium=premium,
    )


# ---------------------------------------------------------------------------
# 2. 获利盘比例，获利盘适中[0.3,0.75]偏多；过高>0.9或大于0.75后快速回落[前一天-今天>0.15]偏空
# ---------------------------------------------------------------------------
def metric_profit_ratio(
    ctx: ScreenContext,
    *,
    bull_low: float = 0.30,
    bull_high: float = 0.75,
    bull_rise: float = 0.15,
    bull_surge_from_max: float = 0.30,
    bear_high: float = 0.90,
    bear_drop: float = 0.15,
) -> MetricPair:
    """获利盘适中断多；从低位快速冲高偏多；极高或从高位快速回落偏空。"""
    profit = _f(ctx.result, "profit_ratio")
    prev_profit = _f(ctx.prev_result, "profit_ratio") if ctx.prev_result else float("nan")
    if not _valid(profit):
        return _pair("获利盘", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    rise = (profit - prev_profit) if _valid(prev_profit) else 0.0
    bull_moderate = bull_low <= profit <= bull_high
    bull_surge = _valid(prev_profit) and prev_profit <= bull_surge_from_max and rise >= bull_rise
    bull_ok = bull_moderate or bull_surge
    if bull_surge:
        bull_strength = min(1.0, rise / bull_rise)
    elif bull_moderate:
        bull_strength = 1.0 - abs(profit - 0.55) / 0.25
        bull_strength = max(0.0, min(1.0, bull_strength))
    else:
        bull_strength = 0.0

    drop = (prev_profit - profit) if _valid(prev_profit) else 0.0
    bear_overheat = profit >= bear_high
    bear_collapse = _valid(prev_profit) and prev_profit >= 0.75 and drop >= bear_drop
    bear_ok = bear_overheat or bear_collapse
    bear_strength = min(1.0, profit) if bear_overheat else min(1.0, drop / bear_drop) if bear_collapse else 0.0

    if bull_surge:
        bull_msg = f"获利盘从 {prev_profit:.1%} 冲高 {rise:.1%} 至 {profit:.1%}"
    elif bull_moderate:
        bull_msg = f"获利盘 {profit:.1%} 处于 {bull_low:.0%}~{bull_high:.0%} 区间"
    else:
        bull_msg = f"获利盘 {profit:.1%}，未触发适中/冲高"
    if bear_overheat:
        bear_msg = f"获利盘过高 {profit:.1%} ≥ {bear_high:.0%}"
    elif bear_collapse:
        bear_msg = f"获利盘从 {prev_profit:.1%} 回落 {drop:.1%}"
    else:
        bear_msg = f"获利盘 {profit:.1%}，未触发高位/回落"

    return _pair("获利盘", _sig(bull_ok, bull_strength, bull_msg), _sig(bear_ok, bear_strength, bear_msg), profit_ratio=profit)


# ---------------------------------------------------------------------------
# 3. 筹码集中度（90%）
# ---------------------------------------------------------------------------
def metric_concentration(
    ctx: ScreenContext,
    *,
    bull_max: float = 0.12,
    bear_min: float = 0.25,
) -> MetricPair:
    """90% 集中度低 = 筹码密集偏多；过高 = 发散偏空。"""
    conc = _f(ctx.result, "p90_concentration")
    width = _f(ctx.result, "p90_width")
    if not _valid(conc):
        return _pair("筹码集中度", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    bull_ok = conc <= bull_max
    bear_ok = conc >= bear_min
    bull_strength = min(1.0, (bull_max - conc) / bull_max) if bull_ok else 0.0
    bear_strength = min(1.0, (conc - bear_min) / (1.0 - bear_min + 1e-9)) if bear_ok else 0.0

    return _pair(
        "筹码集中度",
        _sig(bull_ok, bull_strength, f"90%集中度 {conc:.1%} ≤ {bull_max:.1%}（密集）"),
        _sig(bear_ok, bear_strength, f"90%集中度 {conc:.1%} ≥ {bear_min:.1%}（发散）"),
        p90_concentration=conc,
        p90_width=width,
    )


# ---------------------------------------------------------------------------
# 4. 周期成本排列（上涨：5>10>20>30；下跌：5<10<20<30）
# ---------------------------------------------------------------------------
def metric_period_cost_alignment(
    ctx: ScreenContext,
    *,
    periods: Sequence[int] = DEFAULT_PERIODS,
    min_spread: float = 0.002,
) -> MetricPair:
    """周期成本阶梯：上涨趋势短期成本高于长期；下跌趋势相反。"""
    costs = [ctx.period_costs.get(p) for p in periods]
    if not all(_valid(c) for c in costs):
        return _pair("周期成本排列", _sig(False, 0, "周期成本数据不足"), _sig(False, 0, "数据不足"))

    bull_chain = all(costs[i] > costs[i + 1] * (1.0 + min_spread) for i in range(len(costs) - 1))
    bear_chain = all(costs[i] < costs[i + 1] * (1.0 - min_spread) for i in range(len(costs) - 1))

    spread = (costs[0] - costs[-1]) / costs[-1] if costs[-1] else 0.0
    bull_strength = min(1.0, abs(spread) / 0.05) if bull_chain else 0.0
    bear_strength = min(1.0, abs(spread) / 0.05) if bear_chain else 0.0

    fmt = " > ".join(f"{p}日{c:.2f}" for p, c in zip(periods, costs))
    rev = " < ".join(f"{p}日{c:.2f}" for p, c in zip(periods, costs))

    return _pair(
        "周期成本排列",
        _sig(bull_chain, bull_strength, f"多头排列: {fmt}"),
        _sig(bear_chain, bear_strength, f"空头排列: {rev}"),
        period_costs=dict(zip(periods, costs)),
    )


# ---------------------------------------------------------------------------
# 5. 股价相对周期成本
# ---------------------------------------------------------------------------
def metric_close_vs_period_cost(
    ctx: ScreenContext,
    *,
    periods: Sequence[int] = DEFAULT_PERIODS,
) -> MetricPair:
    """收盘在全部周期成本之上偏多；在全部之下偏空。"""
    close = _f(ctx.result, "close")
    costs = {p: ctx.period_costs.get(p) for p in periods}
    vals = [costs[p] for p in periods if _valid(costs.get(p, float("nan")))]
    if not _valid(close) or len(vals) < 2:
        return _pair("股价vs周期成本", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    above_all = all(close > c for c in vals)
    below_all = all(close < c for c in vals)
    min_c, max_c = min(vals), max(vals)

    if above_all:
        bull_strength = min(1.0, (close - max_c) / max_c / 0.05)
    else:
        bull_strength = 0.0
    if below_all:
        bear_strength = min(1.0, (min_c - close) / min_c / 0.05)
    else:
        bear_strength = 0.0

    return _pair(
        "股价vs周期成本",
        _sig(above_all, bull_strength, f"收盘 {close:.2f} 高于全部周期成本"),
        _sig(below_all, bear_strength, f"收盘 {close:.2f} 低于全部周期成本"),
        close=close,
        period_costs=costs,
    )


# ---------------------------------------------------------------------------
# 6. 平均成本趋势（斜率）
# ---------------------------------------------------------------------------
def metric_avg_cost_slope(
    ctx: ScreenContext,
    *,
    lookback: int = 5,
    min_pct: float = 0.005,
) -> MetricPair:
    """平均成本抬升偏多；下降偏空。"""
    if len(ctx.history) + 1 < lookback:
        return _pair("平均成本趋势", _sig(False, 0, "历史不足"), _sig(False, 0, "历史不足"))

    series = list(ctx.history[-(lookback - 1):]) + [ctx.result]
    avgs = [_f(r, "avg_cost") for r in series]
    if not all(_valid(a) for a in avgs):
        return _pair("平均成本趋势", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    start, end = avgs[0], avgs[-1]#5天前的平均成本和今天的平均成本
    pct = (end - start) / start if start else 0.0
    bull_ok = pct >= min_pct
    bear_ok = pct <= -min_pct
    bull_strength = min(1.0, pct / 0.03) if bull_ok else 0.0
    bear_strength = min(1.0, -pct / 0.03) if bear_ok else 0.0

    return _pair(
        "平均成本趋势",
        _sig(bull_ok, bull_strength, f"{lookback}日平均成本 {start:.2f} → {end:.2f} ({pct:+.2%})"),
        _sig(bear_ok, bear_strength, f"{lookback}日平均成本 {start:.2f} → {end:.2f} ({pct:+.2%})"),
        lookback=lookback,
        slope_pct=pct,
    )


# ---------------------------------------------------------------------------
# 7. 高位风险（获利盘 + 远离成本）
# ---------------------------------------------------------------------------
def metric_high_risk(
    ctx: ScreenContext,
    *,
    min_profit: float = 0.88,
    min_premium: float = 0.12,
) -> MetricPair:
    """高位密集兑现风险：看空侧专用；看多侧为「未过热」。"""
    close = _f(ctx.result, "close")
    avg = _f(ctx.result, "avg_cost")
    profit = _f(ctx.result, "profit_ratio")
    conc = _f(ctx.result, "p90_concentration")
    if not _valid(close, avg, profit):
        return _pair("高位风险", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    premium = close / avg - 1.0 if avg > 0 else 0.0
    bear_ok = profit >= min_profit and premium >= min_premium
    bear_strength = min(1.0, (profit - min_profit) / 0.1 + (premium - min_premium) / 0.1) / 2 if bear_ok else 0.0

    bull_ok = profit < 0.80 and premium < 0.15
    bull_strength = min(1.0, (0.80 - profit) / 0.3 + (0.15 - premium) / 0.15) / 2 if bull_ok else 0.0

    return _pair(
        "高位风险",
        _sig(bull_ok, bull_strength, f"未过热：获利 {profit:.1%}，溢价 {premium:.1%}"),
        _sig(bear_ok, bear_strength, f"高位风险：获利 {profit:.1%}，较成本 +{premium:.1%}，集中度 {conc:.1%}"),
        profit_ratio=profit,
        premium=premium,
    )


# ---------------------------------------------------------------------------
# 8. 回踩支撑（avg_cost / 70%上沿 / 5日成本）
# ---------------------------------------------------------------------------
def metric_pullback_support(
    ctx: ScreenContext,
    *,
    tol_pct: float = 0.03,
    require_uptrend: bool = True,
) -> MetricPair:
    """多头排列中回踩关键支撑企稳偏多；有效跌破支撑偏空。"""
    close = _f(ctx.result, "close")
    avg = _f(ctx.result, "avg_cost")
    p70_hi = _f(ctx.result, "p70_high")
    c5 = ctx.period_costs.get(5, float("nan"))
    if not _valid(close, avg):
        return _pair("回踩支撑", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    supports = [("平均成本", avg), ("70%上沿", p70_hi), ("5日成本", c5)]
    supports = [(n, v) for n, v in supports if _valid(v)]

    uptrend = True
    if require_uptrend and len(ctx.period_costs) >= 2:
        p = DEFAULT_PERIODS
        costs = [ctx.period_costs.get(x) for x in p if _valid(ctx.period_costs.get(x, float("nan")))]
        uptrend = len(costs) >= 2 and costs[0] > costs[-1]#5日成本高于30日成本，说明是上升趋势

    near_support = False
    hit_name, hit_val = "", float("nan")
    for name, val in supports:
        if abs(close - val) / val <= tol_pct:
            near_support = True
            hit_name, hit_val = name, val
            break

    bull_ok = uptrend and near_support and close >= hit_val * (1.0 - tol_pct)
    bull_strength = 1.0 - abs(close - hit_val) / hit_val / tol_pct if bull_ok else 0.0
    bull_strength = max(0.0, min(1.0, bull_strength))

    broken = any(close < val * (1.0 - tol_pct * 1.5) for _, val in supports)
    bear_ok = broken and not uptrend
    if broken and uptrend:
        bear_ok = close < avg * (1.0 - tol_pct)
    bear_strength = min(1.0, (avg - close) / avg / tol_pct) if bear_ok and avg > 0 else 0.0

    bull_msg = f"回踩{hit_name} {hit_val:.2f} 企稳" if bull_ok else "未回踩支撑或未处于上升趋势"
    bear_msg = f"跌破关键支撑，收盘 {close:.2f}" if bear_ok else "支撑有效"

    return _pair("回踩支撑", _sig(bull_ok, bull_strength, bull_msg), _sig(bear_ok, bear_strength, bear_msg))


# ---------------------------------------------------------------------------
# 9. 筹码发散（集中度走阔）
# ---------------------------------------------------------------------------
def metric_chip_divergence(
    ctx: ScreenContext,
    *,
    width_grow_pct: float = 0.08,
    conc_grow: float = 0.03,
) -> MetricPair:
    """集中度走低/走阔且 90% 区间变宽偏空；集中度收敛偏多。"""
    conc = _f(ctx.result, "p90_concentration")
    width = _f(ctx.result, "p90_width")
    prev = ctx.prev_result or {}
    prev_conc = _f(prev, "p90_concentration")
    prev_width = _f(prev, "p90_width")

    if not _valid(conc, width):
        return _pair("筹码发散", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    conc_down = _valid(prev_conc) and (prev_conc - conc) >= conc_grow
    width_up = _valid(prev_width) and prev_width > 0 and (width - prev_width) / prev_width >= width_grow_pct

    bull_ok = conc_down
    bull_strength = min(1.0, (prev_conc - conc) / 0.08) if bull_ok else 0.0

    bear_ok = width_up or (_valid(prev_conc) and (conc - prev_conc) >= conc_grow)
    bear_strength = min(1.0, (width - prev_width) / prev_width / width_grow_pct) if width_up else 0.0
    if bear_ok and not width_up and _valid(prev_conc):
        bear_strength = min(1.0, (conc - prev_conc) / 0.08)

    return _pair(
        "筹码发散",
        _sig(bull_ok, bull_strength, f"集中度收敛 {prev_conc:.1%} → {conc:.1%}" if _valid(prev_conc) else "集中度收敛"),
        _sig(bear_ok, bear_strength, f"区间走阔/集中度上升，宽度 {prev_width:.2f} → {width:.2f}" if _valid(prev_width) else "筹码发散"),
    )


# ---------------------------------------------------------------------------
# 10. 低位密集 + 站上成本（组合型启动）
# ---------------------------------------------------------------------------
def metric_low_dense_launch(
    ctx: ScreenContext,
    *,
    max_conc: float = 0.14,
    min_profit: float = 0.25,
    max_profit: float = 0.70,
) -> MetricPair:
    """低位单峰密集且刚进入获利区偏多；低位密集但跌破成本偏空。"""
    close = _f(ctx.result, "close")
    avg = _f(ctx.result, "avg_cost")
    profit = _f(ctx.result, "profit_ratio")
    conc = _f(ctx.result, "p90_concentration")

    if not _valid(close, avg, profit, conc):
        return _pair("低位密集启动", _sig(False, 0, "数据不足"), _sig(False, 0, "数据不足"))

    dense = conc <= max_conc
    bull_ok = dense and close > avg and min_profit <= profit <= max_profit
    bull_strength = min(1.0, (max_conc - conc) / max_conc) if bull_ok else 0.0

    bear_ok = dense and close < avg and profit < 0.20
    bear_strength = min(1.0, (avg - close) / avg / 0.05) if bear_ok else 0.0

    return _pair(
        "低位密集启动",
        _sig(bull_ok, bull_strength, f"密集区启动：集中度 {conc:.1%}，获利 {profit:.1%}"),
        _sig(bear_ok, bear_strength, f"密集区破位：集中度 {conc:.1%}，收盘低于成本"),
    )


# ---------------------------------------------------------------------------
# 注册表 & 批量筛选
# ---------------------------------------------------------------------------
METRIC_FUNCTIONS = (
    metric_cost_breakout,
    metric_profit_ratio,
    metric_concentration,
    metric_period_cost_alignment,
    metric_close_vs_period_cost,
    metric_avg_cost_slope,
    metric_high_risk,
    metric_pullback_support,
    metric_chip_divergence,
    metric_low_dense_launch,
)


def list_metrics() -> list[str]:
    return [fn.__name__ for fn in METRIC_FUNCTIONS]


def run_metrics(ctx: ScreenContext, metrics: Sequence = METRIC_FUNCTIONS) -> list[MetricPair]:
    return [fn(ctx) for fn in metrics]


def screen_stock(
    stock: Mapping[str, Any],
    idx: int = -1,
    *,
    use_decay: Optional[bool] = None,
    metrics: Sequence = METRIC_FUNCTIONS,
) -> list[MetricPair]:
    """对单只股票运行全部（或指定）指标。"""
    ctx = build_context(stock, idx=idx, use_decay=use_decay)
    return run_metrics(ctx, metrics)


def weighted_score(
    report: Sequence[MetricPair],
    *,
    config: dict[str, tuple[float, float]] = WEIGHT_CONFIG,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> ScoreResult:
    """按权重配置对指标结果做加权打分。

    公式: score_i = (bull_strength * bull_mult_i - bear_strength) * weight_i
         total_score = Σ score_i
    bull_mult_i = 1.0 / 0.7 / 0.3，分别对应真看多 / 次要利好 / 排除型信号。
    看空侧不区分强弱，直接减去 bear_strength，因为看空的时候是真的空。
    Returns:
        ScoreResult，包含总分、明细、建议操作。
    """
    details: list[ScoreDetail] = []
    total = 0.0

    for m in report:
        weight, bull_mult = config.get(m.name, (0.0, 1.0))
        bull_s = m.bull.strength if m.bull.triggered else 0.0
        bear_s = m.bear.strength if m.bear.triggered else 0.0
        raw = bull_s * bull_mult - bear_s
        ws = raw * weight
        total += ws

        details.append(ScoreDetail(
            name=m.name,
            weight=weight,
            bull_mult=bull_mult,
            bull_strength=bull_s,
            bear_strength=bear_s,
            raw_score=raw,
            weighted_score=ws,
        ))

    if total > buy_threshold:
        action = "买入"
    elif total < sell_threshold:
        action = "卖出"
    else:
        action = "观望"

    return ScoreResult(
        total_score=total,
        details=details,
        action=action,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def summarize(report: Sequence[MetricPair]) -> dict[str, Any]:
    """汇总看多/看空触发数与综合强度。"""
    bull_hits = [m for m in report if m.bull.triggered]
    bear_hits = [m for m in report if m.bear.triggered]
    return {
        "bull_count": len(bull_hits),
        "bear_count": len(bear_hits),
        "bull_score": sum(m.bull.strength for m in report),
        "bear_score": sum(m.bear.strength for m in report),
        "bull_names": [m.name for m in bull_hits],
        "bear_names": [m.name for m in bear_hits],
    }


if __name__ == "__main__":
    from chip_new import csv_path_for_code, load_stock

    code = "sz.002384"
    stock = load_stock(
        csv_path_for_code(code), code,
        start_date="2025-01-01",
        fetch_top10=False,   # 自测用 CSV 内已有 decay，避免 AkShare 网络失败
        save_enriched=False,
    )
    report = screen_stock(stock)
    s = summarize(report)
    r = stock["results_raw"][-1]
    print(f"=== {code} {r['trade_date']} 收盘 {r['close']:.2f} ===")
    print(f"看多触发 {s['bull_count']}  看空触发 {s['bear_count']}")
    print(f"看多强度 {s['bull_score']:.2f}  看空强度 {s['bear_score']:.2f}")
    print()
    for m in report:
        b = "✓" if m.bull.triggered else "·"
        s_ = "✓" if m.bear.triggered else "·"
        print(f"  [{b}|{s_}] {m.name}")
        if m.bull.triggered:
            print(f"       多: {m.bull.message}")
        if m.bear.triggered:
            print(f"       空: {m.bear.message}")

    # --- 加权打分 ---
    score_result = weighted_score(report)
    print()
    print(f"  加权总分: {score_result.total_score:+.4f}")
    print(f"  建议操作: {score_result.action}")
    print(f"  (买入阈值 > {score_result.buy_threshold:+.2f}, 卖出阈值 < {score_result.sell_threshold:+.2f})")
    print()
    print("  各指标明细:")
    for d in score_result.details:
        bm_tag = f"×{d.bull_mult}" if d.bull_mult != 1.0 else ""
        print(f"    {d.name:8s}  w={d.weight:.2f}{bm_tag}  "
              f"raw={d.raw_score:+.3f}  加权={d.weighted_score:+.4f}")
