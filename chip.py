# 筹码分布：同花顺那套移动成本模型
# 用法：改 main() 里的 csv 路径，然后 python chip.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 列名别名 → 统一后的标准名
COLUMN_ALIASES = {
    "trade_date": ("date", "trade_date", "datetime", "日期", "交易日期"),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价"),
    "volume": ("volume", "vol", "成交量"),
    "amount": ("amount", "成交额", "成交金额", "turnover"),
    "turnover_rate": ("turnover_rate", "turn", "turnoverratio", "换手率"),
    "float_shares": ("float_shares", "float_share", "circulating_share", "流通股本", "流通股"),
    "code": ("code", "symbol", "股票代码"),
    "tradestatus": ("tradestatus", "trade_status", "交易状态"),
}


def _pick_col(columns, aliases):
    """按别名在表头里找列，忽略大小写和首尾空格。"""
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


#将换手率统一为小数格式并返回float，如果没有换手率但有流通股数，则用成交量除以流通股数计算换手率，如果两者都没有则抛出异常。
def _turnover(row) -> float:
    # prep_df 已将 turnover_rate 转为小数；直接取用
    if "turnover_rate" in row.index and pd.notna(row["turnover_rate"]):
        return float(row["turnover_rate"])

    col = _pick_col(row.index, COLUMN_ALIASES["turnover_rate"])
    if col is not None and pd.notna(row[col]):
        # 原始列为百分数（BaoStock turn 可小于 1，如 0.2143 = 0.2143%）
        return float(row[col]) / 100

    float_col = _pick_col(row.index, COLUMN_ALIASES["float_shares"])
    if float_col is not None and pd.notna(row[float_col]) and float(row[float_col]) > 0:
        return float(row["volume"]) / float(row[float_col])

    vol_col = _pick_col(row.index, COLUMN_ALIASES["volume"]) or "volume"
    vol = float(row[vol_col]) if vol_col in row.index and pd.notna(row[vol_col]) else 0.0
    if vol <= 0:
        return 0.0  # 停牌/无成交：换手率按 0 处理

    date_col = _pick_col(row.index, COLUMN_ALIASES["trade_date"]) or "trade_date"
    raise ValueError(f"{row.get(date_col)}: 缺 turnover_rate/turn 或 float_shares")


def _avg_price(row) -> float:
    if "amount" in row.index and pd.notna(row["amount"]) and row["volume"] > 0:
        return float(row["amount"]) / float(row["volume"])
    return (row["open"] + row["high"] + row["low"] + row["close"]) / 4


# ============================================================
# 历史换手衰减系数 (Historical Turnover Decay Coefficient)
# ============================================================
#
# 背景:
#   BaoStock 的 turn 字段 = volume / 流通股本 × 100
#   这里的"流通股本"是名义流通股本(总股本-限售股本), 包含前十大股东持有的流通股.
#   但前十大股东中相当一部分股份并不活跃交易,
#   导致 BaoStock 的换手率被低估.
#
# 修正:
#   真实换手率 = volume / (流通股本 × (1 - top10_ratio × (1 - active_ratio)))
#             = (volume / 流通股本) × 衰减系数
#   其中: 衰减系数 = 1 / (1 - top10_ratio × (1 - active_ratio))
#
# 参数说明:
#   top10_ratio  : 前十大股东合计持股比例 (0~1), 来自季报, 有约 3 个月披露滞后
#   active_ratio : 前十大股东中实际活跃交易的比例估计值 (0~1, 默认 0.2)
#                  即假设十大股东中有 20% 的股份会参与日常交易, 80% 视为锁定
#
# 特殊情况:
#   - active_ratio = 0  → 退化为原始公式 1 / (1 - top10_ratio)
#   - active_ratio = 1  → 衰减系数 = 1.0 (不做任何调整)
#   - top10_ratio = 0   → 衰减系数 = 1.0
#   - top10_ratio 缺失  → 衰减系数 = 1.0 (回退到无调整)
# ============================================================

# 前十大股东数据的列名别名
TOP10_ALIASES = {
    "report_date": ("report_date", "date", "trade_date", "pub_date", "报告期", "公告日期", "日期"),
    "top10_ratio": (
        "top10_ratio", "top10_hold_ratio", "top10", "hold_ratio_top10",
        "前十大股东持股比例", "十大股东持股比例", "top10_holding",
    ),
}

def load_top10_from_csv(csv_path) -> pd.DataFrame:
    """
    从 CSV 加载前十大股东持股比例数据.
    CSV 格式要求 (列名不区分大小写, 支持中英文):
        - 日期列: report_date / date / 公告日期 / 报告期
        - 比例列: top10_ratio / 前十大股东持股比例 / top10_hold_ratio
    比例可为小数 (0.6532) 或百分数 (65.32), calc_decay_coefficient 会自动识别.
    示例 CSV:
        report_date,top10_ratio
        2024-03-31,0.6532
        2024-06-30,0.6210
        2024-09-30,0.5985
    返回:
        DataFrame, 含 report_date 和 top10_ratio 两列.
    """
    df = pd.read_csv(csv_path)
    return df


def calc_decay_coefficient(top10_ratio, active_ratio: float = 0.2) -> float:
    """
    计算单期历史换手衰减系数.
    公式: decay = 1 / (1 - top10_ratio * (1 - active_ratio))
    参数:
        top10_ratio : 前十大股东合计持股比例, 取值 0~1.
                      传入 None/NaN 时返回 1.0 (不调整).
                      传入 >1.5 的值视为百分数并自动除以 100.
        active_ratio: 前十大股东实际活跃交易比例, 取值 0~1, 默认 0.2.
    返回:
        衰减系数 (float), 恒 >= 1.0.
    """
    if top10_ratio is None or pd.isna(top10_ratio):
        return 1.0

    ratio = float(top10_ratio)
    # 兼容百分数输入 (如 65.3 表示 65.3%)
    if ratio > 1.0:
        ratio = ratio / 100.0
    ratio = max(0.0, min(ratio, 1.0))  # 截断到 [0, 1]

    active = float(active_ratio)
    active = max(0.0, min(active, 1.0))

    locked = ratio * (1.0 - active)  # 锁定筹码比例
    if locked >= 1.0:
        return 1.0  # 极端情况兜底, 避免除零
    return 1.0 / (1.0 - locked)


# 前十大股东数据的披露滞后天数 (季报披露规则: Q1 4月底, Q2 8月底, Q3 10月底, 年报4月底)
# 取 90 天作为统一滞后估计, 即报告期 + 90 天后该数据才"生效"
TOP10_LAG_DAYS = 90
def prepare_decay_series(
    df: pd.DataFrame,
    top10_ratios: pd.DataFrame,
    *,
    active_ratio: float = 0.2,
    lag_days: int = TOP10_LAG_DAYS,
) -> list:
    """
    根据季频的前十大股东持股比例, 生成与日线数据对齐的衰减系数序列.

    参数:
        df           : 日线 DataFrame, 必须含 trade_date 列 (已升序).
        top10_ratios : 前十大股东持股比例表, 至少含:
                       - 日期列 (report_date / date / 公告日期 等)
                       - 比例列 (top10_ratio / 前十大股东持股比例 等)
                       比例可为小数 (0.65) 或百分数 (65.0), 函数会自动识别.
        active_ratio : 前十大股东活跃交易比例, 默认 0.2.
        lag_days     : 披露滞后天数, 默认 90. 即报告期 + lag_days 后该数据才生效.

    返回:
        list[float], 长度等于 len(df), 每个元素为对应交易日的衰减系数.
        若某交易日之前无任何已生效的季报数据, 则该日衰减系数为 1.0.

    说明:
        - 由于季报披露有滞后, Q1 报告 (截至 3/31) 通常 4 月底才公告,
          因此用 report_date + lag_days 作为"生效日期"做 forward-fill.
        - 这样能避免"用未来数据"的 look-ahead bias.
    """
    n = len(df)
    if top10_ratios is None or len(top10_ratios) == 0:
        return [1.0] * n

    date_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["report_date"])
    ratio_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["top10_ratio"])
    if date_col is None or ratio_col is None:
        raise ValueError(
            "top10_ratios 缺少必要列: 需要日期列 (report_date/date/公告日期) "
            "和比例列 (top10_ratio/前十大股东持股比例)"
        )

    ratios = top10_ratios[[date_col, ratio_col]].copy()
    ratios.columns = ["report_date", "top10_ratio"]
    ratios["report_date"] = pd.to_datetime(ratios["report_date"], errors="coerce")
    ratios["top10_ratio"] = pd.to_numeric(ratios["top10_ratio"], errors="coerce")
    ratios = ratios.dropna(subset=["report_date", "top10_ratio"])
    # 如果有真实的公告日期 pub_date，就用它；否则才用 +90天 的保底策略
    if "pub_date" in ratios.columns:
        ratios["effective_date"] = ratios["pub_date"]
    else:
        ratios["effective_date"] = ratios["report_date"] + pd.Timedelta(days=lag_days)
    ratios = ratios.sort_values("effective_date").reset_index(drop=True)

    if ratios.empty:
        return [1.0] * n

    trade_dates = pd.to_datetime(df["trade_date"]).values
    eff_dates = ratios["effective_date"].values
    eff_ratios = ratios["top10_ratio"].values

    # 对每个交易日, 二分查找"生效日期 <= trade_date"的最近一条记录
    import bisect
    decay_series = []
    for td in trade_dates:
        td_ts = pd.Timestamp(td)
        # bisect_right 返回第一个 > td 的位置, 减 1 即最后一个 <= td 的位置
        pos = bisect.bisect_right(eff_dates, td_ts) - 1
        if pos < 0:
            decay_series.append(1.0)
        else:
            decay_series.append(calc_decay_coefficient(eff_ratios[pos], active_ratio))
    return decay_series


def attach_decay_to_df(
    df: pd.DataFrame,
    top10_ratios: pd.DataFrame = None,
    *,
    active_ratio: float = 0.2,
    lag_days: int = TOP10_LAG_DAYS,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    把日频衰减系数作为 'decay' 列附加到日线 DataFrame.
    参数:
        df           : 日线 DataFrame, 必须含 trade_date 列.
        top10_ratios : 前十大股东持股比例表. 若为 None, 则 decay 列全部填 1.0.
        active_ratio : 前十大股东活跃交易比例, 默认 0.2.
        lag_days     : 披露滞后天数, 默认 90.
        inplace      : 是否原地修改 df.
    返回:
        附加了 'decay' 列的 DataFrame.

    说明:
        - _update_chip() 会优先读取行内的 'decay' 列, 若不存在则回退到 decay 参数.
        - 调用本函数后, calc_chip / chip_snapshot 会自动使用每日动态衰减系数.
    """
    out = df if inplace else df.copy()
    if top10_ratios is None or len(top10_ratios) == 0:
        out["decay"] = 1.0
    else:
        out["decay"] = prepare_decay_series(
            out, top10_ratios, active_ratio=active_ratio, lag_days=lag_days
        )
    return out


def prepare_top10_daily_series(
    df: pd.DataFrame,
    top10_ratios: pd.DataFrame,
    *,
    lag_days: int = TOP10_LAG_DAYS,
    as_percent: bool = True,
) -> list:
    """
    根据季频的前十大股东持股比例, 生成与日线数据对齐的日频 top10_ratio 序列.

    与 prepare_decay_series 类似, 但返回的是原始持股比例 (而非衰减系数),
    用于写入 CSV 供用户查看.

    参数:
        df           : 日线 DataFrame, 必须含 trade_date 列 (已升序).
        top10_ratios : 前十大股东持股比例表 (含 report_date 和 top10_ratio 列).
        lag_days     : 披露滞后天数, 默认 90.
        as_percent   : True 返回百分数 (如 48.02 表示 48.02%),
                       False 返回小数 (如 0.4802).

    返回:
        list[float], 长度等于 len(df).
        若某交易日之前无已生效季报, 则该日为 0.0 (或 NaN).
    """
    n = len(df)
    if top10_ratios is None or len(top10_ratios) == 0:
        return [0.0] * n

    date_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["report_date"])
    ratio_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["top10_ratio"])
    if date_col is None or ratio_col is None:
        return [0.0] * n

    ratios = top10_ratios[[date_col, ratio_col]].copy()
    ratios.columns = ["report_date", "top10_ratio"]
    ratios["report_date"] = pd.to_datetime(ratios["report_date"], errors="coerce")
    ratios["top10_ratio"] = pd.to_numeric(ratios["top10_ratio"], errors="coerce")
    ratios = ratios.dropna(subset=["report_date", "top10_ratio"])
    # 如果有真实的公告日期 pub_date，就用它；否则才用 +90天 的保底策略
    if "pub_date" in ratios.columns:
        ratios["effective_date"] = ratios["pub_date"]
    else:
        ratios["effective_date"] = ratios["report_date"] + pd.Timedelta(days=lag_days)
    ratios = ratios.sort_values("effective_date").reset_index(drop=True)

    if ratios.empty:
        return [0.0] * n

    import bisect
    trade_dates = pd.to_datetime(df["trade_date"]).values
    eff_dates = ratios["effective_date"].values
    eff_ratios = ratios["top10_ratio"].values

    series = []
    for td in trade_dates:
        td_ts = pd.Timestamp(td)
        pos = bisect.bisect_right(eff_dates, td_ts) - 1
        if pos < 0:
            series.append(0.0)
        else:
            r = float(eff_ratios[pos])
            # 统一为小数
            if r > 1.5:
                r = r / 100.0
            r = max(0.0, min(r, 1.0))
            series.append(r * 100.0 if as_percent else r)
    return series


def fetch_top10_from_akshare(symbol: str) -> pd.DataFrame:
    import akshare as ak
    raw = ak.stock_circulate_stock_holder(symbol=symbol)
    
    raw["截止日期"] = pd.to_datetime(raw["截止日期"], errors="coerce")
    raw["占流通股比例"] = pd.to_numeric(raw["占流通股比例"], errors="coerce")
    raw["公告日期"] = pd.to_datetime(raw["公告日期"], errors="coerce")
    
    raw = raw.dropna(subset=["截止日期", "占流通股比例", "公告日期"])
    
    # 按报告期聚合，同时取该报告期最早的公告日期
    agg = raw.groupby("截止日期", as_index=False).agg(
        top10_ratio=("占流通股比例", "sum"),
        pub_date=("公告日期", "min")  # 新增：保留真实公告日
    ).rename(columns={"截止日期": "report_date"})
    
    agg["top10_ratio"] = (agg["top10_ratio"] / 100.0).clip(0.0, 1.0)
    agg = agg.sort_values("report_date").reset_index(drop=True)
    return agg


#生成价格网格，从 low 到 high，步长为 step，返回一个价格列表(不包含high)。
def _price_grid(low, high, step):
    n = max(int((high - low) / step), 1)
    return [round(low + i * step, 8) for i in range(n)]

#返回当天价格点的权重分布，权重在所有价格点上均匀分布，权重总和为 1。
def _daily_cost_uniform(low, high, step):
    prices = _price_grid(low, high, step)
    w = 1.0 / len(prices)
    return {p: w for p in prices}


def _daily_cost_triangle(low, high, avg, volume, step):
    avg = min(max(avg, low), high)
    prices = _price_grid(low, high, step)
    if high == low:
        return {round(low, 8): 1.0}

    scale = 2.0 / (high - low)#峰值密度，avg处的高度，三角形密度面积为 (high - low) * scale / 2 = 1。
    raw = {}
    for p in prices:
        left, right = p, min(p + step, high)
        if avg <= low or avg >= high:
            h1 = h2 = scale / max(high - low, step)
        elif right <= avg:
            h1 = scale / max(avg - low, step) * (left - low)
            h2 = scale / max(avg - low, step) * (right - low)
        elif left >= avg:
            h1 = scale / max(high - avg, step) * (high - left)
            h2 = scale / max(high - avg, step) * (high - right)
        else:
            h_left = scale / max(avg - low, step) * (left - low)
            h_right = scale / max(high - avg, step) * (high - right)
            left_area = (avg - left) * (scale + h_left) / 2
            right_area = (right - avg) * (scale + h_right) / 2
            raw[p] = (left_area + right_area) * volume
            continue
        raw[p] = step * (h1 + h2) / 2 * volume

    total = sum(raw.values())
    if total <= 0:
        w = 1.0 / len(prices)
        return {p: w for p in prices}
    return {p: v / total for p, v in raw.items()}#对价格点的权重进行归一化，使得权重总和为 1。

#返回的不是归一化权重，而是每个价格点上，历经岁月换手洗后，仍留在市场上的筹码数量占总筹码数量的比例。
def _update_chip(chip, row, *, step=0.01, decay=1.0, mode="triangle"):
    # 优先使用行内 'decay' 列 (历史换手衰减系数), 支持每日动态衰减;
    # 若行内无 decay 列或为空, 则回退到 decay 参数 (向后兼容).
    if "decay" in row.index and pd.notna(row["decay"]):
        decay = float(row["decay"])
    turnover = _turnover(row)
    moved = min(turnover * decay, 1.0)
    stay = 1.0 - moved

    chip = {p: w * stay for p, w in chip.items()}

    low, high, vol = float(row["low"]), float(row["high"]), float(row["volume"])
    if vol <= 0:
        return chip
    if mode == "uniform":
        daily = _daily_cost_uniform(low, high, step)
    else:
        daily = _daily_cost_triangle(low, high, _avg_price(row), vol, step)

    for p, w in daily.items():
        chip[p] = chip.get(p, 0.0) + w * moved#chip.get(p, 0.0) 获取价格点 p 当前的筹码权重，如果没有则默认为 0.0。然后加上当天新产生的筹码权重 w * moved。
    chip = {p: w for p, w in chip.items() if w > 1e-12}
    if len(chip) > 5000:
        chip = _prune_chip(chip, step)
    return chip


def _prune_chip(chip, step, max_keys=5000):
    """价位档过多时合并到更粗网格，避免长历史股票内存爆炸。"""
    if len(chip) <= max_keys:
        return chip
    coarse = max(step * 2, 0.02)
    binned = {}
    for price, weight in chip.items():
        key = round(round(price / coarse) * coarse, 4)
        binned[key] = binned.get(key, 0.0) + weight
    total = sum(binned.values())
    if total <= 0:
        return chip
    return {p: w / total for p, w in binned.items() if w / total > 1e-12}

#筹码平均成本
def _avg_cost(chip):
    t = sum(chip.values())
    return sum(p * w for p, w in chip.items()) / t if t else 0.0

#筹码获利成本比例，即收盘价下方的筹码占总筹码的比例。
def _profit_ratio(chip, close):
    t = sum(chip.values())
    if not t:
        return 0.0
    below = sum(w for p, w in chip.items() if p <= close)
    return below / t

#筹码峰，返回价格点权重分布中，权重大于 min_ratio 的局部峰值列表，按权重从大到小排序。
def _peaks(chip, min_ratio=0.001):
    if not chip:
        return []
    prices = sorted(chip)
    total = sum(chip.values())
    th = total * min_ratio
    out = []
    for i in range(1, len(prices) - 1):
        p = prices[i]
        w = chip[p]
        if w >= th and w > chip[prices[i - 1]] and w > chip[prices[i + 1]]:
            out.append((p, w))
    return sorted(out, key=lambda x: x[1], reverse=True)#对局部峰值按权重(x[1]表示w)从大到小排序(reverse=True)，返回一个列表，其中每个元素是一个元组，包含价格点和对应的权重。


def _snapshot_from_chip(chip, row):
    close = float(row["close"])
    dist = dict(chip)
    return {
        "trade_date": row["trade_date"],
        "close": close,
        "avg_cost": _avg_cost(dist),
        "profit_ratio": _profit_ratio(dist, close),
        "peaks": _peaks(dist),
        "distribution": dist,
    }


CHECKPOINT_EVERY = 150


def calc_chip(
    df: pd.DataFrame,
    *,
    step=0.01,
    decay=1.0,
    mode="triangle",
    checkpoint_every=CHECKPOINT_EVERY,
    store_distribution=False,
):
    """
    输入日线 DataFrame，按日期升序，返回 (results, checkpoints)。

    默认不保存每日 distribution（省内存），交互看图时用 chip_snapshot 按需计算。
    checkpoints: {行号: 筹码字典}，用于加速 snapshot。
    """
    df = df.sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        return [], {0: {}}

    chip = {}
    results = []
    checkpoints = {0: {}}
    calc_kw = {"step": step, "decay": decay, "mode": mode}

    for i in range(len(df)):
        row = df.iloc[i]
        chip = _update_chip(chip, row, **calc_kw)
        snap = _snapshot_from_chip(chip, row)
        if store_distribution:
            results.append(snap)
        else:
            results.append({k: v for k, v in snap.items() if k != "distribution"})

        if checkpoint_every and i % checkpoint_every == 0:
            checkpoints[i] = dict(chip)

    checkpoints[len(df) - 1] = dict(chip)
    return results, checkpoints


def chip_snapshot(df, idx, checkpoints, *, step=0.01, decay=1.0, mode="triangle"):
    """按行号计算含 distribution 的单日筹码快照。"""
    idx = int(idx)
    if idx < 0 or idx >= len(df):
        raise IndexError(f"idx={idx} 超出范围 0~{len(df) - 1}")

    start = max(k for k in checkpoints if k <= idx)
    chip = dict(checkpoints[start])
    calc_kw = {"step": step, "decay": decay, "mode": mode}
    for i in range(start + 1, idx + 1):
        chip = _update_chip(chip, df.iloc[i], **calc_kw)

    return _snapshot_from_chip(chip, df.iloc[idx])


def prep_df(df: pd.DataFrame, *, skip_halted: bool = True) -> pd.DataFrame:
    """列名对齐，并默认去掉停牌/无成交日（tradestatus=0 或 volume=0）。"""
    df = df.copy()
    rename = {}
    for std_name, aliases in COLUMN_ALIASES.items():
        src = _pick_col(df.columns, aliases)
        if src is not None and src != std_name:
            rename[src] = std_name
    df = df.rename(columns=rename)

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("open", "high", "low", "close", "volume", "amount", "float_shares"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "turnover_rate" in df.columns:
        # BaoStock turn 为百分数（可小于 1，如 0.2143 表示 0.2143%），统一转小数
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce") / 100

    if "tradestatus" in df.columns:
        df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")

    if skip_halted and "volume" in df.columns:
        n_before = len(df)
        ok = df["volume"].fillna(0).gt(0)#空值填充为 0，然后判断是否大于 0，返回一个布尔 Series，表示每行是否有成交量。
        if "tradestatus" in df.columns:
            ok &= df["tradestatus"].fillna(1).ne(0)#空值填充为 1，然后判断是否不等于 0，返回一个布尔 Series，表示每行是否不是停牌状态。&=表示并且
        df = df.loc[ok].reset_index(drop=True)
        n_drop = n_before - len(df)
        if n_drop:
            print(f"跳过停牌/无成交日 {n_drop} 天")

    return df


def _rebin_dist(dist, step=0.05):
    """把细粒度(0.01)筹码合并成粗档位(0.05)，专门用于画图。"""
    binned = {}
    for price, weight in dist.items():
        key = round(round(price / step) * step, 4)#round(x)四舍五入到整数,round(x,n)保留n位小数
        binned[key] = binned.get(key, 0.0) + weight
    return binned


def _chip_ylim(binned, close, avg, peaks, pad=0.5):
    """Y 轴只框住筹码密集区，别从上市最低价拉到最高价。"""
    anchors = [close, avg]
    if peaks:
        anchors.extend(p for p, _ in peaks[:8])#有筹码峰，就取权重最大的前8个峰的价格
    elif binned:#没有筹码峰，就取权重最大的前8个价格点
        top = sorted(binned.items(), key=lambda x: x[1], reverse=True)[:8]
        anchors.extend(p for p, _ in top)

    ylo = min(anchors) - pad
    yhi = max(anchors) + pad

    if binned:
        data_lo = min(binned)
        data_hi = max(binned)
        ylo = max(ylo, data_lo - pad * 0.5)
        yhi = min(yhi, data_hi + pad * 0.5)

    if yhi - ylo < 1.0:
        mid = (yhi + ylo) / 2
        ylo, yhi = mid - 0.6, mid + 0.6
    return ylo, yhi


def _draw_chip_on_ax(ax, result, plot_step=0.05, *, label=None, show_legend=True):
    """在指定 axes 上画单日筹码分布（供静态图和交互图复用）。"""
    dist = result["distribution"]
    close = result["close"]
    avg = result["avg_cost"]
    date = result["trade_date"]
    profit = result["profit_ratio"]
    peaks = result.get("peaks", [])

    binned = _rebin_dist(dist, step=plot_step)
    if not binned:
        raise ValueError("distribution 为空")

    ylo, yhi = _chip_ylim(binned, close, avg, peaks)
    prices = sorted(p for p in binned if ylo <= p <= yhi)
    weights = [binned[p] for p in prices]
    if not prices:
        prices = sorted(binned)
        weights = [binned[p] for p in prices]

    colors = ["#d63b3b" if p <= close else "#3b7ddd" for p in prices]

    ax.clear()
    ax.set_facecolor("#1c1c1c")
    ax.barh(
        prices, weights, height=plot_step, left=0,
        color=colors, align="edge", edgecolor="none",
    )
    ax.axhline(avg, color="white", linewidth=1, zorder=5)
    ax.axhline(close, color="#e8c547", linewidth=1.2, linestyle="--", zorder=5)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0, max(weights) * 1.18)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_ylabel("价格", color="#cccccc", fontsize=9)
    ax.set_xlabel("筹码比例", color="#cccccc", fontsize=9)
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    title = f"{date_str}  筹码分布" if not label else f"{label}\n{date_str}"
    ax.set_title(title, color="#eeeeee", fontsize=10, pad=8)
    ax.text(
        0.98, 0.02,
        f"收盘 {close:.2f}\n平均成本 {avg:.2f}\n获利比例 {profit:.2%}",
        transform=ax.transAxes, ha="right", va="bottom",
        color="#dddddd", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#2a2a2a", edgecolor="#555555"),
    )
    if show_legend:
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color="#d63b3b", lw=6, label="获利盘"),
                Line2D([0], [0], color="#3b7ddd", lw=6, label="套牢盘"),
                Line2D([0], [0], color="white", lw=1.8, label=f"平均成本 {avg:.2f}"),
                Line2D([0], [0], color="#e8c547", lw=1.2, linestyle="--", label=f"收盘 {close:.2f}"),
            ],
            loc="upper right", framealpha=0.85,
            facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#dddddd", fontsize=7,
        )


def _draw_kline_on_ax(ax, df, idx, window=120, *, title=None):
    """画 K 线窗口，idx 为当前选中的行号（0-based）。"""
    ax.clear()
    ax.set_facecolor("#1c1c1c")
    start = max(0, idx - window + 1)
    sub = df.iloc[start: idx + 1].reset_index(drop=True)
    sel = len(sub) - 1

    for x, row in sub.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        up = c >= o
        color = "#d63b3b" if up else "#26a641"
        ax.vlines(x, l, h, color=color, linewidth=0.8)
        body_lo, body_hi = min(o, c), max(o, c)
        ax.bar(x, body_hi - body_lo or 0.01, bottom=body_lo, width=0.65, color=color, edgecolor="none")

    ax.axvline(sel, color="#e8c547", linewidth=1.2, linestyle="--", zorder=5)
    ax.scatter([sel], [float(sub.iloc[sel]["close"])], color="#e8c547", s=28, zorder=6)

    step = max(len(sub) // 8, 1)
    ticks = list(range(0, len(sub), step))
    if ticks[-1] != len(sub) - 1:
        ticks.append(len(sub) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [sub.iloc[i]["trade_date"].strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", color="#aaaaaa", fontsize=8,
    )
    ax.tick_params(axis="y", colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_ylabel("价格", color="#cccccc")
    ax.set_title(title or "日 K（点击切换日期）", color="#eeeeee", fontsize=12, pad=10)
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.6)


def _result_at_date(stock, target_date):
    """取 target_date 当日（或之前最近交易日）的筹码结果（含 distribution）。"""
    df = stock["df"]
    results = stock["results"]
    target = pd.Timestamp(target_date)
    mask = df["trade_date"] <= target
    if not mask.any():
        return None, None
    idx = int(mask.values.nonzero()[0][-1])
    r = results[idx]
    if "distribution" in r:
        return r, idx
    snap = chip_snapshot(df, idx, stock["checkpoints"], **stock["calc_kw"])
    return snap, idx


def _result_at_idx(stock, idx):
    r = stock["results"][idx]
    if "distribution" in r:
        return r
    return chip_snapshot(stock["df"], idx, stock["checkpoints"], **stock["calc_kw"])


def _empty_chip_ax(ax, label, target_date):
    ax.clear()
    ax.set_facecolor("#1c1c1c")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_title(f"{label}\n{pd.Timestamp(target_date).strftime('%Y-%m-%d')}", color="#eeeeee", fontsize=10)
    ax.text(
        0.5, 0.5, "该日无数据", transform=ax.transAxes,
        ha="center", va="center", color="#888888", fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _slider_idx_min(n, slider_days):
    """滑块起始行号：筹码全历史计算，交互只看最近 slider_days 个交易日。"""
    if not slider_days or slider_days >= n:
        return 0
    return n - int(slider_days)


def show_compare_interactive(stocks, *, plot_step=0.05, kline_window=120, slider_days=30):
    """
    多股同屏对比：上方筹码分布并排，下方第一只股票的 K 线，共享日期滑块。
    stocks: list[dict]，每项含 code、df、results，可选 label。
    slider_days: 滑块只看最近 N 个交易日；None 表示不限制。筹码仍自 IPO 全历史递推。
    """
    from matplotlib.widgets import Slider

    if not stocks:
        raise ValueError("stocks 为空")
    if len(stocks) == 1:
        s = stocks[0]
        show_interactive(
            s["df"], s["results"],
            checkpoints=s.get("checkpoints"),
            calc_kw=s.get("calc_kw"),
            plot_step=plot_step, kline_window=kline_window, slider_days=slider_days,
        )
        return

    for s in stocks:
        if len(s["df"]) != len(s["results"]):
            raise ValueError(f"{s['code']}: df 与 results 行数不一致")
        if "checkpoints" not in s and "distribution" not in s["results"][0]:
            raise ValueError(f"{s['code']}: 缺少 checkpoints")

    all_dates = sorted({
        pd.Timestamp(d)
        for s in stocks
        for d in s["df"]["trade_date"]
    })
    if not all_dates:
        raise ValueError("没有可用交易日")
    if slider_days and len(all_dates) > slider_days:
        all_dates = all_dates[-slider_days:]

    state = {"date_idx": len(all_dates) - 1, "kline_stock": 0}

    n = len(stocks)
    fig_w = max(6 * n, 12)
    fig = plt.figure(figsize=(fig_w, 8), facecolor="#1c1c1c")

    chip_axes = []
    chip_w = 0.88 / n
    chip_x0 = 0.06
    for i in range(n):
        ax = fig.add_axes([chip_x0 + i * chip_w, 0.42, chip_w * 0.92, 0.52])
        chip_axes.append(ax)

    ax_k = fig.add_axes([0.06, 0.12, 0.88, 0.24])
    ax_slider = fig.add_axes([0.12, 0.04, 0.76, 0.025], facecolor="#2a2a2a")

    slider = Slider(
        ax_slider, "日期", 0, len(all_dates) - 1,
        valinit=state["date_idx"], valstep=1, color="#e8c547",
    )
    slider.label.set_color("#cccccc")
    slider.valtext.set_color("#cccccc")

    def stock_label(s):
        return s.get("label") or s["code"]

    def refresh():
        target_date = all_dates[state["date_idx"]]
        for ax, s in zip(chip_axes, stocks):
            result, _ = _result_at_date(s, target_date)
            if result is None:
                _empty_chip_ax(ax, stock_label(s), target_date)
            else:
                _draw_chip_on_ax(
                    ax, result, plot_step=plot_step,
                    label=stock_label(s), show_legend=(n <= 2),
                )

        ks = stocks[state["kline_stock"]]
        _, kidx = _result_at_date(ks, target_date)
        if kidx is not None:
            _draw_kline_on_ax(
                ax_k, ks["df"], kidx, window=kline_window,
                title=f"{stock_label(ks)}  日 K（点击筹码列切换）",
            )
        else:
            ax_k.clear()
            ax_k.set_facecolor("#1c1c1c")
            ax_k.set_title(f"{stock_label(ks)}  日 K", color="#eeeeee")
            ax_k.text(0.5, 0.5, "该日无数据", transform=ax_k.transAxes, ha="center", va="center", color="#888888")

        slider.label.set_text(f"日期  {target_date.strftime('%Y-%m-%d')}")
        fig.canvas.draw_idle()

    def set_date_idx(i):
        state["date_idx"] = max(0, min(int(i), len(all_dates) - 1))
        slider.set_val(state["date_idx"])
        refresh()

    def on_chip_click(event):
        if event.inaxes not in chip_axes:
            return
        state["kline_stock"] = chip_axes.index(event.inaxes)
        refresh()

    def on_key(event):
        if event.key == "left":
            set_date_idx(state["date_idx"] - 1)
        elif event.key == "right":
            set_date_idx(state["date_idx"] + 1)

    def on_slider(val):
        state["date_idx"] = int(val)
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_chip_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    slider.on_changed(on_slider)

    refresh()
    if slider_days:
        print(f"滑块范围：最近 {len(all_dates)} 个交易日（筹码自 IPO 全历史递推）")
    print("操作：← → 切换日期 | 拖滑块 | 点击筹码列切换下方 K 线")
    plt.show()


def show_interactive(df, results, *, checkpoints=None, calc_kw=None, plot_step=0.05, kline_window=120, slider_days=30):
    """
    单股联动：左 K 线 + 右筹码。
    - 点击 K 线选日期
    - 键盘 ← → 切换
    - 底部滑块拖动
    slider_days: 滑块只看最近 N 个交易日；None 表示不限制。
    """
    from matplotlib.widgets import Slider

    if calc_kw is None:
        calc_kw = {"step": 0.01, "decay": 1.0, "mode": "triangle"}

    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(results)
    if n == 0:
        raise ValueError("results 为空")
    if n != len(df):
        raise ValueError("results 行数与 df 不一致，请用同一份 df 调用 calc_chip")
    if checkpoints is None and "distribution" not in results[0]:
        raise ValueError("results 无 distribution，请传入 checkpoints")

    idx_min = _slider_idx_min(n, slider_days)
    state = {"idx": n - 1}

    fig = plt.figure(figsize=(13, 8), facecolor="#1c1c1c")
    ax_k = fig.add_axes([0.06, 0.12, 0.52, 0.82])
    ax_c = fig.add_axes([0.62, 0.12, 0.34, 0.82])
    ax_slider = fig.add_axes([0.12, 0.04, 0.76, 0.025], facecolor="#2a2a2a")

    slider = Slider(
        ax_slider, "日期", idx_min, n - 1,
        valinit=state["idx"], valstep=1, color="#e8c547",
    )
    slider.label.set_color("#cccccc")
    slider.valtext.set_color("#cccccc")

    def refresh():
        i = state["idx"]
        _draw_kline_on_ax(ax_k, df, i, window=kline_window)
        if "distribution" in results[i]:
            chip_result = results[i]
        else:
            chip_result = chip_snapshot(df, i, checkpoints, **calc_kw)
        _draw_chip_on_ax(ax_c, chip_result, plot_step=plot_step)
        date_str = pd.Timestamp(results[i]["trade_date"]).strftime("%Y-%m-%d")
        slider.label.set_text(f"日期  {date_str}")
        fig.canvas.draw_idle()

    def set_idx(i):
        state["idx"] = max(idx_min, min(int(i), n - 1))
        slider.set_val(state["idx"])
        refresh()

    def on_click(event):
        if event.inaxes is not ax_k or event.xdata is None:
            return
        start = max(0, state["idx"] - kline_window + 1)
        local_x = int(round(event.xdata))
        global_i = start + local_x
        if idx_min <= global_i < n:
            set_idx(global_i)

    def on_key(event):
        if event.key == "left":
            set_idx(state["idx"] - 1)
        elif event.key == "right":
            set_idx(state["idx"] + 1)

    def on_slider(val):
        state["idx"] = int(val)
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    slider.on_changed(on_slider)

    refresh()
    if slider_days:
        print(f"滑块范围：最近 {n - idx_min} 个交易日（筹码自 IPO 全历史递推）")
    print("操作：点击 K 线 | ← → 切换 | 拖底部滑块")
    plt.show()


def plot_chip(result, save_path=None, plot_step=0.05):
    """画单日静态筹码图。"""
    fig, ax = plt.subplots(figsize=(6, 9), facecolor="#1c1c1c")
    _draw_chip_on_ax(ax, result, plot_step=plot_step)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"已保存: {save_path}")
    plt.show()


def load_stock(
    csv_path,
    code,
    *,
    start_date=None,
    end_date=None,
    top10_csv=None,
    top10_df=None,
    fetch_top10=False,
    active_ratio=0.2,
    lag_days=TOP10_LAG_DAYS,
    use_decay=True,
    save_enriched=True,
    enriched_path=None,
):
    """
    读取并预处理单只股票, 返回 (df, results, checkpoints, calc_kw).

    参数:
        csv_path     : 日线 CSV 路径.
        code         : 股票代码, 如 'sh.600000'.
        start_date   : 起始日期 (含), 字符串或 None.
        end_date     : 结束日期 (含), 字符串或 None.
        top10_csv    : 前十大股东持股比例 CSV 路径. 与 top10_df/fetch_top10 三选一.
                       CSV 需含日期列和比例列 (详见 load_top10_from_csv 文档).
        top10_df     : 前十大股东持股比例 DataFrame. 优先级高于 top10_csv.
        fetch_top10  : 是否从 AkShare 在线拉取前十大流通股东数据 (需 pip install akshare).
                       优先级最低, 仅当 top10_df 和 top10_csv 均为 None 时生效.
                       会从 code 中提取纯数字代码 (如 'sh.600522' → '600522').
        active_ratio : 前十大股东活跃交易比例, 默认 0.2.
        lag_days     : 季报披露滞后天数, 默认 90.
        use_decay    : 是否启用历史换手衰减系数. False 时退化为原始行为 (decay=1.0).
        save_enriched: 是否把 top10_ratio / decay / modified_turn 三列追加保存到 CSV.
                       默认 True. 保存时会保留原始 CSV 的所有列, 仅在末尾追加新列.
        enriched_path: 增强版 CSV 的保存路径. 默认 None → 覆盖原 csv_path.
                       若想保留原始文件, 可指定新路径如 'data/{code}_enriched.csv'.

    返回:
        (df, results, checkpoints, calc_kw)
        其中 df 在启用衰减时会多出 'decay' / 'top10_ratio' / 'modified_turn' 列.

    CSV 新增列说明 (均为百分数, 与 BaoStock turn 同量纲, 便于直接对比):
        - top10_ratio   : 当日生效的前十大流通股东合计持股比例 (%), 如 48.02
        - decay         : 历史换手衰减系数 (无量纲, ≥1.0), 如 1.6237
        - modified_turn : 修正后换手率 (%) = 原始 turn × decay, 如 3.95
    """
    raw = pd.read_csv(csv_path)
    if "code" in raw.columns:
        raw = raw[raw["code"] == code]
    df = prep_df(raw.copy())
    if start_date:
        df = df[df["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]
    df = df.sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{code}: 过滤后无数据")

    # ---- 历史换手衰减系数 ----
    top10 = None
    if use_decay:
        if top10_df is not None:
            top10 = top10_df
        elif top10_csv is not None:
            top10 = load_top10_from_csv(top10_csv)
        elif fetch_top10:
            # 从 code 中提取纯数字代码: 'sh.600522' → '600522', 'sz.002281' → '002281'
            symbol = "".join(ch for ch in code if ch.isdigit())
            print(f"  [decay] 从 AkShare 拉取 {symbol} 前十大流通股东数据...")
            top10 = fetch_top10_from_akshare(symbol)
            print(f"  [decay] 拉取到 {len(top10)} 个报告期, "
                  f"范围 {top10['report_date'].min().date()} ~ "
                  f"{top10['report_date'].max().date()}")

        if top10 is not None and len(top10) > 0:
            df = attach_decay_to_df(
                df, top10, active_ratio=active_ratio, lag_days=lag_days
            )
            # 新增: 日频 top10_ratio 列 (百分数, 供 CSV 查看)
            df["top10_ratio"] = prepare_top10_daily_series(
                df, top10, lag_days=lag_days, as_percent=True
            )
            # 打印衰减系数统计, 方便调试
            d = df["decay"]
            print(
                f"  [decay] active_ratio={active_ratio}, lag_days={lag_days}, "
                f"min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}  "
                f"latest={d.iloc[-1]:.4f}"
            )
        else:
            df["decay"] = 1.0
            df["top10_ratio"] = 0.0
            print(f"  [decay] 未提供前十大股东数据, decay=1.0 (不调整)")
    else:
        # 不启用衰减, 确保无 decay 列干扰 (若已存在则置 1.0)
        df["decay"] = 1.0
        df["top10_ratio"] = 0.0

    # ---- 新增: 修正后换手率 modified_turn ----
    # modified_turn = 原始换手率 × 衰减系数
    # turnover_rate 已是小数 (prep_df 把 turn/100), 转回百分数与 BaoStock turn 同量纲
    if "turnover_rate" in df.columns:
        df["modified_turn"] = (df["turnover_rate"] * df["decay"]) * 100.0
    else:
        df["modified_turn"] = 0.0

    calc_kw = {"step": 0.01, "decay": 1.0, "mode": "triangle"}
    results, checkpoints = calc_chip(df, **calc_kw)

    # ---- 新增: 保存增强版 CSV (top10_ratio / decay / modified_turn) ----
    if save_enriched:
        _save_enriched_csv(raw, df, csv_path, enriched_path, code)

    return df, results, checkpoints, calc_kw


def _save_enriched_csv(raw, df, csv_path, enriched_path, code):
    """
    把 top10_ratio / decay / modified_turn 三列追加到原始 CSV 并保存.

    策略: 保留原始 raw 的所有列和列名 (如 BaoStock 的 date/turn/...),
    仅在末尾追加新列. 用 trade_date 做对齐 (raw 的日期列可能是 date/turnover_rate 等).

    参数:
        raw          : 原始读取的 DataFrame (prep_df 之前).
        df           : 处理后的 DataFrame (含 decay/top10_ratio/modified_turn).
        csv_path     : 原始 CSV 路径.
        enriched_path: 保存路径. None 则覆盖 csv_path.
        code         : 股票代码 (仅用于日志).
    """
    save_path = enriched_path or csv_path

    # 找到 raw 中的日期列
    raw_date_col = _pick_col(raw.columns, COLUMN_ALIASES["trade_date"])
    if raw_date_col is None:
        # 找不到日期列, 直接用 df 的标准化列保存
        out = df.copy()
    else:
        # 用日期做对齐, 把新列合并回 raw
        raw_copy = raw.copy()
        raw_copy[raw_date_col] = pd.to_datetime(raw_copy[raw_date_col], errors="coerce")
        # 取 df 中需要的列
        new_cols = ["trade_date", "top10_ratio", "decay", "modified_turn"]
        new_cols = [c for c in new_cols if c in df.columns]
        df_new = df[new_cols].copy()
        df_new = df_new.rename(columns={"trade_date": raw_date_col})
        # 合并
        out = raw_copy.merge(df_new, on=raw_date_col, how="left")

    # 把日期列格式化为字符串 (避免 2025-01-01 变成 2025-01-01 00:00:00)
    date_col_out = _pick_col(out.columns, COLUMN_ALIASES["trade_date"])
    if date_col_out is not None:
        out[date_col_out] = pd.to_datetime(out[date_col_out]).dt.strftime("%Y-%m-%d")

    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [save] 已保存增强数据到 {save_path} "
          f"(新增列: top10_ratio, decay, modified_turn)")


def csv_path_for_code(code: str, data_dir="data") -> Path:
    return Path(data_dir) / f"{code.replace('.', '_')}_daily.csv"


def chip_to_df(results):
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "distribution"}
        row["top_peak_price"] = r["peaks"][0][0] if r["peaks"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # 多股对比：每项 code + 可选 label；csv 默认 data/{code}_daily.csv
    # 可选 top10_csv: 前十大股东持股比例 CSV, 用于计算历史换手衰减系数
    # 可选 fetch_top10: True 时从 AkShare 在线拉取前十大流通股东数据 (需 pip install akshare)
    # 可选 active_ratio: 前十大股东活跃交易比例, 默认 0.2
    STOCKS = [
        {"code": "sz.002384", "label": "东山精密"},
        {"code": "sz.002281", "label": "光迅科技"},
        {"code": "sh.601869", "label": "长飞光纤"},
        {"code": "sh.600498", "label": "烽火通信"},
        {"code": "sh.600522", "label": "中天科技"},
    ]

    start_date = "2025-01-01"   # "2021-01-01"
    end_date = None     # "2026-06-15"

    # ---- 历史换手衰减系数开关 ----
    # use_decay=True 时, 按优先级获取前十大股东数据:
    #   1) top10_csv (item 中指定 CSV 文件)
    #   2) fetch_top10=True (从 AkShare 在线拉取, 推荐)
    #   3) 都没有 → decay=1.0 (退化为原始行为, 向后兼容)
    use_decay = True
    fetch_top10 = True   # 从 AkShare 自动拉取前十大流通股东数据
    active_ratio = 0.2   # 前十大股东活跃交易比例估计值

    # ---- CSV 增强保存开关 ----
    # save_enriched=True 时, 把 top10_ratio / decay / modified_turn 三列
    # 追加保存到每只股票的 CSV 末尾 (保留原始列, 不破坏 BaoStock 数据格式).
    # enriched_path=None → 覆盖原 CSV; 也可指定新路径保留原文件.
    save_enriched = True

    stocks = []
    for item in STOCKS:
        code = item["code"]
        path = item.get("csv") or csv_path_for_code(code)
        print(f"\n=== {item.get('label') or code} ({path}) ===")
        df, results, checkpoints, calc_kw = load_stock(
            path, code,
            start_date=start_date, end_date=end_date,
            top10_csv=item.get("top10_csv"),
            fetch_top10=item.get("fetch_top10", fetch_top10),
            active_ratio=active_ratio,
            use_decay=use_decay,
            save_enriched=save_enriched,
        )
        print(f"样本: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}，共 {len(df)} 天")
        last = results[-1]
        print(f"最新 {last['trade_date'].date()}  收盘 {last['close']:.2f}  "
              f"平均成本 {last['avg_cost']:.2f}  获利 {last['profit_ratio']:.2%}")
        stocks.append({
            "code": code,
            "label": item.get("label"),
            "df": df,
            "results": results,
            "checkpoints": checkpoints,
            "calc_kw": calc_kw,
        })

    show_compare_interactive(stocks, plot_step=0.05, kline_window=120, slider_days=30)


if __name__ == "__main__":
    main()
