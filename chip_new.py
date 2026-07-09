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
    "report_date": ("report_date", "date", "trade_date", "报告期", "日期"),
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
def _prepare_top10_table(top10_ratios: pd.DataFrame, lag_days: int = TOP10_LAG_DAYS) -> pd.DataFrame:
    """
    清洗前十大股东持股比例表, 生成 effective_date 列.

    生效日期规则:
        - 优先用真实的公告日期 pub_date (若存在). 从 pub_date 起, 该比例生效,
          直到下一次公告发布新数据.
        - 若没有 pub_date, 退化为 report_date + lag_days (季报披露滞后估计).

    返回 DataFrame, 含列: report_date, top10_ratio, effective_date, (可选 pub_date).
    按 effective_date 升序排列.
    """
    date_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["report_date"])
    ratio_col = _pick_col(top10_ratios.columns, TOP10_ALIASES["top10_ratio"])
    if date_col is None or ratio_col is None:
        raise ValueError(
            "top10_ratios 缺少必要列: 需要日期列 (report_date/date/公告日期) "
            "和比例列 (top10_ratio/前十大股东持股比例)"
        )

    cols = [date_col, ratio_col]
    pub_col = _pick_col(top10_ratios.columns, ("pub_date", "公告日期", "announcement_date", "披露日期"))
    if pub_col is not None and pub_col not in cols:
        cols.append(pub_col)

    ratios = top10_ratios[cols].copy()
    new_names = ["report_date", "top10_ratio"]
    if pub_col is not None:
        new_names.append("pub_date")
    ratios.columns = new_names

    ratios["report_date"] = pd.to_datetime(ratios["report_date"], errors="coerce")
    ratios["top10_ratio"] = pd.to_numeric(ratios["top10_ratio"], errors="coerce")
    if "pub_date" in ratios.columns:
        ratios["pub_date"] = pd.to_datetime(ratios["pub_date"], errors="coerce")
    ratios = ratios.dropna(subset=["report_date", "top10_ratio"])

    if "pub_date" in ratios.columns and ratios["pub_date"].notna().any():
        ratios["effective_date"] = ratios["pub_date"]
    else:
        ratios["effective_date"] = ratios["report_date"] + pd.Timedelta(days=lag_days)

    ratios = ratios.sort_values("effective_date").reset_index(drop=True)
    return ratios


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
                       - 可选: 公告日期列 (pub_date / 公告日期), 若存在则用作生效日期.
        active_ratio : 前十大股东活跃交易比例, 默认 0.2.
        lag_days     : 披露滞后天数, 默认 90. 仅当无 pub_date 时用作生效日期估计.

    返回:
        list[float], 长度等于 len(df), 每个元素为对应交易日的衰减系数.
        若某交易日之前无任何已生效的季报数据, 则该日衰减系数为 1.0.

    生效日期规则:
        - 有 pub_date: 从公告日起该比例生效, 直到下次公告发布新数据.
          例: 2025Q4 数据 pub_date=2026-01-28, 2026Q1 数据 pub_date=2026-04-30,
          则 2026-01-28 ~ 2026-04-29 用 2025Q4 比例, 2026-04-30 起用 2026Q1 比例.
        - 无 pub_date: 用 report_date + lag_days 估计生效日 (避免 look-ahead bias).
    """
    n = len(df)
    if top10_ratios is None or len(top10_ratios) == 0:
        return [1.0] * n

    ratios = _prepare_top10_table(top10_ratios, lag_days=lag_days)
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
        top10_ratios : 前十大股东持股比例表 (含 report_date 和 top10_ratio 列,
                       可选 pub_date).
        lag_days     : 披露滞后天数, 默认 90. 仅当无 pub_date 时用作生效日期估计.
        as_percent   : True 返回百分数 (如 48.02 表示 48.02%),
                       False 返回小数 (如 0.4802).

    返回:
        list[float], 长度等于 len(df).
        若某交易日之前无已生效季报, 则该日为 0.0.

    生效日期规则:
        - 有 pub_date: 从公告日起该比例生效, 直到下次公告发布新数据.
        - 无 pub_date: 用 report_date + lag_days 估计生效日.
    """
    n = len(df)
    if top10_ratios is None or len(top10_ratios) == 0:
        return [0.0] * n

    ratios = _prepare_top10_table(top10_ratios, lag_days=lag_days)
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


# ============================================================
# 筹码集中度与分布区间 (同花顺口径)
# ============================================================
#
# 【同花顺"集中度"是什么意思】
#   集中度衡量筹码在价格轴上的"聚拢/分散"程度, 是一个无量纲的比值.
#   同花顺默认显示两个值: 90%集中度 和 70%集中度.
#
# 【计算步骤 (以 90% 集中度为例)】
#   1. 把所有筹码按价格从低到高排序, 逐价位累加权重 (累计分布函数 CDF).
#   2. 找到累计权重 = 5%  的价格点 P5  (即 5% 的筹码在此价格以下)
#      找到累计权重 = 95% 的价格点 P95 (即 95% 的筹码在此价格以下)
#      → [P5, P95] 这个价格区间内恰好包含了 90% 的筹码, 称为"90%筹码分布区间"
#   3. 90% 集中度 = (P95 - P5) / (P95 + P5)
#
#      70% 集中度同理: 用 P15 和 P85, 区间内含 70% 筹码
#      70% 集中度 = (P85 - P15) / (P85 + P15)
#
# 【为什么用 (P95 - P5) / (P95 + P5), 而不是 (P95 - P5) / 均价 ?】
#
#   核心目的: 消除价格量纲, 让不同价位的股票可以横向比较.
#
#   举例: 两只股票, 筹码都集中在 ±0.5 元的窄区间内
#     A 股: 价格 10 元, P5=9.5, P95=10.5
#     B 股: 价格 100 元, P5=99.5, P95=100.5
#
#   如果用绝对宽度 (P95 - P5):
#     A 股宽度 = 1.0 元
#     B 股宽度 = 1.0 元
#     → 看起来一样集中, 但实际上 B 股的 1 元相对于 100 元只占 1%,
#       而 A 股的 1 元相对于 10 元占 10%, A 股明显更分散!
#
#   如果用 (P95 - P5) / 均价:
#     A 股 = 1.0 / 10.0 = 10%
#     B 股 = 1.0 / 100.0 = 1%
#     → 这个能正确反映相对集中度, 但依赖"均价"这个外部参考,
#       而均价本身可能被极端价位拉偏 (如长期套牢盘在很低的价位).
#
#   用 (P95 - P5) / (P95 + P5):
#     A 股 = (10.5 - 9.5) / (10.5 + 9.5) = 1.0 / 20.0 = 5.0%
#     B 股 = (100.5 - 99.5) / (100.5 + 99.5) = 1.0 / 200.0 = 0.5%
#     → 同样能正确反映相对集中度, 且只用 P5/P95 自身,
#       不依赖外部参考, 数值更稳定. 这就是同花顺采用此公式的原因.
#
#   数学上, (P95 - P5) / (P95 + P5) 等价于:
#     令 m = (P95 + P5)/2 (区间中点), d = (P95 - P5)/2 (半宽)
#     则 集中度 = d / m = 半宽 / 中点
#     即"区间宽度相对于区间中心的比例", 是一个标准的相对离散度指标.
#
# 【为什么集中度越小越好 (对多头而言)】
#
#   集中度小 = 筹码高度集中在某个窄价格区间 = 大多数人的成本价很接近.
#   这通常意味着:
#
#   1. 主力控盘程度高: 筹码集中在主力成本区, 散户筹码少,
#      主力拉升时抛压轻, 容易走出主升浪.
#
#   2. 横盘洗盘充分: 长期在一个区间震荡, 浮筹已被清洗,
#      剩下的都是坚定持有者, 一旦突破容易加速.
#
#   3. 支撑/阻力明确: 筹码密集区会成为强支撑 (多头成本线)
#      或强阻力 (套牢盘解套区), 便于判断买卖点.
#
#   反之, 集中度大 = 筹码分散在很宽的价格区间 = 持仓成本差异大:
#     - 多空分歧严重, 上涨时高位套牢盘会解套抛售, 下跌时低位获利盘会止盈
#     - 缺乏一致预期, 走势容易反复震荡, 难以形成趋势
#
#   【经验阈值 (同花顺常见解读)】
#     90% 集中度 < 10%  : 非常集中, 筹码高度锁定 (主力控盘 / 长期横盘)
#     90% 集中度 10~20% : 较为集中
#     90% 集中度 20~30% : 较为分散
#     90% 集中度 > 30%  : 非常分散, 筹码松散 (多空分歧大)
#
#   【注意】"越小越好"是针对多头趋势中的回调/横盘阶段.
#   在高位集中度骤降可能是主力出货 (筹码从集中变分散转移给散户),
#   需结合价格位置和获利比例综合判断.
# ============================================================

def _chip_percentile_price(chip, pct):
    """
    返回筹码累计权重达到 pct% 时对应的价格 (线性插值).

    参数:
        chip : {price: weight} 筹码分布.
        pct  : 目标累计百分比, 0~100. 如 5 表示 5%, 95 表示 95%.

    返回:
        float, 对应价格.

    说明:
        把筹码按价格升序排列, 累加权重, 找到累计权重首次 >= pct% 的价格点.
        在相邻两点间线性插值, 提高精度.
    """
    if not chip:
        return 0.0
    prices = sorted(chip)
    total = sum(chip.values())
    if total <= 0:
        return prices[0]

    target = total * (pct / 100.0)
    cum = 0.0
    prev_price = prices[0]
    prev_cum = 0.0
    for p in prices:
        new_cum = cum + chip[p]
        if new_cum >= target:
            # 在 prev_price 和 p 之间线性插值
            if new_cum > prev_cum and p > prev_price:
                frac = (target - prev_cum) / (new_cum - prev_cum)
                return prev_price + frac * (p - prev_price)
            return p
        prev_price = p
        prev_cum = new_cum
        cum = new_cum
    return prices[-1]


def _concentration(chip):
    """
    计算同花顺口径的筹码集中度与分布区间.

    返回 dict:
        {
            "p90_low":  90%区间下沿 (P5),
            "p90_high": 90%区间上沿 (P95),
            "p90_width": 90%区间宽度 = P95 - P5,
            "p90_concentration": 90%集中度 = (P95-P5)/(P95+P5),
            "p70_low":  70%区间下沿 (P15),
            "p70_high": 70%区间上沿 (P85),
            "p70_width": 70%区间宽度 = P85 - P15,
            "p70_concentration": 70%集中度 = (P85-P15)/(P85+P15),
        }
    """
    if not chip:
        return {
            "p90_low": 0.0, "p90_high": 0.0, "p90_width": 0.0, "p90_concentration": 0.0,
            "p70_low": 0.0, "p70_high": 0.0, "p70_width": 0.0, "p70_concentration": 0.0,
        }

    p5 = _chip_percentile_price(chip, 5)
    p95 = _chip_percentile_price(chip, 95)
    p15 = _chip_percentile_price(chip, 15)
    p85 = _chip_percentile_price(chip, 85)

    def conc(lo, hi):
        denom = lo + hi
        if denom <= 0:
            return 0.0
        return (hi - lo) / denom

    return {
        "p90_low": p5,
        "p90_high": p95,
        "p90_width": p95 - p5,
        "p90_concentration": conc(p5, p95),
        "p70_low": p15,
        "p70_high": p85,
        "p70_width": p85 - p15,
        "p70_concentration": conc(p15, p85),
    }


def _period_cost(df, idx, period):
    """
    计算最近 period 个交易日内的平均建仓成本 (周期成本).

    原理:
        从 idx 往前数 period 天, 用这段时间的成交量加权平均价格 (VWAP)
        作为"周期成本", 反映近期入场资金的平均持仓成本.

    参数:
        df     : 日线 DataFrame (已升序).
        idx    : 当前行号 (0-based).
        period : 周期天数, 如 5/10/20/30.

    返回:
        float, 周期内成交量加权平均价格. 数据不足时返回 NaN.
    """
    start = max(0, idx - period + 1)
    sub = df.iloc[start: idx + 1]
    if sub.empty:
        return float("nan")

    vols = sub["volume"].astype(float).values
    if "amount" in sub.columns:
        amounts = sub["amount"].astype(float).values
        total_vol = vols.sum()
        if total_vol <= 0:
            return float("nan")
        return float(amounts.sum() / total_vol)
    else:
        prices = (sub["open"] + sub["high"] + sub["low"] + sub["close"]).values / 4.0
        total_vol = vols.sum()
        if total_vol <= 0:
            return float("nan")
        return float((prices * vols).sum() / total_vol)


def _snapshot_from_chip(chip, row):
    close = float(row["close"])
    dist = dict(chip)
    conc = _concentration(dist)
    snap = {
        "trade_date": row["trade_date"],
        "close": close,
        "avg_cost": _avg_cost(dist),
        "profit_ratio": _profit_ratio(dist, close),
        "peaks": _peaks(dist),
        "distribution": dist,
    }
    # 合并集中度指标
    snap.update(conc)
    return snap


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


def _chip_ylim(binned, close, avg, peaks, pad=0.5, extra_anchors=None):
    """Y 轴只框住筹码密集区，别从上市最低价拉到最高价。

    extra_anchors: 额外需要纳入 y 轴范围的价格点 (如 90%/70% 区间上下沿),
                   保证这些边界线不会被裁掉。
    """
    anchors = [close, avg]
    if peaks:
        anchors.extend(p for p, _ in peaks[:8])#有筹码峰，就取权重最大的前8个峰的价格
    elif binned:#没有筹码峰，就取权重最大的前8个价格点
        top = sorted(binned.items(), key=lambda x: x[1], reverse=True)[:8]
        anchors.extend(p for p, _ in top)
    if extra_anchors:
        for a in extra_anchors:
            if a is None:
                continue
            try:
                if a == a and a > 0:  # 排除 NaN 和 0
                    anchors.append(float(a))
            except (TypeError, ValueError):
                continue

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


def _draw_chip_on_ax(ax, result, plot_step=0.05, *, label=None, show_legend=True,
                     period_costs=None):
    """在指定 axes 上画单日筹码分布（供静态图和交互图复用）。

    参数:
        result       : _snapshot_from_chip 返回的快照 (含 distribution/concentration 等).
        plot_step    : 画图时的价格档位步长.
        label        : 子图标题前缀 (如股票名称).
        show_legend  : 是否显示图例.
        period_costs : 可选, dict, 如 {5: 10.2, 10: 10.1, 20: 10.0, 30: 9.9}
                       周期成本, 会在图上画水平线并在信息框中显示.
    """
    dist = result["distribution"]
    close = result["close"]
    avg = result["avg_cost"]
    date = result["trade_date"]
    profit = result["profit_ratio"]
    peaks = result.get("peaks", [])

    # 集中度指标 (可能不存在于旧格式 result 中, 用 get 兜底)
    p90_lo = result.get("p90_low", 0.0)
    p90_hi = result.get("p90_high", 0.0)
    p90_conc = result.get("p90_concentration", 0.0)
    p70_lo = result.get("p70_low", 0.0)
    p70_hi = result.get("p70_high", 0.0)
    p70_conc = result.get("p70_concentration", 0.0)

    binned = _rebin_dist(dist, step=plot_step)
    if not binned:
        raise ValueError("distribution 为空")

    # 把 90%/70% 区间上下沿也纳入 y 轴范围, 避免边界线被裁掉
    extra_anchors = [p90_lo, p90_hi, p70_lo, p70_hi]
    ylo, yhi = _chip_ylim(binned, close, avg, peaks, extra_anchors=extra_anchors)
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
    # 平均成本线 (白色实线)
    ax.axhline(avg, color="white", linewidth=1, zorder=5)
    # 收盘价线 (黄色虚线)
    ax.axhline(close, color="#e8c547", linewidth=1.2, linestyle="--", zorder=5)

    # 90% 筹码分布区间 (绿色半透明带 + 点线边界)
    if p90_hi > p90_lo > 0:
        ax.axhspan(p90_lo, p90_hi, color="#22c55e", alpha=0.08, zorder=1)
        ax.axhline(p90_lo, color="#22c55e", linewidth=0.8, linestyle=":", alpha=0.7, zorder=4)
        ax.axhline(p90_hi, color="#22c55e", linewidth=0.8, linestyle=":", alpha=0.7, zorder=4)
    # 70% 筹码分布区间 (橙色半透明带 + 点线边界)
    if p70_hi > p70_lo > 0:
        ax.axhspan(p70_lo, p70_hi, color="#f59e0b", alpha=0.10, zorder=1)
        ax.axhline(p70_lo, color="#f59e0b", linewidth=0.8, linestyle=":", alpha=0.7, zorder=4)
        ax.axhline(p70_hi, color="#f59e0b", linewidth=0.8, linestyle=":", alpha=0.7, zorder=4)

    # 周期成本线 (不同颜色的点划线)
    period_colors = {5: "#a78bfa", 10: "#60a5fa", 20: "#06b6d4", 30: "#fbbf24"}
    if period_costs:
        for p, cost in period_costs.items():
            if cost and cost == cost:  # 排除 NaN
                c = period_colors.get(p, "#888888")
                ax.axhline(cost, color=c, linewidth=1.0, linestyle="-.", alpha=0.8, zorder=4)

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

    # ---- 信息框: 收盘/平均成本/获利比例/集中度/周期成本 ----
    # info_lines = [
    #     f"收盘 {close:.2f}",
    #     f"平均成本 {avg:.2f}",
    #     f"获利比例 {profit:.2%}",
    #     f"90%集中 {p90_conc:.1%}  [{p90_lo:.2f}, {p90_hi:.2f}]",
    #     f"70%集中 {p70_conc:.1%}  [{p70_lo:.2f}, {p70_hi:.2f}]",
    # ]
    # if period_costs:
    #     for p in sorted(period_costs.keys()):
    #         cost = period_costs[p]
    #         if cost and cost == cost:  # 排除 NaN
    #             info_lines.append(f"{p}周期成本 {cost:.2f}")
    # ax.text(
    #     0.98, 0.02,
    #     "\n".join(info_lines),
    #     transform=ax.transAxes, ha="right", va="bottom",
    #     color="#dddddd", fontsize=7.5,#family="monospace",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a2a", edgecolor="#555555", alpha=0.92),
    # )

    if show_legend:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="#d63b3b", lw=6, label="获利盘"),
            Line2D([0], [0], color="#3b7ddd", lw=6, label="套牢盘"),
            Line2D([0], [0], color="white", lw=1.8, label=f"平均成本 {avg:.2f}"),
            Line2D([0], [0], color="#e8c547", lw=1.2, linestyle="--", label=f"收盘 {close:.2f}"),
            Line2D([0], [0], color="#22c55e", lw=1.0, linestyle=":", label=f"90%区间 集中{p90_conc:.1%}"),
            Line2D([0], [0], color="#f59e0b", lw=1.0, linestyle=":", label=f"70%区间 集中{p70_conc:.1%}"),
        ]
        if period_costs:
            for p in sorted(period_costs.keys()):
                c = period_colors.get(p, "#888888")
                legend_handles.append(
                    Line2D([0], [0], color=c, lw=1.0, linestyle="-.", label=f"{p}周期成本")
                )
        ax.legend(
            handles=legend_handles,
            loc="upper right", framealpha=0.85,
            facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#dddddd", fontsize=6.5,
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


ENRICH_COLS = ("top10_ratio", "decay", "modified_turn")


def _coalesce_enrich_cols(df: pd.DataFrame) -> pd.DataFrame:
    """合并 CSV 中 decay / decay_x / decay_y 等重复列，只保留标准三列。"""
    out = df.copy()
    for base in ENRICH_COLS:
        candidates = [c for c in out.columns if c == base or c.startswith(f"{base}_")]
        if not candidates:
            continue
        merged = None
        for c in candidates:
            s = pd.to_numeric(out[c], errors="coerce")
            merged = s if merged is None else merged.fillna(s)
        out[base] = merged
        for c in candidates:
            if c != base and c in out.columns:
                out = out.drop(columns=[c])
    return out


def _apply_top10_decay(df, top10, *, active_ratio, lag_days):
    """对整段日线附加 decay / top10_ratio / modified_turn。"""
    df = attach_decay_to_df(df, top10, active_ratio=active_ratio, lag_days=lag_days)
    df["top10_ratio"] = prepare_top10_daily_series(df, top10, lag_days=lag_days, as_percent=True)
    if "turnover_rate" in df.columns:
        df["modified_turn"] = (df["turnover_rate"] * df["decay"]) * 100.0
    else:
        df["modified_turn"] = 0.0
    return df


def _decay_label(on: bool) -> str:
    return f"十大股东修正: {'开' if on else '关'}"


def _set_decay_mode(stocks, state, on: bool):
    state["decay_on"] = bool(on)
    for s in stocks:
        s["decay_on"] = bool(on and s.get("has_decay_toggle"))
    print(f"  [十大股东修正] {'开' if state['decay_on'] else '关'}")


def _stock_chip_pack(stock):
    """按当前 decay_on 返回 (results, checkpoints, df_chip)。"""
    if stock.get("decay_on") and stock.get("has_decay_toggle"):
        return stock["results_adj"], stock["checkpoints_adj"], stock["df"]
    return stock["results_raw"], stock["checkpoints_raw"], stock["df_raw"]


def _result_at_date(stock, target_date):
    """取 target_date 当日（或之前最近交易日）的筹码结果（含 distribution）。"""
    results, checkpoints, df = _stock_chip_pack(stock)
    target = pd.Timestamp(target_date)
    mask = df["trade_date"] <= target
    if not mask.any():
        return None, None
    idx = int(mask.values.nonzero()[0][-1])
    r = results[idx]
    if "distribution" in r:
        return r, idx
    snap = chip_snapshot(df, idx, checkpoints, **stock["calc_kw"])
    return snap, idx


def _result_at_idx(stock, idx):
    results, checkpoints, df = _stock_chip_pack(stock)
    r = results[idx]
    if "distribution" in r:
        return r
    return chip_snapshot(df, idx, checkpoints, **stock["calc_kw"])


def _add_decay_toggle_button(fig, stocks, state, refresh, *, default_on=False, rect=(0.38, 0.535, 0.24, 0.045)):
    """十大股东换手率修正开关；无十大股东数据时不显示。返回 toggle() 供键盘 T 调用。"""
    if not any(s.get("has_decay_toggle") for s in stocks):
        return None

    from matplotlib.widgets import Button

    _set_decay_mode(stocks, state, default_on)

    ax_btn = fig.add_axes(rect)
    ax_btn.set_zorder(200)
    ax_btn.set_navigate(False)
    btn = Button(ax_btn, _decay_label(state["decay_on"]), color="#2563eb", hovercolor="#3b82f6")
    btn.label.set_color("#ffffff")
    btn.label.set_fontsize(10)
    btn.hovercolor = "#3b82f6"

    def toggle(_event=None):
        _set_decay_mode(stocks, state, not state["decay_on"])
        btn.label.set_text(_decay_label(state["decay_on"]))
        refresh()
        fig.canvas.draw_idle()

    btn.on_clicked(toggle)
    state["_decay_btn"] = btn
    return toggle


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


# def show_compare_interactive(stocks, *, plot_step=0.05, kline_window=120, slider_days=30,
#                              periods=(5, 10, 20, 30)):
#     """
#     多股同屏对比：上方筹码分布并排，下方第一只股票的 K 线，共享日期滑块。
#     stocks: list[dict]，每项含 code、df、results，可选 label。
#     slider_days: 滑块只看最近 N 个交易日；None 表示不限制。筹码仍自 IPO 全历史递推。
#     periods: 周期成本列表, 默认 (5, 10, 20, 30). 传 None 或 () 则不显示周期成本。
#     """
#     from matplotlib.widgets import Slider

#     if not stocks:
#         raise ValueError("stocks 为空")
#     if len(stocks) == 1:
#         s = stocks[0]
#         show_interactive(
#             s["df"], s["results"],
#             checkpoints=s.get("checkpoints"),
#             calc_kw=s.get("calc_kw"),
#             plot_step=plot_step, kline_window=kline_window, slider_days=slider_days,
#             periods=periods,
#         )
#         return

#     for s in stocks:
#         if len(s["df"]) != len(s["results"]):
#             raise ValueError(f"{s['code']}: df 与 results 行数不一致")
#         if "checkpoints" not in s and "distribution" not in s["results"][0]:
#             raise ValueError(f"{s['code']}: 缺少 checkpoints")

#     all_dates = sorted({
#         pd.Timestamp(d)
#         for s in stocks
#         for d in s["df"]["trade_date"]
#     })
#     if not all_dates:
#         raise ValueError("没有可用交易日")
#     if slider_days and len(all_dates) > slider_days:
#         all_dates = all_dates[-slider_days:]

#     state = {"date_idx": len(all_dates) - 1, "kline_stock": 0}

#     n = len(stocks)
#     fig_w = max(6 * n, 12)
#     # 画布加高, 给筹码图和 K 线之间留出更大间距
#     fig = plt.figure(figsize=(fig_w, 9), facecolor="#1c1c1c")

#     # ---- 布局调整: 增加筹码图与 K 线之间的间距 ----
#     # 原布局: 筹码图 y=[0.42, 0.94], K线 y=[0.12, 0.36], 间距 0.06
#     # 新布局: 筹码图 y=[0.46, 0.92] (高0.46), K线 y=[0.08, 0.34] (高0.26), 间距 0.12
#     chip_axes = []
#     chip_w = 0.88 / n
#     chip_x0 = 0.06
#     chip_bottom = 0.46    # 筹码图底部
#     chip_height = 0.46    # 筹码图高度
#     kline_bottom = 0.08   # K线底部
#     kline_height = 0.26   # K线高度
#     for i in range(n):
#         ax = fig.add_axes([chip_x0 + i * chip_w, chip_bottom, chip_w * 0.92, chip_height])
#         chip_axes.append(ax)

#     ax_k = fig.add_axes([0.06, kline_bottom, 0.88, kline_height])
#     ax_slider = fig.add_axes([0.12, 0.03, 0.76, 0.022], facecolor="#2a2a2a")

#     slider = Slider(
#         ax_slider, "日期", 0, len(all_dates) - 1,
#         valinit=state["date_idx"], valstep=1, color="#e8c547",
#     )
#     slider.label.set_color("#cccccc")
#     slider.valtext.set_color("#cccccc")

#     def stock_label(s):
#         return s.get("label") or s["code"]

#     def _calc_period_costs(s, idx):
#         """计算指定股票在 idx 处的周期成本 dict."""
#         if not periods or idx is None:
#             return None
#         return {p: _period_cost(s["df"], idx, p) for p in periods}

#     def refresh():
#         target_date = all_dates[state["date_idx"]]
#         for ax, s in zip(chip_axes, stocks):
#             result, idx = _result_at_date(s, target_date)
#             if result is None:
#                 _empty_chip_ax(ax, stock_label(s), target_date)
#             else:
#                 pc = _calc_period_costs(s, idx)
#                 _draw_chip_on_ax(
#                     ax, result, plot_step=plot_step,
#                     label=stock_label(s), show_legend=(n <= 2),
#                     period_costs=pc,
#                 )

#         ks = stocks[state["kline_stock"]]
#         _, kidx = _result_at_date(ks, target_date)
#         if kidx is not None:
#             _draw_kline_on_ax(
#                 ax_k, ks["df"], kidx, window=kline_window,
#                 title=f"{stock_label(ks)}  日 K（点击筹码列切换）",
#             )
#         else:
#             ax_k.clear()
#             ax_k.set_facecolor("#1c1c1c")
#             ax_k.set_title(f"{stock_label(ks)}  日 K", color="#eeeeee")
#             ax_k.text(0.5, 0.5, "该日无数据", transform=ax_k.transAxes, ha="center", va="center", color="#888888")

#         slider.label.set_text(f"日期  {target_date.strftime('%Y-%m-%d')}")
#         fig.canvas.draw_idle()

#     def set_date_idx(i):
#         state["date_idx"] = max(0, min(int(i), len(all_dates) - 1))
#         slider.set_val(state["date_idx"])
#         refresh()

#     def on_chip_click(event):
#         if event.inaxes not in chip_axes:
#             return
#         state["kline_stock"] = chip_axes.index(event.inaxes)
#         refresh()

#     def on_key(event):
#         if event.key == "left":
#             set_date_idx(state["date_idx"] - 1)
#         elif event.key == "right":
#             set_date_idx(state["date_idx"] + 1)

#     def on_slider(val):
#         state["date_idx"] = int(val)
#         refresh()

#     fig.canvas.mpl_connect("button_press_event", on_chip_click)
#     fig.canvas.mpl_connect("key_press_event", on_key)
#     slider.on_changed(on_slider)

#     refresh()
#     if slider_days:
#         print(f"滑块范围：最近 {len(all_dates)} 个交易日（筹码自 IPO 全历史递推）")
#     print("操作：← → 切换日期 | 拖滑块 | 点击筹码列切换下方 K 线")
#     plt.show()
def show_compare_interactive(stocks, *, plot_step=0.05, kline_window=120, slider_days=30, periods=(5, 10, 20, 30),
                             decay_default_on=False):
    from matplotlib.widgets import Slider
    from matplotlib.lines import Line2D
    
    if not stocks:
        raise ValueError("stocks 为空")
    if len(stocks) == 1:
        show_interactive(
            stocks[0],
            plot_step=plot_step, kline_window=kline_window, slider_days=slider_days, periods=periods,
            decay_default_on=decay_default_on,
        )
        return
    
    for s in stocks:
        if "results_raw" not in s or "checkpoints_raw" not in s:
            raise ValueError(f"{s['code']}: 缺少 results_raw/checkpoints_raw，请用 load_stock 加载")
        results, _, df = _stock_chip_pack(s)
        if len(s["df"]) != len(results):
            raise ValueError(f"{s['code']}: df 与 results 行数不一致")
    
    all_dates = sorted({pd.Timestamp(d) for s in stocks for d in s["df"]["trade_date"]})
    if not all_dates:
        raise ValueError("没有可用交易日")
    if slider_days and len(all_dates) > slider_days:
        all_dates = all_dates[-slider_days:]
    
    state = {"date_idx": len(all_dates) - 1, "kline_stock": 0, "decay_on": decay_default_on}
    n = len(stocks)
    
    # 调整画布尺寸和布局参数
    fig_w = max(6 * n, 12)
    fig = plt.figure(figsize=(fig_w, 10), facecolor="#1c1c1c")
    
    # 布局参数调整
    chip_x0 = 0.06
    chip_width = 0.88 / n
    chip_bottom = 0.60
    chip_height = 0.33
    info_bottom = 0.35#底部位置
    info_height = 0.15# 增加信息框高度以容纳图例
    kline_bottom = 0.10
    kline_height = 0.20
    
    chip_axes = []
    info_axes = []
    
    # 创建筹码图和信息框axes
    for i in range(n):
        ax_chip = fig.add_axes([chip_x0 + i * chip_width, chip_bottom, chip_width * 0.92, chip_height])
        chip_axes.append(ax_chip)
        
        ax_info = fig.add_axes([chip_x0 + i * chip_width, info_bottom, chip_width * 0.92, info_height])
        info_axes.append(ax_info)
    
    # K线图axes
    ax_k = fig.add_axes([0.06, kline_bottom, 0.88, kline_height])
    
    # 滑块axes
    ax_slider = fig.add_axes([0.12, 0.01, 0.76, 0.022], facecolor="#2a2a2a")
    slider = Slider(
        ax_slider, "日期", 0, len(all_dates) - 1, valinit=state["date_idx"], valstep=1, color="#e8c547",
    )
    slider.label.set_color("#cccccc")
    slider.valtext.set_color("#cccccc")
    
    def stock_label(s):
        return s.get("label") or s["code"]
    
    def _calc_period_costs(s, idx):
        if not periods or idx is None:
            return None
        return {p: _period_cost(s["df"], idx, p) for p in periods}
    
    def refresh():
        target_date = all_dates[state["date_idx"]]
        
        # 绘制每个股票的筹码图和信息框
        for ax_chip, ax_info, s in zip(chip_axes, info_axes, stocks):
            result, idx = _result_at_date(s, target_date)
            if result is None:
                _empty_chip_ax(ax_chip, stock_label(s), target_date)
                ax_info.clear()
                ax_info.set_facecolor("#1c1c1c")
                ax_info.set_title(f"{stock_label(s)}\n{target_date.strftime('%Y-%m-%d')}", color="#eeeeee", fontsize=10)
                ax_info.text(0.5, 0.5, "该日无数据", transform=ax_info.transAxes, ha="center", va="center", color="#888888", fontsize=11)
                ax_info.set_xticks([])
                ax_info.set_yticks([])
            else:
                pc = _calc_period_costs(s, idx)
                _draw_chip_on_ax(ax_chip, result, plot_step=plot_step, label=stock_label(s), show_legend=(n <= 2), period_costs=pc)
                
                # 在信息框axes中绘制信息和图例
                ax_info.clear()
                ax_info.set_facecolor("#1c1c1c")
                ax_info.set_title(f"{stock_label(s)}\n{target_date.strftime('%Y-%m-%d')}", color="#eeeeee", fontsize=10)
                
                # 数据部分
                mode_tag = "修正" if state.get("decay_on") and s.get("has_decay_toggle") else "原始"
                data_lines = [
                    f"换手模式 [{mode_tag}]",
                    f"收盘 {result['close']:.2f}",
                    f"平均成本 {result['avg_cost']:.2f}",
                    f"获利比例 {result['profit_ratio']:.2%}",
                    f"90%集中 {result['p90_concentration']:.1%} [{result['p90_low']:.2f}, {result['p90_high']:.2f}]",
                    f"70%集中 {result['p70_concentration']:.1%} [{result['p70_low']:.2f}, {result['p70_high']:.2f}]",
                ]
                if pc:
                    for p in sorted(pc.keys()):
                        cost = pc[p]
                        if cost and cost == cost:
                            data_lines.append(f"{p}周期成本 {cost:.2f}")
                
                # 创建图例线条
                legend_lines = []
                legend_labels = []

                # 收盘价线（黄色虚线）
                legend_lines.append(Line2D([0], [0], color="#e8c547", linewidth=1.2, linestyle="--"))
                legend_labels.append("收盘价线")

                # 平均成本线（白色实线）
                legend_lines.append(Line2D([0], [0], color="white", linewidth=1, linestyle="-"))
                legend_labels.append("平均成本线")

                # 90%筹码分布区间（绿色点线）
                legend_lines.append(Line2D([0], [0], color="#22c55e", linewidth=0.8, linestyle=":"))
                legend_labels.append("90%筹码区间")

                # 70%筹码分布区间（橙色点线）
                legend_lines.append(Line2D([0], [0], color="#f59e0b", linewidth=0.8, linestyle=":"))
                legend_labels.append("70%筹码区间")

                # 周期成本线（每条单独列出）
                period_colors = {5: "#a78bfa", 10: "#60a5fa", 20: "#06b6d4", 30: "#fbbf24"}
                if pc:
                    for p in sorted(pc.keys()):
                        if p in period_colors:
                            legend_lines.append(Line2D([0], [0], color=period_colors[p], linewidth=1, linestyle="-."))
                            legend_labels.append(f"{p}周期成本")

                # 添加图例到信息框
                ax_info.legend(
                    handles=legend_lines,
                    labels=legend_labels,
                    loc="upper right",
                    framealpha=0.85,
                    facecolor="#2a2a2a",
                    edgecolor="#555555",
                    labelcolor="#dddddd",
                    fontsize=7,
                    handletextpad=0.1,
                    handlelength=1.5
                )
                
                # 添加数据文本
                y_pos = 0.85  # 数据文本的y位置
                for line in data_lines:
                    ax_info.text(
                        0.02, y_pos,
                        line,
                        transform=ax_info.transAxes,
                        ha="left",
                        va="center",
                        color="#dddddd",
                        fontsize=7.5,
                #family="monospace"
                    )
                    y_pos -= 0.08  # 每行向下移动
                
                ax_info.set_xticks([])
                ax_info.set_yticks([])
        
        # 绘制K线图
        ks = stocks[state["kline_stock"]]
        _, kidx = _result_at_date(ks, target_date)
        if kidx is not None:
            _draw_kline_on_ax(ax_k, ks["df"], kidx, window=kline_window, title=f"{stock_label(ks)} 日 K（点击筹码列切换）")
        else:
            ax_k.clear()
            ax_k.set_facecolor("#1c1c1c")
            ax_k.set_title(f"{stock_label(ks)} 日 K", color="#eeeeee")
            ax_k.text(0.5, 0.5, "该日无数据", transform=ax_info.transAxes, ha="center", va="center", color="#888888")
        
        slider.label.set_text(f"日期 {target_date.strftime('%Y-%m-%d')}")
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
        elif event.key and event.key.lower() == "t" and toggle_decay is not None:
            toggle_decay()

    def on_slider(val):
        state["date_idx"] = int(val)
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_chip_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    slider.on_changed(on_slider)
    toggle_decay = _add_decay_toggle_button(fig, stocks, state, refresh, default_on=decay_default_on)

    refresh()
    if slider_days:
        print(f"滑块范围：最近 {len(all_dates)} 个交易日（筹码自 IPO 全历史递推）")
    if any(s.get("has_decay_toggle") for s in stocks):
        print("操作：← → 切换日期 | 拖滑块 | 点击筹码列切换 K 线 | 中部蓝色按钮或按 T 切换十大股东修正")
    else:
        print("操作：← → 切换日期 | 拖滑块 | 点击筹码列切换下方 K 线")
    plt.show()


def show_interactive(stock, *, plot_step=0.05, kline_window=120, slider_days=30,
                     periods=(5, 10, 20, 30), decay_default_on=False):
    """
    单股联动：左 K 线 + 右筹码。
    stock: load_stock 返回的字典。
    decay_default_on: 启动时是否开启十大股东换手率修正。
    """
    from matplotlib.widgets import Slider

    if "results_raw" not in stock:
        raise ValueError("stock 须为 load_stock 返回的字典")

    df = stock["df"].sort_values("trade_date").reset_index(drop=True)
    results, _, _ = _stock_chip_pack(stock)
    n = len(results)
    if n == 0:
        raise ValueError("results 为空")
    if n != len(df):
        raise ValueError("results 行数与 df 不一致")

    idx_min = _slider_idx_min(n, slider_days)
    state = {"idx": n - 1, "decay_on": decay_default_on}
    stock["decay_on"] = decay_default_on

    # 画布加宽加高, 左 K 线 + 右筹码, 中间留间距
    fig = plt.figure(figsize=(14, 9), facecolor="#1c1c1c")
    ax_k = fig.add_axes([0.05, 0.10, 0.55, 0.82])
    ax_c = fig.add_axes([0.63, 0.10, 0.34, 0.82])
    ax_slider = fig.add_axes([0.12, 0.03, 0.76, 0.022], facecolor="#2a2a2a")

    slider = Slider(
        ax_slider, "日期", idx_min, n - 1,
        valinit=state["idx"], valstep=1, color="#e8c547",
    )
    slider.label.set_color("#cccccc")
    slider.valtext.set_color("#cccccc")

    def refresh():
        i = state["idx"]
        _draw_kline_on_ax(ax_k, df, i, window=kline_window)
        results, checkpoints, df_chip = _stock_chip_pack(stock)
        if "distribution" in results[i]:
            chip_result = results[i]
        else:
            chip_result = chip_snapshot(df_chip, i, checkpoints, **stock["calc_kw"])
        pc = {p: _period_cost(df, i, p) for p in periods} if periods else None
        _draw_chip_on_ax(ax_c, chip_result, plot_step=plot_step, period_costs=pc)
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
        elif event.key and event.key.lower() == "t" and toggle_decay is not None:
            toggle_decay()

    def on_slider(val):
        state["idx"] = int(val)
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    slider.on_changed(on_slider)
    toggle_decay = _add_decay_toggle_button(fig, [stock], state, refresh, default_on=decay_default_on)

    refresh()
    if slider_days:
        print(f"滑块范围：最近 {n - idx_min} 个交易日（筹码自 IPO 全历史递推）")
    if stock.get("has_decay_toggle"):
        print("操作：点击 K 线 | ← → 切换 | 拖滑块 | 中部蓝色按钮或按 T 切换十大股东修正")
    else:
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
    use_decay=False,
    decay_default_on=None,
    save_enriched=True,
    enriched_path=None,
):
    """
    读取并预处理单只股票，返回 stock 字典（供交互图与对比使用）。

    参数:
        fetch_top10  : 是否从 AkShare 拉取前十大流通股东（用于界面按钮切换，与是否默认开启无关）
        use_decay    : 兼容旧参数，等同 decay_default_on（启动时是否开启十大股东修正）
        decay_default_on : 启动时是否开启十大股东换手率修正，默认 False

    返回 stock 字典，主要字段:
        df, df_raw, results_raw, checkpoints_raw, results_adj, checkpoints_adj,
        calc_kw, has_decay_toggle, decay_on, code
    """
    if decay_default_on is None:
        decay_default_on = use_decay

    raw = pd.read_csv(csv_path)
    if "code" in raw.columns:
        raw = raw[raw["code"] == code]
    df_all = prep_df(raw.copy())
    df_all = _coalesce_enrich_cols(df_all)

    # ---- 前十大股东：先对全历史计算，再按 start_date 截取用于筹码递推 ----
    top10 = None
    if top10_df is not None:
        top10 = top10_df
    elif top10_csv is not None:
        top10 = load_top10_from_csv(top10_csv)
    elif fetch_top10:
        symbol = "".join(ch for ch in code if ch.isdigit())
        print(f"  [decay] 从 AkShare 拉取 {symbol} 前十大流通股东数据...")
        top10 = fetch_top10_from_akshare(symbol)
        print(f"  [decay] 拉取到 {len(top10)} 个报告期, "
              f"范围 {top10['report_date'].min().date()} ~ "
              f"{top10['report_date'].max().date()}")

    has_decay_toggle = False
    if top10 is not None and len(top10) > 0:
        df_all = _apply_top10_decay(df_all, top10, active_ratio=active_ratio, lag_days=lag_days)
        has_decay_toggle = bool((df_all["decay"] > 1.0001).any())
        d = df_all["decay"]
        print(
            f"  [decay] active_ratio={active_ratio}, lag_days={lag_days}, "
            f"min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}  "
            f"latest={d.iloc[-1]:.4f}"
        )
    elif "decay" in df_all.columns and (pd.to_numeric(df_all["decay"], errors="coerce") > 1.0001).any():
        df_all["decay"] = pd.to_numeric(df_all["decay"], errors="coerce").fillna(1.0)
        if "top10_ratio" in df_all.columns:
            df_all["top10_ratio"] = pd.to_numeric(df_all["top10_ratio"], errors="coerce").fillna(0.0)
        else:
            df_all["top10_ratio"] = 0.0
        if "turnover_rate" in df_all.columns:
            df_all["modified_turn"] = (df_all["turnover_rate"] * df_all["decay"]) * 100.0
        else:
            df_all["modified_turn"] = 0.0
        has_decay_toggle = True
        print(f"  [decay] 使用 CSV 内已有 decay 列 (latest={df_all['decay'].iloc[-1]:.4f})")
    else:
        df_all["decay"] = 1.0
        df_all["top10_ratio"] = 0.0
        if "turnover_rate" in df_all.columns:
            df_all["modified_turn"] = df_all["turnover_rate"] * 100.0
        else:
            df_all["modified_turn"] = 0.0
        print("  [decay] 未提供前十大股东数据，界面不显示修正开关")

    if save_enriched and has_decay_toggle:
        _save_enriched_csv(raw, df_all, csv_path, enriched_path, code)

    df = df_all.copy()
    if start_date:
        df = df[df["trade_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["trade_date"] <= pd.to_datetime(end_date)]
    df = df.sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{code}: 过滤后无数据")

    calc_kw = {"step": 0.01, "decay": 1.0, "mode": "triangle"}
    results_adj, checkpoints_adj = calc_chip(df, **calc_kw)

    df_raw = df.copy()
    df_raw["decay"] = 1.0
    if "turnover_rate" in df_raw.columns:
        df_raw["modified_turn"] = df_raw["turnover_rate"] * 100.0
    else:
        df_raw["modified_turn"] = 0.0
    results_raw, checkpoints_raw = calc_chip(df_raw, **calc_kw)

    return {
        "code": code,
        "df": df,
        "df_raw": df_raw,
        "results_raw": results_raw,
        "checkpoints_raw": checkpoints_raw,
        "results_adj": results_adj,
        "checkpoints_adj": checkpoints_adj,
        "calc_kw": calc_kw,
        "has_decay_toggle": has_decay_toggle,
        "decay_on": bool(decay_default_on and has_decay_toggle),
    }


def _save_enriched_csv(raw, df, csv_path, enriched_path, code):
    """把 top10_ratio / decay / modified_turn 写回 CSV（全历史行，按日期对齐）。"""
    save_path = enriched_path or csv_path
    raw_date_col = _pick_col(raw.columns, COLUMN_ALIASES["trade_date"])
    if raw_date_col is None:
        out = df.copy()
    else:
        out = _coalesce_enrich_cols(raw.copy())
        out[raw_date_col] = pd.to_datetime(out[raw_date_col], errors="coerce")
        lookup = df.set_index(pd.to_datetime(df["trade_date"]))
        for col in ENRICH_COLS:
            if col in lookup.columns:
                out[col] = out[raw_date_col].map(lookup[col])
        # 去掉 merge 残留的重复列
        out = _coalesce_enrich_cols(out)

    date_col_out = _pick_col(out.columns, COLUMN_ALIASES["trade_date"])
    if date_col_out is not None:
        out[date_col_out] = pd.to_datetime(out[date_col_out]).dt.strftime("%Y-%m-%d")
    # 固定小数位，避免 Excel / IDE 因前几行 decay=1.0 推断为整数列
    for col in ENRICH_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    out.to_csv(save_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    n_ok = out["decay"].notna().sum() if "decay" in out.columns else 0
    d = pd.to_numeric(out["decay"], errors="coerce")
    latest = d.dropna()
    latest_val = latest.iloc[-1] if len(latest) else float("nan")
    print(f" [save] 已保存到 {save_path.resolve()}")
    print(f" [save] 列: top10_ratio, decay, modified_turn（{n_ok}/{len(out)} 行有值）")
    if d.notna().any():
        print(f" [save] decay 范围 {d.min():.6f} ~ {d.max():.6f}，最新 {latest_val:.6f}")
        print(" [save] 提示: Excel 若把 decay 显示为 1，请选中列→数值→4位小数；或看公式栏真实值")


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

    # ---- 前十大股东数据 ----
    # fetch_top10=True 仅拉取数据供界面按钮切换；默认不开启修正（decay_default_on=False）
    fetch_top10 = True
    active_ratio = 0.2
    decay_default_on = False   # 启动时是否开启十大股东修正

    save_enriched = True

    stocks = []
    for item in STOCKS:
        code = item["code"]
        path = item.get("csv") or csv_path_for_code(code)
        print(f"\n=== {item.get('label') or code} ({path}) ===")
        stock = load_stock(
            path, code,
            start_date=start_date, end_date=end_date,
            top10_csv=item.get("top10_csv"),
            fetch_top10=item.get("fetch_top10", fetch_top10),
            active_ratio=active_ratio,
            decay_default_on=item.get("decay_default_on", decay_default_on),
            save_enriched=save_enriched,
        )
        df = stock["df"]
        results, _, _ = _stock_chip_pack(stock)
        print(f"样本: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}，共 {len(df)} 天")
        last = results[-1]
        mode = "修正" if stock.get("decay_on") else "原始"
        print(f"最新 [{mode}] {last['trade_date'].date()}  收盘 {last['close']:.2f}  "
              f"平均成本 {last['avg_cost']:.2f}  获利 {last['profit_ratio']:.2%}")
        stock["label"] = item.get("label")
        stocks.append(stock)

    show_compare_interactive(
        stocks, plot_step=0.05, kline_window=120, slider_days=30,
        decay_default_on=decay_default_on,
    )


if __name__ == "__main__":
    main()
