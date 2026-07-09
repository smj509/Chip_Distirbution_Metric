"""
筹码加权评分回测：验证得分对未来收益的预测力。

用法示例:
    from backtest import run_backtest, print_bucket_report

    # 单只股票回测
    result = run_backtest("sz.002384")
    print_bucket_report(result)

    # 多只股票合并回测
    codes = ["sz.002384", "sh.600498", "sh.600522"]
    result = run_backtest(codes)
    print_bucket_report(result)

    # 自定义参数
    result = run_backtest("sz.002384", hold_days=[5, 10, 20])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from chip_new import csv_path_for_code, load_stock
from Metric import build_context, screen_stock, weighted_score, WEIGHT_CONFIG


DEFAULT_STOCKS = (
    "sz.002384",
    "sh.600498",
    "sh.600522",
    "sh.601869",
    "sz.002281",
    "sh.600000",
)

DEFAULT_HOLD_DAYS = (5, 10, 20)

DEFAULT_BUCKETS = [
    (-1.00, -0.50),
    (-0.50, -0.20),
    (-0.20,  0.00),
    ( 0.00,  0.25),
    ( 0.25,  0.50),
    ( 0.50,  1.00),
]


def _compute_future_returns(
    df: pd.DataFrame,
    hold_days: Sequence[int] = DEFAULT_HOLD_DAYS,
) -> pd.DataFrame:
    """计算每天的 N 日未来总收益 (close_{t+N} / close_t - 1)。

    用信号日收盘价作为买入价，持有 N 个交易日后卖出。
    最后 N 天无法计算未来收益，设为 NaN。
    """
    close = df["close"].values
    n = len(close)
    out = {}

    for h in hold_days:
        rets = [float("nan")] * n
        for i in range(n - h):
            if close[i] > 0:
                rets[i] = (close[i + h] / close[i]) - 1.0
        out[f"ret_{h}d"] = rets

    return pd.DataFrame(out, index=df.index)


def _score_single_stock(
    stock: dict[str, Any],
    *,
    hold_days: Sequence[int] = DEFAULT_HOLD_DAYS,
    config: dict[str, tuple[float, float]] = WEIGHT_CONFIG,
    use_decay: bool | None = None,
) -> pd.DataFrame:
    """对单只股票每个交易日计算加权得分和未来收益。

    Args:
        use_decay: None=跟随stock默认; True=用修正筹码(results_adj);
                   False=用原始筹码(results_raw)

    Returns:
        DataFrame，列: trade_date, close, total_score, action, ret_5d, ret_10d, ret_20d
    """
    # 确定使用哪套筹码数据
    if use_decay is None:
        use_decay = bool(stock.get("decay_on") and stock.get("has_decay_toggle"))

    if use_decay:
        results = stock.get("results_adj") if stock.get("results_adj") is not None else stock.get("results")
        df = stock.get("df") if stock.get("df") is not None else stock.get("df_raw")
    else:
        results = stock.get("results_raw") if stock.get("results_raw") is not None else stock.get("results")
        df = stock.get("df_raw") if stock.get("df_raw") is not None else stock.get("df")
    if results is None or df is None:
        raise ValueError("stock 字典缺少 results 或 df")

    n = len(results)

    # 计算未来收益
    future_ret = _compute_future_returns(df, hold_days)

    # 逐日打分
    rows = []
    for i in range(n):
        # 需要至少 1 天历史才能构建 context（metric_avg_cost_slope 等）
        if i < 1:
            continue

        report = screen_stock(stock, idx=i, use_decay=use_decay)
        score_result = weighted_score(report, config=config)

        row = {
            "trade_date": results[i].get("trade_date"),
            "close": results[i].get("close"),
            "total_score": score_result.total_score,
            "action": score_result.action,
        }
        for h in hold_days:
            col = f"ret_{h}d"
            if col in future_ret.columns:
                row[col] = future_ret.iloc[i][col] if i < len(future_ret) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def run_backtest(
    codes: str | Sequence[str] = DEFAULT_STOCKS,
    *,
    data_dir: str = "data",
    start_date: str | None = None,
    hold_days: Sequence[int] = DEFAULT_HOLD_DAYS,
    buckets: list[tuple[float, float]] | None = None,
    config: dict[str, tuple[float, float]] = WEIGHT_CONFIG,
    use_decay: bool | None = None,
    compare_decay: bool = False,
) -> dict[str, Any]:
    """运行回测：对指定股票逐日打分，按得分分桶统计未来收益。

    Args:
        codes: 股票代码（单个字符串或列表）
        data_dir: CSV 数据目录
        start_date: 起始日期筛选
        hold_days: 持有天数列表
        buckets: 得分分桶边界 [(lo, hi), ...]
        config: 权重配置
        use_decay: None=跟随stock默认(通常False); True=修正筹码; False=原始筹码
        compare_decay: True=同时跑原始+修正两版，返回对比结果；此时 use_decay 被忽略

    Returns:
        dict，含:
          - "daily": 逐日得分+收益 DataFrame
          - "buckets": 合并分桶统计
          - "per_stock": 每只股票的分桶统计 {code: DataFrame}
          - "codes": 股票代码列表
          - "hold_days": 持有天数
          - "use_decay": 实际使用的模式
        若 compare_decay=True，额外返回:
          - "raw": 上述完整 dict（原始筹码版）
          - "adj": 上述完整 dict（修正筹码版）
    """
    if isinstance(codes, str):
        codes = [codes]

    buckets = buckets or DEFAULT_BUCKETS

    # 对比模式：同时跑两版
    if compare_decay:
        print("\n[对比模式] 原始筹码 vs 修正筹码")
        raw_result = _run_backtest_core(
            codes, data_dir=data_dir, start_date=start_date,
            hold_days=hold_days, buckets=buckets, config=config,
            use_decay=False,
        )
        adj_result = _run_backtest_core(
            codes, data_dir=data_dir, start_date=start_date,
            hold_days=hold_days, buckets=buckets, config=config,
            use_decay=True,
        )
        raw_result["use_decay"] = False
        adj_result["use_decay"] = True
        return {
            "raw": raw_result,
            "adj": adj_result,
            "codes": codes,
            "hold_days": hold_days,
            "compare_decay": True,
        }

    # 单模式
    result = _run_backtest_core(
        codes, data_dir=data_dir, start_date=start_date,
        hold_days=hold_days, buckets=buckets, config=config,
        use_decay=use_decay,
    )
    result["use_decay"] = use_decay if use_decay is not None else False
    return result


def _run_backtest_core(
    codes: list[str],
    *,
    data_dir: str,
    start_date: str | None,
    hold_days: Sequence[int],
    buckets: list[tuple[float, float]],
    config: dict[str, tuple[float, float]],
    use_decay: bool | None,
) -> dict[str, Any]:
    """回测核心逻辑：逐只股票加载、打分、分桶。"""
    label = "修正筹码" if use_decay else "原始筹码"
    print(f"\n--- {label} ---")

    all_rows = []
    per_stock_buckets: dict[str, pd.DataFrame] = {}

    for code in codes:
        csv_path = csv_path_for_code(code, data_dir)
        if not Path(csv_path).exists():
            print(f"  ⚠ 跳过 {code}: CSV 不存在 ({csv_path})")
            continue

        try:
            stock = load_stock(
                str(csv_path), code,
                start_date=start_date,
                fetch_top10=False,
                save_enriched=False,
            )
            print(f"  ✓ {code}: 加载成功")
        except Exception as e:
            print(f"  ⚠ 跳过 {code}: 加载失败 ({e})")
            continue

        try:
            day_df = _score_single_stock(stock, hold_days=hold_days, config=config, use_decay=use_decay)
            day_df["code"] = code
            all_rows.append(day_df)
            print(f"    {len(day_df)} 个交易日已打分")
        except Exception as e:
            print(f"  ⚠ 跳过 {code}: 打分失败 ({e})")
            continue

    if not all_rows:
        raise ValueError(f"没有股票成功打分（{label}）")

    daily = pd.concat(all_rows, ignore_index=True)

    for code in codes:
        code_daily = daily[daily["code"] == code]
        if len(code_daily) == 0:
            continue
        per_stock_buckets[code] = _compute_bucket_stats(
            code_daily, buckets=buckets, hold_days=hold_days
        )

    merged_buckets = _compute_bucket_stats(daily, buckets=buckets, hold_days=hold_days)

    return {
        "daily": daily,
        "buckets": merged_buckets,
        "per_stock": per_stock_buckets,
        "codes": codes,
        "hold_days": hold_days,
    }


def _compute_bucket_stats(
    daily: pd.DataFrame,
    *,
    buckets: list[tuple[float, float]],
    hold_days: Sequence[int],
) -> pd.DataFrame:
    """按得分分桶统计：每个桶的样本数、平均收益、胜率。

    Returns:
        DataFrame，列: bucket, count, avg_ret_5d, win_rate_5d, ...
    """
    rows = []

    for lo, hi in buckets:
        mask = (daily["total_score"] >= lo) & (daily["total_score"] < hi)
        subset = daily[mask]

        row = {
            "bucket": f"[{lo:+.2f}, {hi:+.2f})",
            "count": len(subset),
        }

        for h in hold_days:
            col = f"ret_{h}d"
            if col not in subset.columns:
                continue

            valid = subset[col].dropna()
            if len(valid) > 0:
                avg_ret = valid.mean()
                win_rate = (valid > 0).sum() / len(valid)
            else:
                avg_ret = float("nan")
                win_rate = float("nan")

            row[f"avg_ret_{h}d"] = avg_ret
            row[f"win_rate_{h}d"] = win_rate

        rows.append(row)

    return pd.DataFrame(rows)


def _format_bucket_table(
    buckets_df: pd.DataFrame,
    hold_days: Sequence[int],
    title: str = "",
) -> str:
    """格式化一张分桶统计表。"""
    lines = []
    if title:
        lines.append(title)

    header = "桶              | 样本数"
    for h in hold_days:
        header += f" | 平均{h}日收益 | {h}日胜率"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in buckets_df.iterrows():
        line = f"{row['bucket']:16s} | {row['count']:>6d}"
        for h in hold_days:
            avg_col = f"avg_ret_{h}d"
            win_col = f"win_rate_{h}d"
            avg_str = f"{row[avg_col]:+.2%}" if pd.notna(row.get(avg_col)) else "  N/A"
            win_str = f"{row[win_col]:.0%}" if pd.notna(row.get(win_col)) else "  N/A"
            line += f" | {avg_str:>10s} | {win_str:>6s}"
        lines.append(line)

    return "\n".join(lines)


def _print_high_score_days(
    daily: pd.DataFrame,
    code: str,
    hold_days: Sequence[int],
    threshold: float = 0.50,
    label: str = "",
) -> None:
    """打印得分 >= threshold 的交易日明细。"""
    code_daily = daily[daily["code"] == code]
    high_days = code_daily[code_daily["total_score"] >= threshold].sort_values("total_score", ascending=False)

    if len(high_days) == 0:
        print(f"  [{label}] 得分 >= {threshold:+.2f} 的交易日: 无")
        return

    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}得分 >= {threshold:+.2f} 的交易日 ({len(high_days)} 个):")
    print(f"{prefix}  日期         | 得分     | 收盘   |", end="")
    for h in hold_days:
        print(f" {h}日收益 |", end="")
    print()
    print(f"{prefix}" + "-" * (22 + 10 + 8 + len(hold_days) * 10))

    for _, row in high_days.iterrows():
        date_str = str(row["trade_date"])[:10]
        score_str = f"{row['total_score']:+.4f}"
        close_str = f"{row['close']:.2f}"
        line = f"{prefix}  {date_str} | {score_str} | {close_str} |"
        for h in hold_days:
            ret_col = f"ret_{h}d"
            if pd.notna(row.get(ret_col)):
                line += f" {row[ret_col]:+.2%} |"
            else:
                line += f"    N/A |"
        print(line)


def print_bucket_report(result: dict[str, Any]) -> None:
    """打印分桶回测报告。

    普通模式：按每只股票分别输出，最后附合并汇总。
    对比模式：原始筹码 vs 修正筹码，逐只股票对比 + 合计对比。
    """
    if result.get("compare_decay"):
        _print_compare_report(result)
        return

    hold_days = result["hold_days"]
    codes = result["codes"]
    per_stock = result.get("per_stock", {})
    use_decay = result.get("use_decay", False)
    label = "修正筹码" if use_decay else "原始筹码"

    print("=" * 60)
    print(f"回测报告 — {label} — 每只股票单独统计")
    print("=" * 60)

    # 每只股票单独打印
    for code in codes:
        if code not in per_stock:
            print(f"\n--- {code}: 无数据 ---")
            continue

        code_daily = result["daily"][result["daily"]["code"] == code]
        total = code_daily["total_score"].dropna()

        title = f"\n--- {code} ({label}) | {len(code_daily)} 个交易日 ---"
        table = _format_bucket_table(per_stock[code], hold_days, title)
        print(table)
        print(f"  得分分布: 中位数={total.median():+.4f}, 平均={total.mean():+.4f}")
        print(f"  得分范围: [{total.min():+.4f}, {total.max():+.4f}]")

        # 列出得分 >= 0.50 的交易日明细
        _print_high_score_days(result["daily"], code, hold_days, threshold=0.50, label=label)

    # 合并汇总
    total = result["daily"]["total_score"].dropna()
    title = f"\n--- 合计 ({label}) | {len(result['daily'])} 个交易日 ---"
    table = _format_bucket_table(result["buckets"], hold_days, title)
    print(table)
    print(f"  得分分布: 中位数={total.median():+.4f}, 平均={total.mean():+.4f}")
    print(f"  得分范围: [{total.min():+.4f}, {total.max():+.4f}]")

    # 合计中得分 >= 0.50 的交易日
    print(f"\n  得分 >= +0.50 的所有交易日:")
    high_all = result["daily"][result["daily"]["total_score"] >= 0.50].sort_values("total_score", ascending=False)
    if len(high_all) == 0:
        print("    无")
    else:
        print(f"    日期         | 股票       | 得分     | 收盘   |", end="")
        for h in hold_days:
            print(f" {h}日收益 |", end="")
        print()
        print("    " + "-" * (22 + 12 + 10 + 8 + len(hold_days) * 10))
        for _, row in high_all.iterrows():
            date_str = str(row["trade_date"])[:10]
            line = f"    {date_str} | {row['code']:10s} | {row['total_score']:+.4f} | {row['close']:.2f} |"
            for h in hold_days:
                ret_col = f"ret_{h}d"
                if pd.notna(row.get(ret_col)):
                    line += f" {row[ret_col]:+.2%} |"
                else:
                    line += f"    N/A |"
            print(line)

    print("=" * 60)


def _print_compare_report(result: dict[str, Any]) -> None:
    """打印原始 vs 修正筹码对比报告。"""
    hold_days = result["hold_days"]
    codes = result["codes"]
    raw = result["raw"]
    adj = result["adj"]

    print("=" * 60)
    print("对比报告 — 原始筹码 vs 修正筹码")
    print("=" * 60)

    # 逐只股票对比
    for code in codes:
        raw_ps = raw.get("per_stock", {}).get(code)
        adj_ps = adj.get("per_stock", {}).get(code)
        if raw_ps is None and adj_ps is None:
            print(f"\n--- {code}: 两版均无数据 ---")
            continue

        # 统计行数
        raw_n = len(raw["daily"][raw["daily"]["code"] == code]) if code in raw.get("per_stock", {}) else 0
        adj_n = len(adj["daily"][adj["daily"]["code"] == code]) if code in adj.get("per_stock", {}) else 0

        print(f"\n--- {code} | 原始{raw_n}天 vs 修正{adj_n}天 ---")

        # 原始版
        if raw_ps is not None:
            raw_total = raw["daily"][raw["daily"]["code"] == code]["total_score"].dropna()
            print(f"  【原始筹码】得分中位数={raw_total.median():+.4f}, 范围=[{raw_total.min():+.4f}, {raw_total.max():+.4f}]")
            table = _format_bucket_table(raw_ps, hold_days)
            for line in table.split("\n"):
                print(f"    {line}")

        # 修正版
        if adj_ps is not None:
            adj_total = adj["daily"][adj["daily"]["code"] == code]["total_score"].dropna()
            print(f"  【修正筹码】得分中位数={adj_total.median():+.4f}, 范围=[{adj_total.min():+.4f}, {adj_total.max():+.4f}]")
            table = _format_bucket_table(adj_ps, hold_days)
            for line in table.split("\n"):
                print(f"    {line}")

        # 得分 >= 0.50 的交易日明细
        if raw_ps is not None:
            _print_high_score_days(raw["daily"], code, hold_days, threshold=0.50, label="原始")
        if adj_ps is not None:
            _print_high_score_days(adj["daily"], code, hold_days, threshold=0.50, label="修正")

    # 合计对比
    print(f"\n--- 合计对比 ---")
    raw_total = raw["daily"]["total_score"].dropna()
    adj_total = adj["daily"]["total_score"].dropna()
    print(f"  【原始】中位数={raw_total.median():+.4f}, 范围=[{raw_total.min():+.4f}, {raw_total.max():+.4f}]")
    print(f"  【修正】中位数={adj_total.median():+.4f}, 范围=[{adj_total.min():+.4f}, {adj_total.max():+.4f}]")

    print("\n  【原始筹码 — 合计】")
    table = _format_bucket_table(raw["buckets"], hold_days)
    for line in table.split("\n"):
        print(f"    {line}")

    print("\n  【修正筹码 — 合计】")
    table = _format_bucket_table(adj["buckets"], hold_days)
    for line in table.split("\n"):
        print(f"    {line}")

    print("=" * 60)


if __name__ == "__main__":
    # 对比模式：同时跑原始筹码和修正筹码
    result = run_backtest(DEFAULT_STOCKS, compare_decay=True)
    print_bucket_report(result)
