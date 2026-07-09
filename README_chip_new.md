# chip_new.py — 筹码分布与筹码峰计算说明

基于同花顺「移动成本分布」模型的 Python 实现，在原始日线数据上递推筹码分布，并输出平均成本、获利比例、筹码峰、集中度等指标。相比基础版，**新增前十大股东换手率修正**（历史换手衰减系数）及多股对比交互界面。

---

## 依赖与数据

| 依赖 | 用途 |
|------|------|
| `pandas` | 数据处理 |
| `matplotlib` | K 线 + 筹码分布交互图 |
| `baostock` | 日线行情下载（见 `Demo_BaoStock.py`） |
| `akshare`（可选） | 在线拉取前十大流通股东数据（`fetch_top10=True` 时） |

### 日线 CSV 必需字段

经 `prep_df()` 对齐后需要：

- `trade_date`（或 `date`）
- `open`, `high`, `low`, `close`, `volume`
- `turnover_rate`（或 BaoStock 的 `turn`，百分数，会自动 `/100` 转小数）
- 建议有 `amount`（成交额，用于更精确的当日均价）
- 可选 `tradestatus`（停牌过滤）

### 前十大股东数据（可选）

来源三选一（优先级从高到低）：

1. `top10_df`：传入 DataFrame
2. `top10_csv`：本地 CSV
3. `fetch_top10=True`：AkShare `stock_circulate_stock_holder`

所需列：`report_date`（报告期）、`top10_ratio`（合计持股比例）；可选 `pub_date`（公告日，用作生效日期）。

---

## 计算流程总览

```
日线 CSV (BaoStock)
       │
       ▼
  prep_df()          列名对齐、换手率转小数、过滤停牌日
       │
       ▼
  attach_decay_to_df()   ← 前十大股东 → 日频衰减系数 decay
       │
       ▼
  calc_chip()        从首日逐日递推筹码分布，输出每日指标
       │
       ├── chip_snapshot()   按需计算某日完整分布（含画图用的 distribution）
       │
       ▼
  show_interactive() / show_compare_interactive()   交互查看
```

**重要约定：**

- 筹码默认从传入数据的**第一个交易日**开始递推（通常为 IPO；也可用 `start_date` 截断）。
- 交互滑块 `slider_days` 只限制**查看最近 N 天**，不改变已算好的全历史递推结果（除非 `start_date` 截断了输入数据）。

---

## 核心算法：移动成本分布

同花顺经典公式（每日更新）：

```
moved = min(换手率 × 衰减系数, 1.0)
stay  = 1 - moved

Y(p) = Y'(p) × stay + B(p) × moved
```

| 符号 | 含义 |
|------|------|
| `Y'(p)` | 昨日价格 p 上的筹码权重 |
| `B(p)` | 当日新成交筹码在 [Low, High] 上的分布 |
| `moved` | 当日参与换手的筹码比例 |

### 当日成本分布 B(p)

默认 `mode="triangle"`（三角分布）：

- 在 `[Low, High]` 按 `step=0.01` 元分档
- 峰值在当日均价 `amount/volume`（无 amount 时用 OHLC 四价均值）
- 可选 `mode="uniform"` 均匀分布

实现函数：`_daily_cost_triangle()`、`_daily_cost_uniform()`

### 每日更新

`_update_chip(chip, row)` 完成单日递推：

1. 读取换手率 `_turnover(row)`
2. 读取衰减系数：优先行内 `decay` 列，否则用参数 `decay`
3. 旧筹码 × `stay`，新筹码按 `B(p) × moved` 叠加
4. 价位档超过 5000 时 `_prune_chip()` 合并粗网格，防止长历史内存爆炸

---

## 前十大股东换手率修正（本文件核心增强）

### 背景

BaoStock 的 `turn = volume / 流通股本 × 100`，其中流通股本包含**前十大股东持有的流通股**，但大股东大部分筹码并不活跃交易，导致名义换手率**偏低**，筹码衰减偏慢。

### 修正思路

将前十大股东中「锁定不交易」的部分从换手分母里剔除，等效为放大有效换手率：

```
真实换手 ∝ 原始换手 × decay

decay = 1 / (1 - top10_ratio × (1 - active_ratio))
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top10_ratio` | 来自季报 | 前十大流通股东合计持股比例（0~1） |
| `active_ratio` | `0.2` | 十大股东中估计会参与日常交易的比例 |
| `TOP10_LAG_DAYS` | `90` | 无公告日时，报告期 + 90 天作为数据生效日（避免未来函数） |

特殊情况：

- 有 `pub_date`：从公告日起该期比例生效，直到下一期公告
- `top10_ratio` 缺失 → `decay = 1.0`（不调整）
- `active_ratio = 1` → `decay = 1.0`
- `use_decay=False` → 全程 `decay = 1.0`

### 相关函数

| 函数 | 作用 |
|------|------|
| `calc_decay_coefficient()` | 单期衰减系数 |
| `_prepare_top10_table()` | 清洗季报，生成 `effective_date` |
| `prepare_decay_series()` | 季频 → 与日线对齐的 `decay` 列表 |
| `attach_decay_to_df()` | 写入 `df["decay"]` 列 |
| `prepare_top10_daily_series()` | 日频 `top10_ratio`（供 CSV 查看） |
| `fetch_top10_from_akshare()` | AkShare 拉取并聚合前十大流通股东 |
| `load_top10_from_csv()` | 从 CSV 加载 |

`load_stock()` 启用衰减后，还会在 CSV 末尾追加三列（百分数，与 BaoStock `turn` 同量纲）：

- `top10_ratio`：当日生效的十大股东持股比例
- `decay`：衰减系数
- `modified_turn`：修正换手率 = `turn × decay`

---

## 输出指标说明

每日 `_snapshot_from_chip()` 生成：

### 基础指标

| 指标 | 函数 | 说明 |
|------|------|------|
| 平均成本 | `_avg_cost()` | 筹码加权平均价格 |
| 获利比例 | `_profit_ratio()` | 收盘价**以下**筹码占比（含等于收盘价） |
| 筹码峰 | `_peaks()` | 局部极大值价位，按权重降序 |

### 集中度（同花顺口径）

| 指标 | 说明 |
|------|------|
| 90% 区间 `[p90_low, p90_high]` | 累计 5%~95% 筹码所在价格区间 |
| 90% 集中度 | `(P95-P5)/(P95+P5)`，越小越集中 |
| 70% 区间 / 集中度 | 同理，用 15%~85% 分位 |

实现：`_chip_percentile_price()`、`_concentration()`

### 周期成本（图上辅助线）

`_period_cost(df, idx, period)`：最近 N 日成交量加权均价（VWAP），默认显示 5/10/20/30 日。

---

## 内存与性能策略

长历史（数千交易日 × 多只股票）下：

| 策略 | 实现 |
|------|------|
| 默认不存每日 `distribution` | `calc_chip(store_distribution=False)` |
| 检查点加速 | 每 150 天存 `checkpoints`，`chip_snapshot()` 从最近检查点递推 |
| 价位剪枝 | `_prune_chip()` 超 5000 档时合并 |

---

## 函数索引

### 数据预处理

| 函数 | 说明 |
|------|------|
| `prep_df(df)` | 列名对齐、换手率 `/100`、过滤停牌 |
| `load_stock(...)` | 一站式：读 CSV → 衰减 → 计算 → 可选写回增强 CSV |
| `csv_path_for_code(code)` | `sh.600000` → `data/sh_600000_daily.csv` |

### 筹码计算

| 函数 | 说明 |
|------|------|
| `calc_chip(df)` | 全量递推，返回 `(results, checkpoints)` |
| `chip_snapshot(df, idx, checkpoints)` | 指定日完整快照（含 distribution） |
| `chip_to_df(results)` | 指标转 DataFrame（不含 distribution） |

### 可视化

| 函数 | 说明 |
|------|------|
| `_draw_chip_on_ax()` | 单日筹码横向柱状图 + 成本线 + 集中度区间 |
| `_draw_kline_on_ax()` | K 线窗口图 |
| `show_interactive()` | 单股：左 K 线 + 右筹码 + 滑块 |
| `show_compare_interactive()` | 多股并排对比 + 下方 K 线 + 信息框 |
| `plot_chip()` | 单日静态图 |

### 内部工具

| 函数 | 说明 |
|------|------|
| `_pick_col()` | 列名别名查找 |
| `_turnover()` | 换手率转小数 |
| `_avg_price()` | 当日均价 |
| `_price_grid()` | 价格分档网格 |
| `_rebin_dist()` | 0.01 → 0.05 合并（仅画图） |
| `_chip_ylim()` | 筹码图 Y 轴缩放 |
| `_result_at_date()` | 按日期取最近交易日快照 |
| `_slider_idx_min()` | 滑块起始行号 |

---

## 快速开始

### 1. 下载日线

```bash
python Demo_BaoStock.py
```

### 2. 运行交互对比

```bash
python chip_new.py
```

在 `main()` 中配置：

```python
STOCKS = [
    {"code": "sh.600522", "label": "中天科技"},
    # ...
]

start_date = None          # None = 从 CSV 最早日起算（通常 IPO）
use_decay = True           # 启用十大股东修正
fetch_top10 = True         # AkShare 自动拉十大股东
active_ratio = 0.2         # 十大股东活跃比例估计
save_enriched = True       # 回写 top10_ratio/decay/modified_turn 到 CSV

show_compare_interactive(
    stocks,
    plot_step=0.05,        # 画图价格档位（计算仍用 0.01）
    kline_window=120,      # K 线显示最近 120 根
    slider_days=30,        # 滑块只看最近 30 个交易日
    periods=(5, 10, 20, 30) # 周期成本线
)
```

### 3. 编程调用

```python
from chip_new import load_stock, chip_snapshot, csv_path_for_code

df, results, checkpoints, calc_kw = load_stock(
    csv_path_for_code("sh.600522"),
    "sh.600522",
    fetch_top10=True,
    active_ratio=0.2,
)

# 最新日指标
print(results[-1])

# 某日完整分布（含筹码峰）
snap = chip_snapshot(df, len(df) - 1, checkpoints, **calc_kw)
print(snap["peaks"][:3])          # 前三筹码峰
print(snap["p90_concentration"])  # 90% 集中度
```

---

## 交互操作

| 操作 | 效果 |
|------|------|
| `←` / `→` | 切换日期 |
| 拖动底部滑块 | 切换日期 |
| 点击某列筹码图（多股模式） | 下方 K 线切换到该股票 |
| 点击 K 线（单股模式） | 跳转到对应日期 |

---

## 与同花顺的差异说明

- 筹码分布是**模型推算**，非交易所真实持仓。
- 同花顺 `CM()` 闭源，参数（统计窗口、衰减）可能与本实现不同。
- 十大股东修正依赖季报滞后与 `active_ratio` 估计，属于启发式校准。
- `start_date` 截断 = 从该日**空仓重算**，不等于同花顺「只统计最近 N 日窗口」。

建议以本系统**内部一致性**做筛选和对比，不宜强求与同花顺数值逐位相同。

---

## 文件关系

```
stock_trade/
├── Demo_BaoStock.py      # 多股日线下载（前复权）
├── chip_new.py           # 本文件：计算 + 可视化
├── chip.py               # 早期版本（无十大股东修正）
├── data/
│   └── {code}_daily.csv  # 日线 + 可选增强列
└── README_chip_new.md    # 本文档
```
