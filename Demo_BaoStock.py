
# http://baostock.com/baostock/index.php/A%E8%82%A1K%E7%BA%BF%E6%95%B0%E6%8D%AE
# 下载多只股票上市至今的日线，前复权

import baostock as bs
import pandas as pd
from datetime import date
from pathlib import Path

# 支持多只股票，格式如 sh.600000、sz.000001
CODES = [
    "sz.002384",  # 东山精密
    "sz.002281",  # 光迅科技
    "sh.601869",  # 长飞光纤
    "sh.600498",  # 烽火通信
    "sh.600522",  # 中天科技
]
OUTPUT_DIR = Path("data")
K_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"


def result_to_df(result):
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def output_path(code: str) -> Path:
    """sh.600000 -> data/sh_600000_daily.csv"""
    return OUTPUT_DIR / f"{code.replace('.', '_')}_daily.csv"


def download_one(code: str, end_date: str) -> bool:
    rs_basic = bs.query_stock_basic(code=code)
    basic = result_to_df(rs_basic)
    if basic.empty:
        print(f"[跳过] {code}: 查不到股票基本信息")
        return False

    ipo_date = basic.iloc[0]["ipoDate"]
    print(f"下载 {code}  {ipo_date} ~ {end_date}")

    rs = bs.query_history_k_data_plus(
        code,
        K_FIELDS,
        start_date=ipo_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权，与筹码分布常用口径一致
    )
    if rs.error_code != "0":
        print(f"[失败] {code}: {rs.error_code} {rs.error_msg}")
        return False

    df = result_to_df(rs)
    out = output_path(code)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[完成] {code}: 共 {len(df)} 行 -> {out.resolve()}")
    return True


def main():
    if not CODES:
        print("请在 CODES 列表中填入至少一只股票代码")
        return

    lg = bs.login()
    print("login:", lg.error_code, lg.error_msg)
    if lg.error_code != "0":
        return

    end_date = date.today().strftime("%Y-%m-%d")
    ok, fail = 0, 0
    for code in CODES:
        if download_one(code, end_date):
            ok += 1
        else:
            fail += 1

    bs.logout()
    print(f"全部结束: 成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    main()
