"""
extract_option_chain.py

Reads an NSE option chain CSV and writes selected strikes (Call IV, Call Price,
Strike, Put Price, Put IV) into an Excel file.

Usage
-----
python extract_option_chain.py \
    --csv  inputs/option-chain-ED-NIFTY-13-Apr-2026.csv \
    --xlsx inputs/nifty_options_13apr2026.xlsx \
    --strikes 22450 22500 22550 22600 22650 22700 22750 22800 22850 22900 22950

If --strikes is omitted the script prints all available strikes and exits.
"""

import argparse
import csv
import re
import sys

import openpyxl
import pandas as pd
import numpy as np


def _clean_number(s: str):
    """Strip commas and convert to float; return None if not numeric."""
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan
        


def parse_option_chain_csv(csv_path: str) -> dict:
    """
    Parse NSE option-chain CSV.

    Returns a dict keyed by strike (int) with values:
        {
            "call_iv":    float | None,
            "call_price": float | None,   # LTP, or (bid+ask)/2 if LTP unavailable
            "put_price":  float | None,   # LTP, or (bid+ask)/2 if LTP unavailable
            "put_iv":     float | None,
        }

    CSV column order (0-indexed, first column is blank):
      0  blank
      1  OI (calls)
      2  CHNG IN OI (calls)
      3  VOLUME (calls)
      4  IV (calls)           ← call_iv
      5  LTP (calls)          ← call_price
      6  CHNG (calls)
      7  BID QTY (calls)
      8  BID (calls)
      9  ASK (calls)
      10 ASK QTY (calls)
      11 STRIKE
      12 BID QTY (puts)
      13 BID (puts)
      14 ASK (puts)
      15 ASK QTY (puts)
      16 CHNG (puts)
      17 LTP (puts)           ← put_price
      18 IV (puts)            ← put_iv
      ...
    """
    data = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 19:
                continue

            # Strike is in column 11
            strike_raw = row[11].strip().replace(",", "")
            if not re.match(r"^\d+(\.\d+)?$", strike_raw):
                continue  # header or blank row

            strike = int(float(strike_raw))
            call_iv    = _clean_number(row[4])
            call_price = _clean_number(row[5])
            if call_price is None:
                call_bid = _clean_number(row[8])
                call_ask = _clean_number(row[9])
                if call_bid is not None and call_ask is not None:
                    call_price = (call_bid + call_ask) / 2

            put_price  = _clean_number(row[17])
            if put_price is None:
                put_bid = _clean_number(row[13])
                put_ask = _clean_number(row[14])
                if put_bid is not None and put_ask is not None:
                    put_price = (put_bid + put_ask) / 2

            put_iv     = _clean_number(row[18])

            data[strike] = {
                "call_iv":    call_iv,
                "call_price": call_price,
                "put_price":  put_price,
                "put_iv":     put_iv,
            }

    return data


def get_option_data(csv_path: str, strikes: list[int]) -> pd.DataFrame:
    """
    Read an NSE option chain CSV and return a DataFrame for the requested strikes.

    Parameters
    ----------
    csv_path : str
        Path to the NSE option chain CSV file.
    strikes : list[int]
        Strikes to include. They are returned sorted ascending.

    Returns
    -------
    pd.DataFrame
        Columns: Actual IV Call | Call Price | Strike | Put Price | Actual IV Put
        Strikes not found in the CSV are silently dropped.
    """
    data = parse_option_chain_csv(csv_path)

    rows = []
    missing = []
    for strike in sorted(strikes):
        if strike not in data:
            missing.append(strike)
            continue
        d = data[strike]
        rows.append({
            "Actual IV Call": d["call_iv"],
            "Call Price":     d["call_price"],
            "Strike":         strike,
            "Put Price":      d["put_price"],
            "Actual IV Put":  d["put_iv"],
        })

    if missing:
        print(f"Warning: strikes not found in CSV and skipped: {missing}")

    return pd.DataFrame(rows, columns=["Actual IV Call", "Call Price", "Strike", "Put Price", "Actual IV Put"])


def write_to_excel(data: dict, strikes: list[int], xlsx_path: str) -> None:
    """Write selected strikes to an Excel file (overwrites if exists)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    # Header
    ws.append(["Actual IV Call", "Call Price", "Strike", "Put Price", "Actual IV Put"])

    missing = []
    for strike in sorted(strikes):
        if strike not in data:
            missing.append(strike)
            continue
        row = data[strike]
        ws.append([
            row["call_iv"],
            row["call_price"],
            strike,
            row["put_price"],
            row["put_iv"],
        ])

    wb.save(xlsx_path)

    if missing:
        print(f"Warning: strikes not found in CSV and skipped: {missing}")

    print(f"Wrote {len(strikes) - len(missing)} strikes to {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract option chain data to Excel.")
    parser.add_argument("--csv",  required=True, help="Path to the NSE option chain CSV")
    parser.add_argument("--xlsx", required=True, help="Path for the output Excel file")
    parser.add_argument(
        "--strikes", nargs="*", type=int,
        help="Space-separated list of strikes to include. Omit to list available strikes.",
    )
    args = parser.parse_args()

    data = parse_option_chain_csv(args.csv)

    if not args.strikes:
        print("Available strikes in CSV:")
        for s in sorted(data):
            print(f"  {s}")
        sys.exit(0)

    write_to_excel(data, args.strikes, args.xlsx)


if __name__ == "__main__":
    main()
