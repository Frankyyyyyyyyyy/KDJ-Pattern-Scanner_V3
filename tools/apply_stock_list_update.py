import csv
import json
import os
import re


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_LIST_PATH = os.path.join(BASE_DIR, "stock_list.csv")
FIELDS = ["Ticker", "Name_EN", "Name_CN", "Sector", "Type", "Avg_Volume"]


def _extract_payload(text: str) -> dict:
    if not text:
        raise ValueError("Empty issue body")

    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        return json.loads(m.group(1))

    raise ValueError("No JSON payload found")


def _read_stock_list() -> list:
    if not os.path.exists(STOCK_LIST_PATH):
        return []
    with open(STOCK_LIST_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            r = {k: (row.get(k, "") or "").strip() for k in FIELDS}
            if r["Ticker"]:
                r["Ticker"] = r["Ticker"].upper()
                rows.append(r)
        return rows


def _write_stock_list(rows: list) -> None:
    with open(STOCK_LIST_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in FIELDS})


def main() -> None:
    payload = _extract_payload(os.environ.get("ISSUE_BODY", ""))
    add = payload.get("add") or []
    delete = payload.get("delete") or []

    if not isinstance(add, list) or not isinstance(delete, list):
        raise ValueError("Invalid payload structure")

    rows = _read_stock_list()
    by_ticker = {r["Ticker"].upper(): r for r in rows if r.get("Ticker")}

    for t in delete:
        tt = str(t or "").strip().upper()
        if tt:
            by_ticker.pop(tt, None)

    for item in add:
        if not isinstance(item, dict):
            continue
        t = str(item.get("Ticker") or "").strip().upper()
        if not t:
            continue
        if t in by_ticker:
            continue
        by_ticker[t] = {
            "Ticker": t,
            "Name_EN": str(item.get("Name_EN") or "").strip(),
            "Name_CN": str(item.get("Name_CN") or "").strip(),
            "Sector": str(item.get("Sector") or "").strip(),
            "Type": str(item.get("Type") or "Stock").strip(),
            "Avg_Volume": str(item.get("Avg_Volume") or "0").strip(),
        }

    out = list(by_ticker.values())
    out.sort(key=lambda r: r.get("Ticker", ""))
    _write_stock_list(out)


if __name__ == "__main__":
    main()

