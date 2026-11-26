import io
import math
import requests
import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="BharatFundamenta — Screener Edition")

stock_cache = TTLCache(maxsize=300, ttl=900)
chart_cache = TTLCache(maxsize=500, ttl=900)
NSE_STOCKS = []

def load_nse_stocks():
    global NSE_STOCKS
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        NSE_STOCKS = [
            {"symbol": str(row["SYMBOL"]).strip(), "name": str(row["NAME OF COMPANY"]).strip()}
            for _, row in df.iterrows()
        ]
        print(f"Loaded {len(NSE_STOCKS)} NSE stocks.")
    except Exception as e:
        print(f"Fallback list loaded ({e})")
        NSE_STOCKS = [
            {"symbol": "RELIANCE", "name": "Reliance Industries Limited"},
            {"symbol": "TCS", "name": "Tata Consultancy Services Limited"},
            {"symbol": "HDFCBANK", "name": "HDFC Bank Limited"},
            {"symbol": "INFY", "name": "Infosys Limited"},
            {"symbol": "ITC", "name": "ITC Limited"},
        ]

@app.on_event("startup")
def startup_event():
    load_nse_stocks()

def safe_num(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def extract_statement(df, row_map, is_quarterly=False):
    """Converts a yfinance statement DataFrame into a Screener-style table (rows = metrics, cols = dates)."""
    if df is None or df.empty:
        return {"columns": [], "rows": []}
    cols = list(df.columns)[:5]
    cols.reverse()
    col_labels = [c.strftime("%b %Y") if is_quarterly else f"Mar {c.year}" for c in cols]

    rows = []
    for display_name, possible_keys in row_map:
        matched_key = next((k for k in possible_keys if k in df.index), None)
        if not matched_key:
            continue
        values = []
        for c in cols:
            raw = safe_num(df.loc[matched_key, c], None)
            if raw is None:
                values.append("—")
            else:
                values.append(round(raw / 1e7, 1))  # Convert to ₹ Crores
        rows.append({"metric": display_name, "values": values})

    return {"columns": col_labels, "rows": rows}

def generate_pros_cons(pe, pb, roe, de, rev_growth, div_yield, op_margin):
    pros, cons = [], []
    if 0 <= de <= 0.2:
        pros.append("Company is almost debt free.")
    elif de > 1.2:
        cons.append(f"High debt-to-equity ratio of {de}.")

    if roe >= 18:
        pros.append(f"Strong return on equity (ROE) of {roe}%.")
    elif 0 < roe < 10:
        cons.append(f"Low return on equity of {roe}%.")

    if rev_growth >= 12:
        pros.append(f"Healthy revenue growth of {rev_growth}% YoY.")
    elif rev_growth < 0:
        cons.append(f"Revenue declined by {rev_growth}% recently.")

    if div_yield >= 1.5:
        pros.append(f"Company maintains a healthy dividend yield of {div_yield}%.")

    if op_margin >= 20:
        pros.append(f"Strong operating profit margins of {op_margin}%.")

    if pb > 5:
        cons.append(f"Stock is trading at {pb} times its book value.")
    elif 0 < pb < 1.2:
        pros.append(f"Stock is trading near its book value (P/B: {pb}).")

    if pe > 45:
        cons.append(f"High valuation with P/E ratio of {pe}.")

    if not pros:
        pros.append("Stable market presence in its core sector.")
    if not cons:
        cons.append("No major fundamental red flags detected from primary ratios.")
    return pros, cons

def calculate_health_score(pe, roe, de, rev_growth):
    score = 0
    if 0 < pe <= 25: score += 25
    elif 25 < pe <= 40: score += 15
    if roe >= 15: score += 30
    elif roe >= 10: score += 15
    if 0 <= de <= 0.5: score += 25
    elif 0.5 < de <= 1.0: score += 15
    if rev_growth >= 10: score += 20
    elif rev_growth >= 5: score += 10
    return min(score, 100)

@app.get("/api/search")
def search_stocks(q: str = ""):
    q_clean = q.strip().upper()
    if not q_clean:
        return NSE_STOCKS[:15]
    return [
        s for s in NSE_STOCKS
        if q_clean in s["symbol"].upper() or q_clean in s["name"].upper()
    ][:20]

@app.get("/api/chart/{symbol}")
def get_chart(symbol: str, period: str = "1y"):
    sym = symbol.upper().strip()
    period_map = {"1mo": "1mo", "6mo": "6mo", "1y": "1y", "3y": "2y", "5y": "5y", "max": "max"}
    yf_period = period_map.get(period, "1y")
    cache_key = f"{sym}_{yf_period}"
    if cache_key in chart_cache:
        return chart_cache[cache_key]

    try:
        ticker = yf.Ticker(f"{sym}.NS")
        hist = ticker.history(period=yf_period)
        if hist is None or hist.empty or "Close" not in hist:
            return {"dates": [], "prices": []}

        close_series = hist["Close"].dropna()
        data = {
            "dates": [d.strftime("%d %b %Y") for d in close_series.index],
            "prices": [round(float(p), 2) for p in close_series.values.flatten()],
        }
        chart_cache[cache_key] = data
        return data
    except Exception as e:
        print(f"Chart error for {sym}: {e}")
        return {"dates": [], "prices": []}


def get_item(df, keys, idx=0):
    """Safely fetches a value from a yfinance financial DataFrame by column index (0 = latest year, 1 = prior year)."""
    if df is None or df.empty or idx >= len(df.columns):
        return None
    for k in keys:
        if k in df.index:
            val = safe_num(df.loc[k, df.columns[idx]], None)
            if val is not None:
                return val
    return None

def calculate_pro_metrics(info, inc, bs, cf):
    # --- 1. Piotroski F-Score (0 to 9) ---
    ni_0, ni_1 = get_item(inc, ["Net Income", "Net Income Common Stockholders"], 0), get_item(inc, ["Net Income", "Net Income Common Stockholders"], 1)
    rev_0, rev_1 = get_item(inc, ["Total Revenue", "Operating Revenue"], 0), get_item(inc, ["Total Revenue", "Operating Revenue"], 1)
    gp_0, gp_1 = get_item(inc, ["Gross Profit"], 0), get_item(inc, ["Gross Profit"], 1)
    ta_0, ta_1 = get_item(bs, ["Total Assets"], 0), get_item(bs, ["Total Assets"], 1)
    ca_0, ca_1 = get_item(bs, ["Current Assets"], 0), get_item(bs, ["Current Assets"], 1)
    cl_0, cl_1 = get_item(bs, ["Current Liabilities"], 0), get_item(bs, ["Current Liabilities"], 1)
    ltd_0, ltd_1 = get_item(bs, ["Long Term Debt", "Total Debt"], 0), get_item(bs, ["Long Term Debt", "Total Debt"], 1)
    sh_0, sh_1 = get_item(bs, ["Ordinary Shares Number", "Share Issued"], 0), get_item(bs, ["Ordinary Shares Number", "Share Issued"], 1)
    cfo_0 = get_item(cf, ["Operating Cash Flow"], 0)

    roa_0 = (ni_0 / ta_0) if (ni_0 is not None and ta_0) else None
    roa_1 = (ni_1 / ta_1) if (ni_1 is not None and ta_1) else None
    cr_0 = (ca_0 / cl_0) if (ca_0 is not None and cl_0) else None
    cr_1 = (ca_1 / cl_1) if (ca_1 is not None and cl_1) else None
    gm_0 = (gp_0 / rev_0) if (gp_0 is not None and rev_0) else None
    gm_1 = (gp_1 / rev_1) if (gp_1 is not None and rev_1) else None
    at_0 = (rev_0 / ta_0) if (rev_0 is not None and ta_0) else None
    at_1 = (rev_1 / ta_1) if (rev_1 is not None and ta_1) else None

    f_checks = {
        "Positive Net Profit": bool(ni_0 and ni_0 > 0),
        "Positive Operating Cash Flow": bool(cfo_0 and cfo_0 > 0),
        "Improving ROA": bool(roa_0 is not None and roa_1 is not None and roa_0 > roa_1),
        "Cash Flow > Net Profit (Quality)": bool(cfo_0 is not None and ni_0 is not None and cfo_0 > ni_0),
        "Lower Long-Term Debt": bool(ltd_0 is not None and ltd_1 is not None and ltd_0 <= ltd_1),
        "Improving Current Ratio": bool(cr_0 is not None and cr_1 is not None and cr_0 > cr_1),
        "No Share Dilution": bool(sh_0 is not None and sh_1 is not None and sh_0 <= sh_1),
        "Improving Gross Margin": bool(gm_0 is not None and gm_1 is not None and gm_0 > gm_1),
        "Improving Asset Turnover": bool(at_0 is not None and at_1 is not None and at_0 > at_1),
    }
    f_score = sum(1 for passed in f_checks.values() if passed)

    # --- 2. Altman Z-Score ---
    ebit = get_item(inc, ["EBIT", "Operating Income"], 0)
    re = get_item(bs, ["Retained Earnings"], 0)
    tl = get_item(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"], 0)
    mcap = safe_num(info.get("marketCap"))
    z_score, z_zone = None, "N/A"
    if all(v is not None for v in [ca_0, cl_0, ta_0, re, ebit, tl, rev_0]) and ta_0 > 0 and tl > 0 and mcap > 0:
        wc = ca_0 - cl_0
        z_val = 1.2 * (wc / ta_0) + 1.4 * (re / ta_0) + 3.3 * (ebit / ta_0) + 0.6 * (mcap / tl) + 1.0 * (rev_0 / ta_0)
        z_score = round(z_val, 2)
        z_zone = "Safe" if z_score > 2.99 else ("Grey Zone" if z_score >= 1.81 else "Distress")

    # --- 3. Pro Investor Ratios ---
    interest = get_item(inc, ["Interest Expense"], 0)
    fcf = safe_num(info.get("freeCashflow"))
    cap_emp = (ta_0 - cl_0) if (ta_0 is not None and cl_0 is not None and (ta_0 - cl_0) > 0) else None

    roce = round((ebit / cap_emp) * 100, 2) if (ebit is not None and cap_emp) else "—"
    int_cov = round(ebit / abs(interest), 2) if (ebit is not None and interest and abs(interest) > 0) else "—"
    fcf_yield = round((fcf / mcap) * 100, 2) if (fcf and mcap > 0) else "—"
    cash_conv = round(cfo_0 / ni_0, 2) if (cfo_0 is not None and ni_0 and ni_0 > 0) else "—"
    ev_ebitda = round(safe_num(info.get("enterpriseToEbitda")), 2) or "—"
    peg = round(safe_num(info.get("trailingPegRatio") or info.get("pegRatio")), 2) or "—"

    return {
        "f_score": f_score,
        "f_checks": f_checks,
        "z_score": z_score if z_score is not None else "—",
        "z_zone": z_zone,
        "roce": roce,
        "ev_ebitda": ev_ebitda,
        "peg": peg,
        "int_cov": int_cov,
        "fcf_yield": fcf_yield,
        "cash_conv": cash_conv,
    }
@app.get("/api/stock/{symbol}")
def get_stock_data(symbol: str):
    sym = symbol.upper().strip()
    if sym in stock_cache:
        return stock_cache[sym]

    ticker = yf.Ticker(f"{sym}.NS")
    info = ticker.info or {}

    price = safe_num(info.get("currentPrice") or info.get("regularMarketPrice"))
    if price == 0:
        raise HTTPException(status_code=404, detail=f"No live data found for {sym}.NS")

    prev_close = safe_num(info.get("previousClose"), price)
    change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0.0

    pe = round(safe_num(info.get("trailingPE")), 2)
    pb = round(safe_num(info.get("priceToBook")), 2)
    eps = round(safe_num(info.get("trailingEps")), 2)
    bvps = round(safe_num(info.get("bookValue")), 2)
    roe = round(safe_num(info.get("returnOnEquity")) * 100, 2)
    de = round(safe_num(info.get("debtToEquity")) / 100, 2)
    rev_growth = round(safe_num(info.get("revenueGrowth")) * 100, 2)
    op_margin = round(safe_num(info.get("operatingMargins")) * 100, 2)
    div_yield = round(safe_num(info.get("dividendYield")), 2)
    if div_yield < 0.2 and div_yield > 0:  # handle decimal vs % formats
        div_yield = round(div_yield * 100, 2)

    fcf = safe_num(info.get("freeCashflow"))
    shares = safe_num(info.get("sharesOutstanding"), 1.0)
    fcf_per_share = round(fcf / shares, 2) if shares > 0 else 0.0
    graham_number = round(math.sqrt(22.5 * eps * bvps), 2) if (eps > 0 and bvps > 0) else 0.0
    pros, cons = generate_pros_cons(pe, pb, roe, de, rev_growth, div_yield, op_margin)

    pl_rows = [
        ("Sales / Revenue", ["Total Revenue", "Operating Revenue"]),
        ("Operating Profit", ["Operating Income", "EBIT"]),
        ("Interest", ["Interest Expense"]),
        ("Profit Before Tax", ["Pretax Income"]),
        ("Net Profit", ["Net Income", "Net Income Common Stockholders"]),
    ]
    bs_rows = [
        ("Total Assets", ["Total Assets"]),
        ("Total Equity / Net Worth", ["Stockholders Equity", "Total Equity Gross Minority Interest"]),
        ("Total Debt", ["Total Debt"]),
        ("Cash & Equivalents", ["Cash And Cash Equivalents"]),
    ]
    cf_rows = [
        ("Cash from Operating Activity", ["Operating Cash Flow"]),
        ("Cash from Investing Activity", ["Investing Cash Flow"]),
        ("Cash from Financing Activity", ["Financing Cash Flow"]),
        ("Free Cash Flow", ["Free Cash Flow"]),
    ]

    result = {
        "symbol": sym,
        "name": info.get("longName", sym),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "about": info.get("longBusinessSummary", "No company description available."),
        "price": round(price, 2),
        "change_pct": change_pct,
        "high_52w": round(safe_num(info.get("fiftyTwoWeekHigh")), 2),
        "low_52w": round(safe_num(info.get("fiftyTwoWeekLow")), 2),
        "market_cap_cr": round(safe_num(info.get("marketCap")) / 1e7, 1),
        "pe": pe,
        "pb": pb,
        "roe": roe,
        "debt_to_equity": de,
        "div_yield": div_yield,
        "eps": eps,
        "bvps": bvps,
        "op_margin": op_margin,
        "fcf_per_share": fcf_per_share,
        "graham_number": graham_number,
        "health_score": calculate_health_score(pe, roe, de, rev_growth),
        "pros": pros,
        "cons": cons,
        "quarterly": extract_statement(ticker.quarterly_income_stmt, pl_rows, is_quarterly=True),
        "annual_pl": extract_statement(ticker.income_stmt, pl_rows, is_quarterly=False),
        "balance_sheet": extract_statement(ticker.balance_sheet, bs_rows, is_quarterly=False),
        "cash_flow": extract_statement(ticker.cashflow, cf_rows, is_quarterly=False),
        "pro": calculate_pro_metrics(info, ticker.income_stmt, ticker.balance_sheet, ticker.cashflow),
    }

    stock_cache[sym] = result
    return result

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")