"""SEC EDGAR data source: company facts, filing sections, Form 4 insider data.

All requests observe the 10 req/s courtesy limit via a simple throttle.
No API key required — only a User-Agent header.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Optional

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from .cache import Cache


_UA = "InvestBot/1.0 (personal research; contact@example.com)"
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}

_FACTS_TTL_H = 24 * 90
_FILINGS_TTL_H = 24 * 90   # filings are immutable once published
_FORM4_TTL_H = 24 * 7

_LAST_REQ: list[float] = [0.0]
_MIN_INTERVAL = 0.11  # ~9 req/s to stay under 10 req/s limit


def _get(url: str) -> requests.Response:
    elapsed = time.time() - _LAST_REQ[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    _LAST_REQ[0] = time.time()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

_ticker_to_cik: dict[str, str] = {}


def resolve_cik(ticker: str) -> Optional[str]:
    global _ticker_to_cik
    if not _ticker_to_cik:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        _ticker_to_cik = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
    return _ticker_to_cik.get(ticker.upper())


# ---------------------------------------------------------------------------
# Company facts (XBRL structured fundamentals)
# ---------------------------------------------------------------------------

def fetch_company_facts(ticker: str, cache: Cache) -> dict:
    """Pull XBRL companyfacts — used for peer-universe metric extraction."""
    key = f"edgar_facts_{ticker}"
    cached = cache.get(key, _FACTS_TTL_H)
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return {}

    try:
        data = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()
    except Exception:
        return {}

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    def latest(concept: str) -> Optional[float]:
        entries = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
        annual = []
        for e in entries:
            if e.get("form") not in ("10-K", "10-K/A") or e.get("val") is None:
                continue
            start, end = e.get("start"), e.get("end")
            if start and end:
                # duration concept: keep full fiscal years only
                try:
                    days = (datetime.fromisoformat(end[:10])
                            - datetime.fromisoformat(start[:10])).days
                except ValueError:
                    continue
                if not (340 <= days <= 380):
                    continue
            annual.append(e)
        if not annual:
            return None
        annual.sort(key=lambda e: e.get("end", ""), reverse=True)
        return float(annual[0]["val"])

    result = {
        "revenue": latest("Revenues") or latest("RevenueFromContractWithCustomerExcludingAssessedTax"),
        "net_income": latest("NetIncomeLoss"),
        "gross_profit": latest("GrossProfit"),
        "operating_income": latest("OperatingIncomeLoss"),
        "total_assets": latest("Assets"),
        "total_liabilities": latest("Liabilities"),
        "stockholders_equity": latest("StockholdersEquity"),
        "long_term_debt": latest("LongTermDebt"),
        "ebitda": latest("EarningsBeforeInterestTaxesDepreciationAndAmortization"),
    }
    cache.set(key, result)
    return result


def fetch_company_facts_timeseries(ticker: str, cache: Cache) -> dict:
    """Multi-year revenue and net income for CAGR computation."""
    key = f"edgar_facts_ts_{ticker}"
    cached = cache.get(key, _FACTS_TTL_H)
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return {}

    try:
        data = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()
    except Exception:
        return {}

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    def annual_series(*concepts: str) -> dict[str, float]:
        """Merge full-fiscal-year facts across concepts (companies switch tags
        over time, e.g. Revenues → RevenueFromContractWithCustomer...).
        Filters out quarterly periods that also appear in 10-K filings."""
        seen: dict[str, float] = {}
        for concept in concepts:
            entries = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            for e in entries:
                if e.get("form") not in ("10-K", "10-K/A") or e.get("val") is None:
                    continue
                start, end = e.get("start", ""), e.get("end", "")
                if not start or not end:
                    continue
                try:
                    days = (datetime.fromisoformat(end[:10])
                            - datetime.fromisoformat(start[:10])).days
                except ValueError:
                    continue
                if not (340 <= days <= 380):  # full fiscal year only
                    continue
                seen[end[:10]] = float(e["val"])
        return dict(sorted(seen.items(), reverse=True))

    result = {
        "revenue_by_year": annual_series(
            "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
            "SalesRevenueNet",
        ),
        "net_income_by_year": annual_series("NetIncomeLoss"),
        "fcf_by_year": {},  # FCF not in XBRL; computed from cashflow - capex
    }
    cache.set(key, result)
    return result


# ---------------------------------------------------------------------------
# Filing fetch + section extraction
# ---------------------------------------------------------------------------

def fetch_latest_filing_sections(ticker: str, form_type: str,
                                  cache: Cache, which: int = 0) -> dict[str, str]:
    """Return {item_7: text, item_1a: text} from a 10-K or 10-Q.

    `which` selects which matching filing, newest-first: 0 = latest,
    1 = prior year, etc. Used to pull the prior-year filing for tone-delta analysis.
    """
    key = f"edgar_sections_{ticker}_{form_type}_{which}"
    cached = cache.get(key, _FILINGS_TTL_H)
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return {}

    try:
        submissions = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    except Exception:
        return {}

    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    # Find the `which`-th matching filing (newest-first ordering)
    target_idx = None
    seen = 0
    for i, f in enumerate(forms):
        if f == form_type:
            if seen == which:
                target_idx = i
                break
            seen += 1

    if target_idx is None:
        return {}

    accession_dashes = accessions[target_idx]  # e.g. "0001193125-24-267818"
    accession_nodash = accession_dashes.replace("-", "")
    filing_date = dates[target_idx]

    # primaryDocument is available directly from the submissions response
    primary_docs = filings.get("primaryDocument", [])
    primary_doc = primary_docs[target_idx] if target_idx < len(primary_docs) else None

    # Fallback: query the filing index JSON
    if not primary_doc:
        try:
            idx_data = _get(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                f"/{accession_nodash}/{accession_dashes}-index.json"
            ).json()
            for doc in idx_data.get("directory", {}).get("item", []):
                name = doc.get("name", "")
                if name.endswith((".htm", ".html")) and not name.startswith("R"):
                    primary_doc = name
                    break
        except Exception:
            return {}

    if not primary_doc:
        return {}

    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_doc}"
    try:
        html = _get(doc_url).text
    except Exception:
        return {}

    sections = _extract_sections(html)
    sections["filing_date"] = filing_date
    sections["form_type"] = form_type
    cache.set(key, sections)
    return sections


def _extract_sections(html: str) -> dict[str, str]:
    """Best-effort extraction of Item 7 (MD&A) and Item 1A (Risk Factors).

    Modern EDGAR filings (iXBRL/XHTML) have the Table of Contents at the top;
    we skip ToC hits (where the next non-empty line is a page number) and find
    the real body section.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    lines = text.split("\n")

    item_patterns = {
        "item_7":  re.compile(r"^item\s+7[\.\s]", re.IGNORECASE),
        "item_7a": re.compile(r"^item\s+7a[\.\s]", re.IGNORECASE),
        "item_1a": re.compile(r"^item\s+1a[\.\s]", re.IGNORECASE),
        "item_1b": re.compile(r"^item\s+1b[\.\s]", re.IGNORECASE),
        "item_2":  re.compile(r"^item\s+2[\.\s]", re.IGNORECASE),
        "item_8":  re.compile(r"^item\s+8[\.\s]", re.IGNORECASE),
    }

    # Collect ALL boundary positions per item name
    all_boundaries: dict[str, list[int]] = {k: [] for k in item_patterns}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 3 or len(stripped) > 300:
            continue
        for name, pat in item_patterns.items():
            if pat.match(stripped):
                all_boundaries[name].append(i)

    def _is_toc_entry(idx: int) -> bool:
        """Return True if any of the next several non-empty lines is a bare page number."""
        seen_nonempty = 0
        for j in range(idx + 1, min(idx + 10, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            seen_nonempty += 1
            if re.match(r"^\d{1,3}$", nxt):
                return True
            if seen_nonempty >= 4:
                break
        return False

    def _best_boundary(name: str) -> int | None:
        """Prefer the last non-ToC occurrence; body sections appear after the ToC."""
        hits = all_boundaries.get(name, [])
        if not hits:
            return None
        # Walk from last to first; pick last non-ToC hit
        for idx in reversed(hits):
            if not _is_toc_entry(idx):
                return idx
        # All look like ToC (unlikely) — use the last hit anyway
        return hits[-1]

    def _extract(start_name: str, end_names: list[str]) -> str:
        start_i = _best_boundary(start_name)
        if start_i is None:
            return ""
        end_i: int | None = None
        for end_name in end_names:
            for idx in all_boundaries.get(end_name, []):
                if idx > start_i and (end_i is None or idx < end_i):
                    end_i = idx
        chunk = lines[start_i: end_i if end_i else start_i + 600]
        return "\n".join(chunk)[:8000]

    return {
        "item_7":  _extract("item_7",  ["item_7a", "item_8"]),
        "item_1a": _extract("item_1a", ["item_1b", "item_2"]),
    }


# ---------------------------------------------------------------------------
# Form 4 insider activity
# ---------------------------------------------------------------------------

def fetch_insider_transactions(ticker: str, cache: Cache,
                                lookback_days: int = 180) -> list[dict]:
    """Aggregate Form 4 transactions for the trailing N days."""
    key = f"edgar_form4_{ticker}_{lookback_days}"
    cached = cache.get(key, _FORM4_TTL_H)
    if cached is not None:
        return cached

    cik = resolve_cik(ticker)
    if not cik:
        return []

    try:
        submissions = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    except Exception:
        return []

    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    transactions = []

    for i, f in enumerate(forms):
        if f != "4":
            continue
        if dates[i] < cutoff:
            continue
        accession = accessions[i].replace("-", "")
        # primaryDocument is the XSL-rendered view (e.g. "xslF345X06/form4.xml");
        # stripping the xsl prefix yields the raw XML document.
        doc = primary_docs[i] if i < len(primary_docs) else ""
        raw_doc = doc.split("/")[-1] if doc else "form4.xml"
        try:
            xml = _get(
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{raw_doc}"
            ).text
            tx = _parse_form4_xml(xml, dates[i])
            transactions.extend(tx)
        except Exception:
            continue

    cache.set(key, transactions)
    return transactions


def _parse_form4_xml(xml_text: str, filing_date: str) -> list[dict]:
    soup = BeautifulSoup(xml_text, "lxml-xml")
    transactions = []

    for tx in soup.find_all("nonDerivativeTransaction"):
        code = tx.find("transactionCode")
        shares_el = tx.find("transactionShares", {"type": None}) or tx.find("transactionShares")
        price_el = tx.find("transactionPricePerShare")
        if not code:
            continue

        tx_code = code.get_text(strip=True)
        if tx_code not in ("P", "S"):  # only buys and sales
            continue

        try:
            shares = float(shares_el.find("value").get_text()) if shares_el else 0.0
            price = float(price_el.find("value").get_text()) if price_el else 0.0
        except (AttributeError, ValueError):
            continue

        transactions.append({
            "filing_date": filing_date,
            "transaction_code": tx_code,  # P=purchase, S=sale
            "shares": shares,
            "price_per_share": price,
            "value": shares * price,
            "is_buy": tx_code == "P",
        })

    return transactions
