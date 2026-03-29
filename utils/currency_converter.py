"""
FraudGuard — Currency Converter
=================================
Converts transaction amounts between currencies for threshold evaluation.

Uses static fallback rates from legal_rules.json / policy_config.
Can be overridden via environment variables:
  FX_USD_KES   (default: 129.50)
  FX_EUR_KES   (default: 140.20)
  FX_GBP_KES   (default: 163.80)

No external HTTP dependency — keeps the compliance engine offline-safe.
"""
import os
import logging

log = logging.getLogger(__name__)

# ── Static fallback rates (KES per 1 unit of foreign currency) ─────────────────
# Update these via env vars or legal_rules.json policy_config.currency_conversion

_STATIC_RATES: dict = {
    "USD": float(os.environ.get("FX_USD_KES", "129.50")),
    "EUR": float(os.environ.get("FX_EUR_KES", "140.20")),
    "GBP": float(os.environ.get("FX_GBP_KES", "163.80")),
    "KES": 1.0,
    "UGX": 0.034,   # Uganda Shilling → KES
    "TZS": 0.049,   # Tanzania Shilling → KES
    "RWF": 0.113,   # Rwanda Franc → KES
    "ETB": 2.40,    # Ethiopian Birr → KES
    "ZAR": 7.10,    # South African Rand → KES
    "GHS": 8.50,    # Ghanaian Cedi → KES
    "NGN": 0.085,   # Nigerian Naira → KES
    "AED": 35.25,   # UAE Dirham → KES
    "CNY": 17.90,   # Chinese Yuan → KES
    "INR": 1.56,    # Indian Rupee → KES
    "CAD": 95.20,   # Canadian Dollar → KES
    "AUD": 84.60,   # Australian Dollar → KES
    "CHF": 146.30,  # Swiss Franc → KES
    "JPY": 0.87,    # Japanese Yen → KES
}


def to_kes(amount: float, currency: str) -> float:
    """Convert an amount in any supported currency to KES."""
    cur = (currency or "KES").strip().upper()
    if cur == "KES":
        return float(amount)
    rate = _STATIC_RATES.get(cur)
    if rate is None:
        log.warning(f"[CurrencyConverter] Unknown currency '{cur}' — treating as KES.")
        return float(amount)
    return float(amount) * rate


def to_usd(amount: float, currency: str) -> float:
    """Convert an amount in any supported currency to USD."""
    kes_amount = to_kes(amount, currency)
    usd_rate = _STATIC_RATES.get("USD", 129.50)
    return kes_amount / usd_rate


def get_rate_to_kes(currency: str) -> float:
    """Return the KES exchange rate for a given currency."""
    cur = (currency or "KES").strip().upper()
    return _STATIC_RATES.get(cur, 1.0)


def infer_currency(txn: dict) -> str:
    """
    Infer currency from transaction fields.
    Priority: currency > type hints > default KES.
    """
    for field in ("currency", "Currency", "CURRENCY", "txn_currency"):
        val = txn.get(field)
        if val and isinstance(val, str) and len(val.strip()) in (3, 4):
            return val.strip().upper()
    # If channel suggests international
    channel = str(txn.get("channel", "")).lower()
    if "swift" in channel or "international" in channel or "forex" in channel:
        return "USD"
    return "KES"
