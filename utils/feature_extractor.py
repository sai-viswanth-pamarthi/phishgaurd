import re
import math
import tldextract
from urllib.parse import urlparse, parse_qs
from collections import Counter

# ── Suspicious keyword sets ──────────────────────────────────────────────────
PHISHING_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update", "banking",
    "password", "confirm", "ebayisapi", "webscr", "paypal", "free", "lucky",
    "winner", "prize", "claim", "urgent", "alert", "suspended", "validate",
    "credential", "authorize", "access", "wallet", "crypto", "support",
    "helpdesk", "invoice", "refund", "reset",
}

BRAND_KEYWORDS = {
    "paypal", "google", "apple", "amazon", "facebook", "microsoft", "netflix",
    "instagram", "twitter", "linkedin", "dropbox", "adobe", "yahoo", "outlook",
    "office365", "gmail", "chase", "wellsfargo", "bankofamerica", "citibank",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bl.ink", "lnkd.in", "db.tt", "qr.ae",
    "cutt.ly", "shorte.st", "rebrand.ly", "tiny.cc",
}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".online", ".site", ".info", ".biz", ".click", ".link", ".live",
}


def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_features(url: str) -> dict:
    """
    Extract 30 numerical phishing-related features from a URL.
    Returns a dict; call list(d.values()) to get the feature vector.
    """
    raw_url = url.strip()

    # Normalise – add scheme if missing so urlparse works
    if not re.match(r"https?://", raw_url, re.I):
        norm = "http://" + raw_url
    else:
        norm = raw_url

    parsed = urlparse(norm)
    ext = tldextract.extract(norm)

    netloc   = parsed.netloc or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    domain   = ext.domain or ""
    suffix   = ext.suffix or ""
    subdomain = ext.subdomain or ""
    full_host = netloc.split(":")[0]          # strip port

    url_lower = norm.lower()
    features = {}

    # ── Length-based ─────────────────────────────────────────────────────────
    features["url_length"]        = len(raw_url)
    features["domain_length"]     = len(full_host)
    features["path_length"]       = len(path)
    features["query_length"]      = len(query)
    features["num_params"]        = len(parse_qs(query))

    # ── Character-count ───────────────────────────────────────────────────────
    features["num_dots"]          = raw_url.count(".")
    features["num_hyphens"]       = raw_url.count("-")
    features["num_underscores"]   = raw_url.count("_")
    features["num_slashes"]       = raw_url.count("/")
    features["num_at_signs"]      = raw_url.count("@")
    features["num_ampersands"]    = raw_url.count("&")
    features["num_equals"]        = raw_url.count("=")
    features["num_digits_in_domain"] = sum(c.isdigit() for c in full_host)

    # ── Entropy ───────────────────────────────────────────────────────────────
    features["url_entropy"]       = round(_entropy(raw_url), 4)
    features["domain_entropy"]    = round(_entropy(full_host), 4)

    # ── Security / protocol ───────────────────────────────────────────────────
    features["is_https"]          = int(parsed.scheme.lower() == "https")
    features["has_port"]          = int(bool(re.search(r":\d+", netloc)))
    features["ip_in_url"]         = int(bool(
        re.match(r"^(\d{1,3}\.){3}\d{1,3}$", full_host)
    ))

    # ── Subdomain depth ───────────────────────────────────────────────────────
    features["subdomain_depth"]   = len(subdomain.split(".")) if subdomain else 0
    features["has_www"]           = int(subdomain.lower() == "www")

    # ── Keyword signals ───────────────────────────────────────────────────────
    features["phishing_keywords"] = sum(
        1 for kw in PHISHING_KEYWORDS if kw in url_lower
    )
    features["brand_in_subdomain"] = int(any(
        brand in subdomain.lower() for brand in BRAND_KEYWORDS
    ))
    features["brand_in_path"]     = int(any(
        brand in path.lower() for brand in BRAND_KEYWORDS
    ))

    # ── Structural red-flags ──────────────────────────────────────────────────
    features["has_redirect"]      = int("redirect" in url_lower or "url=" in url_lower)
    features["double_slash_path"] = int("//" in path)
    features["hex_encoding"]      = int("%" in raw_url)
    features["is_shortener"]      = int(full_host.lower() in SHORTENERS)
    features["suspicious_tld"]    = int(
        any(suffix.lower().endswith(t.lstrip(".")) for t in SUSPICIOUS_TLDS)
    )
    features["has_data_uri"]      = int(raw_url.startswith("data:"))

    # ── Ratio features ────────────────────────────────────────────────────────
    digit_count  = sum(c.isdigit() for c in raw_url)
    letter_count = sum(c.isalpha() for c in raw_url)
    features["digit_to_letter_ratio"] = round(
        digit_count / max(letter_count, 1), 4
    )

    return features


def feature_vector(url: str) -> list:
    return list(extract_features(url).values())


FEATURE_NAMES = list(extract_features("http://example.com").keys())
