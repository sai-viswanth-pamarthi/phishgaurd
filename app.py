"""
app.py – Phishing URL Detection Flask Application
"""

import os, json, time
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from utils.feature_extractor import extract_features, feature_vector, FEATURE_NAMES

app = Flask(__name__)

# ── Load model artefacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE, "models", "phishing_model.pkl")
SCALER_PATH  = os.path.join(BASE, "models", "scaler.pkl")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

# ── Threat thresholds ─────────────────────────────────────────────────────────
RISK_BANDS = [
    (0.80, "CRITICAL",  "🔴"),
    (0.60, "HIGH",      "🟠"),
    (0.40, "MEDIUM",    "🟡"),
    (0.20, "LOW",       "🟢"),
    (0.00, "SAFE",      "✅"),
]

def classify_risk(prob: float):
    for threshold, label, icon in RISK_BANDS:
        if prob >= threshold:
            return label, icon
    return "SAFE", "✅"


def analyse_url(url: str) -> dict:
    t0 = time.perf_counter()
    feats  = extract_features(url)
    vec    = np.array(list(feats.values())).reshape(1, -1)
    vec_s  = scaler.transform(vec)
    prob   = float(model.predict_proba(vec_s)[0][1])
    pred   = int(model.predict(vec_s)[0])
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    risk_label, risk_icon = classify_risk(prob)

    # Build a human-readable list of triggered red-flags
    red_flags = []
    if feats["ip_in_url"]:          red_flags.append("IP address used instead of domain")
    if feats["is_https"] == 0:      red_flags.append("No HTTPS – insecure connection")
    if feats["suspicious_tld"]:     red_flags.append("Suspicious top-level domain (TLD)")
    if feats["has_redirect"]:       red_flags.append("Redirect parameter detected")
    if feats["is_shortener"]:       red_flags.append("URL shortener service used")
    if feats["num_at_signs"] > 0:   red_flags.append("@ symbol found in URL (spoofing risk)")
    if feats["phishing_keywords"] >= 2: red_flags.append(f"{feats['phishing_keywords']} phishing keywords detected")
    if feats["brand_in_subdomain"]: red_flags.append("Brand name impersonated in subdomain")
    if feats["brand_in_path"]:      red_flags.append("Brand name appears in path (may be spoofing)")
    if feats["url_length"] > 100:   red_flags.append(f"Unusually long URL ({feats['url_length']} chars)")
    if feats["subdomain_depth"] > 2:red_flags.append(f"Deep subdomain nesting ({feats['subdomain_depth']} levels)")
    if feats["double_slash_path"]:  red_flags.append("Double slash found in path")
    if feats["hex_encoding"]:       red_flags.append("Hex/percent encoding in URL")
    if feats["has_port"]:           red_flags.append("Non-standard port specified")
    if feats["num_dots"] > 5:       red_flags.append(f"Excessive dots in URL ({feats['num_dots']})")

    # Top feature contributions
    importances = model.feature_importances_
    feat_vals   = list(feats.values())
    contributions = sorted(
        [{"name": FEATURE_NAMES[i], "value": feat_vals[i], "importance": round(float(importances[i]), 4)}
         for i in range(len(FEATURE_NAMES))],
        key=lambda x: x["importance"], reverse=True
    )[:10]

    return {
        "url":           url,
        "prediction":    pred,
        "probability":   round(prob * 100, 2),
        "risk_level":    risk_label,
        "risk_icon":     risk_icon,
        "is_phishing":   bool(pred),
        "red_flags":     red_flags,
        "features":      feats,
        "contributions": contributions,
        "scan_time_ms":  elapsed,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        result = analyse_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_scan():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls", [])
    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Provide a list of URLs under 'urls'"}), 400
    if len(urls) > 50:
        return jsonify({"error": "Max 50 URLs per batch request"}), 400
    results = []
    for url in urls:
        try:
            results.append(analyse_url(url.strip()))
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/features", methods=["POST"])
def features_only():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    return jsonify(extract_features(url))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
