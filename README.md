# 🛡️ PhishGuard — AI-Based Phishing URL Detection System

A production-ready machine learning web application that detects phishing URLs in real-time using a Random Forest classifier trained on 10,000+ URLs.

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| ML Model | Scikit-learn Random Forest (200 trees) |
| Feature Engineering | Custom 30-feature extractor (`tldextract`, `urllib`) |
| Frontend | Vanilla HTML/CSS/JS — cybersecurity dark UI |
| Serialization | `joblib` (model + scaler) |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **92%+** |
| ROC-AUC | **0.96+** |
| Cross-Validation | **5-fold CV** |
| Training Dataset | **10,000 URLs** (balanced) |
| Inference Latency | **< 5 ms** per URL |

---

## 🧠 Feature Engineering (30 Features)

### Length-based
- `url_length`, `domain_length`, `path_length`, `query_length`, `num_params`

### Character-count
- `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`
- `num_at_signs`, `num_ampersands`, `num_equals`, `num_digits_in_domain`

### Entropy
- `url_entropy`, `domain_entropy` (Shannon entropy)

### Security / Protocol
- `is_https`, `has_port`, `ip_in_url`

### Subdomain
- `subdomain_depth`, `has_www`

### Keyword Signals
- `phishing_keywords` (30 known phishing terms)
- `brand_in_subdomain`, `brand_in_path` (10 major brands)

### Structural Red-flags
- `has_redirect`, `double_slash_path`, `hex_encoding`
- `is_shortener`, `suspicious_tld`, `has_data_uri`

### Ratio
- `digit_to_letter_ratio`

---

## 📁 Project Structure

```
phishing-detector/
├── app.py                   # Flask application (routes + prediction logic)
├── train_model.py           # Dataset generation + model training
├── requirements.txt
├── models/
│   ├── phishing_model.pkl   # Trained Random Forest
│   ├── scaler.pkl           # StandardScaler
│   └── metrics.json         # Evaluation metrics
├── utils/
│   ├── __init__.py
│   └── feature_extractor.py # 30-feature URL parser
└── templates/
    └── index.html           # Single-page web UI
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Run Flask app
```bash
python app.py
# → http://localhost:5000
```

---

## 🔌 REST API

### Single URL Scan
```bash
POST /api/scan
Content-Type: application/json

{"url": "http://paypal-secure.tk/login?redirect=true"}
```

**Response:**
```json
{
  "url": "http://paypal-secure.tk/login?redirect=true",
  "is_phishing": true,
  "probability": 96.4,
  "risk_level": "CRITICAL",
  "red_flags": [
    "Suspicious top-level domain (TLD)",
    "2 phishing keywords detected",
    "Redirect parameter detected"
  ],
  "scan_time_ms": 2.8
}
```

### Batch Scan (up to 50 URLs)
```bash
POST /api/batch
Content-Type: application/json

{"urls": ["https://google.com", "http://evil-site.tk/login"]}
```

### Feature Extraction Only
```bash
POST /api/features
{"url": "https://example.com"}
```

### Model Metrics
```bash
GET /api/metrics
```

---

## 🎯 Phishing Detection Strategies Modelled

1. **Lookalike domains** — brand + random chars + suspicious TLD
2. **IP-based URLs** — skipping DNS entirely
3. **Subdomain spoofing** — `paypal.secure-xxxx.tk`
4. **URL shorteners** — hiding true destination
5. **Hex-encoded URLs** — obfuscating intent
6. **Excessive parameters** — evading simple filters
7. **@ symbol trick** — `http://legit.com@evil.tk`
8. **Double-slash injection** — `//verify//login`

---

## 📝 Resume Bullet Points (Project Highlights)

- Trained a **Random Forest classifier** on **10,000+ URLs** with **30 engineered phishing-related features**, achieving **92%+ accuracy** and **0.96+ ROC-AUC** on held-out test data.
- Implemented feature extraction techniques for URL analysis including **HTTPS validation**, **redirect detection**, **domain entropy**, **subdomain spoofing signals**, and **keyword frequency analysis**.
- Built a **Flask REST API** with single-URL and **batch scanning** (up to 50 URLs) endpoints delivering inference in **< 5 ms** per URL.
- Designed a **real-time web dashboard** with threat-level scoring, red-flag explanation, and feature-importance transparency.
