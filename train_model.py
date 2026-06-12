"""
train_model.py
Generates a synthetic 10,000-URL dataset, extracts features,
trains a Random Forest classifier, and saves the model + scaler.
"""

import os, random, string, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)

# Make sure utils is importable when running from project root
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.feature_extractor import feature_vector, FEATURE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic URL generators
# ─────────────────────────────────────────────────────────────────────────────

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
    "microsoft.com", "apple.com", "github.com", "stackoverflow.com", "bbc.co.uk",
    "nytimes.com", "espn.com", "cnn.com", "weather.com", "imdb.com",
    "dropbox.com", "adobe.com", "salesforce.com", "shopify.com", "stripe.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/products", "/services", "/blog",
    "/news/today", "/help/faq", "/account/settings", "/search?q=python",
    "/category/technology", "/2024/01/article-title", "/support/ticket/123",
]

PHISHING_WORDS    = ["secure", "login", "verify", "account", "update", "banking",
                     "confirm", "paypal", "amazon", "apple", "microsoft", "support"]
SUSPICIOUS_TLDS   = ["tk", "ml", "xyz", "top", "club", "online", "site", "info"]
SHORTENERS        = ["bit.ly", "tinyurl.com", "ow.ly", "is.gd", "cutt.ly"]

def _rand_str(n, charset=string.ascii_lowercase + string.digits):
    return "".join(random.choices(charset, k=n))

def _legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path   = random.choice(LEGIT_PATHS)
    scheme = "https" if random.random() < 0.92 else "http"
    return f"{scheme}://www.{domain}{path}"

def _phishing_url():
    strategy = random.randint(1, 8)

    if strategy == 1:           # Brand word + random domain + suspicious TLD
        brand = random.choice(PHISHING_WORDS)
        tld   = random.choice(SUSPICIOUS_TLDS)
        dom   = f"{brand}-{_rand_str(6)}.{tld}"
        path  = f"/{''.join(random.choices(PHISHING_WORDS, k=2))}.php?id={_rand_str(8)}"
        return f"http://{dom}{path}"

    elif strategy == 2:         # IP address URL
        ip   = ".".join(str(random.randint(1, 254)) for _ in range(4))
        path = f"/{random.choice(PHISHING_WORDS)}/index.php?cmd={_rand_str(12)}"
        return f"http://{ip}{path}"

    elif strategy == 3:         # Lookalike subdomain spoofing
        brand  = random.choice(PHISHING_WORDS)
        tld    = random.choice(SUSPICIOUS_TLDS)
        sub    = f"{brand}.secure-{_rand_str(4)}"
        return f"http://{sub}.{_rand_str(5)}.{tld}/login?redirect={_rand_str(10)}"

    elif strategy == 4:         # URL shortener
        shortener = random.choice(SHORTENERS)
        return f"http://{shortener}/{_rand_str(6)}"

    elif strategy == 5:         # Hex / encoded
        brand = random.choice(PHISHING_WORDS)
        enc   = "".join(f"%{ord(c):02x}" for c in _rand_str(4))
        tld   = random.choice(SUSPICIOUS_TLDS)
        return f"http://{brand}-{enc}.{tld}/verify?token={_rand_str(16)}"

    elif strategy == 6:         # Very long URL with many params
        brand = random.choice(PHISHING_WORDS)
        tld   = random.choice(SUSPICIOUS_TLDS)
        params = "&".join(f"{_rand_str(4)}={_rand_str(8)}" for _ in range(12))
        return f"http://{brand}.{_rand_str(8)}.{tld}/page?{params}"

    elif strategy == 7:         # @ symbol trick
        brand = random.choice(PHISHING_WORDS)
        tld   = random.choice(SUSPICIOUS_TLDS)
        return f"http://www.legit.com@{brand}.{tld}/login"

    else:                       # Double slash in path
        brand = random.choice(PHISHING_WORDS)
        tld   = random.choice(SUSPICIOUS_TLDS)
        return f"http://{_rand_str(5)}.{tld}//{brand}//verify?id={_rand_str(8)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build dataset  (10,000 URLs – balanced)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n=10000):
    random.seed(42)
    urls, labels = [], []

    half = n // 2
    for _ in range(half):
        urls.append(_legit_url());     labels.append(0)
    for _ in range(half):
        urls.append(_phishing_url());  labels.append(1)

    # shuffle
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)
    return list(urls), list(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(urls):
    X = []
    for url in urls:
        try:
            X.append(feature_vector(url))
        except Exception:
            X.append([0] * len(FEATURE_NAMES))
    return np.array(X)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("🔨 Building dataset (10,000 URLs) …")
    urls, labels = build_dataset(10000)
    y = np.array(labels)

    print("⚙️  Extracting features …")
    X = extract_all(urls)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("🌲 Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred  = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="accuracy")

    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance top-10
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("\nTop-10 Features:")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<30s} {importances[idx]:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,     "models/phishing_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Save metrics for the dashboard
    metrics = {
        "accuracy": round(acc * 100, 2),
        "auc":      round(auc, 4),
        "cv_mean":  round(cv_scores.mean() * 100, 2),
        "cv_std":   round(cv_scores.std()  * 100, 2),
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(importances[i]), 4) for i in indices
        },
    }
    import json
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Model saved → models/phishing_model.pkl")
    print("✅ Scaler saved → models/scaler.pkl")
    print("✅ Metrics saved → models/metrics.json")
    return rf, scaler, metrics


if __name__ == "__main__":
    train()
