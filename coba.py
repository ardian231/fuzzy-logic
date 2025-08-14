# app_fuzzy.py
import firebase_admin
from firebase_admin import credentials, db, firestore
from flask import Flask, jsonify, request

# ============== FIREBASE CONFIG ==============
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
firestore_client = firestore.client()

app = Flask(__name__)

# ============== UTILITIES ==============
def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def tri(x, a, b, c):
    if a == b == c: 
        return 1.0 if x == a else 0.0
    if x <= a or x >= c: 
        return 0.0
    if x < b:  
        return (x - a) / max(b - a, 1e-9)
    return (c - x) / max(c - b, 1e-9)

def trap(x, a, b, c, d):
    if x <= a or x >= d: 
        return 0.0
    if b <= x <= c: 
        return 1.0
    if a < x < b:  
        return (x - a) / max(b - a, 1e-9)
    if c < x < d:  
        return (d - x) / max(d - c, 1e-9)
    return 0.0

def pmt(rate_per_period, nper, pv):
    if nper <= 0: 
        return 0
    if rate_per_period == 0: 
        return pv / nper
    r = rate_per_period
    return pv * (r * (1 + r) ** nper) / ((1 + r) ** nper - 1)

# ============== KEBIJAKAN: MARGIN & TENOR (tanpa input tenor) ==============
MARGIN_DEFAULTS = {
    # Motor
    "motor_baru":   {"min": 12, "max": 48, "margin_tahunan": 0.20},
    "motor_bekas":  {"min": 12, "max": 48, "margin_tahunan": 0.21},

    # Mobil
    "mobil_baru":   {"min": 12, "max": 48, "margin_tahunan": 0.16},
    "mobil_bekas":  {"min": 12, "max": 48, "margin_tahunan": 0.17},

    # AMANAH (Adira Multi Dana Syariah) — umumkan ke 48 bln
    "amanah":       {"min": 12, "max": 48, "margin_tahunan": 0.16},
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def pick_scheme(item_label: str) -> dict:
    it = _norm(item_label)
    if it == "motor baru":   return MARGIN_DEFAULTS["motor_baru"]
    if it == "motor bekas":  return MARGIN_DEFAULTS["motor_bekas"]
    if it == "mobil baru":   return MARGIN_DEFAULTS["mobil_baru"]
    if it == "mobil bekas":  return MARGIN_DEFAULTS["mobil_bekas"]
    if it.startswith("amanah"): return MARGIN_DEFAULTS["amanah"]
    return {"min": 12, "max": 48, "margin_tahunan": 0.18}  # fallback

def infer_nominal(dp: float, nominal: float | None, asumsi_dp_pct: float = 0.20) -> float:
    """Jika nominal pinjaman tidak ada, tebak dari DP dan asumsi %DP kebijakan."""
    if nominal and nominal > 0:
        return float(nominal)
    if dp and dp > 0 and asumsi_dp_pct > 0:
        return float(dp) / asumsi_dp_pct
    return 0.0

def choose_tenor_and_installment(
    gaji: float,
    nominal: float,
    item_label: str,
    dp_pct: float | None = None,
    tenor_hint: int | None = None,
    margin_tahunan: float | None = None,
    target_pti: float = 0.33,
    pti_low: float = 0.25,
    pti_high: float = 0.50,
):
    """
    Pilih tenor otomatis agar PTI mendekati target.
    - Validasi tenor ke rentang kebijakan (min..max).
    - Jika nominal/gaji tidak valid → fallback pakai PTI target.
    - Bias: DP% tinggi condong ke tenor lebih pendek; DP% rendah condong ke tenor lebih panjang.
    Return: (tenor_dipakai, margin_tahun, angsuran_bulanan, pti)
    """
    sc = pick_scheme(item_label)
    n_min, n_max = sc["min"], sc["max"]
    year_rate = float(margin_tahunan) if margin_tahunan is not None else sc["margin_tahunan"]

    if nominal <= 0 or gaji <= 0:
        angs = target_pti * max(gaji, 1.0)
        return int(n_max), year_rate, angs, angs / max(gaji, 1.0)

    if tenor_hint:
        n = max(n_min, min(int(tenor_hint), n_max))
        angs = pmt(year_rate/12.0, n, nominal)
        return n, year_rate, angs, angs / gaji

    best = None  # (penalty, n, angs, pti)
    for n in range(n_min, n_max + 1):
        angs = pmt(year_rate/12.0, n, nominal)
        pti  = angs / gaji
        penalty = abs(pti - target_pti)
        if dp_pct is not None:
            if dp_pct >= 0.24:
                penalty -= 1e-4 * (n_max - n)  # prefer lebih pendek
            elif dp_pct < 0.12:
                penalty -= 1e-4 * (n - n_min)  # prefer lebih panjang
        cand = (penalty, n, angs, pti)
        if best is None or cand < best:
            best = cand

    _, n_sel, angs_sel, pti_sel = best

    if dp_pct is not None and dp_pct >= 0.24:
        candidates = []
        for n in range(n_min, n_max + 1):
            angs = pmt(year_rate/12.0, n, nominal)
            pti  = angs / gaji
            if pti_low <= pti <= target_pti:
                candidates.append((n, angs, pti))
        if candidates:
            n_short, angs_short, pti_short = min(candidates, key=lambda t: t[0])
            return n_short, year_rate, angs_short, pti_short

    return n_sel, year_rate, angs_sel, pti_sel

# ============== MEMBERSHIP ==============
# PTI = Angsuran/Gaji (0..1.5)
def mu_pti(x):
    x = clamp(x, 0, 1.5)
    return {
        "VeryLow": trap(x, 0.00, 0.00, 0.18, 0.25),  # ≤ 25%
        "Low":     tri(x, 0.22, 0.28, 0.33),        # 25–33%
        "Medium":  tri(x, 0.30, 0.40, 0.50),        # 33–50%
        "High":    trap(x, 0.45, 0.60, 1.50, 1.50), # > 50%
    }

# CLI = CicilanLain/Gaji (0..0.6)
def mu_cli(x):
    x = clamp(x, 0, 0.6)
    return {
        "Low":    trap(x, 0.00, 0.00, 0.10, 0.15),
        "Medium": tri(x, 0.10, 0.20, 0.30),
        "High":   trap(x, 0.20, 0.30, 0.60, 0.60),
    }

# DP% = dp/nominal (0..1). Jika tak ada → Medium (netral)
def mu_dp(dp_pct):
    if dp_pct is None:
        return {"Low": 0.0, "Medium": 1.0, "High": 0.0}
    x = clamp(dp_pct, 0, 1)
    return {
        "Low":    trap(x, 0.00, 0.00, 0.12, 0.16),
        "Medium": tri(x, 0.12, 0.20, 0.30),
        "High":   trap(x, 0.24, 0.35, 1.00, 1.00),
    }

# ============== ATURAN KERAS ==============
def aturan_keras(gaji, angsuran):
    if gaji <= 0: 
        return False, "Gaji tidak valid."
    if angsuran <= 0: 
        return False, "Angsuran tidak valid."
    if angsuran >= gaji: 
        return False, "Angsuran ≥ 100% gaji."
    return True, "OK"

# ============== MAMDANI RULE BASE ==============
REP = {"Reject": 25, "Consider": 50, "Approve": 85}
# Akomodasi rule analis: x4 layak, x3 pertimbangan, x2 dipertimbangkan bila tanpa cicilan lain
RULES = [
    (("PTI","VeryLow"), (), (), "Approve"),

    (("PTI","Low"), ("CLI","Low"), (), "Approve"),
    (("PTI","Low"), ("DP","High"), (), "Approve"),

    (("PTI","Medium"), ("CLI","Low"), ("DP","High"), "Approve"),
    (("PTI","Medium"), ("CLI","Low"), (), "Consider"),
    (("PTI","Medium"), ("CLI","Medium"), (), "Consider"),
    (("PTI","Medium"), ("CLI","High"), (), "Reject"),

    (("PTI","High"), ("CLI","Low"), (), "Consider"),  # x2 kalau CLI rendah/0
    (("PTI","High"), ("CLI","Medium"), (), "Reject"),
    (("PTI","High"), ("CLI","High"), (), "Reject"),

    (("DP","Low"), ("CLI","High"), (), "Reject"),
    (("DP","High"), ("CLI","Low"), (), "Consider"),
]

def fuzzy_infer(pti, cli, dp_pct):
    muP, muC, muD = mu_pti(pti), mu_cli(cli), mu_dp(dp_pct)
    num = den = 0.0
    fired = []
    for a, b, c, out in RULES:
        mu_a = muP[a[1]] if a else 1.0
        mu_b = muC[b[1]] if b else 1.0
        mu_c = muD[c[1]] if c else 1.0
        strength = min(mu_a, mu_b, mu_c)
        if strength > 0:
            num += strength * REP[out]
            den += strength
            fired.append({"rule": f"{a}&{b}&{c}->{out}", "strength": round(strength, 3), "output": out})
    score = num / den if den > 0 else 0.0
    label = "Approve" if score >= 60 else ("Consider" if score >= 40 else "Reject")
    return score, label, fired

PENJELASAN_STATUS = {
    "Approve":  ["PTI rendah.", "Beban cicilan lain terkendali.", "DP mendukung."],
    "Consider": ["Sebagian indikator cukup.", "Pertimbangkan tambah DP atau kecilkan nominal."],
    "Reject":   ["PTI/CLI tinggi.", "DP rendah atau penghasilan belum mencukupi."],
}
def risiko(score): 
    return "RENDAH" if score >= 70 else ("SEDANG" if score >= 55 else "TINGGI")

# ============== EVALUASI TANPA INPUT TENOR ==============
def evaluasi_akhir(raw):
    # Ambil input
    gaji         = float(raw.get("gaji", 0) or 0)
    cicilan_lain = float(raw.get("cicilan_lain", raw.get("installment", 0)) or 0)
    nominal_in   = float(raw.get("pengajuan_baru", raw.get("nominal", 0)) or 0)
    dp           = float(raw.get("dp", 0) or 0)
    item_label   = raw.get("jenis_pengajuan", raw.get("item", ""))  # "Motor Bekas", dll
    tenor_hint   = raw.get("tenor")  # opsional, tetap divalidasi bila ada
    margin_hint  = raw.get("margin_tahunan")  # opsional

    # DP% & nominal
    nominal = infer_nominal(dp, nominal_in, asumsi_dp_pct=0.20)
    dp_pct = (dp / nominal) if (dp > 0 and nominal > 0) else None

    # Jika angsuran tidak dikirim, pilih tenor & hitung angsuran otomatis
    angsuran_in = float(raw.get("angsuran", 0) or 0)
    if angsuran_in > 0:
        angsuran = angsuran_in
        tenor_used = int(tenor_hint) if tenor_hint else None
        margin_used = float(margin_hint) if margin_hint else None
    else:
        tenor_used, margin_used, angsuran, _pti_est = choose_tenor_and_installment(
            gaji=gaji,
            nominal=nominal,
            item_label=item_label,
            dp_pct=dp_pct,
            tenor_hint=tenor_hint,
            margin_tahunan=margin_hint,
            target_pti=0.33,
            pti_low=0.25,
            pti_high=0.50
        )

    # Knock-out checks
    ok, alasan = aturan_keras(gaji, angsuran)
    if not ok:
        return {
            "status": "Reject",
            "risiko": "TINGGI",
            "alasan": [alasan],
            "saran": "Perbaiki input / kecilkan nominal / tambah DP.",
            "score": 0,
            "detail": {
                "pti": None, "cli": None, "dp_pct": dp_pct,
                "tenor": tenor_used, "margin_tahun": margin_used,
                "angsuran": round(angsuran)
            },
            "fired_rules": []
        }

    # Rasio inti
    pti = angsuran / gaji if gaji > 0 else 10.0
    cli = cicilan_lain / gaji if gaji > 0 else 10.0

    # Fuzzy infer
    score, label, fired = fuzzy_infer(pti, cli, dp_pct)

    # Catatan singkat PTI
    note = "PTI aman" if pti <= 0.33 else ("PTI sedang" if pti <= 0.5 else "PTI tinggi")

    return {
        "status": label,
        "risiko": risiko(score),
        "alasan": PENJELASAN_STATUS[label],
        "saran": note,
        "score": round(score, 2),
        "detail": {
            "pti": round(pti, 4), "cli": round(cli, 4),
            "dp_pct": round(dp_pct, 4) if dp_pct is not None else None,
            "tenor": tenor_used, "margin_tahun": margin_used,
            "angsuran": round(angsuran)
        },
        "fired_rules": fired[:5]
    }

# ============== API ==============
@app.route("/")
def home():
    return "API fuzzy kelayakan (tenor otomatis) berjalan."

@app.route("/prediksi", methods=["POST"])
def prediksi():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        return jsonify(evaluasi_akhir(payload))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/run_fuzzy", methods=["GET"])
def run_fuzzy():
    data = db.reference("orders").get() or {}
    processed = 0
    for doc_id, r in data.items():
        status = str(r.get("status", "")).lower()
        if status in ["processed", "cancel", "process"]:
            continue
        payload = {
            "gaji": r.get("income", 0),
            "cicilan_lain": r.get("installment", 0),
            "pengajuan_baru": r.get("nominal", 0),
            "dp": r.get("dp", 0),
            "item": r.get("item", ""),           # "Motor Bekas", dll
            "tenor": r.get("tenor"),             # opsional
            "margin_tahunan": r.get("margin_tahunan"),  # opsional
            "angsuran": r.get("angsuran", 0),    # jika ada, dipakai langsung
        }
        hasil = evaluasi_akhir(payload)
        hasil["agent"] = {
            "email": r.get("agentEmail"),
            "nama": r.get("agentName"),
            "telepon": r.get("agentPhone")
        }
        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        processed += 1
    return jsonify({"message": f"Processed {processed} records."})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong", "status": "ok"})

@app.route("/warmup", methods=["GET"])
def warmup():
    try:
        _ = firestore_client.collection("hasil_prediksi").limit(1).get()
        return jsonify({"status": "warm"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
