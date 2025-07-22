import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from datetime import datetime

# ===================== KONFIGURASI =====================
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

firestore_client = firestore.client()

# ===================== APLIKASI FLASK =====================
app = Flask(__name__)

# ===================== DATA STATIS =====================
PEKERJAAN_ALIAS = {
    "pns": [
        "kelurahan", "kecamatan", "dinas", "pemkab", "pemkot", "puskesmas", "bpn", "rsud", "bumn", "bkd",
        "bpjs", "kemen", "perangkat desa", "guru", "sekolah negeri"
    ],
    "karyawan": [
        "pt", "cv", "toko", "minimarket", "indomaret", "alfamart", "supermarket", "mall", "kantor", "plaza",
        "dealer", "kopkar", "perusahaan", "swasta", "asuransi", "karyawan", "bank", "industri", "hotel", "sekolah swasta"
    ],
    "wiraswasta": [
        "usaha", "warung", "dagang", "bengkel", "konter", "resto", "kuliner", "umkm", "cafe", "kosmetik", "laundry",
        "barbershop", "cucian", "perabot", "tokopedia", "shopee", "jualan", "dropship", "jual", "jasa"
    ],
    "driver": [
        "ojol", "gojek", "grab", "sopir", "driver", "kurir", "pengemudi", "angkot", "ojek", "logistik", "ngojek"
    ],
    "freelancer": [
        "freelance", "editor", "fotografer", "youtuber", "desainer", "animator", "konten", "blogger", "musisi",
        "influencer", "penulis"
    ],
    "buruh": [
        "pabrik", "buruh", "kuli", "gudang", "produksi", "tenaga kasar", "angkut", "operator"
    ],
    "profesional": [
        "dokter", "pengacara", "notaris", "arsitek", "akuntan", "psikolog", "insinyur", "konsultan", "bidan",
        "perawat", "auditor"
    ],
    "petani": [
        "petani", "perkebunan", "sawah", "ladang", "tani", "nelayan", "tambak", "peternak"
    ]
}

PENJELASAN_STATUS = {
    "LAYAK": ["Pendapatan memadai.", "Jenis pekerjaan dan cicilan sesuai."],
    "DI PERTIMBANGKAN": ["Data perlu tinjauan lebih lanjut.", "Beberapa indikator belum optimal."],
    "TIDAK LAYAK": ["Pendapatan atau pekerjaan kurang mendukung.", "Beban cicilan terlalu tinggi."],
    "DITOLAK": ["Gagal lolos syarat dasar (aturan keras)."]
}

# ===================== UTILITAS =====================
def normalisasi_pekerjaan(teks):
    teks = teks.lower().strip()
    semua_alias = [s for lst in PEKERJAAN_ALIAS.values() for s in lst]
    tfidf = TfidfVectorizer().fit(semua_alias + [teks])
    v_teks = tfidf.transform([teks])
    return max(PEKERJAAN_ALIAS, key=lambda label: cosine_similarity(v_teks, tfidf.transform(PEKERJAAN_ALIAS[label])).max())

def konversi_uang(teks):
    if pd.isnull(teks): return 0
    teks = str(teks).lower().replace("rp", "").replace(" ", "").replace(".", "").replace(",", ".")
    matches = re.findall(r"(\d+\.?\d*)\s*(k|rb|ribu|jt|juta|m|miliar|b)?", teks)
    satuan_map = {"k":1e3, "rb":1e3, "ribu":1e3, "jt":1e6, "juta":1e6, "m":1e6, "miliar":1e9, "b":1e9}
    return int(sum(float(n) * satuan_map.get(s, 1) for n, s in matches))

def rekomendasi_tenor(gaji, plafon):
    if gaji <= 0 or plafon <= 0: return 12
    ratio = plafon / gaji
    # Membership function
    gr = lambda g: max(0, min(1, (6_000_000 - g)/3_000_000)) if g <= 6_000_000 else max(0, min(1, (g - 6_000_000)/3_000_000))
    rr = lambda r: max(0, min(1, (0.4 - r)/0.2)) if r <= 0.4 else max(0, min(1, (r - 0.4)/0.2))
    rules = [(min(gr(gaji), rr(ratio)), t) for gr, rr, t in [
        (lambda g: 1, lambda r: 1, 36), (lambda g: 1, lambda r: 0.5, 30), (lambda g: 1, lambda r: 0, 24),
        (lambda g: 0.5, lambda r: 1, 30), (lambda g: 0.5, lambda r: 0.5, 24), (lambda g: 0.5, lambda r: 0, 18),
        (lambda g: 0, lambda r: 1, 24), (lambda g: 0, lambda r: 0.5, 18), (lambda g: 0, lambda r: 0, 12)]
    ]
    n, d = sum(w * t for w, t in rules), sum(w for w, _ in rules)
    return min([12, 18, 24, 30, 36], key=lambda x: abs(x - n / d)) if d != 0 else 12

def aturan_keras(data):
    if data["pekerjaan"] == "wiraswasta" and data["gaji"] == 0:
        return True, "Wiraswasta tanpa penghasilan tetap."
    if data["jenis_pengajuan"] == "mobil" and data["gaji"] < 10_000_000 and data["cicilan_lain"] > 0:
        return False, "Gaji <10jt dan ada cicilan lain."
    if (data["pengajuan_baru"] / data["tenor"]) > 0.3 * data["gaji"]:
        return False, "Angsuran melebihi 30% dari gaji."
    return True, "Lolos aturan keras."

def skor_fuzzy(data):
    skor = 10 if data["gaji"] < 3_000_000 else 20 if data["gaji"] < 5_000_000 else 25 if data["gaji"] < 7_000_000 else 30 if data["gaji"] < 10_000_000 else 35
    if data["jenis_pengajuan"] == "mobil":
        if data["cicilan_lain"] > 0: skor += 5 if data["gaji"] > 10_000_000 else -10
        else: skor += 10 if data["gaji"] > 10_000_000 else 5 if data["gaji"] == 10_000_000 else -10
    skor += {"pns":30, "karyawan":25, "profesional":20, "wiraswasta":15, "freelancer":15, "driver":10, "buruh":10}.get(data["pekerjaan"], 10)
    skor += 20 if data["cicilan_lain"] == 0 else 10 if data["cicilan_lain"] < 0.2 * data["gaji"] else 0
    skor += 20 if data["pengajuan_baru"] <= 0.3 * data["gaji"] else 10 if data["pengajuan_baru"] <= 0.5 * data["gaji"] else 0
    return skor

def evaluasi_akhir(data):
    for k in ["gaji", "cicilan_lain", "pengajuan_baru"]:
        data[k] = konversi_uang(data.get(k, 0)) if isinstance(data[k], str) else data[k]
    data["pekerjaan"] = normalisasi_pekerjaan(data["pekerjaan"])
    data["tenor"] = data.get("tenor") or rekomendasi_tenor(data["gaji"], data["pengajuan_baru"])
    angsuran = data["pengajuan_baru"] / data["tenor"]
    data["angsuran"] = int(angsuran)
    data["rasio_angsuran"] = round(angsuran / data["gaji"], 2)
    lolos, _ = aturan_keras(data)
    if data["pekerjaan"] == "wiraswasta" and data["gaji"] == 0 and lolos:
        return {"status": "DI PERTIMBANGKAN", "skor_fuzzy": 50, "risiko": "SEDANG", "saran": "Perlu survey lapangan.", "input": data}
    if not lolos:
        return {"status": "DITOLAK", "skor_fuzzy": 0, "risiko": "TINGGI", "saran": "Tidak memenuhi aturan keras.", "input": data}
    skor = skor_fuzzy(data)
    status = "LAYAK" if skor >= 70 else "DI PERTIMBANGKAN" if skor >= 50 else "TIDAK LAYAK"
    return {"status": status, "skor_fuzzy": skor, "risiko": "RENDAH" if skor >= 75 else "SEDANG" if skor >= 55 else "TINGGI", "saran": "Validasi tambahan diperlukan." if status != "LAYAK" else None, "input": data}

# ===================== ENDPOINT =====================
@app.route("/")
def home(): return "API aktif."

@app.route("/prediksi", methods=["POST"])
def prediksi():
    data = request.form.to_dict()
    data["jenis_pengajuan"] = data.get("item", "").lower()
    return jsonify(evaluasi_akhir(data))

@app.route("/run_fuzzy")
def run_fuzzy():
    data = db.reference("orders").get()
    if not data: return jsonify({"message": "No new data."})
    processed = 0
    for doc_id, record in data.items():
        if record.get("status") in ["processed", "cancel", "process"]: continue
        record.update({
            "gaji": record.get("income", 0),
            "cicilan_lain": record.get("installment", 0),
            "pengajuan_baru": record.get("nominal", 0),
            "pekerjaan": record.get("job", ""),
            "jenis_pengajuan": record.get("item", ""),
            "tenor": rekomendasi_tenor(konversi_uang(record.get("income", 0)), konversi_uang(record.get("nominal", 0)))
        })
        hasil = evaluasi_akhir(record)
        hasil["timestamp"] = datetime.utcnow().isoformat()
        hasil["agent"] = {"email": record.get("agentEmail"), "nama": record.get("agentName"), "telepon": record.get("agentPhone")}
        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        processed += 1
    return jsonify({"message": f"Processed {processed} records."})

@app.route("/ping")
def ping(): return jsonify({"message": "pong"})

@app.route("/warmup")
def warmup():
    try:
        firestore_client.collection("hasil_prediksi").limit(1).get()
        return jsonify({"status": "warm"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
