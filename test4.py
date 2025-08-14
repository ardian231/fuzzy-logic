import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from PIL import Image
from datetime import datetime
import io
import requests
import time

# Firebase Initialization
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

firestore_client = firestore.client()

app = Flask(__name__)

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
    "DITOLAK": ["Gagal lolos syarat dasar (aturan keras).","kurangi jumlah pengajuan atau lunasi cicilan lain."]
}

# ===================== UTILITY =====================
def normalisasi_pekerjaan(teks):
    teks = teks.lower().strip()
    semua_alias = [s for lst in PEKERJAAN_ALIAS.values() for s in lst]
    tfidf = TfidfVectorizer().fit(semua_alias + [teks])
    v_teks = tfidf.transform([teks])

    label_terpilih, skor_tertinggi = teks, 0
    for label, sinonim in PEKERJAAN_ALIAS.items():
        skor = cosine_similarity(v_teks, tfidf.transform(sinonim)).max()
        if skor > skor_tertinggi:
            label_terpilih, skor_tertinggi = label, skor
    return label_terpilih

def konversi_uang(teks):
    if pd.isnull(teks) or str(teks).strip() == "":
        return 0

    teks = str(teks).lower().replace("rp", "").replace(",", ".")
    parts = teks.split()
    hasil = []

    for part in parts:
        if re.match(r"^[0-9.]+$", part):
            if part.count(".") > 1:
                part = part.replace(".", "")
        hasil.append(part)

    teks = " ".join(hasil)
    matches = re.findall(r"([0-9.]+)\s*(k|rb|ribu|jt|juta|m|miliar|b)?", teks)

    total = 0
    for angka, satuan in matches:
        try:
            num = float(angka)
        except ValueError:
            continue

        if satuan in ["k", "rb", "ribu"]:
            total += num * 1_000
        elif satuan in ["jt", "juta"]:
            total += num * 1_000_000
        elif satuan in ["miliar", "m"]:
            total += num * 1_000_000_000
        else:
            total += num
    return int(total)



def rekomendasi_tenor(gaji, plafon=None):
    if gaji <= 0 or plafon is None or plafon <= 0:
        return 12  # fallback default

    ratio = plafon / gaji

    # Membership Gaji
    if gaji <= 3_000_000:
        gaji_rendah = 1
        gaji_sedang = 0
        gaji_tinggi = 0
    elif gaji <= 6_000_000:
        gaji_rendah = (6_000_000 - gaji) / 3_000_000
        gaji_sedang = (gaji - 3_000_000) / 3_000_000
        gaji_tinggi = 0
    elif gaji <= 9_000_000:
        gaji_rendah = 0
        gaji_sedang = (9_000_000 - gaji) / 3_000_000
        gaji_tinggi = (gaji - 6_000_000) / 3_000_000
    else:
        gaji_rendah = 0
        gaji_sedang = 0
        gaji_tinggi = 1

    # Membership Ratio
    if ratio <= 0.2:
        ratio_kecil = 1
        ratio_sedang = 0
        ratio_besar = 0
    elif ratio <= 0.4:
        ratio_kecil = (0.4 - ratio) / 0.2
        ratio_sedang = (ratio - 0.2) / 0.2
        ratio_besar = 0
    elif ratio <= 0.6:
        ratio_kecil = 0
        ratio_sedang = (0.6 - ratio) / 0.2
        ratio_besar = (ratio - 0.4) / 0.2
    else:
        ratio_kecil = 0
        ratio_sedang = 0
        ratio_besar = 1

    # Aturan Sugeno: (bobot, nilai_tenor)
    rules = [
    (min(gaji_rendah, ratio_besar), 36),   
    (min(gaji_rendah, ratio_sedang), 30),
    (min(gaji_rendah, ratio_kecil), 24),   

    (min(gaji_sedang, ratio_besar), 30),
    (min(gaji_sedang, ratio_sedang), 24),
    (min(gaji_sedang, ratio_kecil), 18),

    (min(gaji_tinggi, ratio_besar), 24),
    (min(gaji_tinggi, ratio_sedang), 18),
    (min(gaji_tinggi, ratio_kecil), 12)
]


    numerator = sum(weight * output for weight, output in rules)
    denominator = sum(weight for weight, _ in rules)

    if denominator == 0:
        return 12  # fallback jika semua keanggotaan nol

    hasil = numerator / denominator
    tenor_tersedia = [12, 18, 24, 30, 36,]
    tenor_terdekat = min(tenor_tersedia, key=lambda x: abs(x - hasil))
    return tenor_terdekat

def kategori_risiko(skor):
    return "RENDAH" if skor >= 70 else "SEDANG" if skor >= 55 else "TINGGI"

def saran_perbaikan(status):
    if status == "TIDAK LAYAK":
        return "Kurangi jumlah pengajuan atau pastikan cicilan lain telah lunas."
    elif status == "DI PERTIMBANGKAN":
        return "Perlu tinjauan lapangan atau dokumen tambahan untuk validasi."
    return None
# ===================== ATURAN & PENILAIAN =====================
def aturan_keras(data):
    if data["pekerjaan"] == "wiraswasta" and data["gaji"] == 0:
        return True, "Wiraswasta dengan gaji 0 akan dipertimbangkan untuk survey."

    if data["jenis_pengajuan"] == "mobil" and data["gaji"] < 10_000_000 and data["cicilan_lain"] > 0:
        return False, "Gaji <10jt dan ada cicilan lain."

    angsuran_per_bulan = data["pengajuan_baru"] / data["tenor"]
    if angsuran_per_bulan > 0.30 * data["gaji"]:
        return False, "Angsuran > 30% dari gaji."

    return True, "Lolos aturan keras."

def skor_fuzzy(data):
    skor = 0
    gaji = data["gaji"]
    if gaji >= 10_000_000:
        skor += 35
    elif gaji >= 7_000_000:
        skor += 35
    elif gaji >= 5_000_000:
        skor += 30
    elif gaji >= 3_000_000:
        skor += 25
    elif gaji >= 2_000_000:
        skor += 20
    elif gaji >= 1_000_000:
        skor += 15
    else:
        skor += 10

    # Adjust score based on Adira car loan rules
    if data["jenis_pengajuan"] == "mobil":
        if data["cicilan_lain"] > 0: # If other installments exist
            if data["gaji"] == 10_000_000:
                skor -= 10 # Not recommended, lower score
            elif data["gaji"] > 10_000_000:
                skor += 5 # Considered, slight positive
        else: # No other installments
            if data["gaji"] < 10_000_000:
                skor -= 10 # Not recommended, lower score
            elif data["gaji"] == 10_000_000:
                skor += 5 # Considered, slight positive
            elif data["gaji"] > 10_000_000:
                skor += 10 # Approve, higher score

    # Score from pekerjaan
    map_pekerjaan = {
        'pns': 30, 'karyawan': 25, 'profesional': 25,
        'wiraswasta': 20, 'freelancer': 20,
        'driver': 15, 'buruh': 15, 'petani':15, 'tidak bekerja': 5
    }
    skor += map_pekerjaan.get(data["pekerjaan"], 10)

    if data["cicilan_lain"] == 0:
        skor += 25
    elif data["cicilan_lain"] < 0.20 * data["gaji"]:
        skor += 15
    else:
        skor += 5

    return skor

def evaluasi_akhir(data):
    for k in ["gaji", "cicilan_lain", "pengajuan_baru"]:
        data[k] = konversi_uang(data[k]) if isinstance(data[k], str) else data[k]

    data["pekerjaan"] = normalisasi_pekerjaan(data["pekerjaan"])

    if "tenor" not in data or not data["tenor"]:
        data["tenor"] = rekomendasi_tenor(data["gaji"], data["pengajuan_baru"])

    angsuran_per_bulan = data["pengajuan_baru"] / data["tenor"]
    rasio = angsuran_per_bulan / data["gaji"]

    data["angsuran"] = int(angsuran_per_bulan)
    data["rasio_angsuran"] = round(rasio, 2)

    if rasio > 0.5:
        data["note_approval"] = "Angsuran sangat besar terhadap gaji."
    elif rasio > 0.3:
        data["note_approval"] = "Angsuran cukup tinggi, perlu pertimbangan."
    else:
        data["note_approval"] = "Angsuran wajar terhadap gaji."

    lolos, _ = aturan_keras(data)
    if data["pekerjaan"] == "wiraswasta" and data["gaji"] == 0 and lolos:
        return {
        "status": "DI PERTIMBANGKAN",
        "risiko": "SEDANG",
        "alasan": ["Wiraswasta tanpa gaji tetap."],
        "saran": "Lengkapi data penghasilan atau sertakan bukti usaha.",
        "skor_fuzzy": 55,
        "input": data
    }
    elif not lolos:
        return {
        "status": "DITOLAK",
        "risiko": "TINGGI",
        "alasan": ["gagal lolos aturan dasar (aturan keras)."],
        "saran": "Perbaiki data atau hubungi admin.",
        "skor_fuzzy": 0,
        "input": data
    }


    skor = skor_fuzzy(data)
    if skor >= 70:
        status = "LAYAK"
    elif skor >= 55:
        status = "DI PERTIMBANGKAN"
    elif skor >= 50 and rasio < 0.1 and data["cicilan_lain"] == 0:
        status = "LAYAK"
    elif skor >= 50:
        status = "DI PERTIMBANGKAN"
    else:
        status = "TIDAK LAYAK"

    risiko = kategori_risiko(skor)
    saran = saran_perbaikan(status)
    return {
        "status": status,
        "risiko": risiko,
        "alasan": PENJELASAN_STATUS[status],
        "saran": f"{saran} {data['note_approval']}" if saran else data['note_approval'],
        "skor_fuzzy": skor,
        "input": data
    }

# ===================== FLASK API =====================
@app.route("/")
def home():
    return "API prediksi kelayakan pembiayaan berjalan."

@app.route("/prediksi", methods=["POST"])
def prediksi():
    try:
        input_data = request.form.to_dict()
        input_data["jenis_pengajuan"] = input_data.get("item", "").lower()
        hasil = evaluasi_akhir(input_data)
        return jsonify(hasil)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/run_fuzzy", methods=["GET"])
def run_fuzzy():
    data = db.reference("orders").get()
    if not data:
        return jsonify({"message": "No new data."}), 200

    processed = 0
    for doc_id, record in data.items():
        status = record.get("status", "").lower()
        if status in ["processed", "cancel", "process"]:
            continue

        record["gaji"] = konversi_uang(record.get("income", 0))
        record["cicilan_lain"] = konversi_uang(record.get("installment", 0))
        record["pengajuan_baru"] = konversi_uang(record.get("nominal", 0))
        record["pekerjaan"] = record.get("job", "")
        record["jenis_pengajuan"] = record.get("item", "").lower()
        record["tenor"] = rekomendasi_tenor(record["gaji"], record["pengajuan_baru"])

        hasil = evaluasi_akhir(record)
        hasil["agent"] = {
            "email": record.get("agentEmail"),
            "nama": record.get("agentName"),
            "telepon": record.get("agentPhone")
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
