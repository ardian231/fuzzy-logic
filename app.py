import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
import pytesseract
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
    'pns': ['pns', 'pegawai negeri'],
    'karyawan': ['karyawan', 'pegawai swasta', 'staff', 'pegawai'],
    'profesional': ['dokter', 'pengacara', 'notaris', 'arsitek'],
    'wiraswasta': ['wiraswasta', 'pengusaha', 'dagang', 'usaha sendiri', 'pedagang'],
    'freelancer': ['freelancer', 'pekerja lepas', 'remote', 'freelance'],
    'driver': ['driver', 'sopir', 'ojol', 'ngojek'],
    'buruh': ['buruh', 'kuli', 'tenaga kasar'],
    'tidak bekerja': ['tidak bekerja', 'pengangguran', 'belum kerja']
}

PENJELASAN_STATUS = {
    "LAYAK": ["Pendapatan memadai.", "Jenis pekerjaan dan cicilan sesuai."],
    "PERLU SURVEY": ["Data perlu tinjauan lebih lanjut.", "Beberapa indikator belum optimal."],
    "TIDAK LAYAK": ["Pendapatan atau pekerjaan kurang mendukung.", "Beban cicilan terlalu tinggi."],
    "DITOLAK": ["Gagal lolos syarat dasar (aturan keras)."]
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
    if pd.isnull(teks): return 0
    teks = str(teks).lower().replace('.', '').replace(',', '.').replace('rp', '').strip()
    match = re.search(r'(\d+\.?\d*)\s*(juta|jt|ribu|rb)?', teks)
    if not match: return 0

    num = float(match.group(1))
    satuan = match.group(2)
    return int(num * 1_000_000) if satuan in ['juta', 'jt'] else \
           int(num * 1_000) if satuan in ['ribu', 'rb'] else int(num)

def ekstrak_usia_dari_ktp(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(image)
        match = re.search(r'(\d{2}-\d{2}-\d{4})', text)
        if not match:
            match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if not match:
            match = re.search(r'(\d{2}.\d{2}.\d{4})', text)
        if match:
            tgl = re.sub(r'[^0-9]', '-', match.group(1))
            tgl_lahir = datetime.strptime(tgl, "%d-%m-%Y")
            usia = (datetime.now() - tgl_lahir).days // 365
            return usia
    except Exception as e:
        print("OCR Error:", e)
    return None

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
        (min(gaji_rendah, ratio_besar), 12),
        (min(gaji_rendah, ratio_sedang), 18),
        (min(gaji_rendah, ratio_kecil), 24),
        (min(gaji_sedang, ratio_besar), 18),
        (min(gaji_sedang, ratio_sedang), 24),
        (min(gaji_sedang, ratio_kecil), 36),
        (min(gaji_tinggi, ratio_besar), 24),
        (min(gaji_tinggi, ratio_sedang), 36),
        (min(gaji_tinggi, ratio_kecil), 48)
    ]

    numerator = sum(weight * output for weight, output in rules)
    denominator = sum(weight for weight, _ in rules)

    if denominator == 0:
        return 12  # fallback jika semua keanggotaan nol

    hasil = numerator / denominator
    tenor_tersedia = [12, 18, 24, 36, 48]
    tenor_terdekat = min(tenor_tersedia, key=lambda x: abs(x - hasil))
    return tenor_terdekat

def kategori_risiko(skor):
    return "RENDAH" if skor >= 75 else "SEDANG" if skor >= 60 else "TINGGI"

def saran_perbaikan(status):
    if status == "TIDAK LAYAK":
        return "Kurangi jumlah pengajuan atau pastikan cicilan lain telah lunas."
    elif status == "PERLU SURVEY":
        return "Perlu tinjauan lapangan atau dokumen tambahan untuk validasi."
    return None


# ===================== ATURAN & PENILAIAN =====================
def aturan_keras(data):
    usia_lunas = data['usia'] + data['tenor'] / 12

    if data['tinggal_di_kost']:
        return False, "Tinggal di kost tidak diperbolehkan."
    if data['pekerjaan'] == 'karyawan' and usia_lunas > 55:
        return False, "Usia > 55 untuk karyawan."
    if data['pekerjaan'] == 'wiraswasta' and usia_lunas > 60:
        return False, "Usia > 60 untuk wiraswasta."
    if data['jenis_pengajuan'] == 'mobil' and data['gaji'] < 10_000_000 and data['cicilan_lain'] > 0:
        return False, "Gaji <10jt dan ada cicilan lain."
    if data['pengajuan_baru'] > 0.30 * data['gaji']:
        return False, "Cicilan > 30% dari gaji."
    return True, "Lolos aturan keras."

def skor_fuzzy(data):
    skor = 0
    skor += 30 if data['gaji'] >= 10_000_000 else \
            25 if data['gaji'] >= 7_000_000 else \
            20 if data['gaji'] >= 5_000_000 else \
            15 if data['gaji'] >= 3_000_000 else 10

    map_pekerjaan = {
        'pns': 30, 'karyawan': 25, 'profesional': 20,
        'wiraswasta': 15, 'freelancer': 15,
        'driver': 10, 'buruh': 10, 'tidak bekerja': 5
    }
    skor += map_pekerjaan.get(data['pekerjaan'], 10)

    skor += 20 if data['cicilan_lain'] == 0 else \
            10 if data['cicilan_lain'] < 0.20 * data['gaji'] else 0

    skor += 20 if data['pengajuan_baru'] <= 0.30 * data['gaji'] else \
            10 if data['pengajuan_baru'] <= 0.50 * data['gaji'] else 0

    return skor

def evaluasi_akhir(data):
    for k in ['gaji', 'cicilan_lain', 'pengajuan_baru']:
        data[k] = konversi_uang(data[k]) if isinstance(data[k], str) else data[k]

    data['pekerjaan'] = normalisasi_pekerjaan(data['pekerjaan'])

    if 'tenor' not in data or not data['tenor']:
        data['tenor'] = rekomendasi_tenor(data['gaji'], data['pengajuan_baru'])

    lolos, _ = aturan_keras(data)
    if not lolos:
        return {
            "status": "DITOLAK",
            "alasan": PENJELASAN_STATUS["DITOLAK"],
            "skor_fuzzy": 0,
            "risiko": "TINGGI",
            "saran": "Tidak memenuhi syarat dasar kelayakan.",
            "usia_hasil_ocr": data.get("usia_ocr"),
            "input": data
        }

    skor = skor_fuzzy(data)
    status = "LAYAK" if skor >= 70 else "PERLU SURVEY" if skor >= 50 else "TIDAK LAYAK"
    risiko = kategori_risiko(skor)
    saran = saran_perbaikan(status)

    return {
        "status": status,
        "risiko": risiko,
        "alasan": PENJELASAN_STATUS[status],
        "saran": saran,
        "skor_fuzzy": skor,
        "usia_hasil_ocr": data.get("usia_ocr"),
        "input": data
    }

# ===================== FLASK API =====================

@app.route("/")
def home():
    return "API prediksi kelayakan pembiayaan berjalan."

@app.route("/prediksi", methods=["POST"])
def prediksi():
    input_data = request.form.to_dict()
    file = request.files.get("ktp")

    if file:
        usia = ekstrak_usia_dari_ktp(file.read())
        input_data['usia_ocr'] = usia
    else:
        usia = int(input_data.get("usia", 0))

    input_data['usia'] = usia or 0
    input_data['tinggal_di_kost'] = input_data.get('tinggal_di_kost', 'tidak') == 'ya'
    input_data['jenis_pengajuan'] = input_data.get('item', '').lower()

    hasil = evaluasi_akhir(input_data)
    return jsonify(hasil)

@app.route("/run_fuzzy", methods=["GET"])
def run_fuzzy():
    data = db.reference('orders').get()
    if not data:
        return jsonify({"message": "No new data."}), 200

    processed = 0
    for doc_id, record in data.items():
        status = record.get("status", "").lower()

        if status == "processed":
            # Sudah selesai, skip
            continue
        if status == "cancel":
            # Data dibatalkan, skip juga
            print(f"[INFO] Record {doc_id} dibatalkan, dilewati.")
            continue
        if status == "process":
            # Sedang diproses, skip bisa di sesuaikan
            print(f"[INFO] Record {doc_id} sedang diproses, dilewati.")
            continue

        record['tinggal_di_kost'] = False
        record['jenis_pengajuan'] = record.get('item', '').lower()
        record['usia'] = 30
        record['usia_ocr'] = None
        if 'ktp' in record and record['ktp']:
            try:
                resp = requests.get(record['ktp'])
                if resp.status_code == 200:
                    usia_ocr = ekstrak_usia_dari_ktp(resp.content)
                    if usia_ocr:
                        record['usia'] = usia_ocr
                        record['usia_ocr'] = usia_ocr
                        print(f"[INFO] OCR berhasil: usia = {usia_ocr} dari {record['ktp']}")
            except Exception as e:
                print("Error ambil KTP:", e)

        record['gaji'] = record.get('income', 0)
        record['cicilan_lain'] = record.get('installment', 0)
        record['pengajuan_baru'] = record.get('nominal', 0)
        record['pekerjaan'] = record.get('job', '')
        record['tenor'] = rekomendasi_tenor(
            konversi_uang(record['gaji']),
            konversi_uang(record['pengajuan_baru'])
        )

        hasil = evaluasi_akhir(record)

        hasil["timestamp"] = datetime.utcnow().isoformat()
        hasil["agent"] = {
        "email": record.get("agentEmail"),
        "nama": record.get("agentName"),
        "telepon": record.get("agentPhone")
        }

        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        processed += 1


    return jsonify({"message": f"Processed {processed} records."})

if __name__ == "__main__":
    app.run(debug=True)
