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

def rekomendasi_tenor(gaji, plafon=None, bunga_tahunan=0.07):
    bunga_bulanan = bunga_tahunan / 12
    tenor_list = [12, 24, 36, 48, 60]
    for tenor in tenor_list:
        if plafon:
            cicilan = (plafon * bunga_bulanan) / (1 - (1 + bunga_bulanan) ** -tenor)
        else:
            cicilan = 0.2 * gaji
        if cicilan <= 0.3 * gaji:
            return tenor
    return tenor_list[-1]

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
            "usia_hasil_ocr": data.get("usia_ocr"),
            "input": data
        }

    skor = skor_fuzzy(data)
    status = "LAYAK" if skor >= 70 else "PERLU SURVEY" if skor >= 50 else "TIDAK LAYAK"

    return {
        "status": status,
        "alasan": PENJELASAN_STATUS[status],
        "skor_fuzzy": skor,
        "usia_hasil_ocr": data.get("usia_ocr"),
        "input": data
    }

# ===================== FLASK API =====================
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
        if record.get("status") == "processed":
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
        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        processed += 1

    return jsonify({"message": f"Processed {processed} records."})

if __name__ == "__main__":
    app.run(debug=True)
