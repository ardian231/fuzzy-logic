import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
import pytesseract
from PIL import Image
import cv2
from datetime import datetime

# Firebase Initialization
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

firestore_client = firestore.client()

# Pekerjaan alias
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

# Alasan status
PENJELASAN_STATUS = {
    "LAYAK": ["Pendapatan memadai.", "Jenis pekerjaan dan cicilan sesuai."],
    "PERLU SURVEY": ["Data perlu tinjauan lebih lanjut.", "Beberapa indikator belum optimal."],
    "TIDAK LAYAK": ["Pendapatan atau pekerjaan kurang mendukung.", "Beban cicilan terlalu tinggi."],
    "DITOLAK": ["Gagal lolos syarat dasar (aturan keras)."]
}

# ================================
# Fungsi Utilitas
# ================================
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

# ================================
# OCR & Usia dari KTP
# ================================
def hitung_usia_dari_nik(nik):
    if not nik or len(nik) < 6:
        return None
    try:
        dd = int(nik[0:2])
        mm = int(nik[2:4])
        yy = int(nik[4:6])
        if dd > 40:
            dd -= 40  # perempuan

        tahun_lahir = 1900 + yy if yy > 30 else 2000 + yy
        tanggal_lahir = datetime(tahun_lahir, mm, dd)
        hari_ini = datetime.now()

        usia = hari_ini.year - tanggal_lahir.year - (
            (hari_ini.month, hari_ini.day) < (tanggal_lahir.month, tanggal_lahir.day))
        return usia
    except:
        return None

def hitung_usia_dari_ktp(path_ktp):
    try:
        img = cv2.imread(path_ktp)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='ind')

        nik_match = re.search(r'\b\d{16}\b', text)
        if nik_match:
            nik = nik_match.group(0)
            usia = hitung_usia_dari_nik(nik)
            if usia: return usia

        tgl_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', text)
        if tgl_match:
            tgl = datetime.strptime(tgl_match.group(0), '%d-%m-%Y')
            hari_ini = datetime.now()
            usia = hari_ini.year - tgl.year - ((hari_ini.month, hari_ini.day) < (tgl.month, tgl.day))
            return usia
    except Exception as e:
        print("OCR error:", e)
    return None

def tentukan_usia(record, path_ktp=None):
    if path_ktp:
        usia = hitung_usia_dari_ktp(path_ktp)
        if usia: return usia
    if 'tgl_lahir' in record:
        try:
            tgl = datetime.strptime(record['tgl_lahir'], '%d-%m-%Y')
            hari_ini = datetime.now()
            return hari_ini.year - tgl.year - ((hari_ini.month, hari_ini.day) < (tgl.month, tgl.day))
        except:
            pass
    return 30
# ==========================
# Rekomendasi Tenor
# ==========================
def rekomendasi_tenor(gaji, plafon, bunga_tahunan=0.07):
    bunga_bulanan = bunga_tahunan / 12
    tenor_list = [12, 24, 36, 48, 60]
    for tenor in tenor_list:
        cicilan = (plafon * bunga_bulanan) / (1 - (1 + bunga_bulanan) ** -tenor)
        if cicilan <= 0.3 * gaji:
            return tenor
    return tenor_list[-1]

# ================================
# Validasi & Evaluasi
# ================================
def aturan_keras(data):
    usia_lunas = data['usia'] + data['tenor'] / 12

    if data.get('tinggal_di_kost', False):
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

def evaluasi_akhir(data, path_ktp=None):
    for k in ['gaji', 'cicilan_lain', 'pengajuan_baru']:
        data[k] = konversi_uang(data[k]) if isinstance(data[k], str) else data[k]

    data['pekerjaan'] = normalisasi_pekerjaan(data['pekerjaan'])
    data['usia'] = tentukan_usia(data, path_ktp)

    if 'tenor' not in data:
        data['tenor'] = 24  # default tenor 2 tahun jika tidak ada

    if 'jenis_pengajuan' not in data:
        data['jenis_pengajuan'] = 'motor' if 'item' in data and 'motor' in data['item'].lower() else 'amanah'

    lolos, _ = aturan_keras(data)
    if not lolos:
        return {
            "status": "DITOLAK",
            "alasan": PENJELASAN_STATUS["DITOLAK"],
            "skor_fuzzy": 0,
            "input": data
        }

    skor = skor_fuzzy(data)
    status = "LAYAK" if skor >= 70 else "PERLU SURVEY" if skor >= 50 else "TIDAK LAYAK"

    return {
        "status": status,
        "alasan": PENJELASAN_STATUS[status],
        "skor_fuzzy": skor,
        "input": data
    }

# ================================
# Firebase Integration
# ================================
def ambil_data_baru():
    ref = db.reference('orders')
    data = ref.get()
    return {k: v for k, v in data.items() if v.get('status') != 'processed'} if data else {}

def run_fuzzy(event=None, context=None):
    data = ambil_data_baru()
    if not data:
        return jsonify({"message": "No new data."}), 200

    for doc_id, record in data.items():
        hasil = evaluasi_akhir(record)
        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        db.reference(f'orders/{doc_id}').update({'status': 'processed'})

    return jsonify({"message": f"Processed {len(data)} records."}), 200
