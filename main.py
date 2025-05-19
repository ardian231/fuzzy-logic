# Evaluasi Kelayakan Pembiayaan - Adira Finance dengan Cloud Function + Fuzzy Logic + Firestore

import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify

# Inisialisasi Firebase hanya sekali (untuk Cloud Functions)
FIREBASE_CREDENTIALS = "firebase-key.json"
DATABASE_URL = "https://reseller-form-a616f-default-rtdb.asia-southeast1.firebasedatabase.app/"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })

firestore_client = firestore.client()

alias_pekerjaan = {
    'pns': ['pns', 'pegawai negeri'],
    'karyawan': ['karyawan', 'pegawai swasta', 'staff', 'pegawai'],
    'profesional': ['dokter', 'pengacara', 'notaris', 'arsitek'],
    'wiraswasta': ['wiraswasta', 'pengusaha', 'dagang', 'usaha sendiri', 'pedagang'],
    'freelancer': ['freelancer', 'pekerja lepas', 'remote', 'freelance'],
    'driver': ['driver', 'sopir', 'ojol', 'ngojek'],
    'buruh': ['buruh', 'kuli', 'tenaga kasar'],
    'tidak bekerja': ['tidak bekerja', 'pengangguran', 'belum kerja']
}

fixed_explanations = {
    "LAYAK": ["Pendapatan memadai.", "Jenis pekerjaan dan cicilan sesuai."],
    "PERLU SURVEY": ["Data perlu tinjauan lebih lanjut.", "Beberapa indikator belum optimal."],
    "TIDAK LAYAK": ["Pendapatan atau pekerjaan kurang mendukung.", "Beban cicilan terlalu tinggi."],
    "DITOLAK": ["Gagal lolos syarat dasar (aturan keras)."]
}

def normalisasi_pekerjaan(teks):
    teks = teks.lower().strip()
    semua_alias = [s for lst in alias_pekerjaan.values() for s in lst]
    tfidf = TfidfVectorizer().fit(semua_alias + [teks])
    vec_teks = tfidf.transform([teks])
    skor_tertinggi = 0
    label_terpilih = teks
    for label, sinonim_list in alias_pekerjaan.items():
        skor = cosine_similarity(vec_teks, tfidf.transform(sinonim_list)).max()
        if skor > skor_tertinggi:
            skor_tertinggi = skor
            label_terpilih = label
    return label_terpilih

def extract_amount(text):
    if pd.isnull(text):
        return 0
    text = str(text).lower().replace('.', '').replace(',', '.').replace('rp', '').strip()
    match = re.search(r'(\d+\.?\d*)\s*(juta|jt|ribu|rb)?', text)
    if match:
        num = float(match.group(1))
        satuan = match.group(2)
        if satuan in ['juta', 'jt']:
            return int(num * 1_000_000)
        elif satuan in ['ribu', 'rb']:
            return int(num * 1_000)
        else:
            return int(num)
    return 0

def cek_aturan_keras(data):
    usia_lunas = data['usia'] + data['tenor'] / 12
    if data['tinggal_di_kost']:
        return False, "Tempat tinggal di kost tidak diperbolehkan."
    if data['pekerjaan'] == 'karyawan' and usia_lunas > 55:
        return False, "Usia melebihi batas maksimal (55 tahun) untuk karyawan."
    elif data['pekerjaan'] == 'wiraswasta' and usia_lunas > 60:
        return False, "Usia melebihi batas maksimal (60 tahun) untuk wiraswasta."
    if data['jenis_pengajuan'] == 'mobil':
        if data['gaji'] < 10_000_000 and data['cicilan_lain'] > 0:
            return False, "Gaji <10jt dan ada cicilan lain. Ditolak."
    if data['pengajuan_baru'] > 0.30 * data['gaji']:
        return False, "Cicilan melebihi 30% dari gaji. Tidak disarankan."
    return True, "Lolos validasi aturan keras."

def hitung_fuzzy(data):
    skor = 0
    if data['gaji'] >= 10_000_000:
        skor += 30
    elif data['gaji'] >= 7_000_000:
        skor += 25
    elif data['gaji'] >= 5_000_000:
        skor += 20
    elif data['gaji'] >= 3_000_000:
        skor += 15
    else:
        skor += 10
    pekerjaan_score_map = {
        'pns': 30,
        'karyawan': 25,
        'profesional': 20,
        'wiraswasta': 15,
        'freelancer': 15,
        'driver': 10,
        'buruh': 10,
        'tidak bekerja': 5
    }
    pekerjaan = normalisasi_pekerjaan(data['pekerjaan'])
    skor += pekerjaan_score_map.get(pekerjaan, 10)
    if data['cicilan_lain'] == 0:
        skor += 20
    elif data['cicilan_lain'] < 0.20 * data['gaji']:
        skor += 10
    else:
        skor += 0
    if data['pengajuan_baru'] <= 0.30 * data['gaji']:
        skor += 20
    elif data['pengajuan_baru'] <= 0.50 * data['gaji']:
        skor += 10
    else:
        skor += 0
    return skor

def evaluasi_akhir(data):
    data['gaji'] = extract_amount(data['gaji']) if isinstance(data['gaji'], str) else data['gaji']
    data['cicilan_lain'] = extract_amount(data['cicilan_lain']) if isinstance(data['cicilan_lain'], str) else data['cicilan_lain']
    data['pengajuan_baru'] = extract_amount(data['pengajuan_baru']) if isinstance(data['pengajuan_baru'], str) else data['pengajuan_baru']
    data['pekerjaan'] = normalisasi_pekerjaan(data['pekerjaan'])
    status_rules, _ = cek_aturan_keras(data)
    if not status_rules:
        return {
            "status": "DITOLAK",
            "alasan": fixed_explanations.get("DITOLAK"),
            "skor_fuzzy": 0,
            "input": data
        }
    skor = hitung_fuzzy(data)
    if skor >= 70:
        status = "LAYAK"
    elif skor >= 50:
        status = "PERLU SURVEY"
    else:
        status = "TIDAK LAYAK"
    return {
        "status": status,
        "alasan": fixed_explanations.get(status, []),
        "skor_fuzzy": skor,
        "input": data
    }

def get_realtime_data():
    ref = db.reference('orders')
    data = ref.get()
    if not data:
        return {}
    return {k: v for k, v in data.items() if v.get('status') != 'processed'}

def run_fuzzy(event=None, context=None):
    data = get_realtime_data()
    if not data:
        return jsonify({"message": "No new data."}), 200
    for doc_id, input_data in data.items():
        hasil = evaluasi_akhir(input_data)
        firestore_client.collection("hasil_prediksi").document(doc_id).set(hasil)
        db.reference(f'orders/{doc_id}').update({'status': 'processed'})
    return jsonify({"message": f"Processed {len(data)} records."}), 200
