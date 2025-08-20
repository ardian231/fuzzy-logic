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

# -------------------------
# FuzzyLogicCredit (T=63)
# -------------------------
class FuzzyLogicCredit:
    def __init__(self, threshold=63):
        self.threshold = int(threshold)
        self.setup_membership_functions()
        self.setup_fuzzy_rules()

    def setup_membership_functions(self):
        # Gaji
        self.gaji_membership = {
            'rendah': lambda x: 1 if x <= 2000000 else (3500000 - x) / (3500000 - 2000000) if x <= 3500000 else 0,
            'sedang': lambda x: 0 if x <= 2000000 else (x - 2000000) / (3500000 - 2000000) if x <= 3500000 else 1 if x <= 5000000 else (7000000 - x) / (7000000 - 5000000) if x <= 7000000 else 0,
            'tinggi': lambda x: 0 if x <= 4500000 else (x - 4500000) / (7000000 - 4500000) if x <= 7000000 else 1
        }
        # Pinjaman
        self.pinjaman_membership = {
            'kecil':  lambda x: 1 if x <= 10000000 else (20000000 - x) / (20000000 - 10000000) if x <= 20000000 else 0,
            'sedang': lambda x: 0 if x <= 10000000 else (x - 10000000) / (20000000 - 10000000) if x <= 20000000 else 1 if x <= 30000000 else (40000000 - x) / (40000000 - 30000000) if x <= 40000000 else 0,
            'besar':  lambda x: 0 if x <= 30000000 else (x - 30000000) / (40000000 - 30000000) if x <= 40000000 else 1
        }
        # Cicilan lain
        self.cicilan_membership = {
            'tidak_ada': lambda x: 1 if x <= 0 else (500000 - x) / 500000 if x <= 500000 else 0,
            'sedikit':  lambda x: 0 if x <= 0 else x / 500000 if x <= 500000 else 1 if x <= 1500000 else (2500000 - x) / (2500000 - 1500000) if x <= 2500000 else 0,
            'banyak':   lambda x: 0 if x <= 1500000 else (x - 1500000) / (2500000 - 1500000) if x <= 2500000 else 1
        }
        # Output bands
        self.skor_membership = {
            'rendah': {'min': 40, 'max': 55},
            'sedang': {'min': 50, 'max': 75},
            'tinggi': {'min': 70, 'max': 85}
        }

    def setup_fuzzy_rules(self):
        self.fuzzy_rules = [
            # Gaji Tinggi
            {'gaji': 'tinggi', 'pinjaman': 'kecil', 'cicilan': 'tidak_ada', 'output': 'tinggi'},
            {'gaji': 'tinggi', 'pinjaman': 'kecil', 'cicilan': 'sedikit', 'output': 'tinggi'},
            {'gaji': 'tinggi', 'pinjaman': 'kecil', 'cicilan': 'banyak', 'output': 'sedang'},
            {'gaji': 'tinggi', 'pinjaman': 'sedang', 'cicilan': 'tidak_ada', 'output': 'tinggi'},
            {'gaji': 'tinggi', 'pinjaman': 'sedang', 'cicilan': 'sedikit', 'output': 'sedang'},
            {'gaji': 'tinggi', 'pinjaman': 'sedang', 'cicilan': 'banyak', 'output': 'sedang'},
            {'gaji': 'tinggi', 'pinjaman': 'besar', 'cicilan': 'tidak_ada', 'output': 'sedang'},
            {'gaji': 'tinggi', 'pinjaman': 'besar', 'cicilan': 'sedikit', 'output': 'sedang'},
            {'gaji': 'tinggi', 'pinjaman': 'besar', 'cicilan': 'banyak', 'output': 'rendah'},
            # Gaji Sedang
            {'gaji': 'sedang', 'pinjaman': 'kecil', 'cicilan': 'tidak_ada', 'output': 'tinggi'},
            {'gaji': 'sedang', 'pinjaman': 'kecil', 'cicilan': 'sedikit', 'output': 'sedang'},
            {'gaji': 'sedang', 'pinjaman': 'kecil', 'cicilan': 'banyak', 'output': 'rendah'},
            {'gaji': 'sedang', 'pinjaman': 'sedang', 'cicilan': 'tidak_ada', 'output': 'sedang'},
            {'gaji': 'sedang', 'pinjaman': 'sedang', 'cicilan': 'sedikit', 'output': 'sedang'},
            {'gaji': 'sedang', 'pinjaman': 'sedang', 'cicilan': 'banyak', 'output': 'rendah'},
            {'gaji': 'sedang', 'pinjaman': 'besar', 'cicilan': 'tidak_ada', 'output': 'sedang'},
            {'gaji': 'sedang', 'pinjaman': 'besar', 'cicilan': 'sedikit', 'output': 'rendah'},
            {'gaji': 'sedang', 'pinjaman': 'besar', 'cicilan': 'banyak', 'output': 'rendah'},
            # Gaji Rendah
            {'gaji': 'rendah', 'pinjaman': 'kecil', 'cicilan': 'tidak_ada', 'output': 'sedang'},
            {'gaji': 'rendah', 'pinjaman': 'kecil', 'cicilan': 'sedikit', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'kecil', 'cicilan': 'banyak', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'sedang', 'cicilan': 'tidak_ada', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'sedang', 'cicilan': 'sedikit', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'sedang', 'cicilan': 'banyak', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'besar', 'cicilan': 'tidak_ada', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'besar', 'cicilan': 'sedikit', 'output': 'rendah'},
            {'gaji': 'rendah', 'pinjaman': 'besar', 'cicilan': 'banyak', 'output': 'rendah'}
        ]

    def calculate_membership_values(self, gaji, pinjaman, cicilan_lain):
        g = {
            'rendah': max(0, min(1, self.gaji_membership['rendah'](gaji))),
            'sedang': max(0, min(1, self.gaji_membership['sedang'](gaji))),
            'tinggi': max(0, min(1, self.gaji_membership['tinggi'](gaji)))
        }
        p = {
            'kecil': max(0, min(1, self.pinjaman_membership['kecil'](pinjaman))),
            'sedang': max(0, min(1, self.pinjaman_membership['sedang'](pinjaman))),
            'besar': max(0, min(1, self.pinjaman_membership['besar'](pinjaman)))
        }
        c = {
            'tidak_ada': max(0, min(1, self.cicilan_membership['tidak_ada'](cicilan_lain))),
            'sedikit':   max(0, min(1, self.cicilan_membership['sedikit'](cicilan_lain))),
            'banyak':    max(0, min(1, self.cicilan_membership['banyak'](cicilan_lain)))
        }
        return g, p, c

    def evaluate_rules(self, g, p, c):
        out = []
        for r in self.fuzzy_rules:
            strength = min(g[r['gaji']], p[r['pinjaman']], c[r['cicilan']])
            if strength > 0:
                out.append({**r, 'strength': strength})
        return out

    def defuzzify(self, rules):
        num, den = 0.0, 0.0
        for r in rules:
            band = self.skor_membership[r['output']]
            centroid = (band['min'] + band['max']) / 2
            num += r['strength'] * centroid
            den += r['strength']
        return num / den if den > 0 else 50.0

    def _status_from_threshold(self, s):
        if s >= self.threshold: return 'LAYAK'
        if s >= 55:             return 'DI PERTIMBANGKAN'
        return 'TIDAK LAYAK'

    def _risk_level(self, s):
        if s >= 75: return 'RENDAH'
        if s >= 55: return 'SEDANG'
        return 'TINGGI'

    def calculate_fuzzy_score(self, gaji, pinjaman, cicilan_lain):
        g, p, c = self.calculate_membership_values(gaji, pinjaman, cicilan_lain)
        rules = self.evaluate_rules(g, p, c)
        s = round(self.defuzzify(rules))
        status = self._status_from_threshold(s)
        risiko = self._risk_level(s)
        saran = ("Profil aman. Angsuran wajar terhadap gaji." if status == "LAYAK"
                 else "Perlu verifikasi lapangan/dokumen. Pertimbangkan nominal." if status == "DI PERTIMBANGKAN"
                 else "Turunkan nominal pengajuan atau selesaikan cicilan lain.")
        pred = 1 if s >= self.threshold else 0
        return {
            "final_skor": s, "pred": pred, "status": status,
            "risiko_level": risiko, "saran": saran,
            "gaji_values": g, "pinjaman_values": p, "cicilan_values": c, "rule_strengths": rules
        }

PENJELASAN_STATUS = {
    "LAYAK": ["Pendapatan memadai.", "Beban cicilan terkendali"],
    "DI PERTIMBANGKAN": ["Data perlu tinjauan lebih lanjut.", "Beberapa indikator belum optimal."],
    "TIDAK LAYAK": ["Pendapatan kurang mendukung.", "Beban cicilan terlalu tinggi."],
}



ENGINE = FuzzyLogicCredit(threshold=63)

def _to_int(x):
    try:
        return int(x)
    except:
        return 0

def evaluasi_akhir(data: dict) -> dict:
    gaji     = _to_int(data.get("gaji", 0))
    pinjaman = _to_int(data.get("pengajuan_baru", 0))
    cicilan  = _to_int(data.get("cicilan_lain", 0))
    jenis    = str(data.get("jenis_pengajuan", "")).lower()

    # ===== HARD RULE 1: Mobil + Gaji < 10jt + Cicilan banyak =====
    if "mobil" in jenis and gaji < 10000000 and cicilan >= 2000000:
        status = "TIDAK LAYAK"
        risiko = "TINGGI"
        skor   = 40
        saran = "Penghasilan tidak memadai untuk pengajuan mobil dengan beban cicilan tinggi."
        alasan_list = ["Pendapatan kurang mendukung.", "Beban cicilan terlalu tinggi."]
        data["rasio_angsuran"] = round(_to_int(data.get("angsuran", 0)) / gaji, 2) if gaji > 0 else 0

        return {
            "agent": {
                "email": data.get("agentEmail"),
                "nama": data.get("agentName"),
                "telepon": data.get("agentPhone")
            },
            "alasan": alasan_list,
            "input": data,
            "risiko": risiko,
            "saran": saran,
            "skor_fuzzy": skor,
            "status": status
        }

    # ===== LANJUTKAN KE FUZZY LOGIC =====
    res = ENGINE.calculate_fuzzy_score(gaji=gaji, pinjaman=pinjaman, cicilan_lain=cicilan)
    status = res["status"]
    risiko = res["risiko_level"]
    skor   = int(res["final_skor"])

    note_approval = (data.get("note_approval") or "").strip()
    saran = note_approval if note_approval else res["saran"]

    alasan_list = PENJELASAN_STATUS.get(status, [])

    try:
        angsuran_val = float(data.get("angsuran", 0) or 0)
        data["rasio_angsuran"] = round(angsuran_val / gaji, 2) if gaji > 0 else 0
    except Exception:
        pass

    return {
        "agent": {
            "email": data.get("agentEmail"),
            "nama": data.get("agentName"),
            "telepon": data.get("agentPhone")
        },
        "alasan": alasan_list,
        "input": data,
        "risiko": risiko,
        "saran": saran,
        "skor_fuzzy": skor,
        "status": status
    }


# -----------------------
# ROUTE: proses massal
# -----------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong", "status": "ok"})

@app.route("/run_fuzzy", methods=["GET"])
def run_fuzzy():
    data = db.reference("orders").get()
    if not data:
        return jsonify({"message": "No new data."}), 200

    processed = 0
    for doc_id, record in data.items():
        status_now = str(record.get("status", "")).lower()
        if status_now in ["processed", "cancel", "process"]:
            continue

        # CEK APAKAH SUDAH ADA DI FIRESTORE
        doc_ref = firestore_client.collection("hasil_prediksi").document(doc_id)
        if doc_ref.get().exists:
            continue  # skip kalau sudah pernah diproses

        # Normalisasi
        record["gaji"] = _to_int(record.get("income", 0))
        record["cicilan_lain"] = _to_int(record.get("installment", 0))
        record["pengajuan_baru"] = _to_int(record.get("nominal", 0))
        record["jenis_pengajuan"] = str(record.get("item", "")).lower()

        hasil = evaluasi_akhir(record)
        hasil["agent"] = {
            "email": record.get("agentEmail"),
            "nama": record.get("agentName"),
            "telepon": record.get("agentPhone")
        }

        doc_ref.set(hasil)  # simpan hasil
        processed += 1

    return jsonify({"message": f"Processed {processed} records."}), 200

if __name__ == "__main__":
    app.run(debug=True)
