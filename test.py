import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

class AnalisisKreditFuzzy:
    def __init__(self):
        """Inisialisasi Sistem Analisis Kredit Berbasis Fuzzy (Sugeno 0-order)."""
        self.aturan = self._inisialisasi_aturan()
    
    # =========================
    # Fungsi Keanggotaan (Input)
    # =========================
    def keanggotaan_rasio_dp(self, rasio: float) -> Dict[str, float]:
        """
        Fungsi keanggotaan untuk rasio DP/Nominal.
        Keluaran: dict dengan kunci 'rendah', 'sedang', 'tinggi' (0..1).
        """
        # Rendah: di bawah 0.18 (penuh pada 0.12)
        rendah = 1.0 if rasio <= 0.12 else (0.18 - rasio) / (0.18 - 0.12) if rasio < 0.18 else 0.0
        rendah = max(0.0, min(1.0, rendah))
        
        # Sedang: 0.12–0.30 (puncak 0.21)
        if rasio <= 0.12:
            sedang = 0.0
        elif rasio <= 0.21:
            sedang = (rasio - 0.12) / (0.21 - 0.12)
        elif rasio <= 0.30:
            sedang = (0.30 - rasio) / (0.30 - 0.21)
        else:
            sedang = 0.0
        sedang = max(0.0, min(1.0, sedang))
        
        # Tinggi: di atas 0.24 (penuh pada 0.35)
        tinggi = 0.0 if rasio <= 0.24 else (rasio - 0.24) / (0.35 - 0.24) if rasio < 0.35 else 1.0
        tinggi = max(0.0, min(1.0, tinggi))
        
        return {"rendah": rendah, "sedang": sedang, "tinggi": tinggi}
    
    def keanggotaan_rasio_dti(self, rasio: float) -> Dict[str, float]:
        """
        Fungsi keanggotaan untuk DTI (Cicilan Lain / Gaji).
        DTI makin kecil → makin baik.
        """
        # Rendah (baik): 0%–15% (0% paling baik)
        rendah = 1.0 if rasio <= 0.15 else (0.20 - rasio) / (0.20 - 0.15) if rasio < 0.20 else 0.0
        rendah = max(0.0, min(1.0, rendah))
        
        # Sedang: 10%–35%
        if rasio <= 0.10:
            sedang = 0.0
        elif rasio <= 0.20:
            sedang = (rasio - 0.10) / (0.20 - 0.10)
        elif rasio <= 0.35:
            sedang = (0.35 - rasio) / (0.35 - 0.20)
        else:
            sedang = 0.0
        sedang = max(0.0, min(1.0, sedang))
        
        # Tinggi (kurang baik): mulai 30%
        tinggi = 0.0 if rasio <= 0.30 else (rasio - 0.30) / (0.50 - 0.30) if rasio < 0.50 else 1.0
        tinggi = max(0.0, min(1.0, tinggi))
        
        return {"rendah": rendah, "sedang": sedang, "tinggi": tinggi}
    
    def keanggotaan_rasio_lti(self, rasio: float) -> Dict[str, float]:
        """
        Fungsi keanggotaan untuk LTI (Nominal / Gaji) dalam kelipatan gaji.
        """
        # Rendah (aman): ≤4x gaji (mulai turun dari 3x)
        rendah = 1.0 if rasio <= 3.0 else (4.0 - rasio) / (4.0 - 3.0) if rasio < 4.0 else 0.0
        rendah = max(0.0, min(1.0, rendah))
        
        # Sedang: 3x–8x
        if rasio <= 3.0:
            sedang = 0.0
        elif rasio <= 5.0:
            sedang = (rasio - 3.0) / (5.0 - 3.0)
        elif rasio <= 8.0:
            sedang = (8.0 - rasio) / (8.0 - 5.0)
        else:
            sedang = 0.0
        sedang = max(0.0, min(1.0, sedang))
        
        # Tinggi (berisiko): ≥7x gaji
        tinggi = 0.0 if rasio <= 7.0 else (rasio - 7.0) / (10.0 - 7.0) if rasio < 10.0 else 1.0
        tinggi = max(0.0, min(1.0, tinggi))
        
        return {"rendah": rendah, "sedang": sedang, "tinggi": tinggi}
    
    def keanggotaan_nominal(self, nominal_rp: float) -> Dict[str, float]:
        """
        Fungsi keanggotaan untuk Nominal Pengajuan (Rupiah).
        Keluaran: 'kecil', 'sedang', 'besar'.
        """
        jt = nominal_rp / 1_000_000  # konversi ke juta
        
        # Kecil: ≤15 juta (penuh ≤10 juta)
        kecil = 1.0 if jt <= 10.0 else (15.0 - jt) / (15.0 - 10.0) if jt < 15.0 else 0.0
        kecil = max(0.0, min(1.0, kecil))
        
        # Sedang: 10–35 juta
        if jt <= 10.0:
            sedang = 0.0
        elif jt <= 20.0:
            sedang = (jt - 10.0) / (20.0 - 10.0)
        elif jt <= 35.0:
            sedang = (35.0 - jt) / (35.0 - 20.0)
        else:
            sedang = 0.0
        sedang = max(0.0, min(1.0, sedang))
        
        # Besar: ≥30 juta
        besar = 0.0 if jt <= 30.0 else (jt - 30.0) / (50.0 - 30.0) if jt < 50.0 else 1.0
        besar = max(0.0, min(1.0, besar))
        
        return {"kecil": kecil, "sedang": sedang, "besar": besar}
    
    # ================
    # Basis Aturan
    # ================
    def _inisialisasi_aturan(self) -> List[Dict]:
        """
        Inisialisasi aturan fuzzy (Sugeno 0-order).
        Catatan: kolom 'nominal' opsional. Jika 'apa_saja', nominal tidak mempengaruhi rule.
        """
        return [
            # Kondisi sangat baik
            {"dp": "tinggi", "dti": "rendah", "lti": "rendah", "nominal": "kecil",  "output": 0.90},
            {"dp": "tinggi", "dti": "rendah", "lti": "rendah", "nominal": "sedang", "output": 0.85},
            {"dp": "sedang", "dti": "rendah", "lti": "rendah", "nominal": "kecil",  "output": 0.75},
            
            # Baik tapi moderat pada salah satu faktor
            {"dp": "tinggi", "dti": "rendah", "lti": "sedang", "nominal": "apa_saja", "output": 0.70},
            {"dp": "sedang", "dti": "sedang", "lti": "rendah", "nominal": "apa_saja", "output": 0.60},
            {"dp": "tinggi", "dti": "sedang", "lti": "sedang", "nominal": "apa_saja", "output": 0.55},
            {"dp": "rendah", "dti": "rendah", "lti": "sedang", "nominal": "kecil",    "output": 0.50},
            
            # Risiko meningkat
            {"dp": "tinggi", "dti": "rendah", "lti": "tinggi", "nominal": "apa_saja", "output": 0.35},
            {"dp": "rendah", "dti": "sedang", "lti": "tinggi", "nominal": "apa_saja", "output": 0.25},
            {"dp": "sedang", "dti": "tinggi", "lti": "tinggi", "nominal": "apa_saja", "output": 0.20},
            {"dp": "rendah", "dti": "tinggi", "lti": "tinggi", "nominal": "apa_saja", "output": 0.15},
        ]
    
    # =========================
    # Inferensi & Defuzzifikasi
    # =========================
    def analisis_kredit(self, dp: float, nominal_pengajuan: float, cicilan_lain: float, gaji: float) -> Dict:
        """
        Analisis kredit (Sugeno 0-order):
        - Fuzzifikasi input (rasio DP, DTI, LTI, nominal)
        - Kekuatan rule = min(antecedent)
        - Keluaran akhir = rata-rata terbobot dari output singleton
        """
        # Hitung rasio
        r_dp  = dp / nominal_pengajuan if nominal_pengajuan > 0 else 0.0
        r_dti = cicilan_lain / gaji if gaji > 0 else 0.0
        r_lti = nominal_pengajuan / gaji if gaji > 0 else 0.0
        
        # Keanggotaan
        m_dp      = self.keanggotaan_rasio_dp(r_dp)
        m_dti     = self.keanggotaan_rasio_dti(r_dti)
        m_lti     = self.keanggotaan_rasio_lti(r_lti)
        m_nominal = self.keanggotaan_nominal(nominal_pengajuan)
        
        # Agregasi aturan (Sugeno 0-order)
        pembilang = 0.0
        penyebut  = 0.0
        aturan_aktif = []
        
        for i, rule in enumerate(self.aturan):
            dp_val  = m_dp[rule["dp"]]
            dti_val = m_dti[rule["dti"]]
            lti_val = m_lti[rule["lti"]]
            
            # Nominal opsional
            nk = rule.get("nominal", "apa_saja")
            if nk == "kecil":
                nominal_val = m_nominal["kecil"]
            elif nk == "sedang":
                nominal_val = m_nominal["sedang"]
            elif nk == "besar":
                nominal_val = m_nominal["besar"]
            else:  # "apa_saja"
                nominal_val = 1.0
            
            kekuatan = min(dp_val, dti_val, lti_val, nominal_val)
            if kekuatan > 0:
                pembilang += kekuatan * rule["output"]
                penyebut  += kekuatan
                aturan_aktif.append({
                    "indeks_aturan": i + 1,
                    "aturan": f"JIKA DP={rule['dp']} DAN DTI={rule['dti']} DAN LTI={rule['lti']} "
                              f"DAN Nominal={nk} MAKA Skor={rule['output']}",
                    "kekuatan": kekuatan,
                    "output": rule["output"]
                })
        
        skor_akhir = pembilang / penyebut if penyebut > 0 else 0.0
        
        # Keputusan (batas bisa disesuaikan kebijakan)
        if skor_akhir >= 0.60:
            keputusan = "APPROVE"
        elif skor_akhir >= 0.40:
            keputusan = "DIPERTIMBANGKAN"
        else:
            keputusan = "REJECT"
        
        return {
            "rasio": {"r_dp": r_dp, "r_dti": r_dti, "r_lti": r_lti},
            "keanggotaan": {"dp": m_dp, "dti": m_dti, "lti": m_lti, "nominal": m_nominal},
            "skor_fuzzy": skor_akhir,
            "keputusan": keputusan,
            "aturan_aktif": sorted(aturan_aktif, key=lambda x: x["kekuatan"], reverse=True),
            "input": {
                "dp": dp,
                "nominal_pengajuan": nominal_pengajuan,
                "cicilan_lain": cicilan_lain,
                "gaji": gaji
            }
        }
    
    # =========================
    # Utilitas: Plot & Laporan
    # =========================
    def plot_fungsi_keanggotaan(self):
        """Tampilkan grafik fungsi keanggotaan untuk semua variabel."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fungsi Keanggotaan Fuzzy', fontsize=16)
        
        # DP
        x_dp = np.linspace(0, 0.5, 1000)
        dp_r = [self.keanggotaan_rasio_dp(x)["rendah"] for x in x_dp]
        dp_s = [self.keanggotaan_rasio_dp(x)["sedang"] for x in x_dp]
        dp_t = [self.keanggotaan_rasio_dp(x)["tinggi"] for x in x_dp]
        axes[0,0].plot(x_dp, dp_r, 'r-', label='Rendah', linewidth=2)
        axes[0,0].plot(x_dp, dp_s, 'orange', label='Sedang', linewidth=2)
        axes[0,0].plot(x_dp, dp_t, 'g-', label='Tinggi', linewidth=2)
        axes[0,0].set_title('Rasio DP/Nominal')
        axes[0,0].set_xlabel('Rasio')
        axes[0,0].set_ylabel('Keanggotaan')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # DTI
        x_dti = np.linspace(0, 0.6, 1000)
        dti_r = [self.keanggotaan_rasio_dti(x)["rendah"] for x in x_dti]
        dti_s = [self.keanggotaan_rasio_dti(x)["sedang"] for x in x_dti]
        dti_t = [self.keanggotaan_rasio_dti(x)["tinggi"] for x in x_dti]
        axes[0,1].plot(x_dti, dti_r, 'r-', label='Rendah', linewidth=2)
        axes[0,1].plot(x_dti, dti_s, 'orange', label='Sedang', linewidth=2)
        axes[0,1].plot(x_dti, dti_t, 'g-', label='Tinggi', linewidth=2)
        axes[0,1].set_title('Rasio DTI')
        axes[0,1].set_xlabel('Rasio')
        axes[0,1].set_ylabel('Keanggotaan')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # LTI
        x_lti = np.linspace(0, 12, 1000)
        lti_r = [self.keanggotaan_rasio_lti(x)["rendah"] for x in x_lti]
        lti_s = [self.keanggotaan_rasio_lti(x)["sedang"] for x in x_lti]
        lti_t = [self.keanggotaan_rasio_lti(x)["tinggi"] for x in x_lti]
        axes[1,0].plot(x_lti, lti_r, 'r-', label='Rendah', linewidth=2)
        axes[1,0].plot(x_lti, lti_s, 'orange', label='Sedang', linewidth=2)
        axes[1,0].plot(x_lti, lti_t, 'g-', label='Tinggi', linewidth=2)
        axes[1,0].set_title('Rasio LTI (kelipatan gaji)')
        axes[1,0].set_xlabel('Kelipatan gaji')
        axes[1,0].set_ylabel('Keanggotaan')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Nominal
        x_nom = np.linspace(0, 80, 1000)  # 0..80 juta
        nom_k = [self.keanggotaan_nominal(x * 1_000_000)["kecil"] for x in x_nom]
        nom_s = [self.keanggotaan_nominal(x * 1_000_000)["sedang"] for x in x_nom]
        nom_b = [self.keanggotaan_nominal(x * 1_000_000)["besar"] for x in x_nom]
        axes[1,1].plot(x_nom, nom_k, 'r-', label='Kecil', linewidth=2)
        axes[1,1].plot(x_nom, nom_s, 'orange', label='Sedang', linewidth=2)
        axes[1,1].plot(x_nom, nom_b, 'g-', label='Besar', linewidth=2)
        axes[1,1].set_title('Nominal Pengajuan (juta Rp)')
        axes[1,1].set_xlabel('Juta Rupiah')
        axes[1,1].set_ylabel('Keanggotaan')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def cetak_laporan_analisis(self, hasil: Dict):
        """Cetak laporan analisis kredit secara rinci (Bahasa Indonesia)."""
        print("="*60)
        print("LAPORAN ANALISIS KREDIT FUZZY")
        print("="*60)
        
        inp = hasil["input"]
        rasio = hasil["rasio"]
        
        # Data input
        print(f"\n📊 DATA INPUT:")
        print(f"  • Down Payment (DP)       : Rp {inp['dp']:,.0f}")
        print(f"  • Nominal Pengajuan       : Rp {inp['nominal_pengajuan']:,.0f}")
        print(f"  • Cicilan Lain            : Rp {inp['cicilan_lain']:,.0f}")
        if inp['cicilan_lain'] == 0:
            print(f"    → Tidak ada cicilan lain (BAIK)")
        print(f"  • Gaji                    : Rp {inp['gaji']:,.0f}")
        
        # Rasio
        print(f"\n📈 RASIO TERHITUNG:")
        print(f"  • Rasio DP/Nominal (r_dp) : {rasio['r_dp']:.1%}")
        print(f"  • Rasio DTI (r_dti)       : {rasio['r_dti']:.1%}")
        print(f"  • Rasio LTI (r_lti)       : {rasio['r_lti']:.1f}x gaji")
        
        # Keanggotaan
        print(f"\n🎯 NILAI KEANGGOTAAN:")
        m = hasil["keanggotaan"]
        print(f"  DP    - Rendah: {m['dp']['rendah']:.3f}, Sedang: {m['dp']['sedang']:.3f}, Tinggi: {m['dp']['tinggi']:.3f}")
        print(f"  DTI   - Rendah: {m['dti']['rendah']:.3f}, Sedang: {m['dti']['sedang']:.3f}, Tinggi: {m['dti']['tinggi']:.3f}")
        print(f"  LTI   - Rendah: {m['lti']['rendah']:.3f}, Sedang: {m['lti']['sedang']:.3f}, Tinggi: {m['lti']['tinggi']:.3f}")
        print(f"  Nom   - Kecil: {m['nominal']['kecil']:.3f}, Sedang: {m['nominal']['sedang']:.3f}, Besar: {m['nominal']['besar']:.3f}")
        
        # Aturan aktif
        print(f"\n🔥 ATURAN AKTIF (Top 5):")
        for i, r in enumerate(hasil["aturan_aktif"][:5]):
            print(f"  {i+1}. Aturan {r['indeks_aturan']} (Kekuatan: {r['kekuatan']:.3f})")
            print(f"     {r['aturan']}")
        
        # Hasil akhir
        print(f"\n🎉 HASIL AKHIR:")
        print(f"  • Skor Fuzzy : {hasil['skor_fuzzy']:.1%}")
        
        keputusan = hasil['keputusan']
        if keputusan == "APPROVE":
            print(f"  • Keputusan  : ✅ {keputusan}")
            print(f"    → Kredit dapat disetujui.")
        elif keputusan == "DIPERTIMBANGKAN":
            print(f"  • Keputusan  : ⚠️  {keputusan}")
            print(f"    → Perlu review lebih lanjut (uji lapangan/televerifikasi).")
        else:
            print(f"  • Keputusan  : ❌ {keputusan}")
            print(f"    → Kredit sebaiknya ditolak/ditunda.")
        
        print("="*60)


# =========================
# Contoh Pemakaian
# =========================
if __name__ == "__main__":
    sistem = AnalisisKreditFuzzy()
    
    # Contoh 1: Kandidat Baik (DP tinggi, tanpa cicilan lain)
    print("CONTOH 1: Kandidat Baik")
    hasil1 = sistem.analisis_kredit(
        dp=25_000_000,              # 25 juta
        nominal_pengajuan=100_000_000,  # 100 juta
        cicilan_lain=0,             # tidak ada cicilan lain
        gaji=8_000_000              # gaji 8 juta
    )
    sistem.cetak_laporan_analisis(hasil1)
    
    print("\n" + "="*80 + "\n")
    
    # Contoh 2: Kandidat Berisiko (DP rendah, DTI tinggi, LTI tinggi)
    print("CONTOH 2: Kandidat Berisiko")
    hasil2 = sistem.analisis_kredit(
        dp=5_000_000,               # 5 juta (rendah)
        nominal_pengajuan=150_000_000,  # 150 juta
        cicilan_lain=3_000_000,     # 3 juta cicilan lain
        gaji=6_000_000              # gaji 6 juta
    )
    sistem.cetak_laporan_analisis(hasil2)
    
    # Tampilkan grafik fungsi keanggotaan
    print("\nMenampilkan grafik fungsi keanggotaan...")
    sistem.plot_fungsi_keanggotaan()
