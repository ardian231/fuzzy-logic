from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json

app = Flask(__name__)

class FuzzyLogicCredit:
    def __init__(self):
        self.setup_membership_functions()
        self.setup_fuzzy_rules()
        
    def setup_membership_functions(self):
        """Setup membership functions for all input variables"""
        
        # Gaji membership functions
        self.gaji_membership = {
            'rendah': lambda x: 1 if x <= 2500000 else (4000000 - x) / (4000000 - 2500000) if x <= 4000000 else 0,
            'sedang': lambda x: 0 if x <= 2500000 else (x - 2500000) / (4000000 - 2500000) if x <= 4000000 else 1 if x <= 6000000 else (8000000 - x) / (8000000 - 6000000) if x <= 8000000 else 0,
            'tinggi': lambda x: 0 if x <= 6000000 else (x - 6000000) / (8000000 - 6000000) if x <= 8000000 else 1
        }
        
        # Pinjaman membership functions
        self.pinjaman_membership = {
            'kecil': lambda x: 1 if x <= 10000000 else (20000000 - x) / (20000000 - 10000000) if x <= 20000000 else 0,
            'sedang': lambda x: 0 if x <= 10000000 else (x - 10000000) / (20000000 - 10000000) if x <= 20000000 else 1 if x <= 30000000 else (40000000 - x) / (40000000 - 30000000) if x <= 40000000 else 0,
            'besar': lambda x: 0 if x <= 30000000 else (x - 30000000) / (40000000 - 30000000) if x <= 40000000 else 1
        }
        
        # Cicilan lain membership functions
        self.cicilan_membership = {
            'tidak_ada': lambda x: 1 if x <= 0 else (500000 - x) / 500000 if x <= 500000 else 0,
            'sedikit': lambda x: 0 if x <= 0 else x / 500000 if x <= 500000 else 1 if x <= 1500000 else (2500000 - x) / (2500000 - 1500000) if x <= 2500000 else 0,
            'banyak': lambda x: 0 if x <= 1500000 else (x - 1500000) / (2500000 - 1500000) if x <= 2500000 else 1
        }
        
        # Output membership ranges
        self.skor_membership = {
            'rendah': {'min': 40, 'max': 55},
            'sedang': {'min': 50, 'max': 75},
            'tinggi': {'min': 70, 'max': 85}
        }
    
    def setup_fuzzy_rules(self):
        """Setup fuzzy rules"""
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
        """Calculate membership values for all inputs"""
        gaji_values = {
            'rendah': max(0, min(1, self.gaji_membership['rendah'](gaji))),
            'sedang': max(0, min(1, self.gaji_membership['sedang'](gaji))),
            'tinggi': max(0, min(1, self.gaji_membership['tinggi'](gaji)))
        }
        
        pinjaman_values = {
            'kecil': max(0, min(1, self.pinjaman_membership['kecil'](pinjaman))),
            'sedang': max(0, min(1, self.pinjaman_membership['sedang'](pinjaman))),
            'besar': max(0, min(1, self.pinjaman_membership['besar'](pinjaman)))
        }
        
        cicilan_values = {
            'tidak_ada': max(0, min(1, self.cicilan_membership['tidak_ada'](cicilan_lain))),
            'sedikit': max(0, min(1, self.cicilan_membership['sedikit'](cicilan_lain))),
            'banyak': max(0, min(1, self.cicilan_membership['banyak'](cicilan_lain)))
        }
        
        return gaji_values, pinjaman_values, cicilan_values
    
    def evaluate_rules(self, gaji_values, pinjaman_values, cicilan_values):
        """Evaluate fuzzy rules"""
        rule_strengths = []
        
        for rule in self.fuzzy_rules:
            strength = min(
                gaji_values[rule['gaji']],
                pinjaman_values[rule['pinjaman']],
                cicilan_values[rule['cicilan']]
            )
            if strength > 0:
                rule_strengths.append({**rule, 'strength': strength})
        
        return rule_strengths
    
    def defuzzify(self, rule_strengths):
        """Defuzzification using weighted average method"""
        numerator = 0
        denominator = 0
        
        for rule in rule_strengths:
            output_range = self.skor_membership[rule['output']]
            centroid = (output_range['min'] + output_range['max']) / 2
            numerator += rule['strength'] * centroid
            denominator += rule['strength']
        
        return numerator / denominator if denominator > 0 else 50
    
    def get_recommendation(self, final_skor):
        """Get risk level, status, and recommendation based on final score"""
        if final_skor >= 75:
            return {
                'risiko_level': 'RENDAH',
                'status': 'LAYAK',
                'saran': 'Angsuran wajar terhadap gaji.'
            }
        elif final_skor >= 55:
            return {
                'risiko_level': 'SEDANG',
                'status': 'DI PERTIMBANGKAN',
                'saran': 'Perlu tinjauan lapangan atau dokumen tambahan untuk validasi. Angsuran wajar terhadap gaji.'
            }
        else:
            return {
                'risiko_level': 'TINGGI',
                'status': 'TIDAK LAYAK',
                'saran': 'Kurangi jumlah pengajuan atau pastikan cicilan lain telah lunas. Angsuran wajar terhadap gaji.'
            }
    
    def calculate_fuzzy_score(self, gaji, pinjaman, cicilan_lain):
        """Main method to calculate fuzzy score"""
        # Step 1: Fuzzification
        gaji_values, pinjaman_values, cicilan_values = self.calculate_membership_values(
            gaji, pinjaman, cicilan_lain
        )
        
        # Step 2: Rule evaluation
        rule_strengths = self.evaluate_rules(gaji_values, pinjaman_values, cicilan_values)
        
        # Step 3: Defuzzification
        final_skor = self.defuzzify(rule_strengths)
        
        # Step 4: Get recommendation
        recommendation = self.get_recommendation(final_skor)
        
        return {
            'gaji_values': gaji_values,
            'pinjaman_values': pinjaman_values,
            'cicilan_values': cicilan_values,
            'rule_strengths': rule_strengths,
            'final_skor': round(final_skor),
            **recommendation
        }
    
    def plot_membership_functions(self):
        """Generate membership function plots"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Gaji membership plot
        gaji_range = np.linspace(1000000, 10000000, 1000)
        gaji_rendah = [self.gaji_membership['rendah'](x) for x in gaji_range]
        gaji_sedang = [self.gaji_membership['sedang'](x) for x in gaji_range]
        gaji_tinggi = [self.gaji_membership['tinggi'](x) for x in gaji_range]
        
        axes[0].plot(gaji_range/1000000, gaji_rendah, 'r-', linewidth=2, label='Rendah')
        axes[0].plot(gaji_range/1000000, gaji_sedang, 'y-', linewidth=2, label='Sedang')
        axes[0].plot(gaji_range/1000000, gaji_tinggi, 'g-', linewidth=2, label='Tinggi')
        axes[0].set_title('Fungsi Keanggotaan Gaji', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Gaji (Juta Rupiah)')
        axes[0].set_ylabel('Derajat Keanggotaan')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pinjaman membership plot
        pinjaman_range = np.linspace(5000000, 50000000, 1000)
        pinjaman_kecil = [self.pinjaman_membership['kecil'](x) for x in pinjaman_range]
        pinjaman_sedang = [self.pinjaman_membership['sedang'](x) for x in pinjaman_range]
        pinjaman_besar = [self.pinjaman_membership['besar'](x) for x in pinjaman_range]
        
        axes[1].plot(pinjaman_range/1000000, pinjaman_kecil, 'b-', linewidth=2, label='Kecil')
        axes[1].plot(pinjaman_range/1000000, pinjaman_sedang, 'y-', linewidth=2, label='Sedang')
        axes[1].plot(pinjaman_range/1000000, pinjaman_besar, 'r-', linewidth=2, label='Besar')
        axes[1].set_title('Fungsi Keanggotaan Pinjaman', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Pinjaman (Juta Rupiah)')
        axes[1].set_ylabel('Derajat Keanggotaan')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Cicilan lain membership plot
        cicilan_range = np.linspace(0, 3000000, 1000)
        cicilan_tidak_ada = [self.cicilan_membership['tidak_ada'](x) for x in cicilan_range]
        cicilan_sedikit = [self.cicilan_membership['sedikit'](x) for x in cicilan_range]
        cicilan_banyak = [self.cicilan_membership['banyak'](x) for x in cicilan_range]
        
        axes[2].plot(cicilan_range/1000000, cicilan_tidak_ada, 'g-', linewidth=2, label='Tidak Ada')
        axes[2].plot(cicilan_range/1000000, cicilan_sedikit, 'y-', linewidth=2, label='Sedikit')
        axes[2].plot(cicilan_range/1000000, cicilan_banyak, 'r-', linewidth=2, label='Banyak')
        axes[2].set_title('Fungsi Keanggotaan Cicilan Lain', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Cicilan Lain (Juta Rupiah)')
        axes[2].set_ylabel('Derajat Keanggotaan')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()

# Initialize fuzzy logic system
fuzzy_system = FuzzyLogicCredit()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    """API endpoint to calculate fuzzy score"""
    try:
        data = request.json
        gaji = float(data.get('gaji', 0))
        pinjaman = float(data.get('pinjaman', 0))
        cicilan_lain = float(data.get('cicilan_lain', 0))
        
        result = fuzzy_system.calculate_fuzzy_score(gaji, pinjaman, cicilan_lain)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/plots')
def get_plots():
    """API endpoint to get membership function plots"""
    try:
        plot_base64 = fuzzy_system.plot_membership_functions()
        return jsonify({
            'success': True,
            'plot': plot_base64
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_analysis')
def batch_analysis():
    """Batch analysis page"""
    return render_template('batch_analysis.html')

@app.route('/api/batch_calculate', methods=['POST'])
def batch_calculate():
    """API endpoint for batch calculation"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check required columns
        required_columns = ['gaji', 'pengajuan_baru', 'cicilan_lain']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False, 
                'error': f'Missing columns: {missing_columns}'
            }), 400
        
        # Calculate fuzzy scores for each row
        results = []
        for _, row in df.iterrows():
            result = fuzzy_system.calculate_fuzzy_score(
                row['gaji'], 
                row['pengajuan_baru'], 
                row['cicilan_lain']
            )
            results.append({
                'gaji': row['gaji'],
                'pinjaman': row['pengajuan_baru'],
                'cicilan_lain': row['cicilan_lain'],
                'skor_fuzzy': result['final_skor'],
                'status': result['status'],
                'risiko': result['risiko_level'],
                'saran': result['saran']
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_records': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)