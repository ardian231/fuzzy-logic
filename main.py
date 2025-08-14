import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

class FuzzyCreditAnalysis:
    def __init__(self):
        """Initialize Fuzzy Credit Analysis System"""
        self.rules = self._initialize_rules()
    
    def membership_dp_ratio(self, ratio: float) -> Dict[str, float]:
        """
        Membership function for DP/Nominal ratio
        
        Args:
            ratio: DP/Nominal ratio (0.0 to 1.0)
            
        Returns:
            Dictionary with membership values for low, med, high
        """
        # Low: below 0.18, fully low at 0.12
        low = 1.0 if ratio <= 0.12 else (0.18 - ratio) / (0.18 - 0.12) if ratio < 0.18 else 0.0
        low = max(0.0, min(1.0, low))
        
        # Med: 0.12 to 0.30, peak at 0.21
        if ratio <= 0.12:
            med = 0.0
        elif ratio <= 0.21:
            med = (ratio - 0.12) / (0.21 - 0.12)
        elif ratio <= 0.30:
            med = (0.30 - ratio) / (0.30 - 0.21)
        else:
            med = 0.0
        med = max(0.0, min(1.0, med))
        
        # High: above 0.24, fully high at 0.35
        high = 0.0 if ratio <= 0.24 else (ratio - 0.24) / (0.35 - 0.24) if ratio < 0.35 else 1.0
        high = max(0.0, min(1.0, high))
        
        return {"low": low, "med": med, "high": high}
    
    def membership_dti_ratio(self, ratio: float) -> Dict[str, float]:
        """
        Membership function for DTI (Debt-to-Income) ratio
        DTI = 0 (no other debts) is the best condition
        
        Args:
            ratio: Cicilan Lain/Gaji ratio (0.0 to 1.0+)
            
        Returns:
            Dictionary with membership values for low, med, high
        """
        # Low (good): 0% - 15%, with 0% getting perfect score
        low = 1.0 if ratio <= 0.15 else (0.20 - ratio) / (0.20 - 0.15) if ratio < 0.20 else 0.0
        low = max(0.0, min(1.0, low))
        
        # Med (moderate): 10% - 35%
        if ratio <= 0.10:
            med = 0.0
        elif ratio <= 0.20:
            med = (ratio - 0.10) / (0.20 - 0.10)
        elif ratio <= 0.35:
            med = (0.35 - ratio) / (0.35 - 0.20)
        else:
            med = 0.0
        med = max(0.0, min(1.0, med))
        
        # High (bad): starts from 30%
        high = 0.0 if ratio <= 0.30 else (ratio - 0.30) / (0.50 - 0.30) if ratio < 0.50 else 1.0
        high = max(0.0, min(1.0, high))
        
        return {"low": low, "med": med, "high": high}
    
    def membership_lti_ratio(self, ratio: float) -> Dict[str, float]:
        """
        Membership function for LTI (Loan-to-Income) ratio
        
        Args:
            ratio: Nominal/Gaji ratio (times of salary)
            
        Returns:
            Dictionary with membership values for low, med, high
        """
        # Low (safe): <= 4x salary
        low = 1.0 if ratio <= 3.0 else (4.0 - ratio) / (4.0 - 3.0) if ratio < 4.0 else 0.0
        low = max(0.0, min(1.0, low))
        
        # Med (moderate): 3x - 8x salary
        if ratio <= 3.0:
            med = 0.0
        elif ratio <= 5.0:
            med = (ratio - 3.0) / (5.0 - 3.0)
        elif ratio <= 8.0:
            med = (8.0 - ratio) / (8.0 - 5.0)
        else:
            med = 0.0
        med = max(0.0, min(1.0, med))
        
        # High (risky): >= 7x salary
        high = 0.0 if ratio <= 7.0 else (ratio - 7.0) / (10.0 - 7.0) if ratio < 10.0 else 1.0
        high = max(0.0, min(1.0, high))
        
        return {"low": low, "med": med, "high": high}
    
    def membership_nominal_amount(self, amount: float) -> Dict[str, float]:
        """
        Membership function for Nominal Pengajuan (in millions)
        
        Args:
            amount: Nominal amount in IDR
            
        Returns:
            Dictionary with membership values for small, medium, large
        """
        amount_mil = amount / 1_000_000  # Convert to millions
        
        # Small: <= 15 million
        small = 1.0 if amount_mil <= 10.0 else (15.0 - amount_mil) / (15.0 - 10.0) if amount_mil < 15.0 else 0.0
        small = max(0.0, min(1.0, small))
        
        # Medium: 10 - 35 million
        if amount_mil <= 10.0:
            medium = 0.0
        elif amount_mil <= 20.0:
            medium = (amount_mil - 10.0) / (20.0 - 10.0)
        elif amount_mil <= 35.0:
            medium = (35.0 - amount_mil) / (35.0 - 20.0)
        else:
            medium = 0.0
        medium = max(0.0, min(1.0, medium))
        
        # Large: >= 30 million
        large = 0.0 if amount_mil <= 30.0 else (amount_mil - 30.0) / (50.0 - 30.0) if amount_mil < 50.0 else 1.0
        large = max(0.0, min(1.0, large))
        
        return {"small": small, "medium": medium, "large": large}
    
    def _initialize_rules(self) -> List[Dict]:
        """
        Initialize fuzzy rules based on credit analysis logic
        
        Returns:
            List of fuzzy rules
        """
        return [
            # High approval rules
            {"dp": "high", "dti": "low", "lti": "low", "nominal": "small", "output": 0.9},
            {"dp": "high", "dti": "low", "lti": "low", "nominal": "medium", "output": 0.85},
            {"dp": "high", "dti": "low", "lti": "med", "nominal": "small", "output": 0.8},
            {"dp": "high", "dti": "med", "lti": "low", "nominal": "small", "output": 0.75},
            {"dp": "med", "dti": "low", "lti": "low", "nominal": "small", "output": 0.7},
            
            # Medium approval rules
            {"dp": "med", "dti": "low", "lti": "med", "nominal": "small", "output": 0.65},
            {"dp": "med", "dti": "med", "lti": "low", "nominal": "medium", "output": 0.6},
            {"dp": "high", "dti": "med", "lti": "med", "nominal": "medium", "output": 0.55},
            {"dp": "low", "dti": "low", "lti": "low", "nominal": "small", "output": 0.5},
            {"dp": "med", "dti": "low", "lti": "high", "nominal": "small", "output": 0.45},
            
            # Low approval rules
            {"dp": "low", "dti": "low", "lti": "high", "nominal": "medium", "output": 0.4},
            {"dp": "med", "dti": "high", "lti": "med", "nominal": "large", "output": 0.3},
            {"dp": "low", "dti": "med", "lti": "high", "nominal": "large", "output": 0.2},
            {"dp": "low", "dti": "high", "lti": "high", "nominal": "large", "output": 0.1},
        ]
    
    def analyze_credit(self, dp: float, nominal_pengajuan: float, cicilan_lain: float, gaji: float) -> Dict:
        """
Perform fuzzy credit analysis (Sugeno 0-order)
- Fuzzify input (DP ratio, DTI, LTI, nominal)
- Rule firing strength = min(antecedents)
- Output = weighted average of singleton consequents
"""

        # Calculate ratios
        r_dp = dp / nominal_pengajuan if nominal_pengajuan > 0 else 0
        r_dti = cicilan_lain / gaji if gaji > 0 else 0
        r_lti = nominal_pengajuan / gaji if gaji > 0 else 0
        
        # Get membership values
        dp_membership = self.membership_dp_ratio(r_dp)
        dti_membership = self.membership_dti_ratio(r_dti)
        lti_membership = self.membership_lti_ratio(r_lti)
        nominal_membership = self.membership_nominal_amount(nominal_pengajuan)
        
        # Apply fuzzy rules using Mamdani inference
        numerator = 0.0
        denominator = 0.0
        active_rules = []
        
        for i, rule in enumerate(self.rules):
            # Get membership values for each antecedent
            dp_val = dp_membership[rule["dp"]]
            dti_val = dti_membership[rule["dti"]]
            lti_val = lti_membership[rule["lti"]]
            
            # Handle nominal membership mapping
            if rule["nominal"] == "small":
                nominal_val = nominal_membership["small"]
            elif rule["nominal"] == "medium":
                nominal_val = nominal_membership["medium"]
            else:  # large
                nominal_val = nominal_membership["large"]
            
            # Calculate rule strength (minimum of all antecedents)
            rule_strength = min(dp_val, dti_val, lti_val, nominal_val)
            
            if rule_strength > 0:
                numerator += rule_strength * rule["output"]
                denominator += rule_strength
                active_rules.append({
                    "rule_index": i + 1,
                    "rule": f"IF DP={rule['dp']} AND DTI={rule['dti']} AND LTI={rule['lti']} AND Nominal={rule['nominal']} THEN Approval={rule['output']}",
                    "strength": rule_strength,
                    "output": rule["output"]
                })
        
        # Calculate final output using centroid defuzzification
        final_output = numerator / denominator if denominator > 0 else 0.0
        
        # Determine approval decision
        if final_output >= 0.6:
            decision = "APPROVE"
        elif final_output >= 0.4:
            decision = "DI PERTIMBANGKAN"
        else:
            decision = "REJECT"
        
        return {
            "ratios": {
                "r_dp": r_dp,
                "r_dti": r_dti,
                "r_lti": r_lti
            },
            "memberships": {
                "dp": dp_membership,
                "dti": dti_membership,
                "lti": lti_membership,
                "nominal": nominal_membership
            },
            "fuzzy_score": final_output,
            "decision": decision,
            "active_rules": sorted(active_rules, key=lambda x: x["strength"], reverse=True),
            "inputs": {
                "dp": dp,
                "nominal_pengajuan": nominal_pengajuan,
                "cicilan_lain": cicilan_lain,
                "gaji": gaji
            }
        }
    
    def plot_membership_functions(self):
        """Plot all membership functions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fuzzy Membership Functions', fontsize=16)
        
        # DP Ratio Membership
        x_dp = np.linspace(0, 0.5, 1000)
        dp_low = [self.membership_dp_ratio(x)["low"] for x in x_dp]
        dp_med = [self.membership_dp_ratio(x)["med"] for x in x_dp]
        dp_high = [self.membership_dp_ratio(x)["high"] for x in x_dp]
        
        axes[0,0].plot(x_dp, dp_low, 'r-', label='Low', linewidth=2)
        axes[0,0].plot(x_dp, dp_med, 'orange', label='Med', linewidth=2)
        axes[0,0].plot(x_dp, dp_high, 'g-', label='High', linewidth=2)
        axes[0,0].set_title('DP/Nominal Ratio')
        axes[0,0].set_xlabel('Ratio')
        axes[0,0].set_ylabel('Membership')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # DTI Ratio Membership
        x_dti = np.linspace(0, 0.6, 1000)
        dti_low = [self.membership_dti_ratio(x)["low"] for x in x_dti]
        dti_med = [self.membership_dti_ratio(x)["med"] for x in x_dti]
        dti_high = [self.membership_dti_ratio(x)["high"] for x in x_dti]
        
        axes[0,1].plot(x_dti, dti_low, 'r-', label='Low', linewidth=2)
        axes[0,1].plot(x_dti, dti_med, 'orange', label='Med', linewidth=2)
        axes[0,1].plot(x_dti, dti_high, 'g-', label='High', linewidth=2)
        axes[0,1].set_title('DTI Ratio')
        axes[0,1].set_xlabel('Ratio')
        axes[0,1].set_ylabel('Membership')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # LTI Ratio Membership
        x_lti = np.linspace(0, 12, 1000)
        lti_low = [self.membership_lti_ratio(x)["low"] for x in x_lti]
        lti_med = [self.membership_lti_ratio(x)["med"] for x in x_lti]
        lti_high = [self.membership_lti_ratio(x)["high"] for x in x_lti]
        
        axes[1,0].plot(x_lti, lti_low, 'r-', label='Low', linewidth=2)
        axes[1,0].plot(x_lti, lti_med, 'orange', label='Med', linewidth=2)
        axes[1,0].plot(x_lti, lti_high, 'g-', label='High', linewidth=2)
        axes[1,0].set_title('LTI Ratio')
        axes[1,0].set_xlabel('Ratio (times of salary)')
        axes[1,0].set_ylabel('Membership')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Nominal Amount Membership
        x_nominal = np.linspace(0, 80, 1000)  # 0 to 80 million
        nominal_small = [self.membership_nominal_amount(x * 1_000_000)["small"] for x in x_nominal]
        nominal_med = [self.membership_nominal_amount(x * 1_000_000)["medium"] for x in x_nominal]
        nominal_large = [self.membership_nominal_amount(x * 1_000_000)["large"] for x in x_nominal]
        
        axes[1,1].plot(x_nominal, nominal_small, 'r-', label='Small', linewidth=2)
        axes[1,1].plot(x_nominal, nominal_med, 'orange', label='Medium', linewidth=2)
        axes[1,1].plot(x_nominal, nominal_large, 'g-', label='Large', linewidth=2)
        axes[1,1].set_title('Nominal Pengajuan')
        axes[1,1].set_xlabel('Amount (Million IDR)')
        axes[1,1].set_ylabel('Membership')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_analysis_report(self, result: Dict):
        """Print detailed analysis report"""
        print("="*60)
        print("FUZZY CREDIT ANALYSIS REPORT")
        print("="*60)
        
        inputs = result["inputs"]
        ratios = result["ratios"]
        
        # Input Information
        print(f"\n📊 INPUT DATA:")
        print(f"  • Down Payment (DP)      : Rp {inputs['dp']:,.0f}")
        print(f"  • Nominal Pengajuan       : Rp {inputs['nominal_pengajuan']:,.0f}")
        print(f"  • Cicilan Lain           : Rp {inputs['cicilan_lain']:,.0f}")
        if inputs['cicilan_lain'] == 0:
            print(f"    → Tidak ada cicilan lain (BAGUS!)")
        print(f"  • Gaji                   : Rp {inputs['gaji']:,.0f}")
        
        # Calculated Ratios
        print(f"\n📈 CALCULATED RATIOS:")
        print(f"  • DP/Nominal Ratio (r_dp) : {ratios['r_dp']:.1%}")
        print(f"  • DTI Ratio (r_dti)       : {ratios['r_dti']:.1%}")
        print(f"  • LTI Ratio (r_lti)       : {ratios['r_lti']:.1f}x gaji")
        
        # Membership Values
        print(f"\n🎯 MEMBERSHIP VALUES:")
        memberships = result["memberships"]
        
        print(f"  DP Ratio     - Low: {memberships['dp']['low']:.3f}, Med: {memberships['dp']['med']:.3f}, High: {memberships['dp']['high']:.3f}")
        print(f"  DTI Ratio    - Low: {memberships['dti']['low']:.3f}, Med: {memberships['dti']['med']:.3f}, High: {memberships['dti']['high']:.3f}")
        print(f"  LTI Ratio    - Low: {memberships['lti']['low']:.3f}, Med: {memberships['lti']['med']:.3f}, High: {memberships['lti']['high']:.3f}")
        print(f"  Nominal      - Small: {memberships['nominal']['small']:.3f}, Medium: {memberships['nominal']['medium']:.3f}, Large: {memberships['nominal']['large']:.3f}")
        
        # Active Rules
        print(f"\n🔥 ACTIVE RULES (Top 5):")
        for i, rule in enumerate(result["active_rules"][:5]):
            print(f"  {i+1}. Rule {rule['rule_index']} (Strength: {rule['strength']:.3f})")
            print(f"     {rule['rule']}")
        
        # Final Result
        print(f"\n🎉 FINAL RESULT:")
        print(f"  • Fuzzy Score: {result['fuzzy_score']:.1%}")
        
        decision = result['decision']
        if decision == "APPROVE":
            print(f"  • Decision: ✅ {decision}")
            print(f"    → Kredit dapat disetujui")
        elif decision == "DI PERTIMBANGKAN":
            print(f"  • Decision: ⚠️  {decision}")
            print(f"    → Perlu review lebih lanjut")
        else:
            print(f"  • Decision: ❌ {decision}")
            print(f"    → Kredit sebaiknya ditolak")
        
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    # Initialize fuzzy system
    fuzzy_system = FuzzyCreditAnalysis()
    
    # Example 1: Good candidate (high DP, no other debts)
    print("CONTOH 1: Kandidat Bagus")
    result1 = fuzzy_system.analyze_credit(
        dp=25_000_000,           # 25 juta DP
        nominal_pengajuan=100_000_000,  # 100 juta pengajuan
        cicilan_lain=0,          # Tidak ada cicilan lain
        gaji=8_000_000           # 8 juta gaji
    )
    fuzzy_system.print_analysis_report(result1)
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Risky candidate (low DP, high debt)
    print("CONTOH 2: Kandidat Berisiko")
    result2 = fuzzy_system.analyze_credit(
        dp=5_000_000,            # 5 juta DP (rendah)
        nominal_pengajuan=150_000_000,  # 150 juta pengajuan (tinggi)
        cicilan_lain=3_000_000,  # 3 juta cicilan lain
        gaji=6_000_000           # 6 juta gaji
    )
    fuzzy_system.print_analysis_report(result2)
    
    # Plot membership functions
    print("\nMenampilkan grafik membership functions...")
    fuzzy_system.plot_membership_functions()