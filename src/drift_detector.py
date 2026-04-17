import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats

class DriftDetector:
    PSI_THRESHOLD_WARNING = 0.1
    PSI_THRESHOLD_CRITICAL = 0.2
    KS_PVALUE_THRESHOLD = 0.05

    def __init__(self, reference_data):
        self.reference_data = reference_data.copy()
        self.numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
        self.categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    def calculate_psi(self, reference, current, bins=10):
        ref_clean = reference.dropna()
        curr_clean = current.dropna()
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return 0.0
        breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0
        ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
        curr_counts = np.histogram(curr_clean, bins=breakpoints)[0]
        eps = 1e-06
        ref_pct = ref_counts / len(ref_clean) + eps
        curr_pct = curr_counts / len(curr_clean) + eps
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return psi

    def ks_test(self, reference, current):
        ref_clean = reference.dropna()
        curr_clean = current.dropna()
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return (0.0, 1.0)
        stat, p_value = stats.ks_2samp(ref_clean, curr_clean)
        return (stat, p_value)

    def categorical_drift(self, reference, current):
        ref_dist = reference.value_counts(normalize=True)
        curr_dist = current.value_counts(normalize=True)
        all_cats = set(ref_dist.index) | set(curr_dist.index)
        ref_aligned = ref_dist.reindex(all_cats, fill_value=0)
        curr_aligned = curr_dist.reindex(all_cats, fill_value=0)
        max_diff = (ref_aligned - curr_aligned).abs().max()
        return {'max_category_diff': float(max_diff), 'new_categories': list(set(curr_dist.index) - set(ref_dist.index)), 'missing_categories': list(set(ref_dist.index) - set(curr_dist.index)), 'drifted': max_diff > 0.1}

    def detect_drift(self, current_data):
        report = {'timestamp': datetime.now().isoformat(), 'reference_samples': len(self.reference_data), 'current_samples': len(current_data), 'features': {}, 'overall': {'drifted_features': [], 'warning_features': [], 'stable_features': [], 'total_features_analyzed': 0, 'drift_detected': False, 'severity': 'none'}}
        for col in self.numerical_cols:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            ref_col = self.reference_data[col]
            curr_col = current_data[col]
            psi = self.calculate_psi(ref_col, curr_col)
            ks_stat, ks_pvalue = self.ks_test(ref_col, curr_col)
            if psi >= self.PSI_THRESHOLD_CRITICAL:
                status = 'drift_detected'
                report['overall']['drifted_features'].append(col)
            elif psi >= self.PSI_THRESHOLD_WARNING:
                status = 'warning'
                report['overall']['warning_features'].append(col)
            else:
                status = 'stable'
                report['overall']['stable_features'].append(col)
            report['features'][col] = {'type': 'numerical', 'psi': round(psi, 4), 'ks_statistic': round(ks_stat, 4), 'ks_pvalue': round(ks_pvalue, 6), 'ref_mean': round(float(ref_col.mean()), 2), 'curr_mean': round(float(curr_col.mean()), 2), 'mean_shift': round(float(curr_col.mean() - ref_col.mean()), 2), 'ref_std': round(float(ref_col.std()), 2), 'curr_std': round(float(curr_col.std()), 2), 'status': status}
        for col in self.categorical_cols:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            cat_result = self.categorical_drift(self.reference_data[col], current_data[col])
            status = 'drift_detected' if cat_result['drifted'] else 'stable'
            if status == 'drift_detected':
                report['overall']['drifted_features'].append(col)
            else:
                report['overall']['stable_features'].append(col)
            report['features'][col] = {'type': 'categorical', 'max_category_diff': round(cat_result['max_category_diff'], 4), 'new_categories': cat_result['new_categories'], 'missing_categories': cat_result['missing_categories'], 'status': status}
        total = len(report['features'])
        n_drifted = len(report['overall']['drifted_features'])
        n_warning = len(report['overall']['warning_features'])
        report['overall']['total_features_analyzed'] = total
        report['overall']['drift_percentage'] = round(n_drifted / max(total, 1) * 100, 1)
        if n_drifted >= total * 0.3:
            report['overall']['severity'] = 'critical'
            report['overall']['drift_detected'] = True
            report['overall']['recommendation'] = 'RETRAIN MODEL IMMEDIATELY - significant drift in 30%+ features'
        elif n_drifted > 0 or n_warning >= total * 0.3:
            report['overall']['severity'] = 'warning'
            report['overall']['drift_detected'] = True
            report['overall']['recommendation'] = 'INVESTIGATE - some features showing drift, monitor closely'
        else:
            report['overall']['severity'] = 'none'
            report['overall']['drift_detected'] = False
            report['overall']['recommendation'] = 'Model is stable - no action needed'
        return report

    def print_report(self, report):
        print('\n' + '=' * 65)
        print('              DRIFT DETECTION REPORT')
        print('=' * 65)
        print(f"  Timestamp:          {report['timestamp']}")
        print(f"  Reference samples:  {report['reference_samples']}")
        print(f"  Current samples:    {report['current_samples']}")
        overall = report['overall']
        severity_label = {'none': 'STABLE', 'warning': '!! WARNING !!', 'critical': '!!! CRITICAL - RETRAIN NEEDED !!!'}
        print(f"\n  Status:             {severity_label[overall['severity']]}")
        print(f"  Drift Detected:     {overall['drift_detected']}")
        print(f"  Features Analyzed:  {overall['total_features_analyzed']}")
        print(f"  Drifted Features:   {len(overall['drifted_features'])} ({overall['drift_percentage']}%)")
        print(f"  Warning Features:   {len(overall['warning_features'])}")
        print(f"  Recommendation:     {overall['recommendation']}")
        print(f"\n  {'Feature':<30} {'Type':<12} {'PSI/Diff':<10} {'Status':<15}")
        print('  ' + '-' * 65)
        for feat, details in report['features'].items():
            if details['type'] == 'numerical':
                metric = f"{details['psi']:.4f}"
            else:
                metric = f"{details['max_category_diff']:.4f}"
            marker = {'stable': 'OK', 'warning': 'WARNING', 'drift_detected': 'DRIFT'}
            print(f"  {feat:<30} {details['type']:<12} {metric:<10} {marker[details['status']]:<15}")
        if overall['drifted_features']:
            print(f'\n  Drifted Features Detail:')
            for feat in overall['drifted_features']:
                d = report['features'][feat]
                if d['type'] == 'numerical':
                    direction = 'increased' if d['mean_shift'] > 0 else 'decreased'
                    print(f"    {feat}: mean {direction} from {d['ref_mean']} -> {d['curr_mean']} (shift: {d['mean_shift']:+.2f})")
        print('=' * 65)

def simulate_economic_downturn(df, severity=1.0):
    drifted = df.copy()
    n = len(drifted)
    income_drop = 0.15 * severity
    drifted['person_income'] = (drifted['person_income'] * (1 - income_drop)).astype(int)
    drifted['loan_amnt'] = (drifted['loan_amnt'] * (1 + 0.2 * severity)).clip(500, 35000).astype(int)
    drifted['loan_int_rate'] = (drifted['loan_int_rate'] + 2.5 * severity).clip(5, 25)
    drifted['loan_percent_income'] = (drifted['loan_amnt'] / drifted['person_income']).clip(0, 0.83).round(2)
    flip_mask = np.random.random(n) < 0.15 * severity
    drifted.loc[flip_mask, 'cb_person_default_on_file'] = 'Y'
    rent_mask = np.random.random(n) < 0.1 * severity
    drifted.loc[rent_mask & (drifted['person_home_ownership'] == 'OWN'), 'person_home_ownership'] = 'RENT'
    drifted.loc[rent_mask & (drifted['person_home_ownership'] == 'MORTGAGE'), 'person_home_ownership'] = 'RENT'
    grade_shift = np.random.random(n) < 0.2 * severity
    grade_map = {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F', 'F': 'G', 'G': 'G'}
    drifted.loc[grade_shift, 'loan_grade'] = drifted.loc[grade_shift, 'loan_grade'].map(grade_map)
    return drifted

def run_drift_detection():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    report_dir = os.path.join(base_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    print('Loading reference data...')
    reference = pd.read_csv(os.path.join(processed_dir, 'reference_data.csv'))
    detector = DriftDetector(reference)
    print('\n' + '=' * 65)
    print('  TEST 1: CONTROL - Same Distribution (no drift expected)')
    print('=' * 65)
    raw_path = os.path.join(base_dir, 'data', 'raw', 'credit_risk_dataset.csv')
    full_data = pd.read_csv(raw_path)
    control = full_data[full_data['person_age'] <= 100].copy()
    control = control[control['person_emp_length'].isna() | (control['person_emp_length'] <= 60)]
    income_cap = control['person_income'].quantile(0.995)
    control = control[control['person_income'] <= income_cap]
    control = control.sample(n=3000, random_state=99)
    control_report = detector.detect_drift(control)
    detector.print_report(control_report)
    print('\n' + '=' * 65)
    print('  TEST 2: MILD ECONOMIC DOWNTURN (severity=0.5)')
    print('=' * 65)
    mild_drifted = simulate_economic_downturn(reference, severity=0.5)
    mild_report = detector.detect_drift(mild_drifted)
    detector.print_report(mild_report)
    print('\n' + '=' * 65)
    print('  TEST 3: SEVERE ECONOMIC DOWNTURN (severity=1.5)')
    print('=' * 65)
    severe_drifted = simulate_economic_downturn(reference, severity=1.5)
    severe_report = detector.detect_drift(severe_drifted)
    detector.print_report(severe_report)
    for name, report in [('control', control_report), ('mild', mild_report), ('severe', severe_report)]:
        path = os.path.join(report_dir, f'drift_report_{name}.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    print(f'\nAll reports saved to: {report_dir}/')
    return (control_report, mild_report, severe_report)
if __name__ == '__main__':
    run_drift_detection()