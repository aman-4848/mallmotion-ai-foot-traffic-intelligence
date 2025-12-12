"""
Data Drift Detection Module
Detects changes in data distribution over time
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Some drift detection methods may not work.")


class DriftDetector:
    """Detect data drift in features"""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data
        
        Args:
            reference_data: Reference dataset (baseline)
        """
        self.reference_data = reference_data.copy()
        self.reference_stats = self._calculate_statistics(reference_data)
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for numeric columns"""
        stats_dict = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'q25': float(data[col].quantile(0.25)),
                'q75': float(data[col].quantile(0.75))
            }
        
        return stats_dict
    
    def detect_drift_ks_test(self, current_data: pd.DataFrame, alpha: float = 0.05) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            current_data: Current dataset to compare
            alpha: Significance level
            
        Returns:
            Dictionary with drift detection results
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for KS test"}
        
        results = {
            'method': 'Kolmogorov-Smirnov Test',
            'alpha': alpha,
            'drifted_features': [],
            'stable_features': [],
            'p_values': {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in current_data.columns:
                ref_values = self.reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    try:
                        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                        results['p_values'][col] = float(p_value)
                        
                        if p_value < alpha:
                            results['drifted_features'].append({
                                'feature': col,
                                'p_value': float(p_value),
                                'statistic': float(statistic)
                            })
                        else:
                            results['stable_features'].append(col)
                    except Exception as e:
                        results['p_values'][col] = f"Error: {str(e)}"
        
        results['drift_detected'] = len(results['drifted_features']) > 0
        results['drift_ratio'] = len(results['drifted_features']) / len(numeric_cols) if numeric_cols else 0
        
        return results
    
    def detect_drift_statistical(self, current_data: pd.DataFrame, threshold: float = 0.1) -> Dict:
        """
        Detect drift using statistical comparison
        
        Args:
            current_data: Current dataset to compare
            threshold: Threshold for mean/std difference (0.1 = 10%)
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            'method': 'Statistical Comparison',
            'threshold': threshold,
            'drifted_features': [],
            'stable_features': [],
            'comparisons': {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in current_data.columns and col in self.reference_stats:
                ref_mean = self.reference_stats[col]['mean']
                ref_std = self.reference_stats[col]['std']
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                # Calculate relative differences
                mean_diff = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-10)
                std_diff = abs(curr_std - ref_std) / (abs(ref_std) + 1e-10)
                
                comparison = {
                    'reference_mean': float(ref_mean),
                    'current_mean': float(curr_mean),
                    'mean_diff_pct': float(mean_diff * 100),
                    'reference_std': float(ref_std),
                    'current_std': float(curr_std),
                    'std_diff_pct': float(std_diff * 100)
                }
                
                results['comparisons'][col] = comparison
                
                if mean_diff > threshold or std_diff > threshold:
                    results['drifted_features'].append({
                        'feature': col,
                        'mean_diff_pct': float(mean_diff * 100),
                        'std_diff_pct': float(std_diff * 100)
                    })
                else:
                    results['stable_features'].append(col)
        
        results['drift_detected'] = len(results['drifted_features']) > 0
        results['drift_ratio'] = len(results['drifted_features']) / len(numeric_cols) if numeric_cols else 0
        
        return results
    
    def detect_drift_psi(self, current_data: pd.DataFrame, bins: int = 10, threshold: float = 0.2) -> Dict:
        """
        Detect drift using Population Stability Index (PSI)
        
        Args:
            current_data: Current dataset to compare
            bins: Number of bins for PSI calculation
            threshold: PSI threshold (0.2 = significant drift)
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            'method': 'Population Stability Index (PSI)',
            'threshold': threshold,
            'bins': bins,
            'drifted_features': [],
            'stable_features': [],
            'psi_values': {}
        }
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in current_data.columns:
                ref_values = self.reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    try:
                        # Create bins based on reference data
                        _, bin_edges = pd.cut(ref_values, bins=bins, retbins=True, duplicates='drop')
                        
                        # Calculate distributions
                        ref_dist = pd.cut(ref_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
                        curr_dist = pd.cut(curr_values, bins=bin_edges, include_lowest=True).value_counts(normalize=True)
                        
                        # Calculate PSI
                        psi = 0
                        for bin_name in ref_dist.index:
                            ref_pct = ref_dist[bin_name]
                            curr_pct = curr_dist.get(bin_name, 0.0001)  # Avoid log(0)
                            
                            if ref_pct > 0 and curr_pct > 0:
                                psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
                        
                        results['psi_values'][col] = float(psi)
                        
                        if psi > threshold:
                            results['drifted_features'].append({
                                'feature': col,
                                'psi': float(psi)
                            })
                        else:
                            results['stable_features'].append(col)
                    except Exception as e:
                        results['psi_values'][col] = f"Error: {str(e)}"
        
        results['drift_detected'] = len(results['drifted_features']) > 0
        results['drift_ratio'] = len(results['drifted_features']) / len(numeric_cols) if numeric_cols else 0
        
        return results
    
    def generate_drift_report(self, current_data: pd.DataFrame, output_path: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive drift detection report
        
        Args:
            current_data: Current dataset to compare
            output_path: Optional path to save report
            
        Returns:
            Comprehensive drift detection report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reference_data_shape': self.reference_data.shape,
            'current_data_shape': current_data.shape,
            'detections': {}
        }
        
        # Run all detection methods
        report['detections']['ks_test'] = self.detect_drift_ks_test(current_data)
        report['detections']['statistical'] = self.detect_drift_statistical(current_data)
        report['detections']['psi'] = self.detect_drift_psi(current_data)
        
        # Overall summary
        all_drifted = set()
        for method_result in report['detections'].values():
            if 'drifted_features' in method_result:
                for feature_info in method_result['drifted_features']:
                    if isinstance(feature_info, dict):
                        all_drifted.add(feature_info['feature'])
                    else:
                        all_drifted.add(feature_info)
        
        total_features = len(self.reference_data.select_dtypes(include=[np.number]).columns)
        features_with_drift = len(all_drifted)
        
        # Calculate drift severity
        if total_features > 0:
            drift_ratio = features_with_drift / total_features
            if drift_ratio > 0.3:
                severity = 'high'
            elif drift_ratio > 0.1:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'unknown'
        
        report['summary'] = {
            'total_features_checked': total_features,
            'features_with_drift': features_with_drift,
            'drifted_feature_names': list(all_drifted),
            'drift_severity': severity
        }
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


def main():
    """Example usage"""
    from streamlit_app.utils.data_loader import load_processed_data
    
    print("=" * 60)
    print("Data Drift Detection")
    print("=" * 60)
    
    try:
        # Load reference data (first half)
        df = load_processed_data()
        print(f"Loaded data: {df.shape}")
        
        # Split into reference and current
        split_idx = len(df) // 2
        reference_data = df.iloc[:split_idx].copy()
        current_data = df.iloc[split_idx:].copy()
        
        print(f"Reference data: {reference_data.shape}")
        print(f"Current data: {current_data.shape}")
        
        # Initialize detector
        detector = DriftDetector(reference_data)
        
        # Generate report
        report = detector.generate_drift_report(
            current_data,
            output_path=Path(__file__).parent / "drift_report.json"
        )
        
        print("\n📊 Drift Detection Results:")
        print(f"Features with drift: {report['summary']['features_with_drift']}")
        print(f"Drift severity: {report['summary']['drift_severity']}")
        print(f"\nReport saved to: {Path(__file__).parent / 'drift_report.json'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

