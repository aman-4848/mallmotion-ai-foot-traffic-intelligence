"""
Data Quality Monitoring Module
Monitors data quality metrics and generates reports
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))


class DataQualityMonitor:
    """Monitor data quality metrics"""
    
    def __init__(self):
        """Initialize data quality monitor"""
        self.metrics_history = []
    
    def calculate_quality_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive data quality metrics
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': {
                'rows': int(len(data)),
                'columns': int(len(data.columns))
            },
            'completeness': {},
            'validity': {},
            'consistency': {},
            'uniqueness': {},
            'accuracy': {}
        }
        
        # Completeness metrics
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        metrics['completeness'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'completeness_rate': float(1 - (missing_cells / total_cells)) if total_cells > 0 else 0.0,
            'missing_per_column': {col: int(count) for col, count in data.isnull().sum().items()}
        }
        
        # Validity metrics (check for invalid values)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        metrics['validity'] = {
            'numeric_columns': len(numeric_cols),
            'infinite_values': int(np.isinf(data[numeric_cols]).sum().sum()) if numeric_cols else 0,
            'negative_values': {col: int((data[col] < 0).sum()) for col in numeric_cols if (data[col] < 0).sum() > 0}
        }
        
        # Consistency metrics
        metrics['consistency'] = {
            'duplicate_rows': int(data.duplicated().sum()),
            'duplicate_rate': float(data.duplicated().sum() / len(data)) if len(data) > 0 else 0.0
        }
        
        # Uniqueness metrics
        metrics['uniqueness'] = {
            'unique_values_per_column': {col: int(data[col].nunique()) for col in data.columns}
        }
        
        # Accuracy metrics (statistical checks)
        if numeric_cols:
            metrics['accuracy'] = {
                'outliers_detected': self._detect_outliers(data, numeric_cols),
                'data_range': {col: {
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std())
                } for col in numeric_cols[:10]}  # Limit to first 10 for readability
            }
        
        # Overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _detect_outliers(self, data: pd.DataFrame, numeric_cols: List[str], method: str = 'iqr') -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}
        
        for col in numeric_cols[:20]:  # Limit to first 20 columns
            try:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outlier_count > 0:
                    outliers[col] = int(outlier_count)
            except Exception:
                pass
        
        return outliers
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100.0
        
        # Deduct for missing values
        completeness_rate = metrics['completeness'].get('completeness_rate', 0)
        score *= completeness_rate
        
        # Deduct for duplicates
        duplicate_rate = metrics['consistency'].get('duplicate_rate', 0)
        score *= (1 - duplicate_rate * 0.5)  # 50% penalty for duplicates
        
        # Deduct for infinite values
        infinite_count = metrics['validity'].get('infinite_values', 0)
        if infinite_count > 0:
            score *= 0.9  # 10% penalty
        
        return max(0.0, min(100.0, score))
    
    def generate_quality_report(self, data: pd.DataFrame, output_path: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive data quality report
        
        Args:
            data: DataFrame to analyze
            output_path: Optional path to save report
            
        Returns:
            Quality report dictionary
        """
        report = {
            'report_type': 'Data Quality Report',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.calculate_quality_metrics(data),
            'recommendations': []
        }
        
        # Generate recommendations
        metrics = report['metrics']
        
        if metrics['completeness']['completeness_rate'] < 0.95:
            report['recommendations'].append({
                'issue': 'Low completeness',
                'severity': 'high' if metrics['completeness']['completeness_rate'] < 0.8 else 'medium',
                'message': f"Data completeness is {metrics['completeness']['completeness_rate']*100:.1f}%. Consider imputation or data collection improvement."
            })
        
        if metrics['consistency']['duplicate_rate'] > 0.05:
            report['recommendations'].append({
                'issue': 'High duplicate rate',
                'severity': 'medium',
                'message': f"Duplicate rate is {metrics['consistency']['duplicate_rate']*100:.1f}%. Consider data deduplication."
            })
        
        if metrics['validity']['infinite_values'] > 0:
            report['recommendations'].append({
                'issue': 'Infinite values detected',
                'severity': 'high',
                'message': f"{metrics['validity']['infinite_values']} infinite values found. Check data processing pipeline."
            })
        
        if metrics['quality_score'] < 70:
            report['recommendations'].append({
                'issue': 'Low overall quality score',
                'severity': 'high',
                'message': f"Overall quality score is {metrics['quality_score']:.1f}/100. Review data quality issues."
            })
        
        # Save report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def track_quality_over_time(self, data: pd.DataFrame):
        """Track quality metrics over time"""
        metrics = self.calculate_quality_metrics(data)
        self.metrics_history.append(metrics)
        
        # Keep only last 100 records
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_quality_trends(self) -> Dict:
        """Get quality trends over time"""
        if len(self.metrics_history) < 2:
            return {'message': 'Insufficient history for trend analysis'}
        
        trends = {
            'completeness_trend': [m['completeness']['completeness_rate'] for m in self.metrics_history],
            'quality_score_trend': [m['quality_score'] for m in self.metrics_history],
            'duplicate_rate_trend': [m['consistency']['duplicate_rate'] for m in self.metrics_history]
        }
        
        return trends


def main():
    """Example usage"""
    from streamlit_app.utils.data_loader import load_processed_data
    
    print("=" * 60)
    print("Data Quality Monitoring")
    print("=" * 60)
    
    try:
        # Load data
        df = load_processed_data()
        print(f"Loaded data: {df.shape}")
        
        # Initialize monitor
        monitor = DataQualityMonitor()
        
        # Generate report
        report = monitor.generate_quality_report(
            df,
            output_path=Path(__file__).parent / "data_quality_report.json"
        )
        
        print("\n📊 Data Quality Metrics:")
        print(f"Completeness: {report['metrics']['completeness']['completeness_rate']*100:.1f}%")
        print(f"Quality Score: {report['metrics']['quality_score']:.1f}/100")
        print(f"Duplicate Rate: {report['metrics']['consistency']['duplicate_rate']*100:.1f}%")
        print(f"\nRecommendations: {len(report['recommendations'])}")
        for rec in report['recommendations']:
            print(f"  - [{rec['severity'].upper()}] {rec['issue']}: {rec['message']}")
        
        print(f"\nReport saved to: {Path(__file__).parent / 'data_quality_report.json'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

