"""
Export All Results to Various Formats
Exports results to CSV, JSON, and HTML formats for easy sharing
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

def load_all_results():
    """Load all results from JSON files"""
    results_dir = Path(__file__).parent.parent / "results"
    
    all_results = {}
    
    # Classification
    try:
        with open(results_dir / "classification" / "metrics.json", 'r') as f:
            all_results['classification'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load classification results: {e}")
        all_results['classification'] = {}
    
    # Clustering
    try:
        with open(results_dir / "clustering" / "silhouette_score.json", 'r') as f:
            all_results['clustering'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load clustering results: {e}")
        all_results['clustering'] = {}
    
    # Forecasting
    try:
        with open(results_dir / "forecasting" / "rmse.json", 'r') as f:
            all_results['forecasting'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load forecasting results: {e}")
        all_results['forecasting'] = {}
    
    return all_results

def export_to_csv(all_results, output_dir):
    """Export results to CSV format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Classification CSV
    if all_results['classification']:
        clf_data = []
        for model_name, metrics in all_results['classification'].items():
            clf_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics.get('accuracy', 0) * 100,
                'ROC_AUC': metrics.get('roc_auc', 'N/A')
            })
        df_clf = pd.DataFrame(clf_data)
        df_clf.to_csv(output_dir / "classification_results.csv", index=False)
        print("✅ Exported: classification_results.csv")
    
    # Clustering CSV
    if all_results['clustering']:
        clust_data = []
        for model_name, metrics in all_results['clustering'].items():
            clust_data.append({
                'Model': model_name.upper(),
                'Silhouette_Score': metrics.get('silhouette_score', 0),
                'N_Clusters': metrics.get('n_clusters', 'N/A'),
                'N_Noise': metrics.get('n_noise', 0)
            })
        df_clust = pd.DataFrame(clust_data)
        df_clust.to_csv(output_dir / "clustering_results.csv", index=False)
        print("✅ Exported: clustering_results.csv")
    
    # Forecasting CSV
    if all_results['forecasting']:
        forecast_data = []
        for model_name, metrics in all_results['forecasting'].items():
            forecast_data.append({
                'Model': model_name.upper(),
                'RMSE': metrics.get('rmse', 0),
                'MAE': metrics.get('mae', 0)
            })
        df_forecast = pd.DataFrame(forecast_data)
        df_forecast.to_csv(output_dir / "forecasting_results.csv", index=False)
        print("✅ Exported: forecasting_results.csv")
    
    # Combined summary
    summary_data = []
    
    if all_results['classification']:
        best_clf = max(all_results['classification'].items(),
                      key=lambda x: x[1].get('accuracy', 0))
        summary_data.append({
            'Model_Type': 'Classification',
            'Best_Model': best_clf[0].replace('_', ' ').title(),
            'Metric': 'Accuracy',
            'Value': f"{best_clf[1].get('accuracy', 0)*100:.2f}%"
        })
    
    if all_results['clustering']:
        best_clust = max(all_results['clustering'].items(),
                        key=lambda x: x[1].get('silhouette_score', 0))
        summary_data.append({
            'Model_Type': 'Clustering',
            'Best_Model': best_clust[0].upper(),
            'Metric': 'Silhouette Score',
            'Value': f"{best_clust[1].get('silhouette_score', 0):.4f}"
        })
    
    if all_results['forecasting']:
        best_forecast = min(all_results['forecasting'].items(),
                           key=lambda x: x[1].get('rmse', float('inf')))
        summary_data.append({
            'Model_Type': 'Forecasting',
            'Best_Model': best_forecast[0].upper(),
            'Metric': 'RMSE',
            'Value': f"{best_forecast[1].get('rmse', 0):.2e}"
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / "model_summary.csv", index=False)
        print("✅ Exported: model_summary.csv")

def export_to_html(all_results, output_dir):
    """Export results to HTML format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mall Movement Tracking - Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #1E3A8A;
                border-bottom: 3px solid #3B82F6;
                padding-bottom: 10px;
            }
            h2 {
                color: #1E40AF;
                margin-top: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #1E3A8A;
                color: white;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .metric {
                font-weight: bold;
                color: #3B82F6;
            }
            .best {
                background-color: #D1FAE5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏪 Mall Movement Tracking - Results Report</h1>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    """
    
    # Classification section
    if all_results['classification']:
        html += """
            <h2>🎯 Classification Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>ROC-AUC</th>
                </tr>
        """
        best_clf = max(all_results['classification'].items(),
                      key=lambda x: x[1].get('accuracy', 0))
        for model_name, metrics in all_results['classification'].items():
            acc = metrics.get('accuracy', 0) * 100
            roc_auc = metrics.get('roc_auc', 'N/A')
            if roc_auc != 'N/A':
                roc_auc = f"{roc_auc:.4f}"
            is_best = model_name == best_clf[0]
            row_class = 'class="best"' if is_best else ''
            html += f"""
                <tr {row_class}>
                    <td>{model_name.replace('_', ' ').title()}</td>
                    <td class="metric">{acc:.2f}%</td>
                    <td>{roc_auc}</td>
                </tr>
            """
        html += "</table>"
    
    # Clustering section
    if all_results['clustering']:
        html += """
            <h2>👥 Clustering Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Silhouette Score</th>
                    <th>Clusters</th>
                    <th>Noise Points</th>
                </tr>
        """
        best_clust = max(all_results['clustering'].items(),
                        key=lambda x: x[1].get('silhouette_score', 0))
        for model_name, metrics in all_results['clustering'].items():
            is_best = model_name == best_clust[0]
            row_class = 'class="best"' if is_best else ''
            html += f"""
                <tr {row_class}>
                    <td>{model_name.upper()}</td>
                    <td class="metric">{metrics.get('silhouette_score', 0):.4f}</td>
                    <td>{metrics.get('n_clusters', 'N/A')}</td>
                    <td>{metrics.get('n_noise', 0)}</td>
                </tr>
            """
        html += "</table>"
    
    # Forecasting section
    if all_results['forecasting']:
        html += """
            <h2>📈 Forecasting Models</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                </tr>
        """
        best_forecast = min(all_results['forecasting'].items(),
                           key=lambda x: x[1].get('rmse', float('inf')))
        for model_name, metrics in all_results['forecasting'].items():
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            if rmse > 1e9:
                rmse_str = f"{rmse:.2e}"
            else:
                rmse_str = f"{rmse:.2f}"
            if mae > 1e9:
                mae_str = f"{mae:.2e}"
            else:
                mae_str = f"{mae:.2f}"
            is_best = model_name == best_forecast[0]
            row_class = 'class="best"' if is_best else ''
            html += f"""
                <tr {row_class}>
                    <td>{model_name.upper()}</td>
                    <td class="metric">{rmse_str}</td>
                    <td>{mae_str}</td>
                </tr>
            """
        html += "</table>"
    
    html += """
        </div>
    </body>
    </html>
    """
    
    output_path = output_dir / "results_report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✅ Exported: results_report.html")

def export_to_json(all_results, output_dir):
    """Export all results to a single JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    export_data = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results
    }
    
    output_path = output_dir / "all_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print("✅ Exported: all_results.json")

def main():
    """Main export function"""
    reports_dir = Path(__file__).parent
    exports_dir = reports_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Exporting Results")
    print("=" * 60)
    
    all_results = load_all_results()
    
    print("\n📊 Exporting to CSV...")
    export_to_csv(all_results, exports_dir)
    
    print("\n🌐 Exporting to HTML...")
    export_to_html(all_results, exports_dir)
    
    print("\n📄 Exporting to JSON...")
    export_to_json(all_results, exports_dir)
    
    print("\n" + "=" * 60)
    print("✅ All exports completed!")
    print(f"📁 Location: {exports_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

