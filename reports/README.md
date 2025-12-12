# Reports & Results Export

This directory contains scripts and generated reports for the Mall Movement Tracking project.

## 📊 Available Reports

### 1. PDF Report
Comprehensive PDF report with all project results, metrics, and visualizations.

**Generate:**
```bash
python reports/generate_report.py
```

**Output:** `reports/Project_Report_YYYYMMDD_HHMMSS.pdf`

**Includes:**
- Executive Summary
- Project Overview
- Data Overview
- Feature Engineering Summary
- Classification Results
- Clustering Results
- Forecasting Results
- Model Comparison
- Conclusions & Recommendations

### 2. Markdown Summary
Presentation-ready markdown summary document.

**Generate:**
```bash
python reports/generate_summary.py
```

**Output:** `reports/PROJECT_SUMMARY.md`

**Use Cases:**
- GitHub README
- Documentation
- Presentations
- Quick reference

### 3. Results Export
Export all results to CSV, HTML, and JSON formats.

**Generate:**
```bash
python reports/export_results.py
```

**Outputs:**
- `reports/exports/classification_results.csv`
- `reports/exports/clustering_results.csv`
- `reports/exports/forecasting_results.csv`
- `reports/exports/model_summary.csv`
- `reports/exports/results_report.html`
- `reports/exports/all_results.json`

## 🚀 Quick Start

### Generate All Reports
```bash
# Generate PDF report
python reports/generate_report.py

# Generate markdown summary
python reports/generate_summary.py

# Export all results
python reports/export_results.py
```

### Generate All at Once (Windows)
```powershell
python reports/generate_report.py
python reports/generate_summary.py
python reports/export_results.py
```

## 📋 Requirements

### For PDF Report
```bash
pip install reportlab
```

### For All Exports
- pandas
- json (built-in)

## 📁 Directory Structure

```
reports/
├── generate_report.py      # PDF report generator
├── generate_summary.py     # Markdown summary generator
├── export_results.py        # Results exporter (CSV/HTML/JSON)
├── README.md               # This file
├── exports/                # Exported files (generated)
│   ├── *.csv
│   ├── *.html
│   └── *.json
└── Project_Report_*.pdf    # Generated PDF reports
```

## 📊 Report Contents

### PDF Report Sections
1. **Title Page** - Project information
2. **Table of Contents** - Navigation
3. **Executive Summary** - Key achievements
4. **Project Overview** - Objectives and approach
5. **Data Overview** - Dataset statistics
6. **Feature Engineering** - Created features
7. **Classification Models** - Performance metrics
8. **Clustering Models** - Segmentation results
9. **Forecasting Models** - Prediction metrics
10. **Model Comparison** - Best models summary
11. **Conclusions** - Findings and recommendations

### Markdown Summary Sections
- Executive Summary
- Project Overview
- Model Performance (tables)
- Project Structure
- Usage Instructions
- Key Insights
- Recommendations
- Technical Details

## 🎯 Use Cases

### For Presentations
- Use PDF report for formal presentations
- Use HTML export for web sharing
- Use markdown summary for documentation

### For Sharing
- CSV files for data analysis
- JSON for programmatic access
- HTML for easy viewing

### For Documentation
- Markdown summary for README
- PDF for comprehensive reports
- JSON for API integration

## 🔧 Customization

### Modify PDF Report
Edit `reports/generate_report.py`:
- Change styles and colors
- Add custom sections
- Include images/visualizations

### Modify Summary
Edit `reports/generate_summary.py`:
- Customize markdown format
- Add/remove sections
- Change table formats

### Modify Exports
Edit `reports/export_results.py`:
- Add new export formats
- Customize CSV columns
- Modify HTML styling

## 📝 Notes

- Reports are generated from results in `results/` directory
- Ensure models are trained before generating reports
- PDF generation requires `reportlab` package
- All exports are saved in `reports/exports/` directory

## 🆘 Troubleshooting

### PDF Generation Fails
```bash
pip install reportlab
```

### Missing Results
Ensure models are trained:
```bash
python training/train_classification.py
python training/train_clustering.py
python training/train_forecasting.py
```

### Export Errors
Check that results JSON files exist in `results/` directory.

---

**Last Updated:** 2024  
**Version:** 1.0.0

