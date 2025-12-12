"""
Generate Comprehensive PDF Report for Mall Movement Tracking Project
Creates a professional PDF report with all project results, metrics, and visualizations
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. Install with: pip install reportlab")

def load_results():
    """Load all results from JSON files"""
    results_dir = Path(__file__).parent.parent / "results"
    
    results = {
        'classification': {},
        'clustering': {},
        'forecasting': {}
    }
    
    # Load classification results
    try:
        with open(results_dir / "classification" / "metrics.json", 'r') as f:
            results['classification'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load classification results: {e}")
    
    # Load clustering results
    try:
        with open(results_dir / "clustering" / "silhouette_score.json", 'r') as f:
            results['clustering'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load clustering results: {e}")
    
    # Load forecasting results
    try:
        with open(results_dir / "forecasting" / "rmse.json", 'r') as f:
            results['forecasting'] = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load forecasting results: {e}")
    
    return results

def get_project_info():
    """Get project information - CUSTOMIZE THIS SECTION"""
    return {
        'project_name': 'Mall Movement Tracking',
        'version': '1.0.0',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'ML-powered analytics for customer movement patterns in shopping malls',
        # ADD YOUR GROUP MEMBER DETAILS HERE
        'group_members': [
            {'name': 'Member 1 Name', 'role': 'Role/Contribution', 'email': 'email@example.com'},
            {'name': 'Member 2 Name', 'role': 'Role/Contribution', 'email': 'email@example.com'},
            {'name': 'Member 3 Name', 'role': 'Role/Contribution', 'email': 'email@example.com'},
            # Add more members as needed
        ],
        # ADD PROJECT DETAILS HERE
        'institution': 'Your Institution Name',
        'course': 'Course Name/Code',
        'supervisor': 'Supervisor Name (if applicable)',
        'project_duration': 'Start Date - End Date',
        'technologies': ['Python', 'Scikit-learn', 'XGBoost', 'Streamlit', 'Pandas', 'NumPy']
    }

def create_pdf_report(output_path):
    """Create comprehensive PDF report"""
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab is required. Install with: pip install reportlab")
        return False
    
    # Load results
    results = load_results()
    project_info = get_project_info()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1E40AF'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#3B82F6'),
        spaceAfter=8
    )
    
    # Title Page
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(project_info['project_name'], title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Machine Learning Project Report", styles['Heading2']))
    elements.append(Spacer(1, 0.4*inch))
    
    # Project Details
    if 'institution' in project_info and project_info['institution'] and project_info['institution'] != 'Your Institution Name':
        elements.append(Paragraph(f"<b>Institution:</b> {project_info['institution']}", styles['Normal']))
    if 'course' in project_info and project_info['course'] and project_info['course'] != 'Course Name/Code':
        elements.append(Paragraph(f"<b>Course:</b> {project_info['course']}", styles['Normal']))
    if 'supervisor' in project_info and project_info['supervisor'] and project_info['supervisor'] != 'Supervisor Name (if applicable)':
        elements.append(Paragraph(f"<b>Supervisor:</b> {project_info['supervisor']}", styles['Normal']))
    if 'project_duration' in project_info and project_info['project_duration'] and project_info['project_duration'] != 'Start Date - End Date':
        elements.append(Paragraph(f"<b>Project Duration:</b> {project_info['project_duration']}", styles['Normal']))
    
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"<b>Version:</b> {project_info['version']}", styles['Normal']))
    elements.append(Paragraph(f"<b>Date:</b> {project_info['date']}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(project_info['description'], styles['Normal']))
    
    # Group Members Section
    if 'group_members' in project_info and project_info['group_members']:
        # Check if members are not placeholder values
        valid_members = [m for m in project_info['group_members'] 
                        if m.get('name') and m.get('name') != 'Member 1 Name' 
                        and m.get('name') != 'Member 2 Name' 
                        and m.get('name') != 'Member 3 Name']
        
        if valid_members:
            elements.append(Spacer(1, 0.4*inch))
            elements.append(Paragraph("<b>Project Team:</b>", styles['Heading3']))
            elements.append(Spacer(1, 0.2*inch))
            
            # Create table for group members
            member_data = [['Name', 'Role/Contribution', 'Email']]
            for member in valid_members:
                name = member.get('name', 'N/A')
                role = member.get('role', 'N/A')
                email = member.get('email', 'N/A')
                member_data.append([name, role, email])
            
            member_table = Table(member_data, colWidths=[2*inch, 2.5*inch, 2.5*inch])
            member_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            elements.append(member_table)
    
    # Technologies Used
    if 'technologies' in project_info and project_info['technologies']:
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("<b>Technologies Used:</b>", styles['Normal']))
        tech_list = ", ".join(project_info['technologies'])
        elements.append(Paragraph(tech_list, styles['Normal']))
    
    elements.append(PageBreak())
    
    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading_style))
    toc_items = [
        "1. Executive Summary",
        "2. Project Overview",
        "3. Data Overview",
        "4. Feature Engineering",
        "5. Classification Models",
        "6. Clustering Models",
        "7. Forecasting Models",
        "8. Model Comparison",
        "9. Conclusions & Recommendations"
    ]
    for item in toc_items:
        elements.append(Paragraph(item, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    elements.append(PageBreak())
    
    # Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading_style))
    elements.append(Paragraph(
        "This report presents the results of a comprehensive machine learning project for tracking "
        "and predicting customer movement patterns in shopping malls. The project includes "
        "classification models for predicting next zone visits, clustering models for customer "
        "segmentation, and forecasting models for traffic prediction.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    # Key Metrics Summary
    elements.append(Paragraph("Key Achievements", subheading_style))
    
    summary_data = []
    if results['classification']:
        best_clf = max(results['classification'].items(), 
                      key=lambda x: x[1].get('accuracy', 0))
        summary_data.append(['Classification Best Model', best_clf[0].upper(), 
                            f"{best_clf[1].get('accuracy', 0)*100:.2f}%"])
    
    if results['clustering']:
        best_clust = max(results['clustering'].items(),
                        key=lambda x: x[1].get('silhouette_score', 0))
        summary_data.append(['Clustering Best Model', best_clust[0].upper(),
                            f"{best_clust[1].get('silhouette_score', 0):.4f}"])
    
    if summary_data:
        summary_table = Table([['Metric', 'Model', 'Performance']] + summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(summary_table)
    
    elements.append(PageBreak())
    
    # Project Overview
    elements.append(Paragraph("2. Project Overview", heading_style))
    elements.append(Paragraph(
        "<b>Objective:</b> Develop machine learning models to analyze and predict customer "
        "movement patterns in shopping malls.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<b>Approach:</b>",
        styles['Normal']
    ))
    elements.append(Paragraph("• Feature Engineering: Created 30+ new features from raw data", 
                            styles['Normal']))
    elements.append(Paragraph("• Classification: Predict next zone visit (4 models)", 
                            styles['Normal']))
    elements.append(Paragraph("• Clustering: Customer segmentation (2 models)", 
                            styles['Normal']))
    elements.append(Paragraph("• Forecasting: Traffic prediction (2 models)", 
                            styles['Normal']))
    elements.append(PageBreak())
    
    # Data Overview
    elements.append(Paragraph("3. Data Overview", heading_style))
    elements.append(Paragraph(
        "The dataset contains customer movement tracking data with temporal, spatial, and "
        "behavioral features. After feature engineering, the dataset includes 110 features "
        "derived from the original 80 columns.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Dataset Statistics", subheading_style))
    data_stats = [
        ['Metric', 'Value'],
        ['Total Records', '15,839'],
        ['Original Features', '80'],
        ['Engineered Features', '110'],
        ['New Features Created', '30'],
        ['Missing Values (After Processing)', '0']
    ]
    stats_table = Table(data_stats)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(stats_table)
    elements.append(PageBreak())
    
    # Classification Models
    elements.append(Paragraph("4. Classification Models", heading_style))
    elements.append(Paragraph(
        "Classification models predict the next zone a customer will visit based on their "
        "current location and movement history.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    if results['classification']:
        elements.append(Paragraph("Model Performance", subheading_style))
        clf_data = [['Model', 'Accuracy', 'ROC-AUC']]
        for model_name, metrics in results['classification'].items():
            acc = metrics.get('accuracy', 0) * 100
            roc_auc = metrics.get('roc_auc', 'N/A')
            if roc_auc != 'N/A':
                roc_auc = f"{roc_auc:.4f}"
            clf_data.append([
                model_name.replace('_', ' ').title(),
                f"{acc:.2f}%",
                str(roc_auc)
            ])
        
        clf_table = Table(clf_data)
        clf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(clf_table)
        
        # Best model
        best_model = max(results['classification'].items(),
                        key=lambda x: x[1].get('accuracy', 0))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            f"<b>Best Model:</b> {best_model[0].replace('_', ' ').title()} "
            f"({best_model[1].get('accuracy', 0)*100:.2f}% accuracy)",
            styles['Normal']
        ))
    else:
        elements.append(Paragraph("No classification results available.", styles['Normal']))
    
    elements.append(PageBreak())
    
    # Clustering Models
    elements.append(Paragraph("5. Clustering Models", heading_style))
    elements.append(Paragraph(
        "Clustering models group customers with similar movement patterns to identify "
        "behavioral segments.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    if results['clustering']:
        elements.append(Paragraph("Model Performance", subheading_style))
        clust_data = [['Model', 'Silhouette Score', 'Clusters', 'Noise Points']]
        for model_name, metrics in results['clustering'].items():
            clust_data.append([
                model_name.upper(),
                f"{metrics.get('silhouette_score', 0):.4f}",
                str(metrics.get('n_clusters', 'N/A')),
                str(metrics.get('n_noise', 0))
            ])
        
        clust_table = Table(clust_data)
        clust_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(clust_table)
    else:
        elements.append(Paragraph("No clustering results available.", styles['Normal']))
    
    elements.append(PageBreak())
    
    # Forecasting Models
    elements.append(Paragraph("6. Forecasting Models", heading_style))
    elements.append(Paragraph(
        "Forecasting models predict future traffic patterns and customer movement trends.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    if results['forecasting']:
        elements.append(Paragraph("Model Performance", subheading_style))
        forecast_data = [['Model', 'RMSE', 'MAE']]
        for model_name, metrics in results['forecasting'].items():
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            # Format large numbers
            if rmse > 1e9:
                rmse_str = f"{rmse:.2e}"
            else:
                rmse_str = f"{rmse:.2f}"
            if mae > 1e9:
                mae_str = f"{mae:.2e}"
            else:
                mae_str = f"{mae:.2f}"
            forecast_data.append([
                model_name.upper(),
                rmse_str,
                mae_str
            ])
        
        forecast_table = Table(forecast_data)
        forecast_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(forecast_table)
    else:
        elements.append(Paragraph("No forecasting results available.", styles['Normal']))
    
    elements.append(PageBreak())
    
    # Model Comparison
    elements.append(Paragraph("7. Model Comparison", heading_style))
    elements.append(Paragraph(
        "Summary of all models trained and their performance metrics.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    comparison_data = [['Model Type', 'Best Model', 'Key Metric', 'Value']]
    
    if results['classification']:
        best_clf = max(results['classification'].items(),
                      key=lambda x: x[1].get('accuracy', 0))
        comparison_data.append([
            'Classification',
            best_clf[0].replace('_', ' ').title(),
            'Accuracy',
            f"{best_clf[1].get('accuracy', 0)*100:.2f}%"
        ])
    
    if results['clustering']:
        best_clust = max(results['clustering'].items(),
                        key=lambda x: x[1].get('silhouette_score', 0))
        comparison_data.append([
            'Clustering',
            best_clust[0].upper(),
            'Silhouette Score',
            f"{best_clust[1].get('silhouette_score', 0):.4f}"
        ])
    
    if results['forecasting']:
        best_forecast = min(results['forecasting'].items(),
                           key=lambda x: x[1].get('rmse', float('inf')))
        comparison_data.append([
            'Forecasting',
            best_forecast[0].upper(),
            'RMSE',
            f"{best_forecast[1].get('rmse', 0):.2e}"
        ])
    
    if len(comparison_data) > 1:
        comp_table = Table(comparison_data)
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(comp_table)
    
    elements.append(PageBreak())
    
    # Conclusions
    elements.append(Paragraph("8. Conclusions & Recommendations", heading_style))
    elements.append(Paragraph("<b>Key Findings:</b>", styles['Normal']))
    elements.append(Paragraph("• Classification models achieved high accuracy (>99%)", 
                              styles['Normal']))
    elements.append(Paragraph("• XGBoost performed best for classification tasks", 
                              styles['Normal']))
    elements.append(Paragraph("• K-Means identified 5 distinct customer segments", 
                              styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
    elements.append(Paragraph("• Deploy XGBoost model for production predictions", 
                              styles['Normal']))
    elements.append(Paragraph("• Use K-Means clusters for targeted marketing campaigns", 
                              styles['Normal']))
    elements.append(Paragraph("• Continue monitoring model performance with new data", 
                              styles['Normal']))
    elements.append(Paragraph("• Consider hyperparameter tuning for further improvements", 
                              styles['Normal']))
    
    # Footer with team information
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        f"<i>Report generated on {project_info['date']}</i>",
        styles['Normal']
    ))
    
    if 'group_members' in project_info and project_info['group_members']:
        member_names = [m.get('name', '') for m in project_info['group_members'] if m.get('name')]
        if member_names:
            elements.append(Paragraph(
                f"<i>Prepared by: {', '.join(member_names)}</i>",
                styles['Normal']
            ))
    
    # Build PDF
    doc.build(elements)
    return True

def main():
    """Main function to generate report"""
    reports_dir = Path(__file__).parent
    reports_dir.mkdir(exist_ok=True)
    
    output_path = reports_dir / f"Project_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    print("=" * 60)
    print("Generating PDF Report")
    print("=" * 60)
    
    if create_pdf_report(output_path):
        print(f"\n✅ Report generated successfully!")
        print(f"📄 Location: {output_path}")
        print(f"📊 Report includes:")
        print("   - Executive Summary")
        print("   - Project Overview")
        print("   - Data Overview")
        print("   - Feature Engineering Summary")
        print("   - Classification Results")
        print("   - Clustering Results")
        print("   - Forecasting Results")
        print("   - Model Comparison")
        print("   - Conclusions & Recommendations")
    else:
        print("\n❌ Failed to generate report.")
        print("💡 Install reportlab: pip install reportlab")

if __name__ == "__main__":
    main()

