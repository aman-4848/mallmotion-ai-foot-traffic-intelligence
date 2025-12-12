"""
Generate Architecture Diagram for Mall Movement Tracking Project
Creates a comprehensive system architecture diagram as PNG
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080 at 100 DPI
ax.set_xlim(0, 19.2)
ax.set_ylim(0, 10.8)
ax.axis('off')

# Color scheme
COLORS = {
    'data': '#3B82F6',        # Blue
    'feature': '#10B981',     # Green
    'model': '#F59E0B',       # Orange
    'results': '#8B5CF6',     # Purple
    'app': '#EF4444',         # Red
    'support': '#6B7280',     # Gray
    'arrow': '#1F2937',       # Dark gray
    'bg': '#FFFFFF',          # White
    'text': '#111827'         # Dark text
}

# Helper function to create rounded rectangle
def create_box(ax, x, y, width, height, color, label, text_size=10):
    """Create a rounded rectangle box with label"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=1.5,
        alpha=0.8
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center',
            fontsize=text_size, weight='bold',
            color='white' if color != '#FFFFFF' else 'black',
            wrap=True)
    return box

# Helper function to create arrow
def create_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow'], style='solid'):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        mutation_scale=20,
        linewidth=2,
        color=color,
        linestyle=style,
        alpha=0.7
    )
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(9.6, 10.3, 'Mall Movement Tracking - System Architecture',
        ha='center', va='center',
        fontsize=24, weight='bold', color=COLORS['text'])

# ==================== LAYER 1: DATA LAYER (Bottom) ====================
y_data = 1.5
box_height = 0.8

# Data sources
create_box(ax, 1, y_data, 2.5, box_height, COLORS['data'], 
           'Raw Data\nmerged data set.csv', 9)
create_box(ax, 4, y_data, 2.5, box_height, COLORS['data'],
           'Engineered Data\nengineered_features.csv', 9)
create_box(ax, 7, y_data, 2.5, box_height, COLORS['data'],
           'Sample Data\ndata/sample/', 9)

# ==================== LAYER 2: FEATURE ENGINEERING ====================
y_feature = 3.5

create_box(ax, 2, y_feature, 3.5, box_height, COLORS['feature'],
           'Feature Engineering\nPipeline', 10)
create_box(ax, 6.5, y_feature, 2.5, box_height, COLORS['feature'],
           'Feature Config\nfeature_config.yaml', 9)

# Feature engineering steps (small boxes)
y_steps = 4.8
step_width = 1.8
steps = ['Missing\nValues', 'DateTime\nExtract', 'Categorical\nEncode', 
         'Outlier\nDetect', 'Domain\nFeatures', 'Binning']
for i, step in enumerate(steps):
    x_step = 1.5 + i * step_width
    create_box(ax, x_step, y_steps, 1.5, 0.6, COLORS['feature'],
               step, 8)

# ==================== LAYER 3: MODEL TRAINING ====================
y_training = 6.5

# Training scripts
create_box(ax, 1, y_training, 2.5, box_height, COLORS['model'],
           'Classification\nTraining', 9)
create_box(ax, 4.2, y_training, 2.5, box_height, COLORS['model'],
           'Clustering\nTraining', 9)
create_box(ax, 7.4, y_training, 2.5, box_height, COLORS['model'],
           'Forecasting\nTraining', 9)

# Models (above training)
y_models = 8.2
model_width = 1.8

# Classification models
models_class = ['RF', 'DT', 'XGB', 'SVM']
for i, model in enumerate(models_class):
    x_model = 1.2 + i * (model_width + 0.2)
    create_box(ax, x_model, y_models, model_width, 0.6, COLORS['model'],
               model, 8)

# Clustering models
models_cluster = ['K-Means', 'DBSCAN']
for i, model in enumerate(models_cluster):
    x_model = 4.4 + i * (model_width + 0.2)
    create_box(ax, x_model, y_models, model_width, 0.6, COLORS['model'],
               model, 8)

# Forecasting models
models_forecast = ['ARIMA', 'Prophet']
for i, model in enumerate(models_forecast):
    x_model = 7.6 + i * (model_width + 0.2)
    create_box(ax, x_model, y_models, model_width, 0.6, COLORS['model'],
               model, 8)

# ==================== LAYER 4: RESULTS ====================
y_results = 9.5

create_box(ax, 1, y_results, 2.5, box_height, COLORS['results'],
           'Classification\nResults', 9)
create_box(ax, 4.2, y_results, 2.5, box_height, COLORS['results'],
           'Clustering\nResults', 9)
create_box(ax, 7.4, y_results, 2.5, box_height, COLORS['results'],
           'Forecasting\nResults', 9)

# ==================== LAYER 5: APPLICATIONS (Top) ====================
y_app = 11.5

create_box(ax, 2, y_app, 3.5, box_height, COLORS['app'],
           'Streamlit\nDashboard', 10)
create_box(ax, 6.5, y_app, 3.5, box_height, COLORS['app'],
           'FastAPI\nREST API', 10)

# ==================== SUPPORTING COMPONENTS (Right Side) ====================
x_support = 12
y_support_start = 8

create_box(ax, x_support, y_support_start, 2.5, box_height, COLORS['support'],
           'Monitoring\nData Quality\nDrift Detection', 9)
create_box(ax, x_support, y_support_start - 2, 2.5, box_height, COLORS['support'],
           'Testing\nUnit Tests\nIntegration', 9)
create_box(ax, x_support, y_support_start - 4, 2.5, box_height, COLORS['support'],
           'Notebooks\nEDA, Analysis\nExperiments', 9)

# ==================== ARROWS (Data Flow) ====================

# Data → Feature Engineering
create_arrow(ax, 2.25, 2.3, 3.75, 3.5)
create_arrow(ax, 5.25, 2.3, 3.75, 3.5)
create_arrow(ax, 8.25, 2.3, 3.75, 3.5)

# Feature Engineering → Training
create_arrow(ax, 3.75, 4.3, 2.25, 6.5)
create_arrow(ax, 3.75, 4.3, 5.45, 6.5)
create_arrow(ax, 3.75, 4.3, 8.65, 6.5)

# Training → Models
create_arrow(ax, 2.25, 7.3, 2.1, 8.2)
create_arrow(ax, 2.25, 7.3, 3.9, 8.2)
create_arrow(ax, 2.25, 7.3, 5.7, 8.2)
create_arrow(ax, 2.25, 7.3, 7.5, 8.2)
create_arrow(ax, 5.45, 7.3, 5.3, 8.2)
create_arrow(ax, 5.45, 7.3, 7.1, 8.2)
create_arrow(ax, 8.65, 7.3, 8.5, 8.2)
create_arrow(ax, 8.65, 7.3, 10.3, 8.2)

# Models → Results
create_arrow(ax, 2.1, 8.8, 2.25, 9.5)
create_arrow(ax, 5.3, 8.8, 5.45, 9.5)
create_arrow(ax, 8.5, 8.8, 8.65, 9.5)

# Results → Applications
create_arrow(ax, 2.25, 10.3, 3.75, 11.5)
create_arrow(ax, 5.45, 10.3, 3.75, 11.5)
create_arrow(ax, 8.65, 10.3, 3.75, 11.5)
create_arrow(ax, 2.25, 10.3, 8, 11.5)
create_arrow(ax, 5.45, 10.3, 8, 11.5)
create_arrow(ax, 8.65, 10.3, 8, 11.5)

# Supporting components → All layers (dashed)
create_arrow(ax, 13.25, 8.4, 10, 9.5, style='dashed')
create_arrow(ax, 13.25, 6.4, 10, 7.3, style='dashed')
create_arrow(ax, 13.25, 4.4, 10, 4.3, style='dashed')

# ==================== LEGEND ====================
legend_x = 15.5
legend_y = 8
legend_items = [
    ('Data Layer', COLORS['data']),
    ('Feature Engineering', COLORS['feature']),
    ('Model Training', COLORS['model']),
    ('Results', COLORS['results']),
    ('Applications', COLORS['app']),
    ('Supporting', COLORS['support'])
]

for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - i * 0.4
    create_box(ax, legend_x, y_pos, 0.5, 0.3, color, '', 8)
    ax.text(legend_x + 0.6, y_pos + 0.15, label,
            fontsize=9, va='center', color=COLORS['text'])

# ==================== LAYER LABELS ====================
ax.text(0.3, 1.9, 'DATA LAYER', fontsize=12, weight='bold', 
        rotation=90, ha='center', va='center', color=COLORS['text'])
ax.text(0.3, 3.9, 'FEATURE ENGINEERING', fontsize=12, weight='bold',
        rotation=90, ha='center', va='center', color=COLORS['text'])
ax.text(0.3, 6.9, 'MODEL TRAINING', fontsize=12, weight='bold',
        rotation=90, ha='center', va='center', color=COLORS['text'])
ax.text(0.3, 9.9, 'RESULTS', fontsize=12, weight='bold',
        rotation=90, ha='center', va='center', color=COLORS['text'])
ax.text(0.3, 11.9, 'APPLICATIONS', fontsize=12, weight='bold',
        rotation=90, ha='center', va='center', color=COLORS['text'])

# ==================== SAVE ====================
output_path = Path(__file__).parent / 'architecture_diagram.png'

# Remove old file if exists
if output_path.exists():
    output_path.unlink()

plt.tight_layout()

# Save with explicit format and parameters
try:
    # Ensure we're using the correct backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Save the figure
    plt.savefig(str(output_path), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.1,
                transparent=False)
    
    # Flush to ensure file is written
    plt.savefig(str(output_path), format='png')
    
    # Verify file was created
    if output_path.exists() and output_path.stat().st_size > 0:
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✅ Architecture diagram saved successfully!")
        print(f"   Location: {output_path}")
        print(f"   Size: {file_size:.2f} KB")
        print(f"   Dimensions: 1920x1080 pixels (300 DPI)")
    else:
        print(f"❌ Error: File was not created properly")
except Exception as e:
    print(f"❌ Error saving diagram: {e}")
    import traceback
    traceback.print_exc()
finally:
    plt.close('all')  # Close all figures
    import matplotlib.pyplot as plt
    plt.close('all')  # Ensure all figures are closed


