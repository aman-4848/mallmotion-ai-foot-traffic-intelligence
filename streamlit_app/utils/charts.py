"""
Chart Utilities
Reusable visualization functions for Streamlit dashboard
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Dark theme colors
DARK_THEME_COLORS = {
    'main_bg': '#0F172A',
    'sidebar_bg': '#1E293B',
    'text_color': '#FFFFFF',
    'accent_color': '#38BDF8',
    'card_bg': '#1A2238'
}

def setup_style():
    """Setup matplotlib and seaborn style for dark theme"""
    # Set dark background style
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {
        'axes.facecolor': DARK_THEME_COLORS['card_bg'],
        'figure.facecolor': DARK_THEME_COLORS['main_bg'],
        'axes.edgecolor': DARK_THEME_COLORS['accent_color'],
        'axes.labelcolor': DARK_THEME_COLORS['text_color'],
        'xtick.color': DARK_THEME_COLORS['text_color'],
        'ytick.color': DARK_THEME_COLORS['text_color'],
        'text.color': DARK_THEME_COLORS['text_color'],
        'grid.color': DARK_THEME_COLORS['sidebar_bg']
    })
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = DARK_THEME_COLORS['main_bg']
    plt.rcParams['axes.facecolor'] = DARK_THEME_COLORS['card_bg']
    plt.rcParams['axes.edgecolor'] = DARK_THEME_COLORS['accent_color']
    plt.rcParams['axes.labelcolor'] = DARK_THEME_COLORS['text_color']
    plt.rcParams['xtick.color'] = DARK_THEME_COLORS['text_color']
    plt.rcParams['ytick.color'] = DARK_THEME_COLORS['text_color']
    plt.rcParams['text.color'] = DARK_THEME_COLORS['text_color']

def create_bar_chart(data, x, y, title, color=None, figsize=(10, 6)):
    """Create a bar chart with dark theme"""
    setup_style()
    if color is None:
        color = DARK_THEME_COLORS['accent_color']
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_THEME_COLORS['main_bg'])
    ax.set_facecolor(DARK_THEME_COLORS['card_bg'])
    ax.bar(data[x], data[y], color=color, alpha=0.8, edgecolor=DARK_THEME_COLORS['accent_color'])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=DARK_THEME_COLORS['accent_color'])
    ax.set_xlabel(x, fontsize=12, color=DARK_THEME_COLORS['text_color'])
    ax.set_ylabel(y, fontsize=12, color=DARK_THEME_COLORS['text_color'])
    ax.grid(True, alpha=0.3, axis='y', color=DARK_THEME_COLORS['sidebar_bg'])
    plt.xticks(rotation=45, ha='right', color=DARK_THEME_COLORS['text_color'])
    plt.yticks(color=DARK_THEME_COLORS['text_color'])
    plt.tight_layout()
    return fig

def create_heatmap(data, title, cmap='viridis', figsize=(12, 8)):
    """Create a heatmap with dark theme"""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_THEME_COLORS['main_bg'])
    ax.set_facecolor(DARK_THEME_COLORS['card_bg'])
    sns.heatmap(data, cmap=cmap, annot=True, fmt='d', 
                cbar_kws={'label': 'Count'}, ax=ax,
                cmap='viridis' if cmap == 'viridis' else cmap)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=DARK_THEME_COLORS['accent_color'])
    plt.tight_layout()
    return fig

def create_line_plot(data, x, y, title, figsize=(12, 6)):
    """Create a line plot with dark theme"""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_THEME_COLORS['main_bg'])
    ax.set_facecolor(DARK_THEME_COLORS['card_bg'])
    ax.plot(data[x], data[y], linewidth=2, color=DARK_THEME_COLORS['accent_color'])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=DARK_THEME_COLORS['accent_color'])
    ax.set_xlabel(x, fontsize=12, color=DARK_THEME_COLORS['text_color'])
    ax.set_ylabel(y, fontsize=12, color=DARK_THEME_COLORS['text_color'])
    ax.grid(True, alpha=0.3, color=DARK_THEME_COLORS['sidebar_bg'])
    plt.xticks(rotation=45, ha='right', color=DARK_THEME_COLORS['text_color'])
    plt.yticks(color=DARK_THEME_COLORS['text_color'])
    plt.tight_layout()
    return fig

