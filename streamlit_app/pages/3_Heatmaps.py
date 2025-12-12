"""
Heatmaps Page
Visualize movement patterns using heatmaps and map-type visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Try to import plotly for interactive maps
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Install with: pip install plotly")

st.title("🗺️ Movement Heatmaps")
st.markdown("---")
# Load data
with st.spinner("Loading data..."):
    try:
        df = load_processed_data()
        fe = FeatureEngineer()
        df = fe.engineer_features(df)
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
st.markdown("---")
# Zone Popularity Heatmap
st.header("🔥 Zone Popularity Heatmap")
# Detect zone column
zone_cols = [col for col in df.columns if 'zone' in col.lower() or 'space' in col.lower() or 'location' in col.lower()]
user_cols = [col for col in df.columns if 'user' in col.lower() or 'id' in col.lower()]
if zone_cols and user_cols:
    zone_col = zone_cols[0]
    user_col = user_cols[0]
    # Zone popularity
    zone_counts = df[zone_col].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(zone_counts.values.reshape(-1, 1), 
               yticklabels=zone_counts.index,
               cmap='YlOrRd', 
               annot=True, 
               fmt='d',
               cbar_kws={'label': 'Visit Count'},
               ax=ax)
    ax.set_title('Top 20 Most Popular Zones', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")
    # Zone-User Interaction Heatmap
    st.header("👥 Zone-User Interaction Heatmap")
    top_n_zones = st.slider("Select top N zones:", 5, 20, 10)
    top_n_users = st.slider("Select top N users:", 5, 20, 10)
    top_zones = df[zone_col].value_counts().head(top_n_zones).index
    top_users = df[user_col].value_counts().head(top_n_users).index
    df_filtered = df[df[zone_col].isin(top_zones) & df[user_col].isin(top_users)]
    heatmap_data = df_filtered.groupby([zone_col, user_col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data, 
               cmap='YlOrRd', 
               annot=False,
               cbar_kws={'label': 'Visit Count'},
               ax=ax)
    ax.set_title(f'Zone-User Interaction Heatmap (Top {top_n_zones} Zones × Top {top_n_users} Users)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('User', fontsize=12)
    ax.set_ylabel('Zone', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")
    # Temporal Heatmap
    st.header("⏰ Temporal Movement Heatmap")
    # Check for temporal features
    temporal_cols = [col for col in df.columns if any(x in col.lower() for x in ['hour', 'day', 'week', 'month'])]
    if temporal_cols:
        time_col = st.selectbox("Select time dimension:", temporal_cols)
        if time_col in df.columns:
            time_zone_data = df.groupby([time_col, zone_col]).size().unstack(fill_value=0)
            # Select top zones
            top_zones_for_time = df[zone_col].value_counts().head(15).index
            time_zone_data = time_zone_data[top_zones_for_time]
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(time_zone_data.T, 
                       cmap='YlOrRd', 
                       annot=False,
                       cbar_kws={'label': 'Visit Count'},
                       ax=ax)
            ax.set_title(f'Movement Patterns by {time_col}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(time_col, fontsize=12)
            ax.set_ylabel('Zone', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
else:
    st.warning("Zone or user columns not detected. Heatmaps may not work correctly.")
st.markdown("---")
# Map-Type Visualizations
st.header("🗺️ Map-Type Visualizations")

if zone_cols and PLOTLY_AVAILABLE:
    map_type = st.selectbox(
        "Select map visualization type:",
        ["Zone Layout Map", "Zone Network Graph", "Zone Density Map", "Interactive Zone Map"]
    )
    
    if map_type == "Zone Layout Map":
        # Create a grid-based zone layout map
        zone_counts = df[zone_col].value_counts()
        top_zones = zone_counts.head(30)
        
        # Create grid coordinates for zones
        n_zones = len(top_zones)
        grid_size = int(np.ceil(np.sqrt(n_zones)))
        
        # Create heatmap data
        heatmap_data = np.zeros((grid_size, grid_size))
        zone_labels = {}
        
        for idx, (zone_id, count) in enumerate(top_zones.items()):
            row = idx // grid_size
            col = idx % grid_size
            if row < grid_size and col < grid_size:
                heatmap_data[row, col] = count
                zone_labels[(row, col)] = f"Zone {zone_id}"
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale='Viridis',
            text=[[zone_labels.get((i, j), '') for j in range(grid_size)] for i in range(grid_size)],
            texttemplate='%{text}<br>%{z}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Zone Layout Map - Top 30 Zones',
            xaxis_title='Column',
            yaxis_title='Row',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif map_type == "Zone Network Graph":
        # Create network graph showing zone connections
        # Get zone transitions
        if 'USERID' in df.columns:
            user_col = 'USERID'
        else:
            user_col = user_cols[0] if user_cols else None
        
        if user_col:
            # Create transition matrix
            transitions = []
            for user_id in df[user_col].unique()[:100]:  # Limit for performance
                user_data = df[df[user_col] == user_id].sort_values(by=df.columns[0] if len(df.columns) > 0 else zone_col)
                zones = user_data[zone_col].values
                for i in range(len(zones) - 1):
                    transitions.append((zones[i], zones[i+1]))
            
            if transitions:
                transition_df = pd.DataFrame(transitions, columns=['from_zone', 'to_zone'])
                transition_counts = transition_df.groupby(['from_zone', 'to_zone']).size().reset_index(name='count')
                
                # Create network graph
                fig = go.Figure()
                
                # Get unique zones
                all_zones = list(set(transition_counts['from_zone'].unique()) | set(transition_counts['to_zone'].unique()))
                zone_positions = {}
                
                # Create circular layout
                n = len(all_zones)
                for i, zone in enumerate(all_zones):
                    angle = 2 * np.pi * i / n
                    zone_positions[zone] = (np.cos(angle), np.sin(angle))
                
                # Add edges
                for _, row in transition_counts.head(50).iterrows():  # Limit edges
                    from_zone = row['from_zone']
                    to_zone = row['to_zone']
                    count = row['count']
                    
                    x0, y0 = zone_positions[from_zone]
                    x1, y1 = zone_positions[to_zone]
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=count/10, color='rgba(56, 189, 248, 0.3)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                # Add nodes
                for zone, (x, y) in zone_positions.items():
                    zone_visits = df[df[zone_col] == zone].shape[0]
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(size=zone_visits/100, color='#38BDF8', line=dict(width=2, color='#FFFFFF')),
                        text=[f"Zone {zone}"],
                        textposition="middle center",
                        name=f"Zone {zone}",
                        hovertemplate=f"Zone {zone}<br>Visits: {zone_visits}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title='Zone Network Graph - Movement Patterns',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif map_type == "Zone Density Map":
        # Create density-based zone map
        zone_counts = df[zone_col].value_counts()
        
        # Create scatter plot with zone density
        fig = go.Figure()
        
        # Create positions for zones (circular layout)
        n_zones = len(zone_counts)
        for i, (zone_id, count) in enumerate(zone_counts.head(50).items()):
            angle = 2 * np.pi * i / n_zones
            x = np.cos(angle) * (count / zone_counts.max())
            y = np.sin(angle) * (count / zone_counts.max())
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=count/50,
                    color=count,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Visit Count")
                ),
                text=[f"Zone {zone_id}"],
                textposition="middle center",
                name=f"Zone {zone_id}",
                hovertemplate=f"Zone {zone_id}<br>Visits: {count}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Zone Density Map - Visit Frequency',
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif map_type == "Interactive Zone Map":
        # Create interactive choropleth-style map
        # Count visits per zone
        zone_stats = df.groupby(zone_col).size().to_frame('visit_count')
        
        # Add unique users if user_col exists
        if user_col:
            zone_stats['unique_users'] = df.groupby(zone_col)[user_col].nunique()
        else:
            zone_stats['unique_users'] = zone_stats['visit_count']
        
        # Create bar chart with zone statistics
        fig = go.Figure()
        
        top_n = st.slider("Show top N zones:", 10, 50, 20, key="top_zones_map")
        top_zones = zone_stats.nlargest(top_n, 'visit_count')
        
        fig.add_trace(go.Bar(
            x=[f"Zone {idx}" for idx in top_zones.index],
            y=top_zones['visit_count'],
            name='Visit Count',
            marker_color='#38BDF8',
            text=top_zones['visit_count'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Zone {idx}" for idx in top_zones.index],
            y=top_zones['unique_users'],
            name='Unique Users',
            marker_color='#1E293B',
            text=top_zones['unique_users'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Interactive Zone Map - Top {top_n} Zones',
            xaxis_title='Zone',
            yaxis_title='Count',
            barmode='group',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Zone details table
        st.subheader("Zone Details")
        display_stats = top_zones.reset_index()
        display_stats.columns = ['Zone', 'Visit Count', 'Unique Users']
        st.dataframe(display_stats, use_container_width=True)

elif not PLOTLY_AVAILABLE:
    st.info("💡 Install Plotly for map-type visualizations: `pip install plotly`")
    st.markdown("---")

st.markdown("---")
# Statistics
st.header("📊 Heatmap Statistics")
if zone_cols:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Zones", df[zone_col].nunique())
    with col2:
        st.metric("Total Visits", len(df))
    with col3:
        st.metric("Avg Visits/Zone", f"{len(df) / df[zone_col].nunique():.1f}")