import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="🏏 IPL Player Battle Analyzer",
    page_icon="🏏",
    layout="wide"
)

@st.cache_data(ttl=3600, show_spinner="Loading IPL data...")
def load_data():
    """Load data from local CSV files"""
    try:
        deliveries = pd.read_csv('deliveries.csv')
        matches = pd.read_csv('matches.csv')

        # Merge with match details
        merged = deliveries.merge(
            matches[['id', 'season', 'venue', 'date']],
            left_on='match_id',
            right_on='id',
            how='left'
        )

        # Convert date format
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')

        return merged.drop(columns='id'), matches

    except FileNotFoundError:
        st.error("Error: Missing files - Ensure 'deliveries.csv' and 'matches.csv' exist!")
        return None, None
    except Exception as e:
        st.error(f"Data Loading Error: {str(e)}")
        return None, None

def calculate_pvp_stats(data, batter, bowler, venue_filter=None):
    """Compute head-to-head stats between a batter and a bowler"""
    if data is None:
        return None, None
    
    filtered = data.copy()
    
    if venue_filter and venue_filter != "All":
        filtered = filtered[filtered['venue'] == venue_filter]
    
    matchup = filtered[(filtered['batter'] == batter) & (filtered['bowler'] == bowler)]
    
    if matchup.empty:
        return None, None
    
    # Core Stats
    total_runs = matchup['batsman_runs'].sum()
    total_balls = matchup.shape[0]
    dismissals = matchup['is_wicket'].sum()
    
    # Advanced Stats
    dot_balls = (matchup['batsman_runs'] == 0).sum()
    boundaries = matchup[matchup['batsman_runs'].isin([4, 6])].shape[0]

    # Calculations
    strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
    average = total_runs / dismissals if dismissals > 0 else np.inf
    economy = (total_runs / total_balls) * 6 if total_balls > 0 else 0
    boundary_percent = (boundaries / total_balls) * 100 if total_balls > 0 else 0
    dot_percent = (dot_balls / total_balls) * 100 if total_balls > 0 else 0

    # Yearly Strike Rate
    yearly_stats = matchup.groupby('season').agg(
        total_runs=('batsman_runs', 'sum'),
        total_balls=('batsman_runs', 'count')
    ).reset_index()
    yearly_stats['strike_rate'] = (yearly_stats['total_runs'] / yearly_stats['total_balls']) * 100

    return {
        'total_runs': total_runs,
        'total_balls': total_balls,
        'dismissals': dismissals,
        'strike_rate': strike_rate,
        'average': average,
        'economy': economy,
        'boundary_percent': boundary_percent,
        'dot_percent': dot_percent,
        'match_data': matchup
    }, yearly_stats

def create_visualizations(stats, batter, bowler):
    """Create interactive visualizations"""
    if not stats:
        return None
        
    # Create subplots with adjusted layout
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"colspan": 2}, None]],
        subplot_titles=(f"{batter} Batting Metrics", f"{bowler} Bowling Metrics", 
                       "Match Progression"),
        vertical_spacing=0.3,
        horizontal_spacing=0.2,
        row_heights=[0.4, 0.6]
    )

    # Batting indicator with smaller font
    fig.add_trace(go.Indicator(
        mode="number",
        value=stats['strike_rate'],
        title={
            'text': "Strike Rate",
            'font': {'size': 16}
        },
        number={
            'font': {'size': 24},
            'valueformat': ".1f"
        },
        domain={'row': 1, 'column': 1}
    ), row=1, col=1)

    # Bowling indicator with smaller font
    fig.add_trace(go.Indicator(
        mode="number",
        value=stats['economy'],
        title={
            'text': "Economy Rate",
            'font': {'size': 16}
        },
        number={
            'font': {'size': 24},
            'suffix': " RPO",
            'valueformat': ".1f"
        },
        domain={'row': 1, 'column': 2}
    ), row=1, col=2)

    # Match progression
    match_stats = stats['match_data'].groupby('date').agg(
        runs=('batsman_runs', 'sum')
    ).reset_index()
    
    fig.add_trace(go.Scatter(
        x=match_stats['date'],
        y=match_stats['runs'].cumsum(),
        mode='lines+markers',
        name='Cumulative Runs',
        line=dict(color='#FF4B4B')
    ), row=2, col=1)

    # Update layout with better spacing
    fig.update_layout(
        height=500,
        margin=dict(t=60, b=20, l=20, r=20),
        showlegend=False,
        template="plotly_white",
        font=dict(family="Arial")
    )
    
    return fig

def plot_strike_rate_vs_year(yearly_stats):
    """Create a line chart for strike rate trends"""
    if yearly_stats.empty:
        return None

    fig = px.line(
        yearly_stats,
        x='season',
        y='strike_rate',
        markers=True,
        title="📊 Strike Rate Over the Years",
        labels={'season': 'Year', 'strike_rate': 'Strike Rate'},
        line_shape='linear'
    )
    fig.update_traces(line=dict(color='blue', width=2))
    fig.update_layout(template="plotly_white")

    return fig

# Streamlit UI
st.title("🏏 IPL Player Battle Analyzer")
data, matches = load_data()

if data is not None:
    with st.sidebar:
        st.header("⚙️ Select Players")
        
        batters = sorted(data['batter'].unique())
        bowlers = sorted(data['bowler'].unique())

        batter = st.selectbox("Choose Batter", batters, 
                              index=batters.index('V Kohli') if 'V Kohli' in batters else 0)
        bowler = st.selectbox("Choose Bowler", bowlers, 
                              index=bowlers.index('JJ Bumrah') if 'JJ Bumrah' in bowlers else 0)
        
        venues = ["All"] + sorted(data['venue'].unique())
        venue_filter = st.selectbox("Filter by Venue", venues)

    # Compute stats
    stats, yearly_stats = calculate_pvp_stats(data, batter, bowler, venue_filter)

    if stats:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏏 Batting Stats")
            st.metric("Total Runs", stats['total_runs'])
            st.metric("Boundary %", f"{stats['boundary_percent']:.1f}%")
            st.metric("Dot Ball %", f"{stats['dot_percent']:.1f}%")

        with col2:
            st.subheader("🎯 Bowling Stats")
            st.metric("Economy Rate", f"{stats['economy']:.1f}")
            st.metric("Dismissals", stats['dismissals'])
            st.metric("Balls Bowled", stats['total_balls'])

        st.divider()
        st.plotly_chart(create_visualizations(stats, batter, bowler), use_container_width=True)

        # Strike Rate vs Year Chart
        if yearly_stats is not None and not yearly_stats.empty:
            st.plotly_chart(plot_strike_rate_vs_year(yearly_stats), use_container_width=True)

        # Show match details
        with st.expander("📜 Detailed Match Data"):
            st.dataframe(stats['match_data'][['season', 'venue', 'over', 'ball', 
                                              'batsman_runs', 'is_wicket']], height=250)
    else:
        st.warning("No historical matchups found for the selected players.")
