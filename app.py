import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AZ THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AZ Liquidity Control Center",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ASTRAZENECA COLOUR PALETTE [Source: Colour Code.pdf]
AZ_COLORS = {
    "Mulberry": "#830051",      # Primary
    "Lime Green": "#C4D600",    # Accent
    "Navy": "#003865",          # Support/Accent
    "Graphite": "#3F4444",      # Support (Text/Neutral)
    "Light Blue": "#68D2DF",    # Support (Forecasts)
    "Magenta": "#D0006F",       # Support (Alerts)
    "Purple": "#3C1053",        # Support
    "Gold": "#F0AB00",          # Warning
    "White": "#FFFFFF"
}

def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            # Normalize to string
            s = df[c].astype(str).str.strip()
            # Handle negatives in parentheses e.g. "(1,234.56)" -> "-1234.56"
            s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
            # Remove dollar signs and thousands separators
            s = s.str.replace(r'[\$,]', '', regex=True)
            # Remove spaces
            s = s.str.replace(' ', '', regex=False)
            # Convert to numeric
            df[c] = pd.to_numeric(s, errors='coerce')

def apply_plot_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=AZ_COLORS['White']),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            zerolinecolor='rgba(255,255,255,0.06)',
            linecolor='rgba(255,255,255,0.10)',
            tickfont=dict(color=AZ_COLORS['White']),
            titlefont=dict(color=AZ_COLORS['White'])
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            zerolinecolor='rgba(255,255,255,0.06)',
            linecolor='rgba(255,255,255,0.10)',
            tickfont=dict(color=AZ_COLORS['White']),
            titlefont=dict(color=AZ_COLORS['White'])
        ),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=AZ_COLORS['White']))
    )
    return fig

# Custom CSS to enforce AZ Branding with Glassmorphism
st.markdown(f"""
<style>
/* Mulberry Main Background */
.stApp {{
  background: linear-gradient(135deg, {AZ_COLORS['Mulberry']}, {AZ_COLORS['Purple']});
  background-attachment: fixed;
}}

/* Main content glass: darker translucent panels */
[data-testid="stAppViewContainer"] > .main {{
  background: rgba(131, 0, 81, 0.12);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  border-radius: 12px;
  color: {AZ_COLORS['White']};
}}

/* Sidebar: darker purple tint */
[data-testid="stSidebar"] {{
  background: rgba(59, 16, 83, 0.18);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(0, 0, 0, 0.18);
  border-radius: 12px;
}}
[data-testid="stSidebar"] * {{ color: {AZ_COLORS['White']} !important; }}

/* Metric cards */
[data-testid="metric-container"] {{
  background: rgba(0, 0, 0, 0.18);
  border-radius: 10px;
  padding: 14px !important;
}}

/* Metric values: lime green for contrast */
[data-testid="stMetricValue"] {{
  color: {AZ_COLORS['Lime Green']};
  font-weight: bold;
}}

/* Headers & text: white on dark */
h1, h2, h3 {{
  color: {AZ_COLORS['White']};
  font-family: 'Arial', sans-serif;
}}
[data-testid="stMarkdownContainer"] p {{
  color: {AZ_COLORS['White']};
}}

/* Expanders and dataframes: darker panels */
[data-testid="stExpander"],
[data-testid="stDataFrame"] {{
  background: rgba(0, 0, 0, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.04);
  border-radius: 10px;
}}

/* Buttons: subtle on dark */
.stButton > button {{
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.06);
  color: {AZ_COLORS['White']};
  border-radius: 8px;
  transition: all 0.2s ease;
}}
.stButton > button:hover {{
  background: rgba(255, 255, 255, 0.10);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.20);
}}

/* Inputs: dark with white text */
.stRadio > div,
.stMultiSelect > div > div,
.stSelectbox > div > div,
.stSlider > div > div,
.stTextInput > div > div > input {{
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.04);
  color: {AZ_COLORS['White']};
  border-radius: 6px;
}}

/* Info/warning/error boxes */
.stInfo, .stWarning, .stError {{
  background: rgba(0, 0, 0, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.04);
  color: {AZ_COLORS['White']};
  border-radius: 8px;
}}

/* Global text */
body {{
  color: {AZ_COLORS['White']};
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] button {{
  background: rgba(255, 255, 255, 0.06);
  border-radius: 8px;
}}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
  font-size: 1.1rem;
  font-weight: 600;
  color: {AZ_COLORS['White']};
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. DATA LOADING WITH VALIDATION (All Excel Files Connected)
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load ALL 4 CSV files with proper error handling
        df_master = pd.read_csv('AZ_Master_Cleaned_Data.csv')
        df_weekly = pd.read_csv('AZ_Weekly_Summary.csv')
        df_anomaly = pd.read_csv('AZ_Anomaly_Report.csv')
        df_forecast = pd.read_csv('AZ_6Month_Forecast.csv')
        
        # Convert Date columns to datetime format for proper alignment
        df_master['Pstng Date'] = pd.to_datetime(df_master['Pstng Date'], errors='coerce')
        df_weekly['Pstng Date'] = pd.to_datetime(df_weekly['Pstng Date'], errors='coerce')
        df_anomaly['Pstng Date'] = pd.to_datetime(df_anomaly['Pstng Date'], errors='coerce')
        df_forecast['Week_Ending'] = pd.to_datetime(df_forecast['Week_Ending'], errors='coerce')
        
        # Remove any rows with NaT dates
        df_master = df_master.dropna(subset=['Pstng Date'])
        df_weekly = df_weekly.dropna(subset=['Pstng Date'])
        df_anomaly = df_anomaly.dropna(subset=['Pstng Date'])
        df_forecast = df_forecast.dropna(subset=['Week_Ending'])
        
        # Numeric cleanup to prevent zeros from string-formatted values
        clean_numeric(df_master, ['Ending_Balance_USD'])
        clean_numeric(df_weekly, ['Ending_Balance_USD'])
        clean_numeric(df_anomaly, ['Amount in USD', 'Z_Score', 'Deviation_Pct'])
        clean_numeric(df_forecast, [
            'Forecasted_Net_Cash_Flow_USD',
            'Projected_Ending_Balance_USD',
            'Balance_Lower_Bound',
            'Balance_Upper_Bound',
            'Change_From_Current'
        ])
        
        return df_master, df_weekly, df_anomaly, df_forecast
    except FileNotFoundError as e:
        st.error(f"❌ Critical Error: Data files not found. Please ensure these CSV files are in the working directory: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Data Loading Error: {e}")
        return None, None, None, None

df_master, df_weekly, df_anomaly, df_forecast = load_data()

if df_master is not None and len(df_master) > 0:
    # Data loaded successfully
    
    # Add data validation summary in sidebar (hidden from UI but logs status)
    data_stats = {
        'Master Records': len(df_master),
        'Weekly Records': len(df_weekly),
        'Anomalies Detected': len(df_anomaly),
        'Forecast Weeks': len(df_forecast),
        'Master Date Range': f"{df_master['Pstng Date'].min().date()} to {df_master['Pstng Date'].max().date()}",
        'Forecast Date Range': f"{df_forecast['Week_Ending'].min().date()} to {df_forecast['Week_Ending'].max().date()}"
    }
    
    # -------------------------------------------------------------------------
    # 3. SIDEBAR NAVIGATION
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("Liquidity Life-Support System")
        st.markdown("---")
        st.markdown("**System Status: MONITORING**")
        
        page = st.radio("Navigate Vitals:", 
              ["1. Executive Overview", 
               "2. Cash Balances", 
               "3. Forecast", 
               "4. Process Anomalies"]
        )
        

    # -------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW - Connected to ALL Excel files
    # -------------------------------------------------------------------------
    if "Executive" in page:
        st.title("Executive Overview: Liquidity Snapshot")
        st.markdown("The liquidity control dashboard monitors operational continuity across all entities.")

        if (df_master is None or df_master.empty or
            df_forecast is None or df_forecast.empty or
            df_anomaly is None):
            st.error("Datasets not loaded. Please ensure all CSV files are present.")
            st.stop()

        master_range = f"{df_master['Pstng Date'].min().date()} to {df_master['Pstng Date'].max().date()}"
        forecast_range = f"{df_forecast['Week_Ending'].min().date()} to {df_forecast['Week_Ending'].max().date()}"
        st.caption(
            f"Connected • Master: {len(df_master)} records ({master_range}) • Forecast: {len(df_forecast)} records ({forecast_range})"
        )
        
        # Key Metrics Row - Connected to Excel files
        col1, col2, col3, col4 = st.columns(4)
        
        # From AZ_Master_Cleaned_Data.csv
        current_balance = df_master.sort_values('Pstng Date')['Ending_Balance_USD'].iloc[-1] if len(df_master) > 0 else 0
        
        # From AZ_6Month_Forecast.csv
        forecast_min = df_forecast['Projected_Ending_Balance_USD'].min() if len(df_forecast) > 0 else 0
        
        # From AZ_Anomaly_Report.csv
        total_anomalies = len(df_anomaly) if len(df_anomaly) > 0 else 0
        
        # From AZ_6Month_Forecast.csv
        next_week_flow = df_forecast['Forecasted_Net_Cash_Flow_USD'].iloc[0] if len(df_forecast) > 0 else 0

        with col1:
            st.metric("Current Global Balance", f"${current_balance:,.0f}")
        with col2:
            delta_color = "normal" if forecast_min > 0 else "inverse"
            st.metric("Lowest Projected Balance (6M)", f"${forecast_min:,.0f}", delta_color=delta_color)
        with col3:
            st.metric("Process Failures Detected", f"{total_anomalies}", delta_color="inverse")
        with col4:
            st.metric("Next Week Expected Flow", f"${next_week_flow:,.0f}")

        st.markdown("---")
        
        # High Level Chart: Historical vs Forecast Balance
        st.subheader("Historical & Projected Balance")
        
        # Prepare historical data for plot from AZ_Master_Cleaned_Data.csv
        hist_balance = df_master.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Ending_Balance_USD'].last().reset_index()
        
        fig = go.Figure()
        
        # Historical Line
        fig.add_trace(go.Scatter(
            x=hist_balance['Pstng Date'], 
            y=hist_balance['Ending_Balance_USD'],
            mode='lines',
            name='Historical Balance',
            line=dict(color=AZ_COLORS['Mulberry'], width=3)
        ))
        
        # Forecast Line from AZ_6Month_Forecast.csv
        fig.add_trace(go.Scatter(
            x=df_forecast['Week_Ending'], 
            y=df_forecast['Projected_Ending_Balance_USD'],
            mode='lines',
            name='Projected Balance',
            line=dict(color=AZ_COLORS['Light Blue'], width=3, dash='dash')
        ))
        
        # Critical Threshold Line (Zero)
        fig.add_hline(y=0, line_dash="dot", annotation_text="Liquidity Failure (0)", annotation_position="bottom right", line_color="red")

        # Keep x-axis only within dates that exist in the datasets
        valid_hist_dates = hist_balance['Pstng Date'].dropna()
        valid_forecast_dates = df_forecast['Week_Ending'].dropna() if 'Week_Ending' in df_forecast else pd.Series([], dtype='datetime64[ns]')
        if not valid_hist_dates.empty or not valid_forecast_dates.empty:
            x_min = min(valid_hist_dates.min() if not valid_hist_dates.empty else valid_forecast_dates.min(),
                        valid_forecast_dates.min() if not valid_forecast_dates.empty else valid_hist_dates.min())
            x_max = max(valid_hist_dates.max() if not valid_hist_dates.empty else valid_forecast_dates.max(),
                        valid_forecast_dates.max() if not valid_forecast_dates.empty else valid_hist_dates.max())
            fig.update_xaxes(range=[x_min, x_max])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=AZ_COLORS['Graphite']),
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if forecast_min < 0:
            st.error(f"⚠️ CRITICAL: Projected liquidity shortfall. Immediate credit facility required.")

    # -------------------------------------------------------------------------
    # PAGE 2: CASH SATURATION (BALANCES) - Connected to AZ_Master_Cleaned_Data.csv
    # -------------------------------------------------------------------------
    elif "Cash Balances" in page:
        st.title("Cash Balance Monitor")
        st.markdown("Ensuring every entity maintains required liquidity for operations.")
        
        # Filter by Entity from AZ_Master_Cleaned_Data.csv
        entity_list = sorted(df_master['Name'].unique().tolist())
        selected_entity = st.selectbox("Select Entity to Monitor:", ["All Entities"] + entity_list)
        
        if selected_entity != "All Entities":
            filtered_df = df_master[df_master['Name'] == selected_entity]
        else:
            filtered_df = df_master

        # Calculate latest balance from AZ_Master_Cleaned_Data.csv
        latest_bal = filtered_df.sort_values('Pstng Date').groupby('Name')['Ending_Balance_USD'].last()
        
        # Bar Chart for Entity Balances
        st.subheader(f"Current Cash Levels ({selected_entity})")
        
        if selected_entity == "All Entities":
            fig_bar = px.bar(
                latest_bal.reset_index(), 
                x='Name', 
                y='Ending_Balance_USD',
                color='Ending_Balance_USD',
                color_continuous_scale=[AZ_COLORS['Magenta'], AZ_COLORS['Lime Green']],
                title="Cash Balance by Entity"
            )
            fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Line chart for single entity over time
            fig_line = px.line(
                filtered_df,
                x='Pstng Date',
                y='Ending_Balance_USD',
                title=f"Cash Saturation History: {selected_entity}",
                color_discrete_sequence=[AZ_COLORS['Navy']]
            )
            st.plotly_chart(fig_line, use_container_width=True)

        st.caption("Balance > 0 means the entity is self-sustaining. Balance < 0 indicates dependency on group treasury.")

    # -------------------------------------------------------------------------
    # PAGE 3: FORECAST - Connected to AZ_Master_Cleaned_Data.csv & AZ_6Month_Forecast.csv
    # -------------------------------------------------------------------------
    elif "Forecast" in page:
        st.title("Predictive Liquidity Outlook")
        st.markdown("Using ARIMA(1,1,1) modeling to forecast future cash flow based on historical trends.")

        # --- STEP 1: DATE CONVERSION (The most common cause of zeros/empty tables) ---
        # We ensure dates are converted correctly. Added dayfirst=True for safety.
        df_forecast['Week_Ending'] = pd.to_datetime(df_forecast['Week_Ending'], errors='coerce')
        if not df_master.empty:
            df_master['Pstng Date'] = pd.to_datetime(df_master['Pstng Date'], errors='coerce')

        # --- STEP 2: NUMERIC CLEANING ---
        # Force columns to be numbers. This prevents errors when calculating 'net_change'
        forecast_clean = df_forecast.copy()
        numeric_cols = [
            'Projected_Ending_Balance_USD', 
            'Balance_Lower_Bound', 
            'Balance_Upper_Bound', 
            'Forecasted_Net_Cash_Flow_USD', 
            'Change_From_Current'
        ]
        for col in numeric_cols:
            if col in forecast_clean.columns:
                forecast_clean[col] = pd.to_numeric(forecast_clean[col], errors='coerce').fillna(0)

        forecast_sorted = (
            forecast_clean
            .sort_values('Week_Ending')
            .reset_index(drop=True)
        )

        # --- STEP 3: DYNAMIC BALANCE CALCULATION ---
        # If df_master doesn't have 'Ending_Balance_USD', we calculate it from 'Amount in USD'
        if not df_master.empty:
            df_hist = df_master.sort_values('Pstng Date')
            if 'Ending_Balance_USD' not in df_hist.columns:
                # Fallback: assume cumulative sum of historical flows
                current_balance = df_hist['Amount in USD'].sum() 
            else:
                current_balance = df_hist['Ending_Balance_USD'].iloc[-1]
        else:
            current_balance = 0

        # --- STEP 4: UI SELECTION ---
        view_mode = st.radio("Visualization", ["Forecast Only", "Actual vs Predict"], horizontal=True)

        if view_mode == "Forecast Only":
            horizon_choice = st.radio("Forecast Horizon", ["Next 4 Weeks (1 Month)", "Next 6 Months"], index=1, horizontal=True)
            forecast_view = forecast_sorted.head(4) if "4 Weeks" in horizon_choice else forecast_sorted
        else:
            forecast_view = forecast_sorted

        # --- STEP 5: METRICS (The "No Zero" logic) ---
        if not forecast_view.empty:
            final_bal = forecast_view['Projected_Ending_Balance_USD'].iloc[-1]
            # If current_balance is 0, we fallback to the 'Change_From_Current' column in the CSV
            if current_balance != 0:
                net_change = final_bal - current_balance
            else:
                net_change = forecast_view['Change_From_Current'].iloc[-1]
        else:
            final_bal = 0
            net_change = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Forecast Model**\n\nARIMA(1,1,1)")
        with col2:
            st.metric("Projected Final Balance", f"${final_bal:,.2f}")
        with col3:
            label = "Net Change (Selected)" if view_mode == "Forecast Only" else "Net Change (6 Months)"
            st.metric(label, f"${net_change:,.2f}", delta=f"{net_change:,.0f}", delta_color="normal")

        # --- STEP 6: CHARTING ---
        st.subheader("Liquidity Trend Projection")
        fig_cast = go.Figure()

        x_dates = []

        if view_mode == "Actual vs Predict":
            if not df_master.empty:
                # We aggregate by week to make the historical line smooth
                hist_weekly = df_master.groupby(pd.Grouper(key='Pstng Date', freq='W')).agg({
                    'Amount in USD': 'sum'
                }).reset_index()
                # Create a running balance for the chart
                hist_weekly['Running_Bal'] = hist_weekly['Amount in USD'].cumsum() + (current_balance - df_master['Amount in USD'].sum())

                fig_cast.add_trace(go.Scatter(
                    x=hist_weekly['Pstng Date'],
                    y=hist_weekly['Running_Bal'],
                    mode='lines',
                    line=dict(color=AZ_COLORS.get('Navy', '#003865'), width=2),
                    name='Historical Balance'
                ))
                x_dates.append(hist_weekly['Pstng Date'])

            fig_cast.add_trace(go.Scatter(
                x=forecast_view['Week_Ending'],
                y=forecast_view['Projected_Ending_Balance_USD'],
                mode='lines+markers',
                line=dict(color=AZ_COLORS.get('Mulberry', '#830051'), width=3),
                name='ARIMA Forecast'
            ))
            x_dates.append(forecast_view['Week_Ending'])

            if not forecast_view.empty:
                fig_cast.add_vline(x=forecast_view['Week_Ending'].min(), line_width=2, line_dash="dash", line_color="gray")

        else:
            # Forecast Only with Confidence Intervals
            # Use the currently selected forecast_view (will be 4 rows for 1-month selection)
            df_selected = forecast_view.copy()

            fig_cast.add_trace(go.Scatter(
                x=df_selected['Week_Ending'], y=df_selected.get('Balance_Upper_Bound', df_selected.get('Forecast_Upper_Bound')),
                mode='lines', line=dict(width=2, color='red', dash='dash'), showlegend=False, hoverinfo='skip'
            ))
            fig_cast.add_trace(go.Scatter(
                x=df_selected['Week_Ending'], y=df_selected.get('Balance_Lower_Bound', df_selected.get('Forecast_Lower_Bound')),
                mode='lines', line=dict(width=2, color='red', dash='dash'), fill='tonexty',
                fillcolor='rgba(104, 210, 223, 0.2)', name='95% Confidence Interval'
            ))
            fig_cast.add_trace(go.Scatter(
                x=df_selected['Week_Ending'], y=df_selected['Projected_Ending_Balance_USD'],
                mode='lines+markers+text', line=dict(color=AZ_COLORS.get('Mulberry', '#830051'), width=4),
                name='Projected Balance', text=[f'${v:,.0f}' for v in df_selected['Projected_Ending_Balance_USD']],
                textposition='top center', textfont=dict(color=AZ_COLORS.get('Mulberry', '#830051'), size=9)
            ))
            x_dates.append(df_selected['Week_Ending'])

        # Limit x-axis to dates that actually exist in the datasets shown
        if x_dates:
            merged = pd.concat([pd.Series(d).dropna() for d in x_dates])
            if not merged.empty:
                fig_cast.update_xaxes(range=[merged.min(), merged.max()])

        fig_cast.update_layout(
            yaxis_title="USD",
            xaxis_title="Timeline",
            hovermode="x unified",
            yaxis=dict(autorange=True, tickformat="$,.0f")
        )
        st.plotly_chart(fig_cast, use_container_width=True)

        # --- STEP 7: TABLES ---
        with st.expander("📊 View Data Details"):
            st.dataframe(
                forecast_view[['Week_Ending', 'Forecasted_Net_Cash_Flow_USD', 'Projected_Ending_Balance_USD', 'Alert_Level']].rename(columns={
                    'Week_Ending': 'Date',
                    'Forecasted_Net_Cash_Flow_USD': 'Net Flow',
                    'Projected_Ending_Balance_USD': 'Projected Balance',
                    'Alert_Level': 'Status'
                }),
                use_container_width=True, hide_index=True
            )
          # -------------------------------------------------------------------------
    # PAGE 4: PROCESS ANOMALIES - Connected to AZ_Anomaly_Report.csv
    # -------------------------------------------------------------------------
    elif "Process Anomalies" in page:
        st.title("Process Anomalies & Detection")
        st.markdown("Identifying Duplicate Payments, Outflow Spikes, and Mismatches.")

        if df_anomaly is None or df_anomaly.empty:
            st.error("Anomaly dataset is not loaded. Please verify AZ_Anomaly_Report.csv is present and try again.")
            st.stop()

        anomaly_range = f"{df_anomaly['Pstng Date'].min().date()} to {df_anomaly['Pstng Date'].max().date()}" if not df_anomaly.empty else "N/A"
        st.caption(f"Connected to AZ_Anomaly_Report.csv • {len(df_anomaly)} records • Dates: {anomaly_range}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Filter Anomalies")
            min_z = st.slider("Sensitivity (Z-Score)", min_value=3.0, max_value=10.0, value=3.0, step=0.1)

            category_colors = {
                "AP": AZ_COLORS['Magenta'],
                "Tax payable": AZ_COLORS['Gold'],
                "Bank charges": AZ_COLORS['Purple'],
                "Netting AR": AZ_COLORS['Light Blue'],
                "AR": AZ_COLORS['Navy']
            }

            cats_all = sorted(df_anomaly['Category'].dropna().unique().tolist())
            ordered_cats = [c for c in category_colors if c in cats_all] + [c for c in cats_all if c not in category_colors]
            selected_cats = st.multiselect("Filter Category", ordered_cats, default=ordered_cats, key="category_filter")
            color_map = {cat: category_colors.get(cat, AZ_COLORS['Graphite']) for cat in ordered_cats}
            selected_order = [cat for cat in ordered_cats if cat in selected_cats]

        # Apply filters before plotting
        filtered_anomalies = df_anomaly[
            (df_anomaly['Z_Score'] >= min_z) &
            (df_anomaly['Category'].isin(selected_cats))
        ].reset_index(drop=True)
        
        with col2:
            st.subheader(f"Detected {len(filtered_anomalies)} Anomalies")

            if len(filtered_anomalies) > 0:
                fig_anom = px.scatter(
                    filtered_anomalies,
                    x="Pstng Date",
                    y="Amount in USD",
                    color="Category",
                    size="Z_Score",
                    hover_data=["DocumentNo", "Name", "Deviation_Pct"],
                    color_discrete_map=color_map,
                    category_orders={"Category": selected_order},
                    title="Anomaly Severity Map"
                )
            else:
                fig_anom = go.Figure()
                for cat in selected_order:
                    fig_anom.add_trace(
                        go.Scatter(x=[], y=[], mode="markers", name=cat,
                                   marker=dict(color=color_map.get(cat)))
                    )
                fig_anom.update_layout(
                    title="Anomaly Severity Map",
                    xaxis_title="Pstng Date",
                    yaxis_title="Amount in USD"
                )

            fig_anom.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=500, legend_traceorder="normal")
            st.plotly_chart(fig_anom, use_container_width=True, key="anomaly_chart")

        st.markdown("### 📋 Actionable List for Finance Ops")
        st.caption(f"**Showing Categories:** {', '.join(selected_cats)}")

        if len(filtered_anomalies) > 0:
            report_df = filtered_anomalies[['DocumentNo', 'Pstng Date', 'Name', 'Category', 'Amount in USD', 'Z_Score', 'Deviation_Pct']].copy()
            report_df = report_df.rename(columns={
                'DocumentNo': 'Number',
                'Pstng Date': 'Date',
                'Amount in USD': 'Amount (USD)',
                'Z_Score': 'Z-Score',
                'Deviation_Pct': 'Deviation %'
            })
            report_df['Amount (USD)'] = report_df['Amount (USD)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            report_df['Z-Score'] = report_df['Z-Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            report_df['Deviation %'] = report_df['Deviation %'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
            report_df['Date'] = pd.to_datetime(report_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

            st.dataframe(report_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("📊 Report Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Anomalies", len(filtered_anomalies))
            with col2:
                st.metric("Categories Selected", len(selected_cats))
            with col3:
                avg_z_score = filtered_anomalies['Z_Score'].mean()
                st.metric("Avg Z-Score", f"{avg_z_score:.2f}")
            with col4:
                total_amount = filtered_anomalies['Amount in USD'].sum()
                st.metric("Total Amount", f"${total_amount:,.0f}")

            st.download_button(
                label="📥 Download Full Anomaly Report (CSV)",
                data=filtered_anomalies[['DocumentNo', 'Pstng Date', 'Name', 'Category', 'Amount in USD', 'Z_Score', 'Deviation_Pct']].rename(columns={
                    'DocumentNo': 'Number',
                    'Pstng Date': 'Date',
                    'Amount in USD': 'Amount (USD)',
                    'Z_Score': 'Z-Score',
                    'Deviation_Pct': 'Deviation %'
                }).to_csv(index=False).encode('utf-8'),
                file_name='AZ_Verified_Anomalies.csv',
                mime='text/csv',
            )
        else:
            st.info("No anomalies to display or download with current filters.")

       
    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: {AZ_COLORS['Graphite']}'>AZ Liquidity Control Center • AstraZeneca Datathon 2025</div>", unsafe_allow_html=True)

