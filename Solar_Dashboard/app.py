import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import yeojohnson
from scipy.stats.mstats import winsorize
from scipy import stats

st.set_page_config(page_title="MoonLight Energy Solar Comparison", layout="wide")
st.title("🌞 MoonLight Energy Solutions")
st.markdown("### Strategic Solar Site Comparison Dashboard — Benin · Togo · Sierra Leone")

@st.cache_data
def load_and_clean():
    paths = {
        "Benin-Malanville": "data/benin-malanville.csv",
        "Togo-Dapaong": "data/togo-dapaong_qc.csv",
        "SierraLeone-Bumbuna": "data/sierraleone-bumbuna.csv"
    }
    
    dfs = []
    for site, path in paths.items():
        df = pd.read_csv(path)
        # Clean negatives
        for col in ['GHI','DNI','DHI']:
            df[col] = df[col].clip(lower=0)
        df = df.drop(['Comments'], axis=1, errors='ignore')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
        df['Site'] = site
        dfs.append(df)
    
    full_df = pd.concat(dfs)
    
    # Your exact cleaning steps from interim report
    transform_cols = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust']
    for col in transform_cols:
        full_df[f'{col}_yj'], _ = yeojohnson(full_df[col] + 1)
    
    full_df['WS_winsor'] = winsorize(full_df['WS'], limits=[0.005, 0.005])
    full_df['WSgust_winsor'] = winsorize(full_df['WSgust'], limits=[0.005, 0.005])
    
    z_cols = ['Tamb', 'RH', 'BP', 'WS', 'WSgust']
    for col in z_cols:
        full_df[f'{col}_zscore'] = full_df.groupby('Site')[col].transform(lambda x: stats.zscore(x, nan_policy='omit'))
    full_df = full_df[full_df[[f'{c}_zscore' for c in z_cols]].abs().max(axis=1) <= 4]
    
    return full_df

df = load_and_clean()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
sites = st.sidebar.multiselect("Select Sites (default = all)", 
                               options=df['Site'].unique(), 
                               default=df['Site'].unique())

date_range = st.sidebar.date_input("Date Range", 
                                   value=(df.index.min().date(), df.index.max().date()),
                                   min_value=df.index.min().date(),
                                   max_value=df.index.max().date())

plot_type = st.sidebar.selectbox("Quick Plot", [
    "Monthly GHI", "Daily Total Irradiation", "Hourly Profile", 
    "Specific Yield Estimate", "Cleaning Events", "Soiling Check (ModA vs GHI)"
])

df = df[df['Site'].isin(sites)]
df = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])]

# ================== MAIN DASHBOARD ==================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estimated Specific Yield (kWh/kWp/year)")
    df_day = df.between_time('06:00','18:00')
    yield_est = {}
    for site in df['Site'].unique():
        ghi_annual = df_day[df_day['Site']==site]['GHI'].sum() / 60 / 1000
        avg_tmod = df_day[df_day['Site']==site]['TModA'].mean()
        temp_loss = 1 - 0.004 * (avg_tmod - 25)
        specific_yield = ghi_annual * 0.85 * temp_loss
        yield_est[site] = round(specific_yield, 0)
    yield_df = pd.DataFrame(list(yield_est.items()), columns=['Site', 'Yield'])
    fig = px.bar(yield_df.sort_values('Yield', ascending=False), x='Site', y='Yield', 
                 text='Yield', title="Annual Energy Yield per kWp", height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Total Cleaning Events (Soiling Proxy)")
    cleaning = df.groupby('Site')['Cleaning'].sum().reset_index()
    fig = px.bar(cleaning, x='Site', y='Cleaning', text='Cleaning', color='Site')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Quick plots section
if plot_type == "Monthly GHI":
    monthly = df.groupby([df.index.month, 'Site'])['GHI'].mean().reset_index()
    fig = px.line(monthly, x='month', y='GHI', color='Site', markers=True, 
                  title="Monthly Average GHI")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Daily Total Irradiation":
    daily = df.groupby([df.index.date, 'Site'])['GHI'].sum().reset_index()
    daily['kWh/m²/day'] = daily['GHI'] / 60 / 1000
    fig = px.line(daily, x='date', y='kWh/m²/day', color='Site', 
                  title="Daily Total Irradiation (kWh/m²/day)")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Hourly Profile":
    df['hour'] = df.index.hour
    hourly = df.groupby(['Site', 'hour'])['GHI'].mean().reset_index()
    fig = px.line(hourly, x='hour', y='GHI', color='Site', markers=True,
                  title="Average Hourly GHI Profile")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Soiling Check (ModA vs GHI)":
    fig = make_subplots(rows=1, cols=3, subplot_titles=df['Site'].unique())
    for i, site in enumerate(df['Site'].unique()):
        sub = df[df['Site']==site].sample(20000)
        fig.add_trace(go.Scatter(x=sub['ModA'], y=sub['GHI'], mode='markers', opacity=0.6, showlegend=False), 
                      row=1, col=i+1)
        fig.add_trace(go.Scatter(x=[0,1400], y=[0,1400], mode='lines', 
                                line=dict(color='red', dash='dash'), showlegend=False), row=1, col=i+1)
    fig.update_layout(title_text="GHI vs ModA → Tightest = Least Soiling (Togo wins!)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# Full screen plots section
st.markdown("---")
st.subheader("Deep Dive Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Yield & Performance", "Soiling & O&M", "Temperature & Cooling", "Humidity & Cloud Impact"])

with tab1:
    st.plotly_chart(px.violin(df, x='Site', y='GHI', color='Site', box=True, 
                              title="GHI Distribution (Violin)"), use_container_width=True)

with tab2:
    st.plotly_chart(px.scatter(df.sample(100000), x='ModA', y='GHI', color='Site', 
                               title="GHI vs Module Irradiance → Togo has perfect alignment"), 
                    use_container_width=True)

with tab3:
    st.plotly_chart(px.scatter(df.sample(80000), x='WS', y='TModA', color='Site', trendline="ols",
                               title="Wind Speed vs Module Temp → Benin gets best cooling"), 
                    use_container_width=True)

with tab4:
    st.plotly_chart(px.scatter(df.sample(100000), x='RH', y='GHI', color='Site',
                               title="Humidity vs GHI → Sierra Leone trapped in cloud hell"),
                    use_container_width=True)

# Final recommendation
st.markdown("---")
st.success("""
**FINAL RECOMMENDATION**

→ 50% Benin-Malanville (highest yield)  
→ 50% Togo-Dapaong (lowest soiling, best reliability)  
→ 0% Sierra Leone-Bumbuna (loses on both yield and O&M)

This dashboard proves it with data.
""")