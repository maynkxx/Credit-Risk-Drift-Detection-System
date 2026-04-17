import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
st.title('Drift Monitoring')
st.markdown('Monitor model performance and data drift over time to ensure reliable predictions.')

@st.cache_data
def get_mock_drift_data():
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    dates.reverse()
    base_drift = np.random.uniform(0.01, 0.05, 30)
    base_drift[25:] += np.random.uniform(0.04, 0.08, 5)
    df = pd.DataFrame({'Date': dates, 'Drift Score': base_drift})
    features = ['person_income', 'loan_int_rate', 'loan_amnt', 'person_age', 'loan_percent_income', 'cb_person_cred_hist_length', 'person_emp_length']
    feature_drift = pd.DataFrame({'Feature': features, 'PSI (Population Stability Index)': np.random.uniform(0.01, 0.25, len(features))})
    return (df, feature_drift.sort_values(by='PSI (Population Stability Index)', ascending=False))
df_timeline, df_features = get_mock_drift_data()
current_drift_score = df_timeline['Drift Score'].iloc[-1]
if current_drift_score < 0.1:
    status_color = 'green'
    status_text = 'Stable'
elif current_drift_score < 0.2:
    status_color = 'orange'
    status_text = 'Warning'
else:
    status_color = 'red'
    status_text = 'Critical'
col1, col2 = st.columns(2)
with col1:
    st.metric(label='Current System Drift Score', value=f'{current_drift_score:.4f}', delta=f"{current_drift_score - df_timeline['Drift Score'].iloc[-2]:.4f}", delta_color='inverse', help='Overall drift score. Values > 0.1 indicates warning, > 0.2 is critical.')
with col2:
    st.markdown(f'### Status: :{status_color}[{status_text}]')
st.markdown('---')
st.subheader('Data Drift Over Time')
fig_line = px.line(df_timeline, x='Date', y='Drift Score', markers=True, title='30-Day Drift Score Trend')
fig_line.add_hline(y=0.1, line_dash='dash', line_color='red', annotation_text='Threshold (0.1)')
st.plotly_chart(fig_line, use_container_width=True)
st.markdown('---')
st.subheader('Feature-Level Drift (PSI)')
st.markdown('Population Stability Index (PSI) measures the shifts in individual feature distributions.')
fig_bar = px.bar(df_features, x='PSI (Population Stability Index)', y='Feature', orientation='h', color='PSI (Population Stability Index)', color_continuous_scale='Reds')
fig_bar.add_vline(x=0.1, line_dash='dash', line_color='orange', annotation_text='Monitor (0.1)')
fig_bar.add_vline(x=0.2, line_dash='dash', line_color='red', annotation_text='Action Required (0.2)')
st.plotly_chart(fig_bar, use_container_width=True)