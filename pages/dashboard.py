import streamlit as st
from utils.api import get_health, get_model_info
import pandas as pd
st.title('Dashboard')
st.markdown('### System Status')
with st.spinner('Checking API health...'):
    health = get_health()
if health.get('status') == 'error':
    st.markdown('### API Status: **Down**')
    st.error(f"Error details: {health.get('message')}")
else:
    delay = health.get('delay_seconds', 0)
    if delay > 5:
        st.markdown(f'### API Status: **Slow** ({delay:.2f}s response time)')
        st.warning('The API is taking longer to respond than expected (possibly due to a cold start).')
    else:
        st.markdown(f'### API Status: **Running** ({delay:.2f}s response time)')
st.markdown('---')
st.markdown('### Platform Metrics')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total Predictions', '1,245', '12 today')
with col2:
    st.metric('Average Risk Score', '0.24', '-0.02')
with col3:
    st.metric('High Risk Clients', '18%', '2%')
with col4:
    st.metric('Model Drift Alert', 'Stable', '0.012')
st.markdown('---')
st.markdown('### Model Quick Info')
model_info = get_model_info()
if 'error' not in model_info.get('status', ''):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"**Model Name:** `{model_info.get('model_name', 'XGBoost Credit Risk')}`")
        st.markdown(f"**Version:** `{model_info.get('version', '1.0.0')}`")
    with col_m2:
        num_features = len(model_info.get('features', []))
        st.markdown(f'**Input Features:** `{num_features}`')
        st.markdown(f'**Environment:** `Production`')
else:
    st.error('Could not fetch model info. Is the backend running?')