import streamlit as st
from utils.api import get_model_info
st.title('Model Info')
st.markdown('Detailed information about the currently deployed credit risk prediction model.')
info = get_model_info()
if info.get('status') == 'error':
    st.error(f"Failed to fetch model info from API: {info.get('message')}")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('General Details')
        st.markdown(f"**Name:** `{info.get('model_name', 'XGBoostClassifier')}`")
        st.markdown(f"**Version:** `{info.get('version', '1.0.0')}`")
        st.markdown(f"**Type:** `{info.get('model_type', 'Gradient Boosting')}`")
        st.markdown(f"**Description:** `{info.get('description', 'Predicts likelihood of loan default.')}`")
    with col2:
        features = info.get('features', [])
        st.subheader(f'Input Features ({len(features)})')
        if features:
            for feat in features:
                st.markdown(f'- `{feat}`')
        else:
            st.info('Feature list not provided by backend.')
    st.markdown('---')
    st.subheader('Raw JSON Response')
    st.json(info)