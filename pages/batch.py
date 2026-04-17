import streamlit as st
import pandas as pd
from io import BytesIO
from utils.api import predict_batch
st.title('Batch Prediction')
st.markdown('Upload a CSV file containing multiple loan applications to get bulk predictions.')
st.info('Ensure your CSV has the required columns: `cb_person_cred_hist_length`, `cb_person_default_on_file`, `loan_amnt`, `loan_grade`, `loan_int_rate`, `loan_intent`, `loan_percent_income`, `person_age`, `person_emp_length`, `person_home_ownership`, `person_income`')
uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.subheader('Data Preview')
    st.dataframe(df_preview.head())
    if st.button('Run Batch Prediction', type='primary'):
        with st.spinner('Processing batch...'):
            uploaded_file.seek(0)
            result = predict_batch(uploaded_file)
        if result.get('status') == 'error':
            st.error(f"Batch prediction failed: {result.get('message')}")
        else:
            st.success('Batch processing complete!')
            if 'predictions' in result:
                preds_df = pd.DataFrame(result['predictions'])
            else:
                preds_df = pd.DataFrame(result)
            st.subheader('Prediction Results')
            st.dataframe(preds_df, use_container_width=True)
            csv = preds_df.to_csv(index=False).encode('utf-8')
            st.download_button(label='Download Predictions', data=csv, file_name='batch_predictions.csv', mime='text/csv')