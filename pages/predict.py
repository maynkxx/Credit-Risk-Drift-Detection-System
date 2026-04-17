import streamlit as st
import time
from utils.api import predict_single
st.title('Predict Risk')
st.markdown('Enter applicant details below to evaluate credit risk.')
with st.form('predict_form'):
    st.subheader('Applicant Information')
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=120, value=30, help='Age of the applicant in years.')
        income = st.number_input('Income ($)', min_value=0.0, value=50000.0, step=1000.0, help='Annual income of the applicant.')
        emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=60, value=5, help='Number of years employed.')
        home_ownership = st.selectbox('Home Ownership', options=['RENT', 'OWN', 'MORTGAGE'], help="Applicant's current housing situation.")
        cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, max_value=60, value=5, help="Length of applicant's credit history.")
        default_on_file = st.selectbox('Default on File', options=['N', 'Y'], help='Does the applicant have a history of default?')
    with col2:
        st.subheader('Loan Details')
        loan_amnt = st.number_input('Loan Amount ($)', min_value=100.0, value=10000.0, step=500.0, help='Requested loan amount.')
        loan_intent = st.selectbox('Loan Intent', options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE'], help='Purpose of the loan.')
        loan_grade = st.selectbox('Loan Grade', options=['A', 'B', 'C', 'D', 'E'], help='Assigned loan grade based on creditworthiness.')
        loan_int_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1, help='Interest rate of the loan.')
        loan_percent_income = st.number_input('Loan Percent Income (0-1)', min_value=0.0, max_value=1.0, value=0.2, step=0.01, help='Ratio of loan amount to annual income.')
    submitted = st.form_submit_button('Predict Risk', type='primary')
if submitted:
    payload = {'cb_person_cred_hist_length': int(cred_hist_length), 'cb_person_default_on_file': default_on_file, 'loan_amnt': float(loan_amnt), 'loan_grade': loan_grade, 'loan_int_rate': float(loan_int_rate), 'loan_intent': loan_intent, 'loan_percent_income': float(loan_percent_income), 'person_age': int(age), 'person_emp_length': int(emp_length), 'person_home_ownership': home_ownership, 'person_income': float(income)}
    with st.spinner('Analyzing risk profile...'):
        time.sleep(0.5)
        result = predict_single(payload)
    if result.get('status') == 'error':
        st.error(f"Failed to get prediction: {result.get('message')}")
    else:
        st.markdown('---')
        st.subheader('Prediction Result')
        raw_risk = result.get('risk_level', 'UNKNOWN')
        risk = str(raw_risk).upper()
        risk_mapping = {'LOW': 'LOW', 'MODERATE': 'MEDIUM', 'MEDIUM': 'MEDIUM', 'HIGH': 'HIGH', 'VERY_HIGH': 'VERY_HIGH'}
        mapped_risk = risk_mapping.get(risk, 'UNKNOWN')
        ui_map = {'LOW': ('Low Risk', st.success), 'MEDIUM': ('Medium Risk', st.warning), 'HIGH': ('High Risk', st.error), 'VERY_HIGH': ('Very High Risk', st.error)}
        if mapped_risk in ui_map:
            label, display_func = ui_map[mapped_risk]
            display_func(f'{label}')
        else:
            st.info(f'Unknown Risk: {raw_risk}')
        st.caption(f'Model Risk Level: `{raw_risk}`')
        raw_prob = result.get('default_probability', 0)
        try:
            prob = float(raw_prob)
        except (ValueError, TypeError):
            prob = 0.0
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            st.metric(label='Default Probability', value=f'{prob * 100:.2f}%', help='Higher probability means higher risk of default.')
        with st.expander('Raw API Response'):
            st.write(result)