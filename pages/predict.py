import streamlit as st
from utils.api import predict_single

st.title('Predict Risk')
st.markdown('Enter applicant details below to evaluate credit risk.')

with st.form('predict_form'):
    st.subheader('Applicant Information')

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=120, value=30)
        income = st.number_input('Income ($)', min_value=0.0, value=50000.0, step=1000.0)
        emp_length = st.number_input('Employment Length (years)', min_value=0, max_value=60, value=5)
        home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE'])
        cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, max_value=60, value=5)
        default_on_file = st.selectbox('Default on File', ['N', 'Y'])

    with col2:
        st.subheader('Loan Details')

        loan_amnt = st.number_input('Loan Amount ($)', min_value=100.0, value=10000.0, step=500.0)
        loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE'])
        loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E'])
        loan_int_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        loan_percent_income = st.number_input('Loan Percent Income (0-1)', min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    submitted = st.form_submit_button('Predict Risk', type='primary')

if submitted:
    payload = {
        'cb_person_cred_hist_length': int(cred_hist_length),
        'cb_person_default_on_file': default_on_file,
        'loan_amnt': float(loan_amnt),
        'loan_grade': loan_grade,
        'loan_int_rate': float(loan_int_rate),
        'loan_intent': loan_intent,
        'loan_percent_income': float(loan_percent_income),
        'person_age': int(age),
        'person_emp_length': int(emp_length),
        'person_home_ownership': home_ownership,
        'person_income': float(income)
    }

    with st.spinner('Waking up model... first request may take 20–30 seconds'):
        result = predict_single(payload)

    # 🚨 Fixed error handling
    if result is None or result.get("status") == "error" or "prediction" not in result:
        st.error(result.get("message", "Backend is slow or restarting. Please try again."))
    else:
        st.markdown('---')
        st.subheader('Prediction Result')

        # Risk mapping
        raw_risk = result.get('risk_level', 'UNKNOWN')
        risk = str(raw_risk).upper()

        risk_mapping = {
            'LOW': 'LOW',
            'MODERATE': 'MEDIUM',
            'MEDIUM': 'MEDIUM',
            'HIGH': 'HIGH',
            'VERY_HIGH': 'VERY_HIGH'
        }

        mapped_risk = risk_mapping.get(risk, 'UNKNOWN')

        ui_map = {
            'LOW': ('Low Risk', st.success),
            'MEDIUM': ('Medium Risk', st.warning),
            'HIGH': ('High Risk', st.error),
            'VERY_HIGH': ('Very High Risk', st.error)
        }

        if mapped_risk in ui_map:
            label, display_func = ui_map[mapped_risk]
            display_func(label)
        else:
            st.info(f'Unknown Risk: {raw_risk}')

        st.caption(f'Model Risk Level: `{raw_risk}`')

        # Probability handling
        raw_prob = result.get('default_probability', 0)

        try:
            prob = float(raw_prob)
        except (ValueError, TypeError):
            prob = 0.0

        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

        with col_res2:
            st.metric(
                label='Default Probability',
                value=f'{prob * 100:.2f}%'
            )

        with st.expander('Raw API Response'):
            st.write(result)