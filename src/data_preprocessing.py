"""
Data Preprocessing Pipeline for Credit Risk Dataset.
Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Real-world issues handled:
- Outliers (age=144, income=6M, emp_length=123)
- Missing values in person_emp_length (895) and loan_int_rate (3116)
- Categorical encoding for 4 categorical features
- Feature engineering for domain-specific risk metrics
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class CreditRiskPreprocessor:
    """Reusable preprocessing pipeline for credit risk data."""

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False

        self.categorical_cols = [
            'person_home_ownership', 'loan_intent',
            'loan_grade', 'cb_person_default_on_file'
        ]
        self.numerical_cols = [
            'person_age', 'person_income', 'person_emp_length',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length'
        ]

    def _remove_outliers(self, df):
        """
        Remove unrealistic outlier rows.

        Real issues found in EDA:
        - person_age max = 144 (impossible)
        - person_emp_length max = 123 (impossible)
        - person_income max = 6,000,000 (extreme outlier)
        """
        df = df.copy()
        original_len = len(df)

        # Age: realistic range 18-100
        df = df[df['person_age'] <= 100]

        # Employment length: can't exceed age-18, cap at 60
        df = df[df['person_emp_length'].isna() | (df['person_emp_length'] <= 60)]

        # Income: remove extreme outliers (> 99.5th percentile)
        income_cap = df['person_income'].quantile(0.995)
        df = df[df['person_income'] <= income_cap]

        removed = original_len - len(df)
        print(f"  Outliers removed: {removed} rows ({removed/original_len*100:.1f}%)")

        return df.reset_index(drop=True)

    def _engineer_features(self, df):
        """Create domain-specific risk features."""
        df = df.copy()

        # 1. Income-to-loan ratio (higher = safer)
        df['income_to_loan'] = df['person_income'] / (df['loan_amnt'] + 1)

        # 2. Age-to-credit-history ratio (lower = started credit young)
        df['age_credit_ratio'] = df['person_age'] / (df['cb_person_cred_hist_length'] + 1)

        # 3. Employment stability (emp_length relative to age)
        df['emp_stability'] = df['person_emp_length'] / (df['person_age'] - 18 + 1)

        # 4. High risk flag: high loan percentage + high interest rate
        df['high_risk_flag'] = (
            (df['loan_percent_income'] > 0.3) & (df['loan_int_rate'] > 15)
        ).astype(int)

        # 5. Interest rate category buckets (captures non-linear risk)
        df['rate_bucket'] = pd.cut(
            df['loan_int_rate'],
            bins=[0, 8, 12, 16, 25],
            labels=[0, 1, 2, 3],
            right=True
        ).astype(float)

        return df

    def fit_transform(self, df, target_col='loan_status'):
        """Fit the preprocessor on training data and transform it."""
        print("Preprocessing pipeline starting...")

        # Step 1: Remove outliers
        print("\nStep 1: Removing outliers...")
        df = self._remove_outliers(df)

        # Step 2: Feature engineering
        print("Step 2: Engineering features...")
        df = self._engineer_features(df)

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Step 3: Encode categorical variables
        print("Step 3: Encoding categorical variables...")
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col}: {list(le.classes_)}")

        # Step 4: Handle missing values
        print("Step 4: Imputing missing values...")
        engineered_cols = [
            'income_to_loan', 'age_credit_ratio', 'emp_stability',
            'high_risk_flag', 'rate_bucket'
        ]
        all_num_cols = self.numerical_cols + engineered_cols
        num_cols_present = [c for c in all_num_cols if c in X.columns]

        missing_before = X[num_cols_present].isnull().sum()
        print(f"  Missing before imputation:\n{missing_before[missing_before > 0].to_string()}")

        X[num_cols_present] = self.num_imputer.fit_transform(X[num_cols_present])

        # Step 5: Scale numerical features
        print("Step 5: Scaling features...")
        X[num_cols_present] = self.scaler.fit_transform(X[num_cols_present])

        self.feature_names = list(X.columns)
        self.is_fitted = True

        print(f"\nFinal feature set: {len(self.feature_names)} features")
        print(f"Samples: {X.shape[0]}")

        return X, y

    def transform(self, df, has_target=True):
        """Transform new data using the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = self._engineer_features(df)

        if has_target and 'loan_status' in df.columns:
            X = df.drop(columns=['loan_status'])
            y = df['loan_status']
        else:
            X = df.copy()
            y = None

        # Encode categoricals (handle unseen categories)
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            X[col] = X[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        # Impute and scale
        engineered_cols = [
            'income_to_loan', 'age_credit_ratio', 'emp_stability',
            'high_risk_flag', 'rate_bucket'
        ]
        all_num_cols = self.numerical_cols + engineered_cols
        num_cols_present = [c for c in all_num_cols if c in X.columns]

        X[num_cols_present] = self.num_imputer.transform(X[num_cols_present])
        X[num_cols_present] = self.scaler.transform(X[num_cols_present])

        X = X[self.feature_names]

        if y is not None:
            return X, y
        return X

    def save(self, path):
        """Save fitted preprocessor."""
    import __main__
    __main__.CreditRiskPreprocessor = CreditRiskPreprocessor
    joblib.dump(self, path)
    print(f"Preprocessor saved: {path}")

    @staticmethod
    def load(path):
        """Load fitted preprocessor."""
        return joblib.load(path)


def run_preprocessing():
    """Main preprocessing pipeline."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'credit_risk_dataset.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load raw data
    print("=" * 60)
    print("  CREDIT RISK DATA PREPROCESSING")
    print("=" * 60)
    df = pd.read_csv(raw_path)
    print(f"\nRaw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Default rate: {df['loan_status'].mean():.2%}")

    # Fit and transform
    preprocessor = CreditRiskPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples (default rate: {y_train.mean():.2%})")
    print(f"Test set:  {X_test.shape[0]} samples (default rate: {y_test.mean():.2%})")

    # Save processed data
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

    # Save reference data for drift detection (raw, before preprocessing)
    # We use a sample of the original data as reference baseline
    df_clean = df[df['person_age'] <= 100].copy()
    df_clean = df_clean[df_clean['person_emp_length'].isna() | (df_clean['person_emp_length'] <= 60)]
    income_cap = df_clean['person_income'].quantile(0.995)
    df_clean = df_clean[df_clean['person_income'] <= income_cap]

    ref_data = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)
    ref_data.to_csv(os.path.join(processed_dir, 'reference_data.csv'), index=False)
    print(f"Reference data saved: {ref_data.shape[0]} samples")

    # Save preprocessor
    preprocessor.save(os.path.join(model_dir, 'preprocessor.pkl'))

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing()
