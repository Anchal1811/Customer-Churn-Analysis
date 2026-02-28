import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def clean_and_feature_engineer(filepath):
    df = pd.read_csv(filepath)
    
    # 1. DROP THE ID COLUMN (This is what caused your error)
    # We use errors='ignore' in case it was already dropped
    cols_to_drop = ['customerID', 'id', 'id_column'] 
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 2. Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 3. Handle Categorical Data
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # 4. Save to the PROCESSED folder
    output_path = '../Data/processed/cleaned_churn.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data (no IDs) saved to: {output_path}")

if __name__ == "__main__":
    clean_and_feature_engineer('../Data/Telco-Customer-Churn.csv')