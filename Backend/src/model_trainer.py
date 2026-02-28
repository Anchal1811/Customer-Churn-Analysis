import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 1. DEFINE THE FUNCTION FIRST
def train_production_model(data_path):
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Run data_processor.py first!")
        return

    print("📊 Loading processed data...")
    df = pd.read_csv(data_path)
    
    X = df.drop(['Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("⚖️ Balancing classes with SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print("🧠 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # Ensure Model folder exists
    os.makedirs('../Model', exist_ok=True)
    
    # 2. SAVE THE FILES (This creates the .pkl files you need)
    joblib.dump(model, '../Model/churn_model.pkl')
    joblib.dump(X.columns.tolist(), '../Model/features_list.pkl')
    print("🚀 Success! model.pkl and features_list.pkl created in Backend/Model/")

# 3. CALL THE FUNCTION AT THE VERY END
if __name__ == "__main__":
    path = '../Data/processed/cleaned_churn.csv'
    train_production_model(path)