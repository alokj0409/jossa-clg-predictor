import os
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, classification_report

def train_and_save_models(data_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data for training...")
    # Load regression datasets
    train_iit_reg = pd.read_csv(os.path.join(data_dir, "train_iit_reg.csv"))
    test_iit_reg = pd.read_csv(os.path.join(data_dir, "test_iit_reg.csv"))
    train_non_iit_reg = pd.read_csv(os.path.join(data_dir, "train_non_iit_reg.csv"))
    test_non_iit_reg = pd.read_csv(os.path.join(data_dir, "test_non_iit_reg.csv"))
    
    # Load classification datasets
    train_iit_clf = pd.read_csv(os.path.join(data_dir, "train_iit_clf.csv"))
    test_iit_clf = pd.read_csv(os.path.join(data_dir, "test_iit_clf.csv"))
    train_non_iit_clf = pd.read_csv(os.path.join(data_dir, "train_non_iit_clf.csv"))
    test_non_iit_clf = pd.read_csv(os.path.join(data_dir, "test_non_iit_clf.csv"))
    
    # Combine all data to generate the ultimate choices lookup dictionary for the API/web app.
    # The lookup dictionary will hold historical stats for each choice.
    print("Generating choices lookup table...")
    df_all = pd.concat([
        train_iit_reg, test_iit_reg,
        train_non_iit_reg, test_non_iit_reg
    ])
    
    # Get the latest year row for each choice to get the most updated historical features
    choices_lookup = {}
    grouped = df_all.groupby(['type', 'institute', 'program', 'quota', 'category', 'gender'])
    for name, group in grouped:
        latest_row = group.sort_values('year').iloc[-1]
        choices_lookup[name] = {
            'hist_crank_mean': float(latest_row['hist_crank_mean']),
            'hist_crank_min': float(latest_row['hist_crank_min']),
            'hist_crank_max': float(latest_row['hist_crank_max']),
            'last_closing_rank': int(latest_row['crank']),
            'last_year': int(latest_row['year'])
        }
        
    with open(os.path.join(models_dir, "choices_lookup.pkl"), "wb") as f:
        pickle.dump(choices_lookup, f)
    print(f"Saved {len(choices_lookup)} choices to choices_lookup.pkl")
    
    # Define categorical columns to encode
    cat_cols = ['institute', 'program', 'quota', 'category', 'gender']
    
    # We will build categorical mappings using all unique values in the raw dataset
    print("Building categorical mappings...")
    mappings = {}
    for col in cat_cols:
        all_vals = df_all[col].astype(str).unique()
        all_vals = sorted(all_vals)
        
        # Create map: val -> int
        mapping = {val: idx for idx, val in enumerate(all_vals)}
        # Add a special token for unknown values
        mapping['UNKNOWN'] = len(all_vals)
        mappings[col] = mapping
        
    # Save mappings
    with open(os.path.join(models_dir, "categorical_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    print("Saved categorical mappings.")
    
    def encode_df(df, mappings):
        df_encoded = df.copy()
        for col in cat_cols:
            mapping = mappings[col]
            df_encoded[col] = df_encoded[col].astype(str).map(
                lambda x: mapping.get(x, mapping['UNKNOWN'])
            )
        return df_encoded

    # Encode all dataframes
    train_iit_reg_enc = encode_df(train_iit_reg, mappings)
    test_iit_reg_enc = encode_df(test_iit_reg, mappings)
    train_non_iit_reg_enc = encode_df(train_non_iit_reg, mappings)
    test_non_iit_reg_enc = encode_df(test_non_iit_reg, mappings)
    
    train_iit_clf_enc = encode_df(train_iit_clf, mappings)
    test_iit_clf_enc = encode_df(test_iit_clf, mappings)
    train_non_iit_clf_enc = encode_df(train_non_iit_clf, mappings)
    test_non_iit_clf_enc = encode_df(test_non_iit_clf, mappings)
    
    # Feature columns
    reg_features = cat_cols + ['hist_crank_mean', 'hist_crank_min', 'hist_crank_max']
    clf_features = reg_features + ['candidate_rank', 'rank_diff', 'rank_ratio']
    
    # ------------------ IIT Models ------------------
    print("\n--- Training IIT Regressor ---")
    X_train_iit_reg = train_iit_reg_enc[reg_features]
    y_train_iit_reg = train_iit_reg_enc['crank']
    X_test_iit_reg = test_iit_reg_enc[reg_features]
    y_test_iit_reg = test_iit_reg_enc['crank']
    
    iit_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    iit_reg.fit(X_train_iit_reg, y_train_iit_reg)
    
    # Eval IIT Regressor
    preds = iit_reg.predict(X_test_iit_reg)
    mae = mean_absolute_error(y_test_iit_reg, preds)
    r2 = r2_score(y_test_iit_reg, preds)
    print(f"IIT Regressor R2 Score on 2025 Test Set: {r2:.4f}")
    print(f"IIT Regressor MAE: {mae:.2f} ranks")
    
    # Train IIT Classifier
    print("\n--- Training IIT Classifier ---")
    X_train_iit_clf = train_iit_clf_enc[clf_features]
    y_train_iit_clf = train_iit_clf_enc['admitted']
    X_test_iit_clf = test_iit_clf_enc[clf_features]
    y_test_iit_clf = test_iit_clf_enc['admitted']
    
    iit_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_seed=42) # Wait, random_seed or random_state? random_state is standard for XGBoost
    iit_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    iit_clf.fit(X_train_iit_clf, y_train_iit_clf)
    
    # Eval IIT Classifier
    probs = iit_clf.predict_proba(X_test_iit_clf)[:, 1]
    auc = roc_auc_score(y_test_iit_clf, probs)
    preds_clf = iit_clf.predict(X_test_iit_clf)
    print(f"IIT Classifier ROC-AUC on 2025 Test Set: {auc:.4f}")
    print(classification_report(y_test_iit_clf, preds_clf))
    
    # ------------------ Non-IIT Models ------------------
    print("\n--- Training Non-IIT Regressor ---")
    X_train_non_iit_reg = train_non_iit_reg_enc[reg_features]
    y_train_non_iit_reg = train_non_iit_reg_enc['crank']
    X_test_non_iit_reg = test_non_iit_reg_enc[reg_features]
    y_test_non_iit_reg = test_non_iit_reg_enc['crank']
    
    non_iit_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    non_iit_reg.fit(X_train_non_iit_reg, y_train_non_iit_reg)
    
    # Eval Non-IIT Regressor
    preds = non_iit_reg.predict(X_test_non_iit_reg)
    mae = mean_absolute_error(y_test_non_iit_reg, preds)
    r2 = r2_score(y_test_non_iit_reg, preds)
    print(f"Non-IIT Regressor R2 Score on 2025 Test Set: {r2:.4f}")
    print(f"Non-IIT Regressor MAE: {mae:.2f} ranks")
    
    # Train Non-IIT Classifier
    print("\n--- Training Non-IIT Classifier ---")
    X_train_non_iit_clf = train_non_iit_clf_enc[clf_features]
    y_train_non_iit_clf = train_non_iit_clf_enc['admitted']
    X_test_non_iit_clf = test_non_iit_clf_enc[clf_features]
    y_test_non_iit_clf = test_non_iit_clf_enc['admitted']
    
    non_iit_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    non_iit_clf.fit(X_train_non_iit_clf, y_train_non_iit_clf)
    
    # Eval Non-IIT Classifier
    probs = non_iit_clf.predict_proba(X_test_non_iit_clf)[:, 1]
    auc = roc_auc_score(y_test_non_iit_clf, probs)
    preds_clf = non_iit_clf.predict(X_test_non_iit_clf)
    print(f"Non-IIT Classifier ROC-AUC on 2025 Test Set: {auc:.4f}")
    print(classification_report(y_test_non_iit_clf, preds_clf))
    
    # Save all models
    print("\nSaving trained models to disk...")
    with open(os.path.join(models_dir, "xgb_iit_regressor.pkl"), "wb") as f:
        pickle.dump(iit_reg, f)
    with open(os.path.join(models_dir, "xgb_iit_classifier.pkl"), "wb") as f:
        pickle.dump(iit_clf, f)
    with open(os.path.join(models_dir, "xgb_non_iit_regressor.pkl"), "wb") as f:
        pickle.dump(non_iit_reg, f)
    with open(os.path.join(models_dir, "xgb_non_iit_classifier.pkl"), "wb") as f:
        pickle.dump(non_iit_clf, f)
        
    print("All models successfully saved!")

if __name__ == '__main__':
    train_and_save_models(r"d:\jeerankpred\data\processed", r"d:\jeerankpred\models")
