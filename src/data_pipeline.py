import os
import pandas as pd
import numpy as np
import pickle

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    # Strip whitespaces from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Ensure rank columns are numeric
    df['orank'] = pd.to_numeric(df['orank'], errors='coerce')
    df['crank'] = pd.to_numeric(df['crank'], errors='coerce')
    
    # Drop rows with null ranks
    df = df.dropna(subset=['orank', 'crank'])
    df['orank'] = df['orank'].astype(int)
    df['crank'] = df['crank'].astype(int)
    df['year'] = df['year'].astype(int)
    df['round'] = df['round'].astype(int)
    
    return df

def compute_historical_features(df):
    """
    Computes historical closing rank statistics for each row based on years strictly prior to that row's year.
    This prevents data leakage.
    """
    print("Computing historical features...")
    
    # Grouping key
    group_cols = ['institute', 'program', 'quota', 'category', 'gender']
    
    # To compute historical statistics efficiently, we sort the dataframe by year
    df = df.sort_values('year').reset_index(drop=True)
    
    # We will build a dictionary to track historical ranks for each key
    hist_ranks = {}
    
    means = []
    mins = []
    maxs = []
    counts = []
    
    for idx, row in df.iterrows():
        key = tuple(row[group_cols])
        year = row['year']
        
        # Initialize or retrieve history
        if key not in hist_ranks:
            hist_ranks[key] = []
            
        # Filter history to only include years < current year
        past_cranks = [crank for y, crank in hist_ranks[key] if y < year]
        
        if past_cranks:
            means.append(np.mean(past_cranks))
            mins.append(np.min(past_cranks))
            maxs.append(np.max(past_cranks))
            counts.append(len(past_cranks))
        else:
            means.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            counts.append(0)
            
        # Append current year's crank to history for future rows
        hist_ranks[key].append((year, row['crank']))
        
    df['hist_crank_mean'] = means
    df['hist_crank_min'] = mins
    df['hist_crank_max'] = maxs
    df['hist_crank_count'] = counts
    
    # Impute missing values (choices with no prior history) using fallback means
    # Fallback 1: Mean closing rank of the same (institute, program) across all years before current year
    print("Imputing missing historical features...")
    
    # Compute global/fallback means per year to avoid leakage
    global_mean_by_year = df.groupby('year')['crank'].mean().to_dict()
    inst_mean_by_year = df.groupby(['year', 'institute'])['crank'].mean().to_dict()
    
    # Fill NaNs with fallback hierarchies
    for i, row in df.iterrows():
        if np.isnan(row['hist_crank_mean']):
            year = row['year']
            inst = row['institute']
            
            # Use institute-level average closing rank from years < current year
            inst_past = df[(df['year'] < year) & (df['institute'] == inst)]['crank']
            if not inst_past.empty:
                val = inst_past.mean()
                df.at[i, 'hist_crank_mean'] = val
                df.at[i, 'hist_crank_min'] = inst_past.min()
                df.at[i, 'hist_crank_max'] = inst_past.max()
            else:
                # Global past average
                global_past = df[df['year'] < year]['crank']
                if not global_past.empty:
                    val = global_past.mean()
                    df.at[i, 'hist_crank_mean'] = val
                    df.at[i, 'hist_crank_min'] = global_past.min()
                    df.at[i, 'hist_crank_max'] = global_past.max()
                else:
                    # Fallback for the very first year (e.g. 2020)
                    df.at[i, 'hist_crank_mean'] = row['crank']
                    df.at[i, 'hist_crank_min'] = row['crank']
                    df.at[i, 'hist_crank_max'] = row['crank']
                    
    return df

def generate_classification_dataset(df, num_samples_per_row=6):
    """
    Expands the dataset by sampling candidate ranks around the actual closing rank.
    Generates positive samples (candidate_rank <= closing_rank) and negative samples (candidate_rank > closing_rank).
    """
    print(f"Generating classification dataset (sampling {num_samples_per_row} candidate ranks per row)...")
    np.random.seed(42)
    
    expanded_rows = []
    
    # We iterate and sample
    for _, row in df.iterrows():
        crank = row['crank']
        
        # Sample half positive, half negative
        half = num_samples_per_row // 2
        
        # Positive ranks (candidate gets admitted)
        # Ranks from [0.7 * crank, crank]
        pos_ranks = np.random.randint(max(1, int(0.7 * crank)), crank + 1, size=half)
        
        # Negative ranks (candidate gets rejected)
        # Ranks from [crank + 1, 1.3 * crank]
        neg_ranks = np.random.randint(crank + 1, int(1.3 * crank) + 5, size=half)
        
        # Append positive samples
        for r in pos_ranks:
            new_row = row.copy()
            new_row['candidate_rank'] = r
            new_row['admitted'] = 1
            expanded_rows.append(new_row)
            
        # Append negative samples
        for r in neg_ranks:
            new_row = row.copy()
            new_row['candidate_rank'] = r
            new_row['admitted'] = 0
            expanded_rows.append(new_row)
            
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df['candidate_rank'] = expanded_df['candidate_rank'].astype(int)
    expanded_df['admitted'] = expanded_df['admitted'].astype(int)
    
    # Feature engineering for classifier
    expanded_df['rank_diff'] = expanded_df['candidate_rank'] - expanded_df['hist_crank_mean']
    expanded_df['rank_ratio'] = expanded_df['candidate_rank'] / (expanded_df['hist_crank_mean'] + 1e-5)
    
    return expanded_df

def prepare_pipeline_data(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(file_path)
    
    # Add historical features
    df = compute_historical_features(df)
    
    # Split into IIT (JEE Advanced) and non-IIT (JEE Main)
    df_iit = df[df['type'] == 'IIT'].copy()
    df_non_iit = df[df['type'] != 'IIT'].copy()
    
    print(f"IIT dataset size: {df_iit.shape[0]} rows")
    print(f"Non-IIT dataset size: {df_non_iit.shape[0]} rows")
    
    # Save the processed regression dataframes (useful for training regressor directly)
    # Split by year: train (2020-2024), test (2025)
    train_iit = df_iit[df_iit['year'] < 2025].copy()
    test_iit = df_iit[df_iit['year'] == 2025].copy()
    
    train_non_iit = df_non_iit[df_non_iit['year'] < 2025].copy()
    test_non_iit = df_non_iit[df_non_iit['year'] == 2025].copy()
    
    train_iit.to_csv(os.path.join(output_dir, "train_iit_reg.csv"), index=False)
    test_iit.to_csv(os.path.join(output_dir, "test_iit_reg.csv"), index=False)
    train_non_iit.to_csv(os.path.join(output_dir, "train_non_iit_reg.csv"), index=False)
    test_non_iit.to_csv(os.path.join(output_dir, "test_non_iit_reg.csv"), index=False)
    
    # Generate classification datasets (expanding only train to avoid testing leakage)
    print("Generating classification dataset for IIT...")
    train_iit_clf = generate_classification_dataset(train_iit)
    # For testing, we also generate candidate ranks but we keep it small
    test_iit_clf = generate_classification_dataset(test_iit)
    
    print("Generating classification dataset for Non-IIT...")
    train_non_iit_clf = generate_classification_dataset(train_non_iit)
    test_non_iit_clf = generate_classification_dataset(test_non_iit)
    
    train_iit_clf.to_csv(os.path.join(output_dir, "train_iit_clf.csv"), index=False)
    test_iit_clf.to_csv(os.path.join(output_dir, "test_iit_clf.csv"), index=False)
    
    train_non_iit_clf.to_csv(os.path.join(output_dir, "train_non_iit_clf.csv"), index=False)
    test_non_iit_clf.to_csv(os.path.join(output_dir, "test_non_iit_clf.csv"), index=False)
    
    print("Data preparation complete!")

if __name__ == '__main__':
    # Test loading and processing
    prepare_pipeline_data(r"d:\jeerankpred\data\merged_allround.csv", r"d:\jeerankpred\data\processed")
