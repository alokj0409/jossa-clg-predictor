import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Define the state mapping for Home State quota matching
STATE_KEYWORDS = {
    "Tiruchirappalli": "Tamil Nadu",
    "Trichy": "Tamil Nadu",
    "Jaipur": "Rajasthan",
    "Allahabad": "Uttar Pradesh",
    "MNNIT": "Uttar Pradesh",
    "Bhopal": "Madhya Pradesh",
    "MANIT": "Madhya Pradesh",
    "Calicut": "Kerala",
    "Delhi": "Delhi",
    "Durgapur": "West Bengal",
    "Goa": "Goa",
    "Hamirpur": "Himachal Pradesh",
    "Surathkal": "Karnataka",
    "Karnataka": "Karnataka",
    "Kurukshetra": "Haryana",
    "Jalandhar": "Punjab",
    "Jamshedpur": "Jharkhand",
    "Nagpur": "Maharashtra",
    "VNIT": "Maharashtra",
    "Patna": "Bihar",
    "Raipur": "Chhattisgarh",
    "Rourkela": "Odisha",
    "Silchar": "Assam",
    "Srinagar": "Jammu & Kashmir",
    "Surat": "Gujarat",
    "SVNIT": "Gujarat",
    "Warangal": "Telangana",
    "Agartala": "Tripura",
    "Arunachal": "Arunachal Pradesh",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Puducherry": "Puducherry",
    "Pondicherry": "Puducherry",
    "Sikkim": "Sikkim",
    "Uttarakhand": "Uttarakhand",
    "Andhra": "Andhra Pradesh",
    "Shibpur": "West Bengal",
    "IIEST": "West Bengal",
    "Gwalior": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh",
    "Kancheepuram": "Tamil Nadu",
    "Vadodara": "Gujarat",
    "Kota": "Rajasthan",
    "Sri City": "Andhra Pradesh",
    "Guwahati": "Assam",
    "Kalyani": "West Bengal",
    "Lucknow": "Uttar Pradesh",
    "Pune": "Maharashtra",
    "Surat": "Gujarat"
}

def get_institute_state(institute_name: str) -> str:
    """Infers the state of an institute based on keywords in its name."""
    for kw, state in STATE_KEYWORDS.items():
        if kw.lower() in institute_name.lower():
            return state
    return "Other"

app = FastAPI(title="JoSAA ML College Predictor API")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Load models and mappings on startup
encoders_mapping = None
choices_lookup = None
iit_reg = None
iit_clf = None
non_iit_reg = None
non_iit_clf = None

@app.on_event("startup")
def load_assets():
    global encoders_mapping, choices_lookup, iit_reg, iit_clf, non_iit_reg, non_iit_clf
    
    print("Loading models and lookups...")
    try:
        with open(os.path.join(MODELS_DIR, "categorical_mappings.pkl"), "rb") as f:
            encoders_mapping = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, "choices_lookup.pkl"), "rb") as f:
            choices_lookup = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, "xgb_iit_regressor.pkl"), "rb") as f:
            iit_reg = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, "xgb_iit_classifier.pkl"), "rb") as f:
            iit_clf = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, "xgb_non_iit_regressor.pkl"), "rb") as f:
            non_iit_reg = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, "xgb_non_iit_classifier.pkl"), "rb") as f:
            non_iit_clf = pickle.load(f)
            
        print("All assets loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

class PredictRequest(BaseModel):
    type: str  # IIT or NIT, IIIT, etc.
    institute: str
    program: str
    quota: str
    category: str
    gender: str
    candidate_rank: int

class RecommendRequest(BaseModel):
    type: str  # IIT or NIT-IIIT-GFTI
    rank: int
    category: str
    gender: str
    home_state: str
    institute_types: Optional[List[str]] = None   # e.g. ["NIT", "IIIT", "GFTI"]
    branch_keywords: Optional[List[str]] = None   # e.g. ["Computer Science", "Electronics"]

def encode_features(features_dict):
    cat_cols = ['institute', 'program', 'quota', 'category', 'gender']
    encoded = {}
    for col in cat_cols:
        mapping = encoders_mapping[col]
        val = features_dict[col]
        encoded[col] = mapping.get(str(val).strip(), mapping['UNKNOWN'])
    return encoded

@app.post("/predict")
def predict_admission(req: PredictRequest):
    if choices_lookup is None:
        raise HTTPException(status_code=500, detail="Models not loaded yet.")
        
    key = (req.type, req.institute, req.program, req.quota, req.category, req.gender)
    if key not in choices_lookup:
        # Check if we can find a matching choice with a fallback gender or category
        # to see if the combination exists at all.
        raise HTTPException(status_code=404, detail="Choice combination not found in historical data.")
        
    stats = choices_lookup[key]
    hist_mean = stats['hist_crank_mean']
    hist_min = stats['hist_crank_min']
    hist_max = stats['hist_crank_max']
    
    # Feature engineering
    rank_diff = req.candidate_rank - hist_mean
    rank_ratio = req.candidate_rank / (hist_mean + 1e-5)
    
    encoded = encode_features({
        'institute': req.institute,
        'program': req.program,
        'quota': req.quota,
        'category': req.category,
        'gender': req.gender
    })
    
    # Predict closing rank
    reg_features = [
        encoded['institute'], encoded['program'], encoded['quota'],
        encoded['category'], encoded['gender'], hist_mean, hist_min, hist_max
    ]
    
    clf_features = reg_features + [req.candidate_rank, rank_diff, rank_ratio]
    
    # Run appropriate model
    if req.type == 'IIT':
        pred_closing_rank = iit_reg.predict([reg_features])[0]
        prob = iit_clf.predict_proba([clf_features])[0][1]
    else:
        pred_closing_rank = non_iit_reg.predict([reg_features])[0]
        prob = non_iit_clf.predict_proba([clf_features])[0][1]
        
    return {
        "admission_probability": float(prob),
        "predicted_closing_rank": float(pred_closing_rank),
        "historical_mean": hist_mean,
        "historical_min": hist_min,
        "historical_max": hist_max,
        "last_closing_rank": stats['last_closing_rank'],
        "last_year": stats['last_year']
    }

@app.post("/recommend")
def recommend_colleges(req: RecommendRequest):
    if choices_lookup is None:
        raise HTTPException(status_code=500, detail="Models not loaded yet.")
        
    is_iit = (req.type == 'IIT')
    
    # Filters
    category_filter = req.category.strip()
    gender_filter = req.gender.strip() # 'Gender-Neutral' or 'Female-only'
    
    results = []
    
    # Pre-encode common category/gender features to speed up loop
    for key, stats in choices_lookup.items():
        choice_type, inst, prog, quota, cat, gen = key
        
        # Check if type matches (IIT vs non-IIT)
        if is_iit and choice_type != 'IIT':
            continue
        if not is_iit and choice_type == 'IIT':
            continue

        # Institute type filter (NIT / IIIT / GFTI) — only applies for JEE Main mode
        if not is_iit and req.institute_types:
            # The dataset stores 'NIT', '3IT' (for IIIT), 'CFI' (for GFTI/Other)
            # Map user-friendly labels to dataset values
            type_map = {'NIT': 'NIT', 'IIIT': '3IT', 'GFTI': 'CFI'}
            allowed_types = {type_map.get(t, t) for t in req.institute_types}
            if choice_type not in allowed_types:
                continue

        # Branch keyword filter — case-insensitive substring match on program
        if req.branch_keywords:
            prog_lower = prog.lower()
            if not any(kw.lower() in prog_lower for kw in req.branch_keywords):
                continue
            
        # Category match
        if cat != category_filter:
            continue
            
        # Gender match:
        # If user is Female, they can access both 'Female-only' and 'Gender-Neutral'
        # If user is Male/Neutral, they can only access 'Gender-Neutral'
        if gender_filter == 'Gender-Neutral' and gen != 'Gender-Neutral':
            continue
        if gender_filter == 'Female-only' and gen not in ['Female-only', 'Gender-Neutral']:
            continue
            
        # Quota eligibility for NIT/IIIT/GFTI:
        # IITs are always AI (All India)
        if not is_iit:
            inst_state = get_institute_state(inst)
            is_home_state = (inst_state.lower() == req.home_state.lower())
            
            # If it is their home state college, they should get the HS quota row.
            # If it is NOT their home state college, they should get OS (Other State) or AI (All India).
            if is_home_state:
                # Keep HS (Home State) quota. Ignore OS.
                if quota == 'OS':
                    continue
            else:
                # Keep OS or AI quota. Ignore HS.
                if quota == 'HS':
                    continue
                    
        # Choice is eligible! Let's calculate prediction features
        hist_mean = stats['hist_crank_mean']
        hist_min = stats['hist_crank_min']
        hist_max = stats['hist_crank_max']
        
        rank_diff = req.rank - hist_mean
        rank_ratio = req.rank / (hist_mean + 1e-5)
        
        encoded = encode_features({
            'institute': inst,
            'program': prog,
            'quota': quota,
            'category': cat,
            'gender': gen
        })
        
        reg_features = [
            encoded['institute'], encoded['program'], encoded['quota'],
            encoded['category'], encoded['gender'], hist_mean, hist_min, hist_max
        ]
        clf_features = reg_features + [req.rank, rank_diff, rank_ratio]
        
        results.append({
            'type': choice_type,
            'institute': inst,
            'program': prog,
            'quota': quota,
            'category': cat,
            'gender': gen,
            'reg_features': reg_features,
            'clf_features': clf_features,
            'historical_mean': hist_mean,
            'last_closing_rank': stats['last_closing_rank']
        })
        
    if not results:
        return {"dream": [], "reach": [], "safe": []}
        
    # Batch predict to be faster
    df_res = pd.DataFrame(results)
    
    # Encode values
    reg_features_list = df_res['reg_features'].tolist()
    clf_features_list = df_res['clf_features'].tolist()
    
    if is_iit:
        pred_closing_ranks = iit_reg.predict(reg_features_list)
        probs = iit_clf.predict_proba(clf_features_list)[:, 1]
    else:
        pred_closing_ranks = non_iit_reg.predict(reg_features_list)
        probs = non_iit_clf.predict_proba(clf_features_list)[:, 1]
        
    df_res['predicted_closing_rank'] = pred_closing_ranks
    df_res['admission_probability'] = probs
    
    # Drop temp columns
    df_res = df_res.drop(columns=['reg_features', 'clf_features'])
    
    # Categorize choices
    dream = []
    reach = []
    safe = []
    
    for _, row in df_res.iterrows():
        p = row['admission_probability']
        item = {
            "institute": row['institute'],
            "program": row['program'],
            "quota": row['quota'],
            "gender": row['gender'],
            "last_closing_rank": int(row['last_closing_rank']),
            "predicted_closing_rank": int(row['predicted_closing_rank']),
            "historical_mean": float(row['historical_mean']),
            "admission_probability": float(p)
        }
        
        if p >= 0.80:
            safe.append(item)
        elif 0.30 <= p < 0.80:
            reach.append(item)
        elif 0.02 <= p < 0.30:
            dream.append(item)
            
    # Sort results
    # For safe and reach, sort by last_closing_rank ascending (better colleges first)
    safe = sorted(safe, key=lambda x: x['last_closing_rank'])
    reach = sorted(reach, key=lambda x: x['last_closing_rank'])
    # For dream, sort by probability descending (highest chance of dream colleges first)
    dream = sorted(dream, key=lambda x: x['admission_probability'], reverse=True)
    
    # Limit to top 50 in each category
    return {
        "dream": dream[:50],
        "reach": reach[:50],
        "safe": safe[:50]
    }
