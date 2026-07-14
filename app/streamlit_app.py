import streamlit as st
import requests
import pandas as pd
import json

# Setup page config
st.set_page_config(
    page_title="JoSAA ML College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS for aesthetics (sleek dark glassmorphism theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Title Styling */
    .title-gradient {
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Card design */
    .card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Card */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin-top: 15px;
    }
    .metric-item {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        flex: 1;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #818cf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 5rem;
        padding: 20px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Table headers */
    thead tr th {
        background-color: #312e81 !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="title-gradient">JoSAA ML College Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning-Based Choice Recommendation and Cutoff Predictor</div>', unsafe_allow_html=True)

# API Endpoint definition
API_URL = "http://localhost:8000"

# Sidebar controls
st.sidebar.markdown("### 👤 Candidate Profile")

exam_type = st.sidebar.selectbox(
    "Select Exam & Colleges",
    ["NIT-IIIT-GFTI (JEE Main)", "IIT (JEE Advanced)"],
    index=0
)
model_type = "IIT" if exam_type.startswith("IIT") else "NIT"

rank = st.sidebar.number_input(
    "Enter JEE Rank",
    min_value=1,
    max_value=1500000,
    value=15000,
    step=100
)

category = st.sidebar.selectbox(
    "Seat Category",
    ["OPEN", "OBC-NCL", "SC", "ST", "EWS", "OPEN (PwD)", "OBC-NCL (PwD)", "SC (PwD)", "ST (PwD)", "EWS (PwD)"]
)

gender = st.sidebar.selectbox(
    "Gender Pool",
    ["Gender-Neutral", "Female-only"]
)

# Standard Indian States
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", 
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", 
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", 
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", 
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", 
    "Puducherry", "Chandigarh", "Ladakh"
]

home_state = st.sidebar.selectbox(
    "Home State (For Quota)",
    sorted(INDIAN_STATES),
    index=sorted(INDIAN_STATES).index("Delhi") if "Delhi" in INDIAN_STATES else 0
)

# ---- Filters (only shown when JEE Main mode selected) ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# Institute type filter (JEE Main only)
if model_type == "NIT":
    institute_types = st.sidebar.multiselect(
        "Institute Type",
        ["NIT", "IIIT", "GFTI"],
        default=["NIT", "IIIT", "GFTI"],
        help="Filter by National Institutes of Technology (NIT), Indian Institutes of Information Technology (IIIT), or Government Funded Technical Institutes (GFTI)."
    )
else:
    institute_types = []  # not applicable for IITs

# Branch / Domain filter
BRANCH_CATEGORIES = {
    "Computer Science & IT": ["Computer Science", "Information Technology", "Computing", "Artificial Intelligence", "Machine Learning", "Data Science", "Software"],
    "Electronics & Communication": ["Electronics", "Communication", "VLSI", "Microelectronics"],
    "Electrical Engineering": ["Electrical", "Power", "Instrumentation"],
    "Mechanical Engineering": ["Mechanical", "Manufacturing", "Industrial", "Production", "Aerospace", "Automobile"],
    "Civil Engineering": ["Civil", "Structural", "Environmental", "Geotechnical"],
    "Chemical Engineering": ["Chemical", "Biochemical", "Petrochemical", "Polymer"],
    "Metallurgy & Materials": ["Metallurgy", "Materials", "Mining"],
    "Mathematics & Computing": ["Mathematics", "Statistics", "Mathematical"],
    "Physics & Engineering Physics": ["Physics", "Engineering Physics", "Photonics"],
    "Biotechnology & Biomedical": ["Biotechnology", "Biomedical", "Bioscience", "Life Sciences"],
    "Architecture & Planning": ["Architecture", "Planning", "Design"],
    "Other Engineering": ["Ocean", "Naval", "Textile", "Ceramic", "Food", "Agricultural"],
}

selected_branch_cats = st.sidebar.multiselect(
    "Branch / Domain",
    list(BRANCH_CATEGORIES.keys()),
    default=[],
    help="Leave blank to include all branches. Select one or more to filter results."
)

# Flatten keywords from selected categories
branch_keywords = []
for cat_name in selected_branch_cats:
    branch_keywords.extend(BRANCH_CATEGORIES[cat_name])
# Deduplicate
branch_keywords = list(set(branch_keywords))

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Dynamically filters out incompatible quotas and gender pools.
2. Runs your rank through an **XGBoost Classifier** to compute admission probability.
3. Runs your rank through an **XGBoost Regressor** to predict expected round 6 closing ranks.
""")

# Action trigger
if st.sidebar.button("🔮 Predict Admissions", use_container_width=True):
    with st.spinner("Analyzing historical trends and running ML models..."):
        try:
            # Format payload
            payload = {
                "type": model_type,
                "rank": rank,
                "category": category,
                "gender": gender,
                "home_state": home_state,
                "institute_types": institute_types if institute_types else None,
                "branch_keywords": branch_keywords if branch_keywords else None,
            }
            
            response = requests.post(f"{API_URL}/recommend", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if results are empty
                if not data['safe'] and not data['reach'] and not data['dream']:
                    st.warning("No colleges found matching your profile. Please check if your rank/category combinations are correct.")
                else:
                    # Tabs for results
                    tab1, tab2, tab3 = st.tabs([
                        "🎯 Safe Choices (High Probability)",
                        "📈 Reach Choices (Medium Probability)",
                        "✨ Dream Choices (Low Probability / Reach Target)"
                    ])
                    
                    def format_results_df(choices_list):
                        if not choices_list:
                            return None
                        
                        df = pd.DataFrame(choices_list)
                        # Reorder columns
                        cols = ['institute', 'program', 'quota', 'last_closing_rank', 'predicted_closing_rank', 'admission_probability']
                        df = df[cols]
                        
                        # Rename columns
                        df.columns = [
                            'Institute Name', 'Branch / Program', 'Quota',
                            '2025 Closing Rank', 'Predicted Closing Rank', 'Admission Chance'
                        ]
                        
                        # Format percentage
                        df['Admission Chance'] = df['Admission Chance'].map(lambda x: f"{x * 100:.1f}%")
                        return df

                    # Active filter summary shown above tabs
                    filter_labels = []
                    if model_type == 'NIT' and institute_types:
                        filter_labels.append(f"📌 Institute types: {', '.join(institute_types)}")
                    if selected_branch_cats:
                        filter_labels.append(f"🎓 Branches: {', '.join(selected_branch_cats)}")
                    if filter_labels:
                        st.info('  •  '.join(filter_labels))
                    
                    with tab1:
                        st.markdown("### Safe Colleges")
                        st.write("These choices have a **high probability (80%+)** of admission based on historical closing ranks and ML forecasting.")
                        df_safe = format_results_df(data['safe'])
                        if df_safe is not None:
                            st.dataframe(df_safe, use_container_width=True, hide_index=True)
                        else:
                            st.info("No colleges fall in the 'Safe' category for this rank.")
                            
                    with tab2:
                        st.markdown("### Reach Colleges")
                        st.write("These choices have a **moderate probability (30% - 80%)** of admission. Highly recommended to place these in your choice list.")
                        df_reach = format_results_df(data['reach'])
                        if df_reach is not None:
                            st.dataframe(df_reach, use_container_width=True, hide_index=True)
                        else:
                            st.info("No colleges fall in the 'Reach' category for this rank.")
                            
                    with tab3:
                        st.markdown("### Dream Colleges")
                        st.write("These choices are competitive but have a **slight probability (2% - 30%)** of admission. Worth adding to the top of your choice list.")
                        df_dream = format_results_df(data['dream'])
                        if df_dream is not None:
                            st.dataframe(df_dream, use_container_width=True, hide_index=True)
                        else:
                            st.info("No colleges fall in the 'Dream' category for this rank.")
                            
                    # Add Trend Analysis Tool
                    st.markdown("---")
                    st.markdown("### 📊 Trend Analysis & Insights")
                    
                    # Gather unique colleges from recommendations for visualization
                    all_recs = data['safe'] + data['reach'] + data['dream']
                    unique_insts = sorted(list(set([r['institute'] for r in all_recs])))
                    
                    if unique_insts:
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_inst = st.selectbox("Select Institute for details", unique_insts)
                            
                        # Filter programs for this institute
                        inst_programs = sorted(list(set([r['program'] for r in all_recs if r['institute'] == selected_inst])))
                        with col2:
                            selected_prog = st.selectbox("Select Branch / Program", inst_programs)
                            
                        # Find the choice details
                        selected_choice = next(
                            (r for r in all_recs if r['institute'] == selected_inst and r['program'] == selected_prog),
                            None
                        )
                        
                        if selected_choice:
                            # Card showing details
                            prob_pct = selected_choice['admission_probability'] * 100
                            st.markdown(f"""
                            <div class="card">
                                <h4>{selected_inst}</h4>
                                <p style="color:#94a3b8; font-size:1.05rem;">{selected_prog}</p>
                                <div class="metric-container">
                                    <div class="metric-item">
                                        <div class="metric-value">{prob_pct:.1f}%</div>
                                        <div class="metric-label">Admission Chance</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">{selected_choice['predicted_closing_rank']}</div>
                                        <div class="metric-label">Predicted 2026 Cutoff</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">{selected_choice['last_closing_rank']}</div>
                                        <div class="metric-label">2025 Cutoff</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">{int(selected_choice['historical_mean'])}</div>
                                        <div class="metric-label">5-Year Average Cutoff</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please make sure the API server is running on http://localhost:8000")

else:
    # Initial landing view
    st.markdown("""
    <div class="card" style="text-align:center; padding: 40px;">
        <h3 style="color:#818cf8;">Welcome to the JoSAA Seat Predictor</h3>
        <p style="color:#94a3b8; font-size:1.1rem; max-width:600px; margin: 15px auto;">
            Enter your JEE Main/Advanced rank, category, gender, and home state in the sidebar on the left, and click <b>Predict Admissions</b> to run our machine learning recommendations.
        </p>
        <img src="https://images.unsplash.com/photo-1523050854058-8df90110c9f1?q=80&w=400&auto=format&fit=crop" style="border-radius:12px; margin-top:20px; max-width:350px; opacity:0.85;" />
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with FastAPI, Streamlit & XGBoost. JoSAA ML Predictor © 2026.</div>', unsafe_allow_html=True)
