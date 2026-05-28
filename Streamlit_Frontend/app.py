import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Define Base Directory and Absolute Paths ---
BASE_DIR = r"Streamlit_Frontend"

# --- Configuration and Load Resources ---
@st.cache_resource
def load_resources():
    model_path = os.path.join(BASE_DIR, 'final_house_price_model.joblib')
    features_path = os.path.join(BASE_DIR, 'feature_names.joblib')
    kmeans_path = os.path.join(BASE_DIR, 'kmeans_object.joblib')
    address_map_path = os.path.join(BASE_DIR, 'address_mapping.joblib')
    global_mean_path = os.path.join(BASE_DIR, 'global_mean_price_log.joblib')

    # Initialize variables to None
    model = None
    feature_names = None
    kmeans_model = None
    address_mapping = None
    global_mean_price_log = None
    
    current_loading_file = ""

    try:
        status_placeholder = st.empty()
        
        current_loading_file = "final_house_price_model.joblib"
        status_placeholder.info(f"⏳ Processing: {current_loading_file} ...")
        model = joblib.load(model_path)
        
        current_loading_file = "feature_names.joblib"
        status_placeholder.info(f"⏳ Processing: {current_loading_file} ...")
        feature_names = joblib.load(features_path)
        
        current_loading_file = "kmeans_object.joblib"
        status_placeholder.info(f"⏳ Processing: {current_loading_file} ...")
        kmeans_model = joblib.load(kmeans_path)
        
        current_loading_file = "address_mapping.joblib"
        status_placeholder.info(f"⏳ Processing: {current_loading_file} ...")
        address_mapping = joblib.load(address_map_path)
        
        current_loading_file = "global_mean_price_log.joblib"
        status_placeholder.info(f"⏳ Processing: {current_loading_file} ...")
        global_mean_price_log = joblib.load(global_mean_path)
        
        status_placeholder.empty() # Clear if successful

    except Exception as e:
        st.error("❌ **Pickle Serialization / Corruption Error Detected!**")
        st.markdown(
            f"""
            ### Broken File Spotted: `{current_loading_file}`
            
            The script stopped while trying to deserialize **`{current_loading_file}`**. The file data is incomplete or unreadable.
            
            **How to quickly fix this specific artifact:**
            1. Go back to the Notebook where you performed KMeans clustering.
            2. Locate your trained KMeans object and export it fresh using:
               ```python
               joblib.dump(kmeans, 'kmeans_object.joblib', compress=3)
               ```
            3. Replace the old file inside `{BASE_DIR}` with your freshly saved one.
            4. Click the **three dots $(\dots)$** in the top-right corner of this app, click **"Clear cache"**, and refresh!
            """
        )
        st.stop()

    return model, feature_names, kmeans_model, address_mapping, global_mean_price_log

# Execute resource loader safely
model, feature_names, kmeans_model, address_mapping, global_mean_price_log = load_resources()

# --- Helper Functions for Preprocessing ---
def preprocess_inputs(user_inputs, feature_names, kmeans_model, address_mapping, global_mean_price_log):
    input_df = pd.DataFrame([user_inputs])
    processed_input = pd.DataFrame(0, index=[0], columns=feature_names)

    for col in ['BHK_NO.', 'SQUARE_FT', 'LONGITUDE', 'LATITUDE']:
        if col in input_df.columns:
            processed_input.loc[0, col] = input_df.loc[0, col]

    processed_input.loc[0, 'UNDER_CONSTRUCTION'] = int(input_df.loc[0, 'UNDER_CONSTRUCTION'])
    processed_input.loc[0, 'RERA'] = int(input_df.loc[0, 'RERA'])
    processed_input.loc[0, 'READY_TO_MOVE'] = int(input_df.loc[0, 'READY_TO_MOVE'])
    processed_input.loc[0, 'RESALE'] = int(input_df.loc[0, 'RESALE'])

    if user_inputs['POSTED_BY'] == 'Dealer':
        processed_input.loc[0, 'POSTED_BY_Dealer'] = 1
    elif user_inputs['POSTED_BY'] == 'Owner':
        processed_input.loc[0, 'POSTED_BY_Owner'] = 1

    if user_inputs['BHK_OR_RK'] == 'RK':
        processed_input.loc[0, 'BHK_OR_RK_RK'] = 1

    coords = np.array([[user_inputs['LONGITUDE'], user_inputs['LATITUDE']]])
    processed_input.loc[0, 'Location_Cluster'] = kmeans_model.predict(coords)[0]

    address = user_inputs['ADDRESS']
    processed_input.loc[0, 'ADDRESS_Target_Encoded'] = address_mapping.get(address, global_mean_price_log)
    processed_input = processed_input[feature_names]

    return processed_input

# --- Streamlit App UI Configuration ---
st.set_page_config(page_title="House Price Predictor", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
    .stApp { background-color: #F4F6F8; }
    h1 { color: #0A2540 !important; font-weight: 700; }
    h2, h3 { color: #639FAB !important; }
    .stMarkdown, p, label { color: #0A2540 !important; }
    div.stButton > button:first-child {
        background-color: #2A9D8F !important;
        color: white !important;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🏡 House Price Prediction App")
st.markdown("--- ✨ Predict the market value of your property using Machine Learning ✨ ---")

st.sidebar.header("Input House Features")
all_locations = sorted(list(address_mapping.keys())) if address_mapping else []

with st.sidebar.form("input_form"):
    st.subheader("Property Details")
    bhk_no = st.slider("Number of BHK", min_value=1, max_value=10, value=2, step=1)
    square_ft = st.slider("Square Footage (sq ft)", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0, format="%.1f")
    bhk_or_rk = st.radio("BHK or RK", options=['BHK', 'RK'], index=0)

    st.subheader("Location Details")
    address = st.selectbox("Select Property Address Location", options=all_locations, index=0 if all_locations else None)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.597960, format="%.6f")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.969910, format="%.6f")

    st.subheader("Property Status")
    under_construction = st.checkbox("Under Construction", value=False)
    rera = st.checkbox("RERA Certified", value=False)
    ready_to_move = st.checkbox("Ready To Move", value=True)
    resale = st.checkbox("Resale Property", value=True)

    st.subheader("Posted By")
    posted_by = st.radio("Posted By", options=['Owner', 'Dealer', 'Builder'], index=0)

    st.markdown("--- ")
    submit_button = st.form_submit_button("✨ Predict Price ✨")

st.markdown("## Prediction Results")

if submit_button:
    user_inputs = {
        'BHK_NO.': bhk_no, 'SQUARE_FT': square_ft, 'LONGITUDE': longitude, 'LATITUDE': latitude,
        'UNDER_CONSTRUCTION': under_construction, 'RERA': rera, 'READY_TO_MOVE': ready_to_move,
        'RESALE': resale, 'POSTED_BY': posted_by, 'BHK_OR_RK': bhk_or_rk, 'ADDRESS': address 
    }

    try:
        processed_data = preprocess_inputs(user_inputs, feature_names, kmeans_model, address_mapping, global_mean_price_log)
        log_predicted_price = model.predict(processed_data)[0]
        predicted_price = np.expm1(log_predicted_price)

        st.success(f"**Predicted House Price:** {predicted_price:,.2f} Lacs")
        st.info("*(Prices are displayed in Indian Lacs)*")
        st.dataframe(processed_data)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("--- ")
st.caption("Developed by Almas Parwaiz Data Scientist for house price prediction.")
