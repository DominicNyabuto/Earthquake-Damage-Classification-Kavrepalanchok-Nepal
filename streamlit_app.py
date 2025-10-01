# # STREAMLIT WEB APP


# # app.py
# import streamlit as st
# import pandas as pd
# import lightgbm as lgb
# import plotly.express as px 
# import joblib

# # --- Load trained LightGBM model ---
# best_lgbm = joblib.load(
#     "results/models/best_lightgbm_model.pkl"
# )

# # --- Load example dataframe for defaults & category info ---
# df_clean = pd.read_csv(
#     "data/kavrepalanchok_test.csv"
# )

# # Drop 'b_id' if present
# if 'b_id' in df_clean.columns:
#     df_clean = df_clean.drop('b_id', axis=1)

# # --- Identify categorical features from training ---
# categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
# numeric_features = df_clean.select_dtypes(exclude=['object']).columns.tolist()

# # --- Convert categorical columns to category type (match training) ---
# for col in categorical_features:
#     # In training, these columns were converted to category
#     df_clean[col] = df_clean[col].astype('category')

# # --- Streamlit page config ---
# st.set_page_config(page_title="Earthquake Damage Predictor", layout="wide")
# st.title("🏚️🚨Earthquake Damage Prediction Dashboard🌀🌏")
# st.markdown("Enter building details below to predict earthquake damage severity.")

# # --- Sidebar inputs dynamically generated ---
# st.sidebar.header("Input Building Features")

# def user_input_features():
#     data = {}
#     for col in numeric_features:
#         min_val = int(df_clean[col].min()) if df_clean[col].dtype == 'int64' else float(df_clean[col].min())
#         max_val = int(df_clean[col].max()) if df_clean[col].dtype == 'int64' else float(df_clean[col].max())
#         median_val = int(df_clean[col].median()) if df_clean[col].dtype == 'int64' else float(df_clean[col].median())
#         data[col] = st.sidebar.number_input(
#             col, min_value=min_val, max_value=max_val, value=median_val
#         )
#     for col in categorical_features:
#         data[col] = st.sidebar.selectbox(
#             col, df_clean[col].cat.categories
#         )
#     return pd.DataFrame(data, index=[0])

# input_df = user_input_features()

# # --- Convert input categorical columns to category type ---
# for col in categorical_features:
#     input_df[col] = pd.Categorical(input_df[col], categories=df_clean[col].cat.categories)

# st.subheader("Input Building Features")
# st.write(input_df)

# # --- Make prediction ---
# prediction = best_lgbm.predict(input_df)
# prediction_proba = best_lgbm.predict_proba(input_df)

# st.subheader("Prediction")
# st.write(f"Predicted Severe Damage Class: **{prediction[0]}**")
# st.write("Prediction Probabilities:")
# prob_df = pd.DataFrame(prediction_proba, columns=best_lgbm.classes_)
# st.dataframe(prob_df)

# # --- Feature importance visualization ---
# st.subheader("Feature Importance (Top 10)")
# importance_df = pd.DataFrame({
#     'Feature': best_lgbm.feature_name_,
#     'Importance': best_lgbm.feature_importances_
# }).sort_values(by='Importance', ascending=False).head(10)

# fig = px.bar(
#     importance_df,
#     x='Importance',
#     y='Feature',
#     orientation='h',
#     text='Importance',
#     title="LightGBM Feature Importance (Top 10)"
# )
# fig.update_layout(yaxis={'categoryorder':'total ascending'})
# st.plotly_chart(fig, use_container_width=True)


# =================================== 2ND ITERATION ==============================


# # ---------- Page: About ----------
# elif page == "About":
#     st.title("ℹ️ About this app")
#     st.markdown("""
#     **Purpose:** Provide an end-user-focused interface for producing earthquake damage severity predictions and explaining them.

#     **Model:** Random Forest (best_rf) — place model file in `results/figures/best_rf_model.pkl` or the app will try `best_rf_model.pkl`.

#     **Required CSV structure (exact column names):**
#     """)
#     st.markdown("""
#     **Note:** Your CSV file should have the following structure to work with the model:

#     - **Columns:** 
#         1. `age_building` (int) – Age of the building in years
#         2. `plinth_area_sq_ft` (int) – Total plinth area in square feet
#         3. `height_ft_pre_eq` (int) – Building height in feet before earthquake
#         4. `land_surface_condition` (object) – Condition of the land (e.g., flat, slope)
#         5. `foundation_type` (object) – Type of foundation (e.g., mud, stone, reinforced)
#         6. `roof_type` (object) – Roof material type (e.g., RCC, timber, metal)
#         7. `ground_floor_type` (object) – Material/type of ground floor
#         8. `other_floor_type` (object) – Material/type of upper floors
#         9. `position` (object) – Position of the building (e.g., attached, detached)
#         10. `plan_configuration` (object) – Plan layout (e.g., rectangular, L-shape)
#         11. `superstructure` (object) – Material of main structural system
    
#     - **Important:** Column names must match exactly as above. Remove any extra columns (like `b_id`) before uploading.  
#     - **Data types:** Ensure numeric columns are integers/floats and categorical columns are strings/objects.
#     """)

#     st.markdown("---")
#     st.markdown("**Notes & tips**")
#     st.write("• If SHAP plots fail to render, install shap with `pip install shap` and restart the app.")
#     st.write("• The app assumes the model expects the columns listed above in the given order/format.")
#     st.write("• For production, consider putting the model behind an API and restricting file sizes for uploads.")

#     st.markdown("---")
#     st.markdown("Made with ❤️ by DominicNyabuto.")




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# =====================
# Load Model and Data
# =====================
MODEL_PATHS = [
    "/results/models/best_rf_model.pkl",
    "results/models/best_rf_model.pkl",
    "best_rf_model.pkl"
]
model = None
for path in MODEL_PATHS:
    try:
        model = joblib.load(path)
        break
    except:
        pass
if model is None:
    st.error("❌ Could not load best_rf model. Ensure the model file exists.")

# Load test data
TEST_PATH = "data/kavrepalanchok_test.csv"
df_test = pd.read_csv(TEST_PATH)
if "b_id" in df_test.columns:
    df_test = df_test.drop(columns=["b_id"])

# Predictions
probs = model.predict_proba(df_test)
preds = model.predict(df_test)
df_results = df_test.copy()
df_results["Predicted_Severity"] = preds
df_results[[f"Prob_{i}" for i in range(probs.shape[1])]] = probs

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Nepal Earthquake Damage Prediction", layout="wide")

st.title("🏠 Nepal Earthquake Building Damage Prediction")

# Predictions page content
st.markdown("""
This app predicts the **severity of building damage** in Kavrepalanchok, Nepal, using a trained Random Forest model.
It uses the provided **kavrepalanchok_test.csv** dataset — you don’t need to upload anything.

**Note:** Your CSV file should have the following structure to work with the model (already satisfied here):
- `age_building`, `plinth_area_sq_ft`, `height_ft_pre_eq`, `land_surface_condition`, `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`, `position`, `plan_configuration`, `superstructure`
""")

# Tabs
page = st.sidebar.radio("Navigation", ["Predictions", "Aggregated Summaries", "Dataset Explorer", "Model Insights", "About"])

# =====================
# Predictions Tab
# =====================
if page == "Predictions":
    st.header("📊 Predictions Dashboard")
    st.dataframe(df_results.head(50))

    # Download option
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Highlight most at-risk buildings
    st.subheader("⚠️ Most At-Risk Buildings")
    severe_class = df_results["Predicted_Severity"].max()
    risky = df_results.sort_values(by=f"Prob_{severe_class}", ascending=False).head(10)
    st.dataframe(risky)

# =====================
# Aggregated Summaries
# =====================

elif page == "Aggregated Summaries":
    st.header("📈 Aggregated Summaries")
    counts = df_results["Predicted_Severity"].value_counts()
    st.plotly_chart(px.pie(values=counts.values, names=counts.index, title="Predicted Damage Distribution"))
    
    severe_class = df_results["Predicted_Severity"].max()

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("% Severe Damage", f"{(df_results['Predicted_Severity'] == severe_class).mean()*100:.1f}%")
    with col2:
        st.metric("Average Severe Damage Probability", f"{df_results[f'Prob_{severe_class}'].mean():.2f}")
    with col3:
        st.metric("Most Common Class", counts.idxmax())

# =====================
# Dataset Explorer
# =====================
elif page == "Dataset Explorer":
    st.header("🔍 Dataset Explorer")

    # Correlation heatmap
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    corr = df_results[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    # Feature distributions
    feat = st.selectbox("Select a numeric feature to explore", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(data=df_results, x=feat, hue="Predicted_Severity", kde=True, ax=ax)
    st.pyplot(fig)

# =====================
# Model Insights
# =====================
elif page == "Model Insights":
    st.header("🧠 Model Insights")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": df_test.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_imp.set_index("Feature"))

# =====================
# About Page
# =====================
elif page == "About":
    st.header("ℹ️ About")
    st.markdown("""
    - **Dataset**: Kavrepalanchok, Nepal earthquake building records
    - **Model**: Random Forest Classifier trained to predict building damage severity
    - **Developer**: Streamlit demo app generated with love 💙
    """)
