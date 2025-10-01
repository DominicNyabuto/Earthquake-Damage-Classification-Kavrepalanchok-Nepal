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


# app.py
# All-in-one Streamlit app for Earthquake Damage Predictions (Random Forest best_rf)
# Features: Predictions Dashboard, Aggregated Summaries, Feature Exploration,
# Model Insights (feature importance + SHAP), Interactive predictions, EDA, downloads.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

st.set_page_config(page_title="Earthquake Damage — Predictions & Insights", layout="wide")

# -------------------- Helper functions --------------------
@st.cache_resource
def load_model(paths=["results/models/best_rf_model.pkl", "best_rf_model.pkl", "/results/models/best_rf_model.pkl"]):
    for p in paths:
        try:
            model = joblib.load(p)
            return model, p
        except Exception:
            continue
    return None, None

@st.cache_data
def load_test_data(path):
    df = pd.read_csv(path)
    if "b_id" in df.columns:
        df = df.drop(columns=["b_id"])
    return df

@st.cache_data
def prepare_dataframe_for_model(df, required_cols):
    # Make sure required cols exist; if extra columns exist, keep them but model will only use required cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def download_df(df, filename="predictions.csv"):
    return st.download_button("📥 Download predictions (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name=filename, mime='text/csv')


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# SHAP import handling
shap_available = True
try:
    import shap
except Exception:
    shap_available = False

# -------------------- Constants / Required Columns --------------------
REQUIRED_COLUMNS = [
    'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'land_surface_condition',
    'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
    'position', 'plan_configuration', 'superstructure'
]

# -------------------- Load model + data --------------------
model, model_path = load_model()

# Attempt to load default X_test path provided by user
DEFAULT_TEST_PATH = "/workspaces/Earthquake-Damage-Classification-Kavrepalanchok-Nepal/data/kavrepalanchok_test.csv"
try:
    X_test = load_test_data(DEFAULT_TEST_PATH)
    test_load_msg = f"Loaded X_test from {DEFAULT_TEST_PATH}"
except Exception:
    X_test = None
    test_load_msg = "No default X_test found at provided path. Upload a file or set path in sidebar."

# -------------------- Sidebar --- navigation & uploads --------------------
st.sidebar.title("🔧 Controls")
page = st.sidebar.radio("Navigate", ["Home", "Predictions", "Summaries", "Model Insights", "Dataset Explorer", "About"])

st.sidebar.markdown("---")
# Allow user to override test data path or upload
upload_file = st.sidebar.file_uploader("Upload X_test CSV (optional)", type=["csv"], help="File must match required columns. Remove b_id before upload.")
if upload_file is not None:
    try:
        X_test = pd.read_csv(upload_file)
        if "b_id" in X_test.columns:
            X_test = X_test.drop(columns=["b_id"])
        st.sidebar.success("Uploaded X_test (preview available on Dataset Explorer)")
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")

# Load model status
if model is None:
    st.sidebar.error("Model not found. Place best_rf model at results/figures/best_rf_model.pkl or best_rf_model.pkl")
else:
    st.sidebar.success(f"Loaded model from: {model_path}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Required columns for uploaded CSV**")
st.sidebar.code(
"""
age_building, plinth_area_sq_ft, height_ft_pre_eq, land_surface_condition,
foundation_type, roof_type, ground_floor_type, other_floor_type,
position, plan_configuration, superstructure
"""
)
st.sidebar.caption("Remove extra columns like b_id. Numeric columns must be numbers; categories as text.")

# ---------- Page: Home ----------
if page == "Home":
    st.title("🏠 Earthquake Damage — Predictions & Insights")
    st.markdown("""
    This app is designed for **end-users** who have building records (X_test) and want **predictions** of earthquake damage severity using the Random Forest model (`best_rf`).

    The app focuses on **what the model predicts** and **why** (SHAP explanations), plus user-friendly exports and interactive "what-if" tools.

    **Quick start:**
    1. Upload `X_test` (CSV) with required columns, or use the default test dataset if available.
    2. Go to the *Predictions* page to run bulk predictions and download results.
    3. Use *Model Insights* for global & local explanations.
    """)

    st.markdown("**Status**")
    st.write(test_load_msg)
    if not shap_available:
        st.warning("SHAP library is not installed. To enable detailed explanations install with: `pip install shap`.")

# ---------- Page: Predictions ----------
elif page == "Predictions":
    st.title("📝 Predictions Dashboard")

    st.markdown("Upload a CSV or use the uploaded/default dataset to generate predictions for each building.")

    if X_test is None:
        st.info("No X_test available. Upload a CSV on the sidebar to proceed.")
    else:
        # Validate columns but allow model to accept subset (we will use required features only)
        try:
            df_for_model = prepare_dataframe_for_model(X_test, REQUIRED_COLUMNS)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Generating predictions..."):
            preds = model.predict(df_for_model[REQUIRED_COLUMNS])
            probs = model.predict_proba(df_for_model[REQUIRED_COLUMNS])
            # For multiclass: get max prob and corresponding class prob
            top_prob = probs.max(axis=1)
            # build output
            output = X_test.copy()
            output['predicted_severity'] = preds
            output['predicted_confidence'] = np.round(top_prob, 4)

        # KPI row
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        pct_severe = np.mean(output['predicted_severity']==1) * 100 if 1 in output['predicted_severity'].unique() else 0
        col1.metric("% Predicted Severe", f"{pct_severe:.1f}%")
        col2.metric("Avg Severe Prob.", f"{(output.loc[output['predicted_severity']==1,'predicted_confidence'].mean() if (output['predicted_severity']==1).any() else 0):.2f}")
        col3.metric("Total Rows", f"{len(output)}")
        col4.metric("Most common class", f"{output['predicted_severity'].mode()[0]}")

        st.markdown("#### Predicted rows (top risks first)")
        topN = st.slider("Show top N at-risk buildings by predicted probability", 5, 200, 20)
        top_risk = output.sort_values('predicted_confidence', ascending=False).head(topN)

        # Color-coded table: use plotly table for nicer style
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(top_risk.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[top_risk[c] for c in top_risk.columns], fill_color='lavender', align='left')
        )])
        st.plotly_chart(fig, use_container_width=True)

        # Download
        download_df(output, filename="predictions_with_confidence.csv")

        # Uncertainty flagging
        st.markdown("#### Low-confidence predictions (near 50%)")
        thresh = st.slider("Uncertainty threshold (probability close to 0.5)", 0.0, 0.49, 0.1)
        low_conf = output[(output['predicted_confidence'] >= 0.5 - thresh) & (output['predicted_confidence'] <= 0.5 + thresh)]
        st.write(f"Predictions flagged as low confidence: {len(low_conf)} rows")
        if not low_conf.empty:
            st.dataframe(low_conf)

        # Aggregated distribution (also available on Summaries page)
        st.markdown("#### Predicted class distribution")
        dist = output['predicted_severity'].value_counts().sort_index()
        fig2 = px.pie(names=dist.index.astype(str), values=dist.values, title="Predicted Severity Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Page: Summaries ----------
elif page == "Summaries":
    st.title("📈 Aggregated Summaries & Storytelling")
    if X_test is None:
        st.info("Upload X_test to generate summaries.")
    else:
        try:
            df_for_model = prepare_dataframe_for_model(X_test, REQUIRED_COLUMNS)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        preds = model.predict(df_for_model[REQUIRED_COLUMNS])
        probs = model.predict_proba(df_for_model[REQUIRED_COLUMNS])
        output = X_test.copy()
        output['predicted_severity'] = preds
        output['predicted_confidence'] = probs.max(axis=1)

        # 3 narrative visuals
        st.markdown("### Visual Storytelling — 3 Key Visuals")
        st.markdown("1) Older buildings tend to have higher predicted severity (trend line)")
        # Trend: avg predicted_confidence by age bucket
        output['age_bucket'] = pd.cut(output['age_building'], bins=[-1,10,20,30,50,100], labels=["0-10","11-20","21-30","31-50","50+"])
        trend = output.groupby('age_bucket')['predicted_confidence'].mean().reset_index()
        fig_trend = px.line(trend, x='age_bucket', y='predicted_confidence', markers=True, title='Avg predicted severe probability by age bucket')
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("2) Foundation type vs predicted severity (stacked)")
        stacked = pd.crosstab(output['foundation_type'], output['predicted_severity'])
        fig_stacked = stacked.plot(kind='bar', stacked=True, figsize=(10,4)).get_figure()
        st.pyplot(fig_stacked)

        st.markdown("3) Height vs Plinth area colored by predicted severity")
        fig_scatter = px.scatter(output, x='height_ft_pre_eq', y='plinth_area_sq_ft', color='predicted_severity', size='predicted_confidence', title='Height vs Plinth Area (colored by predicted severity)')
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Aggregated KPIs
        st.markdown("### KPIs")
        col1, col2, col3 = st.columns(3)
        col1.metric("% predicted severe", f"{(output['predicted_severity']==1).mean()*100:.1f}%")
        col2.metric("Avg predicted confidence", f"{output['predicted_confidence'].mean():.2f}")
        col3.metric("Rows", f"{len(output)}")

        # Export filtered
        if st.button("Export high-risk subset (predicted_confidence >= 0.8)"):
            high_risk = output[output['predicted_confidence'] >= 0.8]
            if not high_risk.empty:
                st.download_button("Download high-risk CSV", data=high_risk.to_csv(index=False).encode('utf-8'), file_name='high_risk_subset.csv')
            else:
                st.info("No rows meet the threshold.")

# ---------- Page: Model Insights ----------
elif page == "Model Insights":
    st.title("🧠 Model Insights & Explainability")
    st.markdown("This page focuses on communicating *why* the model predicts what it predicts. SHAP is used for explanations when available.")

    if model is None:
        st.info("No model loaded. Place model file and reload.")
    elif X_test is None:
        st.info("Upload X_test to compute SHAP and insights.")
    else:
        try:
            df_for_model = prepare_dataframe_for_model(X_test, REQUIRED_COLUMNS)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # Feature importance (global)
        st.markdown("#### Global Feature Importance (Random Forest)")
        try:
            importances = pd.Series(model.feature_importances_, index=REQUIRED_COLUMNS).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8,5))
            importances.plot(kind='barh', ax=ax)
            ax.set_title('Feature importances')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not compute feature importances: {e}")

        # SHAP global
        if shap_available:
            st.markdown("#### SHAP — Global summary")
            explainer = shap.TreeExplainer(model)
            # Use a sample to keep responsiveness
            sample = df_for_model[REQUIRED_COLUMNS].sample(min(500, len(df_for_model)), random_state=42)
            shap_values = explainer.shap_values(sample)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig_shap = shap.summary_plot(shap_values, sample, show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.warning("SHAP not available. Install via `pip install shap` and restart the app to enable SHAP visualizations.")

        # Local explanation: pick a row or let user pick by index
        st.markdown("#### Local explanation (single building)")
        idx = st.number_input('Pick an index (0-based) from X_test to explain', min_value=0, max_value=len(X_test)-1 if X_test is not None else 0, value=0, step=1)
        chosen_row = df_for_model[REQUIRED_COLUMNS].iloc[[int(idx)]]
        pred = model.predict(chosen_row)[0]
        prob = model.predict_proba(chosen_row).max()
        st.write(f"Predicted severity: **{pred}** — confidence **{prob:.3f}**")

        if shap_available:
            shap_vals_row = explainer.shap_values(chosen_row)
            st.markdown("SHAP waterfall / force plot for selected row")
            try:
                # For multiclass shap_values is a list. Use predicted class index
                if isinstance(shap_vals_row, list):
                    class_index = int(pred)
                    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[class_index], shap_vals_row[class_index][0], feature_names=REQUIRED_COLUMNS)
                    st.pyplot(bbox_inches='tight')
                else:
                    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_vals_row[0], feature_names=REQUIRED_COLUMNS)
                    st.pyplot(bbox_inches='tight')
            except Exception:
                st.info("Could not render waterfall plot in this environment. You can still inspect shap_values array programmatically.")

# ---------- Page: Dataset Explorer ----------
elif page == "Dataset Explorer":
    st.title("🔍 Dataset Explorer & EDA")

    if X_test is None:
        st.info("Upload X_test to explore dataset.")
    else:
        st.markdown("#### Data preview")
        st.dataframe(X_test.head(50))

        # Numeric columns
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()

        st.markdown("#### Correlation heatmap (numeric features)")
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8,6))
            corr = X_test[numeric_cols].corr()
            im = ax.matshow(corr)
            fig.colorbar(im)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=90)
            ax.set_yticklabels(numeric_cols)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

        st.markdown("#### Feature distributions (histogram & boxplot)")
        feature = st.selectbox("Select numeric feature to inspect", numeric_cols)
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        axes[0].hist(X_test[feature].dropna(), bins=30)
        axes[0].set_title(f'Histogram: {feature}')
        axes[1].boxplot(X_test[feature].dropna())
        axes[1].set_title(f'Boxplot: {feature}')
        st.pyplot(fig)

        st.markdown("#### Categorical stacked bar: compare with predicted (if predictions exist)")
        if 'predicted_severity' in X_test.columns:
            base_df = X_test.copy()
        else:
            # create predictions to allow charts
            try:
                df_for_model = prepare_dataframe_for_model(X_test, REQUIRED_COLUMNS)
                base_df = X_test.copy()
                base_df['predicted_severity'] = model.predict(df_for_model[REQUIRED_COLUMNS])
            except Exception:
                base_df = X_test.copy()
                base_df['predicted_severity'] = np.nan

        cat_feature = st.selectbox("Select categorical feature", cat_cols)
        stacked = pd.crosstab(base_df[cat_feature], base_df['predicted_severity']).fillna(0)
        st.write(stacked)
        fig = stacked.plot(kind='bar', stacked=True, figsize=(10,4)).get_figure()
        st.pyplot(fig)

        # Geospatial if available
        if 'latitude' in X_test.columns and 'longitude' in X_test.columns:
            st.markdown("#### Geospatial: map of predicted severity")
            try:
                map_df = X_test.copy()
                if 'predicted_severity' not in map_df.columns:
                    df_for_model = prepare_dataframe_for_model(X_test, REQUIRED_COLUMNS)
                    map_df['predicted_severity'] = model.predict(df_for_model[REQUIRED_COLUMNS])
                fig_map = px.scatter_mapbox(map_df, lat='latitude', lon='longitude', color='predicted_severity', hover_data=[*REQUIRED_COLUMNS], zoom=10)
                fig_map.update_layout(mapbox_style='open-street-map')
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.info(f"Could not render map: {e}")

# ---------- Page: About ----------
elif page == "About":
    st.title("ℹ️ About this app")
    st.markdown("""
    **Purpose:** Provide an end-user-focused interface for producing earthquake damage severity predictions and explaining them.

    **Model:** Random Forest (best_rf) — place model file in `results/figures/best_rf_model.pkl` or the app will try `best_rf_model.pkl`.

    **Required CSV structure (exact column names):**
    """)
    st.markdown("""
    **Note:** Your CSV file should have the following structure to work with the model:

    - **Columns:** 
        1. `age_building` (int) – Age of the building in years
        2. `plinth_area_sq_ft` (int) – Total plinth area in square feet
        3. `height_ft_pre_eq` (int) – Building height in feet before earthquake
        4. `land_surface_condition` (object) – Condition of the land (e.g., flat, slope)
        5. `foundation_type` (object) – Type of foundation (e.g., mud, stone, reinforced)
        6. `roof_type` (object) – Roof material type (e.g., RCC, timber, metal)
        7. `ground_floor_type` (object) – Material/type of ground floor
        8. `other_floor_type` (object) – Material/type of upper floors
        9. `position` (object) – Position of the building (e.g., attached, detached)
        10. `plan_configuration` (object) – Plan layout (e.g., rectangular, L-shape)
        11. `superstructure` (object) – Material of main structural system
    
    - **Important:** Column names must match exactly as above. Remove any extra columns (like `b_id`) before uploading.  
    - **Data types:** Ensure numeric columns are integers/floats and categorical columns are strings/objects.
    """)

    st.markdown("---")
    st.markdown("**Notes & tips**")
    st.write("• If SHAP plots fail to render, install shap with `pip install shap` and restart the app.")
    st.write("• The app assumes the model expects the columns listed above in the given order/format.")
    st.write("• For production, consider putting the model behind an API and restricting file sizes for uploads.")

    st.markdown("---")
    st.markdown("Made with ❤️ by DominicNyabuto.")
