# STREAMLIT WEB APP


# app.py
import streamlit as st
import pandas as pd
import lightgbm as lgb
import plotly.express as px 
import joblib

# --- Load trained LightGBM model ---
best_lgbm = joblib.load(
    "/workspaces/Earthquake-Damage-Classification-Kavrepalanchok-Nepal/results/models/best_lightgbm_model.pkl"
)

# --- Load example dataframe for defaults & category info ---
df_clean = pd.read_csv(
    "/workspaces/Earthquake-Damage-Classification-Kavrepalanchok-Nepal/data/kavrepalanchok_test.csv"
)

# Drop 'b_id' if present
if 'b_id' in df_clean.columns:
    df_clean = df_clean.drop('b_id', axis=1)

# --- Identify categorical features from training ---
categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
numeric_features = df_clean.select_dtypes(exclude=['object']).columns.tolist()

# --- Convert categorical columns to category type (match training) ---
for col in categorical_features:
    # In training, these columns were converted to category
    df_clean[col] = df_clean[col].astype('category')

# --- Streamlit page config ---
st.set_page_config(page_title="Earthquake Damage Predictor", layout="wide")
st.title("Earthquake Damage Prediction Dashboard")
st.markdown("Enter building details below to predict earthquake damage severity.")

# --- Sidebar inputs dynamically generated ---
st.sidebar.header("Input Building Features")

def user_input_features():
    data = {}
    for col in numeric_features:
        min_val = int(df_clean[col].min()) if df_clean[col].dtype == 'int64' else float(df_clean[col].min())
        max_val = int(df_clean[col].max()) if df_clean[col].dtype == 'int64' else float(df_clean[col].max())
        median_val = int(df_clean[col].median()) if df_clean[col].dtype == 'int64' else float(df_clean[col].median())
        data[col] = st.sidebar.number_input(
            col, min_value=min_val, max_value=max_val, value=median_val
        )
    for col in categorical_features:
        data[col] = st.sidebar.selectbox(
            col, df_clean[col].cat.categories
        )
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Convert input categorical columns to category type ---
for col in categorical_features:
    input_df[col] = pd.Categorical(input_df[col], categories=df_clean[col].cat.categories)

st.subheader("Input Building Features")
st.write(input_df)

# --- Make prediction ---
prediction = best_lgbm.predict(input_df)
prediction_proba = best_lgbm.predict_proba(input_df)

st.subheader("Prediction")
st.write(f"Predicted Severe Damage Class: **{prediction[0]}**")
st.write("Prediction Probabilities:")
prob_df = pd.DataFrame(prediction_proba, columns=best_lgbm.classes_)
st.dataframe(prob_df)

# --- Feature importance visualization ---
st.subheader("Feature Importance (Top 10)")
importance_df = pd.DataFrame({
    'Feature': best_lgbm.feature_name_,
    'Importance': best_lgbm.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    text='Importance',
    title="LightGBM Feature Importance (Top 10)"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)
