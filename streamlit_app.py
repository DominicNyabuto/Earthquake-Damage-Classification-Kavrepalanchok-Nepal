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
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
# import shap

# Load model
best_rf = joblib.load("results/models/best_rf_model.pkl")

# Load test data
X_test = pd.read_csv("/workspaces/Earthquake-Damage-Classification-Kavrepalanchok-Nepal/data/kavrepalanchok_test.csv")
if "b_id" in X_test.columns:
    X_test = X_test.drop("b_id", axis=1)

# Sidebar multipage navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Insights", "Dataset Explorer", "About"])

# ===== Home Page =====
if page == "Home":
    st.title("🏠 Earthquake Damage Prediction App")
    st.markdown("""
    **Purpose:** Predict earthquake damage severity for buildings in Kavrepalanchok, Nepal using Random Forest.  
    Explore data, test scenarios, and interpret predictions.
    """)

# ===== Prediction Page =====
elif page == "Prediction":
    st.title("📝 Predict Damage Severity")
    
    uploaded_file = st.file_uploader("Upload CSV for bulk prediction", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "b_id" in df.columns:
            df = df.drop("b_id", axis=1)
        preds = best_rf.predict(df)
        probs = best_rf.predict_proba(df).max(axis=1)
        df["Predicted_Severity"] = preds
        df["Prediction_Confidence"] = probs
        
        st.success("Predictions done!")
        st.dataframe(df.head(10))
        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            file_name="predictions.csv"
        )

    st.markdown("#### What-If Analysis")
    st.markdown("Adjust features for a single building and see predicted severity.")
    # Example with number of floors
    floors = st.slider("Number of floors", 1, 5, 2)
    roof_types = st.selectbox("Roof type", X_test["roof_type"].unique())
    foundation_types = st.selectbox("Foundation type", X_test["foundation_type"].unique())
    
    single_building = pd.DataFrame({
        "no_of_floors": [floors],
        "roof_type": [roof_types],
        "foundation_type": [foundation_types]
    })
    
    pred = best_rf.predict(single_building)[0]
    st.write(f"Predicted Damage Severity: **{pred}**")

# ===== Model Insights =====
elif page == "Model Insights":
    st.title("📊 Model Insights")
    
    st.markdown("#### Feature Importance")
    importances = pd.Series(best_rf.feature_importances_, index=X_test.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)
    
    st.markdown("#### Confusion Matrix & Classification Report")
    y_true = pd.read_csv("/workspaces/Earthquake-Damage-Classification-Kavrepalanchok-Nepal/data/kavrepalanchok_test_labels.csv")["severe_damage"]
    y_pred = best_rf.predict(X_test)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))
    
    # ROC curve (for severe damage vs rest)
    st.markdown("#### ROC Curve")
    y_bin = (y_true == 1).astype(int)  # example: severe damage=1
    y_score = best_rf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_bin, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    
    # # SHAP explanation
    # st.markdown("#### SHAP Feature Contribution (Local Interpretability)")
    # explainer = shap.TreeExplainer(best_rf)
    # shap_values = explainer.shap_values(X_test.iloc[:50])
    # st_shap = st.pyplot(shap.summary_plot(shap_values, X_test.iloc[:50], show=False))

# ===== Dataset Explorer / EDA =====
elif page == "Dataset Explorer":
    st.title("🔍 Dataset Explorer")
    
    st.markdown("#### Class Distribution")
    counts = X_test["severe_damage"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, title="Damage Severity Distribution")
    st.plotly_chart(fig)
    
    st.markdown("#### Feature Correlation Heatmap")
    numeric_cols = X_test.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots()
    sns.heatmap(X_test[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.markdown("#### Histograms & Boxplots")
    feature = st.selectbox("Select numeric feature", numeric_cols)
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.histplot(X_test[feature], ax=ax[0], kde=True)
    sns.boxplot(x=X_test["severe_damage"], y=X_test[feature], ax=ax[1])
    st.pyplot(fig)
    
    st.markdown("#### Stacked Bar Plot (Categorical Features)")
    cat_feature = st.selectbox("Select categorical feature", ["roof_type", "foundation_type"])
    stacked = pd.crosstab(X_test[cat_feature], X_test["severe_damage"])
    stacked.plot(kind="bar", stacked=True)
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    
    # Optional: Geospatial plot if lat/lon present
    if "latitude" in X_test.columns and "longitude" in X_test.columns:
        st.markdown("#### Geospatial Visualization")
        fig = px.scatter_mapbox(
            X_test, lat="latitude", lon="longitude",
            color="severe_damage", size_max=15, zoom=10,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig)

# ===== About Page =====
elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    **Data Source:** Kavrepalanchok Building Survey  
    **Model:** Random Forest trained on building features  
    **Author:** Dominic N Nyabuto  
    **Purpose:** Predict earthquake damage severity, explore EDA, and provide scenario testing.
    """)
