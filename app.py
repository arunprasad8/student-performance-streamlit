import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("student_dataset.csv")

st.title("🎓 Student Performance Predictor")

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

# Define pass/fail (you can customize this based on dataset)
if 'Grade' in df_encoded.columns:
    df_encoded['pass'] = df_encoded['G3'].apply(lambda x: 1 if x >= 10 else 0)
else:
    st.error("Column 'Grade' not found in the dataset.")
    st.stop()

# Train/test split
X = df_encoded.drop(['Grade', 'pass'], axis=1)
y = df_encoded['pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Visualizations
st.subheader("📊 Feature Correlation Heatmap")
fig_corr = px.imshow(df_encoded.corr(), text_auto=True)
st.plotly_chart(fig_corr)

st.subheader("🎯 Model Accuracy")
st.success(f"Accuracy: {acc:.2f}")

# Prediction form
st.subheader("📋 Make a Prediction")
input_data = {}
for col in X.columns:
    if col in label_encoders:
        options = df[col].unique().tolist()
        input_data[col] = st.selectbox(f"{col}", options)
    else:
        input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    result = "🎉 Pass" if prediction == 1 else "⚠️ Fail"
    st.subheader(f"Prediction: {result}")
