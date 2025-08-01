import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_dataset.csv")  # Replace with your CSV filename

# Display dataset
st.title("🎓 Student Performance Classifier")
st.write("### Raw Data")
st.dataframe(df)

# Define Pass/Fail based on average of Math, Physics, Chemistry
df['Average'] = df[['Math', 'Physics', 'Chemistry']].mean(axis=1)
df['Pass'] = df['Average'].apply(lambda x: 1 if x >= 40 else 0)  # You can change the threshold

# Split data
X = df[['Math', 'Physics', 'Chemistry']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
st.write(f"### Model Accuracy: {acc:.2f}")

# Prediction input
st.write("## 📋 Predict Student Outcome")
math = st.slider("Math Score", 0, 100, 50)
physics = st.slider("Physics Score", 0, 100, 50)
chem = st.slider("Chemistry Score", 0, 100, 50)

if st.button("Predict"):
    input_data = pd.DataFrame([[math, physics, chem]], columns=['Math', 'Physics', 'Chemistry'])
    prediction = model.predict(input_data)[0]
    result = "🎉 Pass" if prediction == 1 else "❌ Fail"
    st.success(f"Prediction: {result}")
