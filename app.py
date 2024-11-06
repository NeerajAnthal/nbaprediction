import streamlit as st
import pickle
import pandas as pd

# Load the model from the pickle file
with open("kmeans.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Performance Prediction App")

# Define cluster-to-label mapping
cluster_labels = {
    0: "bad",
    1: "average",
    2: "good",
    3: "very good"
}

# Create input fields for each feature
feature1 = st.number_input("Enter value for pts", min_value=0.0, max_value=100.0, step=1.0)
feature2 = st.number_input("Enter value for reb", min_value=0.0, max_value=100.0, step=1.0)
feature3 = st.number_input("Enter value for ast", min_value=0.0, max_value=100.0, step=1.0)
feature4 = st.number_input("Enter value for net_rating", min_value=-100.0, max_value=100.0, step=1.0)

feature5 = st.number_input("Enter value for usg_pct", min_value=0.0, max_value=100.0, step=1.0)
feature6 = st.number_input("Enter value for ts_pct", min_value=0.0, max_value=100.0, step=1.0)

# Create a DataFrame with user inputs
input_data = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6]], 
                          columns=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"])

# Predict and show the performance label
if st.button("Predict"):
    cluster_number = model.predict(input_data)[0]  # Get cluster number
    performance_label = cluster_labels.get(cluster_number, "Unknown")  # Map to label
    st.write("Performance Label:", performance_label)
