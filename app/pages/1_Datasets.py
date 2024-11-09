import streamlit as st
import pandas as pd


from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name.split('.')[0]
    dataset = Dataset.from_dataframe(
            data=df,
            name=dataset_name,
            asset_path=uploaded_file.name
        )

    automl.registry.register(dataset)
    st.write("Dataset added successfully")

    # Refresh dataset list after adding a new dataset
    datasets = automl.registry.list(type="dataset")
