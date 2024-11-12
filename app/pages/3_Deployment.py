import pickle

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline

st.write("# 🚀 Deployment")
st.write("In this section, you can see and load your saved pipelines.")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")
if not pipelines:
    st.write("No pipelines found. Please train a pipeline first.")
    st.stop()

def load_pipeline(selected_pipeline_name: str) -> Pipeline:
    """Load a pipeline from the registry."""
    selected_pipeline = next(
        pipe for pipe in pipelines if pipe.name == selected_pipeline_name
    )
    selected_pipeline_id = selected_pipeline.id

    selected_artifact = automl.registry.get(selected_pipeline_id)
    pipeline_data = pickle.loads(selected_artifact.data)
    pipeline = Pipeline(
        metrics=pipeline_data["metrics"],
        dataset=pipeline_data["dataset"],
        model=pipeline_data["model"],
        input_features=pipeline_data["input_features"],
        target_feature=pipeline_data["target_feature"],
        split=pipeline_data["split"],
    )

    return pipeline


def check_dataset(dataset: Dataset, pipeline: Pipeline) -> bool:
    """Check if the dataset contains the required features for the pipeline."""
    data = dataset.read()
    for name in data.columns:
        for feature in pipeline._input_features:
            if name == feature.name:
                return True
    return False


if pipelines:
    st.write("### Saved Pipelines")
    pipeline_names = [pipeline.name for pipeline in pipelines]
    for name in pipeline_names:
        if st.checkbox(name):
            pipeline = load_pipeline(name)

            st.write("##### Pipeline Details")
            st.write(str(pipeline))

    st.write("### Load Pipeline")
    selected = st.selectbox("Pipeline", pipeline_names)

    if st.button("Load Pipeline"):
        pipeline = load_pipeline(selected)
        st.session_state["pipeline"] = pipeline
        st.write("Pipeline loaded successfully."
                 "\nProvide a file to make predictions.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name.split(".")[0]
        dataset = Dataset.from_dataframe(
            data=df, name=dataset_name, asset_path=uploaded_file.name
        )
        st.write("Dataset loaded successfully")
        if st.button("Train Pipeline"):
            if "pipeline" in st.session_state:
                pipeline = st.session_state["pipeline"]
            try:
                pipeline._dataset = dataset
                if check_dataset(dataset, pipeline):
                    prediction = pipeline.execute()
                    st.write("### Prediction")
                    st.write(prediction)
                else:
                    st.write(
                        "Dataset does not contain the required features."
                    )
            except Exception as e:
                st.write(f"Error executing pipeline: {e}")
    else:
        st.write("Please upload a CSV file to make predictions.")

else:
    st.write("No saved pipelines found.")
