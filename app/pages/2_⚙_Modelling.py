import streamlit as st  # noqa

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.model.classification.decisiontree import DecisionTreeModel
from autoop.core.ml.model.classification.knearest import KNearestNeighborsModel
from autoop.core.ml.model.classification.logisticregression import LogisticRegression
from autoop.core.ml.model.regression.lasso import LassoRegression
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.ridge import RidgeRegression
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types, improved_detect_features

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """Write helper text to the page."""
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."  # noqa
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
st.write("### Select a Dataset")
dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Dataset", dataset_names)

selected_dataset = next(
    dataset for dataset in datasets if dataset.name == selected_dataset_name
)
selected_dataset_id = selected_dataset.id

selected_artifact = automl.registry.get(selected_dataset_id)
selected_dataset = Dataset(
    name=selected_artifact.name,
    asset_path=selected_artifact.asset_path,
    data=selected_artifact.data,
    version=selected_artifact.version,
    tags=selected_artifact.tags,
    metadata=selected_artifact.metadata,
)

if selected_dataset:
    st.write("### Feature Selection")

    features = improved_detect_features(selected_dataset)
    feature_names = [feature.name for feature in features]
    target_feature = st.selectbox("Select Target Feature", feature_names)
    rest_feature_names = [
        feature for feature in feature_names if feature != target_feature
    ]
    input_features = st.multiselect("Select Input Features", rest_feature_names)
    input_features = [feature for feature in features if feature.name in input_features]

    st.write("### Detected Task Type")
    target = next(feature for feature in features if feature.name == target_feature)
    if target.type == "categorical":
        st.write("##### Classification Task")
    else:
        st.write("##### Regression Task")

    st.write("Select a Model based on the task type")
    regression_models = [
        "Multiple Linear Regression",
        "Lasso Regression",
        "Ridge Regression",
    ]
    classification_models = [
        "Logistic Regression",
        "K Nearest Neighbors",
        "Decision Tree",
    ]

    if target.type == "categorical":
        model = st.selectbox("Select Model", classification_models)
    else:
        model = st.selectbox("Select Model", regression_models)

    if model == "Multiple Linear Regression":
        model = MultipleLinearRegression()
    elif model == "Lasso Regression":
        model = LassoRegression()
    elif model == "Ridge Regression":
        model = RidgeRegression()
    elif model == "Logistic Regression":
        model = LogisticRegression()
    elif model == "K Nearest Neighbors":
        model = KNearestNeighborsModel()
    elif model == "Decision Tree":
        model = DecisionTreeModel()

    st.write("### Split Data")
    split = st.slider("Select Split Ratio", 0.1, 0.9, 0.8, 0.1)

    st.write("### Metrics")
    metrics = METRICS
    metric_names = [metric for metric in metrics]
    selected_metrics = st.multiselect("Select Metrics", METRICS)
    metrics = []
    for metric_name in selected_metrics:
        try:
            metric = get_metric(metric_name)
            metrics.append(metric)
        except ValueError as e:
            st.write(f"Error: {e}")

    st.write("### Pipeline Summary")
    pipeline = Pipeline(metrics, selected_dataset, model, input_features, target, split)
    st.write(str(pipeline))

    if st.button("Train Model"):
        pipeline.execute()
        st.write("Model trained successfully!")

    st.write("### Save Pipeline")
    name = st.text_input("Enter pipeline name")
    version = st.text_input("Enter pipeline version", value="1.0.0")
    if name and st.button("Save"):
        pipeline_artifact = pipeline._save(name, version)
        automl.registry.register(pipeline_artifact)
        st.write("Pipeline saved successfully!")

    st.write("### Make Predictions")
    if st.button("Make Predictions"):
        predictions = pipeline.execute()
        st.write(predictions)
        st.write("Predictions made successfully!")

else:
    st.write("No datasets found. Please upload a dataset in the previous step.")
