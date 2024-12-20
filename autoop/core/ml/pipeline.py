import pickle
from typing import List

import numpy as np

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline:
    """Pipeline class used to train and evaluate models."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """Initialize the pipeline with the given parameters."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError(
                "Model type must be classification for categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self):
        """Get the string representation of the pipeline."""
        return f"""
Pipeline(
    \nmodel={self._model.type},
    \ninput_features={list(map(str, self._input_features))},
    \ntarget_feature={str(self._target_feature)},
    \nsplit={self._split},
    \nmetrics={list(map(str, self._metrics))},
\n)
"""

    @property
    def model(self):
        """Used to get the model used in the pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """Register an artifact with the pipeline."""
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """Preprocesses the features in the dataset."""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

    def _split_data(self):
        """Split the data into training and testing sets."""
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Compacts the input vectors into a single matrix."""
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """Trains the model on the training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self):
        """Evaluate the model on the test data."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self):
        """Execute the pipeline and returns the results."""
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        test_metrics_results = self._metrics_results
        test_predictions = self._predictions

        train_X = self._compact_vectors(self._train_X)
        train_Y = self._train_y
        train_metrics_results = []
        train_predictions = self._model.predict(train_X)
        for metric in self._metrics:
            result = metric.evaluate(train_predictions, train_Y)
            train_metrics_results.append((metric, result))

        return {
            "train_metrics": train_metrics_results,
            "train_predictions": train_predictions,
            "test_metrics": test_metrics_results,
            "test_predictions": test_predictions,
        }

    def _save(self, name: str, version: str = "1.0.0"):
        """Save the pipeline as an artifact."""
        pipeline_data = {
            "dataset": self._dataset,
            "model": self._model,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
            "metrics": self._metrics,
        }

        # Serialize pipeline data to bytes using pickle
        artifact_data = pickle.dumps(pipeline_data)
        path = f"{name}.pkl"

        pipeline_artifact = Artifact(
            name=name,
            version=version,
            type="pipeline",
            data=artifact_data,
            asset_path=path,
        )

        return pipeline_artifact
