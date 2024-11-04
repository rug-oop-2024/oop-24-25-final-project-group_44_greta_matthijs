import unittest

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric.extensions import Precision
from autoop.functional.feature import detect_feature_types


class TestFeatures(unittest.TestCase):
    """
    Tests ml features.
    """
    def setUp(self) -> None:
        pass

    def test_detect_features_continuous(self):
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")

    def test_detect_features_with_categories(self):
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns, features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(
            lambda x: x.name in categorical_columns, features
        ):
            self.assertEqual(detected_feature.type, "categorical")


class TestPrecision(unittest.TestCase):
    def setUp(self):
        self.precision = Precision()

    def test_evaluate_default(self):
        predictions = [True, False, True, True, False]
        ground_truth = [True, False, False, True, False]
        result = self.precision._evaluate(predictions, ground_truth)
        self.assertEqual(result, 2 / 3)

    def test_evaluate_pos_int(self):
        predictions = [1, -1, 2, 3, -2]
        ground_truth = [1, -1, 1, 3, -2]
        result = self.precision._evaluate(predictions, ground_truth, positive="pos_int")
        self.assertEqual(result, 2 / 3)

    def test_evaluate_neg_int(self):
        predictions = [-1, -2, -3, 1, 2]
        ground_truth = [-1, -2, -1, 1, 2]
        result = self.precision._evaluate(predictions, ground_truth, positive="neg_int")
        self.assertEqual(result, 2 / 3)

    def test_evaluate_true(self):
        predictions = [True, False, True, True, False]
        ground_truth = [True, False, False, True, False]
        result = self.precision._evaluate(predictions, ground_truth, positive=True)
        self.assertEqual(result, 2 / 3)

    def test_evaluate_false(self):
        predictions = [True, False, True, True, False]
        ground_truth = [True, False, False, True, False]
        result = self.precision._evaluate(predictions, ground_truth, positive=False)
        self.assertEqual(result, 1.0)

    def test_evaluate_invalid_positive(self):
        predictions = [True, False, True, True, False]
        ground_truth = [True, False, False, True, False]
        with self.assertRaises(TypeError):
            self.precision._evaluate(predictions, ground_truth, positive="invalid")
