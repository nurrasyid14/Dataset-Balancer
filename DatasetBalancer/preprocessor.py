import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple
import logging
import warnings


class Preprocessor:
    def __init__(self, df: pd.DataFrame, target_column: str, random_state: int = 42):
        """
        Handles preprocessing tasks:
        - Encodes target labels into consecutive integers
        - Provides utilities for splitting, imputing, and scaling

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target column
        random_state : int, optional
            Random state for reproducibility
        """
        self.df = df.copy()
        self.target_column = target_column
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

        # Encode target labels into 0...n-1
        self.df[self.target_column] = self.label_encoder.fit_transform(
            self.df[self.target_column]
        )

        # Features and target
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings("ignore")

    def split_data(
        self, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits dataset into train and test sets with stratification.
        """
        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

    def get_imputer(self, strategy: str = "mean") -> SimpleImputer:
        """
        Returns an imputer for missing values.
        """
        return SimpleImputer(strategy=strategy)

    def get_scaler(self, method: str = "standard"):
        """
        Returns a scaler for feature scaling.
        """
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method")

    def decode_labels(self, y_pred):
        """
        Converts encoded labels back to original form.
        """
        return self.label_encoder.inverse_transform(y_pred)