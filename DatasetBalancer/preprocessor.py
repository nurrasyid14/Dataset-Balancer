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
        Melakukan preprocessing pada dataset:
            - Mengubah label target menjadi bilangan bulat berurutan
            - Menyediakan utilitas untuk splitting, imputasi, dan scaling


        Parameter:
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
        Melakukan Splitting pada dataset menjadi Test & Train dengan Stratifikasi
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
        Mengisi missing values dengan Mean.
        """
        return SimpleImputer(strategy=strategy)

    def get_scaler(self, method: str = "standard"):
        """
        Memanggil StandardScaler untuk standarisasi Dataset.
        """
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError("Metode scaling Invalid. Gunakan 'standard' atau 'minmax'.")

    def decode_labels(self, y_pred):
        """
        Mengubah kembali Label ke bentuk semula.
        """
        return self.label_encoder.inverse_transform(y_pred)