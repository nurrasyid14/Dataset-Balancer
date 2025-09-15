import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from typing import Tuple
import plotly.express as px
import logging
import warnings

class Balancer:
    def __init__(self, df: pd.DataFrame, target_column: str, random_state: int = 42):
        """
        Menyeimangkan dataset menggunakan berbagai teknik balancing:
            - SMOTE : Synthetic Minority Over-sampling Technique
            - RandomUnderSampler : Random Under Sampling
            - SMOTEENN : Kombinasi SMOTE dan Edited Nearest Neighbors
            - SMOTETomek : Kombinasi SMOTE dan Tomek Links
        Parameter:
        ----------
        df : pd.DataFrame
            Dataset input
        target_column : str
            Nama kolom target
        random_state : int, optional
            Random state for reproducibility

        """
        self.df = df
        self.target_column = target_column
        self.random_state = random_state
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings("ignore")

    def plot_class_distribution(self, title: str = "Class Distribution", use_streamlit: bool = False):
        """
        Plot Distribusi Kelas
        Parameter:
        ----------
        title : str, optional
            Title of the plot
        use_streamlit : bool, optional
            Whether to use Streamlit for plotting
        """
        import streamlit as st
        class_counts = self.y.value_counts()
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title=title
        )
        if use_streamlit:
            st.plotly_chart(fig, use_container_width=True)
        else:
            return fig

    def balance_data(self, method: str = 'SMOTE') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Menyeimbangkan dataset menggunakan metode yang dipilih.
        Parameter:
        method : str, optional
            Metode balancing yang digunakan. Pilihan: 'SMOTE', 'RandomUnderSampler', 'SMOTEENN', 'SMOTETomek'
        Returns:
        -------
        X_res : pd.DataFrame
            Fitur setelah balancing
        y_res : pd.Series
            Target setelah balancing
        
        """
        if method == 'SMOTE':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=self.random_state)
        elif method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError("Unsupported balancing method")
        
        X_res, y_res = sampler.fit_resample(self.X, self.y)
        self.logger.info(f"Resampled dataset shape: {Counter(y_res)}")
        return X_res, y_res
