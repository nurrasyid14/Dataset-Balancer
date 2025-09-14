import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import logging
import warnings

class Evaluator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'SVC': SVC(random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'Gaussian NB': GaussianNB(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings("ignore")

    def evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        results = {}
        for model_name, model in self.models.items():
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model_name] = report
            self.logger.info(f"Model: {model_name}\n{classification_report(y_test, y_pred)}")
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        return results
