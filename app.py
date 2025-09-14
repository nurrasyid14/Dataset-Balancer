import streamlit as st
import pandas as pd

from DatasetBalancer.preprocessor import Preprocessor
from DatasetBalancer.balancer import Balancer
from DatasetBalancer.evaluator import Evaluator

st.set_page_config(page_title="Dataset Balancer", layout="wide")

st.title("Dataset Balancer & Model Evaluator")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Your Dataset")
    st.write(df.head())

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if target_column:
        balance_method = st.selectbox(
            "Choose a Balancing Method",
            ["SMOTE", "RandomUnderSampler", "SMOTEENN", "SMOTETomek"]
        )
        test_size = st.slider("Test Size (for train/test split)", 0.1, 0.5, 0.2)

        if st.button("Run Balancer & Evaluate Models"):
            with st.spinner("Processing..."):
                # --- Step 1: Preprocessing ---
                st.markdown("### Step 1: Data Preprocessing")
                st.write(
                    f"We use the **Preprocessor** class to split the dataset into train/test sets. "
                    f"Also, the target column **`{target_column}`** is encoded into consecutive integers "
                    f"(fixing any issues like classes `[1, 2, 4]` → `[0, 1, 2]`)."
                )
                preprocessor = Preprocessor(df, target_column)
                X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=test_size)
                st.success("✅ Data has been split successfully.")

                # --- Step 2: Initial class distribution ---
                st.markdown("### Step 2: Initial Class Distribution")
                st.write(
                    "Here we call **Balancer.plot_class_distribution()**. "
                    "This shows how many samples belong to each class before balancing. "
                    "If one class has much fewer samples, the dataset is imbalanced, "
                    "which usually hurts model performance."
                )
                balancer = Balancer(df, target_column)
                st.plotly_chart(
                    balancer.plot_class_distribution("Initial Class Distribution"),
                    use_container_width=True
                )

                # --- Step 3: Evaluate on imbalanced data ---
                st.markdown("### Step 3: Model Evaluation on Imbalanced Data")
                st.write(
                    "We now call **Evaluator.evaluate_models()**. "
                    "This function trains multiple classifiers using a pipeline "
                    "(`imputer → scaler → model`) and generates metrics:"
                )
                st.markdown("""
                - **Precision**: Of the predicted positives, how many are actually correct?  
                - **Recall**: Of the actual positives, how many did we correctly identify?  
                - **F1-score**: Harmonic mean of precision and recall (balance of both).  
                - **Support**: Number of samples in each class.  
                """)
                evaluator = Evaluator()
                imbalanced_results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)

                for model_name, metrics in imbalanced_results.items():
                    st.markdown(f"#### Results for {model_name} (Imbalanced Data)")
                    df_metrics = pd.DataFrame(metrics).transpose()
                    st.dataframe(df_metrics.style.background_gradient(cmap="Blues"))

                # --- Step 4: Balance dataset ---
                st.markdown("### Step 4: Balancing the Dataset")
                st.write(
                    f"We now call **Balancer.balance_data(method='{balance_method}')** "
                    "to create a more even class distribution. "
                    "This helps the model pay equal attention to all classes."
                )
                X_res, y_res = balancer.balance_data(method=balance_method)
                balanced_df = pd.DataFrame(X_res, columns=df.drop(columns=[target_column]).columns)
                balanced_df[target_column] = y_res
                st.success("✅ Dataset has been balanced successfully.")

                # --- Step 5: Balanced class distribution ---
                st.markdown("### Step 5: Balanced Class Distribution")
                st.write(
                    "Here’s the class distribution **after balancing**. "
                    "Each class now has roughly the same number of samples, "
                    "which should improve recall for minority classes."
                )
                balanced_balancer = Balancer(balanced_df, target_column)
                st.plotly_chart(
                    balanced_balancer.plot_class_distribution("Balanced Class Distribution"),
                    use_container_width=True
                )

                # --- Step 6: Evaluate on balanced data ---
                st.markdown("### Step 6: Model Evaluation on Balanced Data")
                st.write(
                    "We call **Evaluator.evaluate_models()** again, this time on the balanced dataset. "
                    "Compare these results to Step 3. "
                    "You should see better recall/F1 for minority classes, "
                    "though sometimes precision drops slightly due to synthetic samples."
                )
                preprocessor_bal = Preprocessor(balanced_df, target_column)
                X_train_res, X_test_res, y_train_res, y_test_res = preprocessor_bal.split_data(test_size=test_size)
                balanced_results = evaluator.evaluate_models(X_train_res, X_test_res, y_train_res, y_test_res)

                for model_name, metrics in balanced_results.items():
                    st.markdown(f"#### Results for {model_name} (Balanced Data)")
                    df_metrics = pd.DataFrame(metrics).transpose()
                    st.dataframe(df_metrics.style.background_gradient(cmap="Greens"))

            st.success("Fine")

else:
    st.info("📂 Please upload a CSV file to get started.")
