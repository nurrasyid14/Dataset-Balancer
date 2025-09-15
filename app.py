import streamlit as st
import pandas as pd

from DatasetBalancer.preprocessor import Preprocessor
from DatasetBalancer.balancer import Balancer
from DatasetBalancer.evaluator import Evaluator

st.set_page_config(page_title="Dataset Balancer", layout="wide")

st.title("Dataset Balancer & Model Evaluator")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV/XLS/XLSX/TSV/TSV.GZ)", 
    type=["csv", "xls", "xlsx", "tsv", "tsv.gz"]
)

if uploaded_file:
    file_name = uploaded_file.name.lower()

    # Baca sesuai tipe file
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".tsv", ".tsv.gz")):
        df = pd.read_csv(uploaded_file, sep="\t")
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("⚠️ Format file tidak didukung.")
        df = None

    if df is not None:
        st.subheader("Preview of Your Dataset")
        st.write(df.head())


    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:
        balance_method = st.selectbox(
            "Choose a Balancing Method",
            ["SMOTE", "RandomUnderSampler", "SMOTEENN", "SMOTETomek"]
        )
        test_size = st.slider("Test Size (for train/test split)", 0.1, 0.5, 0.2)

        if st.button("Run Balancer & Evaluate Models"):
            with st.spinner("Processing..."):
                try:
                    # --- Step 1: Preprocessing ---
                    st.markdown("### Step 1: Data Preprocessing")
                    preprocessor = Preprocessor(df, target_column)
                    X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=test_size)
                    st.success("✅ Data has been split successfully.")

                    # --- Step 2: Initial class distribution ---
                    st.markdown("### Step 2: Initial Class Distribution")
                    balancer = Balancer(df, target_column)
                    st.plotly_chart(
                        balancer.plot_class_distribution("Initial Class Distribution"),
                        use_container_width=True
                    )

                    # --- Step 3: Evaluate on imbalanced data ---
                    st.markdown("### Step 3: Model Evaluation on Imbalanced Data")
                    evaluator = Evaluator()
                    imbalanced_results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)

                    for model_name, metrics in imbalanced_results.items():
                        st.markdown(f"#### Results for {model_name} (Imbalanced Data)")
                        df_metrics = pd.DataFrame(metrics).transpose()
                        st.dataframe(df_metrics.style.background_gradient(cmap="Blues"))

                    # --- Step 4: Balance dataset ---
                    st.markdown("### Step 4: Balancing the Dataset")
                    X_res, y_res = balancer.balance_data(method=balance_method)
                    balanced_df = pd.DataFrame(X_res, columns=df.drop(columns=[target_column]).columns)
                    balanced_df[target_column] = y_res
                    st.success("Dataset has been balanced successfully.")

                    # Tambahkan tombol download CSV
                    csv_download = balanced_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Balanced Dataset (CSV)",
                        data=csv_download,
                        file_name=f"balanced_{balance_method}.csv",
                        mime="text/csv",
                    )

                    # --- Step 5: Balanced class distribution ---
                    st.markdown("### Step 5: Balanced Class Distribution")
                    balanced_balancer = Balancer(balanced_df, target_column)
                    st.plotly_chart(
                        balanced_balancer.plot_class_distribution("Balanced Class Distribution"),
                        use_container_width=True
                    )

                    # --- Step 6: Evaluate on balanced data ---
                    st.markdown("### Step 6: Model Evaluation on Balanced Data")
                    preprocessor_bal = Preprocessor(balanced_df, target_column)
                    X_train_res, X_test_res, y_train_res, y_test_res = preprocessor_bal.split_data(test_size=test_size)
                    balanced_results = evaluator.evaluate_models(X_train_res, X_test_res, y_train_res, y_test_res)

                    for model_name, metrics in balanced_results.items():
                        st.markdown(f"#### Results for {model_name} (Balanced Data)")
                        df_metrics = pd.DataFrame(metrics).transpose()
                        st.dataframe(df_metrics.style.background_gradient(cmap="Greens"))

                    st.success("✅ Done! Models evaluated, dataset balanced and available for download.")

                except ValueError as e:
                    st.error(f"Error: {e}")
                    st.info("Tip: Pastikan setiap kelas target punya minimal 2 sampel agar bisa di-split.")

else:
    st.info("📂 Please upload a CSV file to get started.")
