import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from DatasetBalancer.preprocessor import Preprocessor
from DatasetBalancer.balancer import Balancer
from DatasetBalancer.evaluator import Evaluator

# -----------------------------
# Load dataset langsung
# -----------------------------
DATA_PATH = "klasifikasi__UKT_komplit.csv"  # pastikan file ada di folder project
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Dataset Balancer (Fixed Dataset)", layout="wide")

st.title("Dataset Balancer & Model Evaluator (Dataset: Klasifikasi UKT)")

st.subheader("Preview of Dataset")
st.write(df.head())

# -----------------------------
# Pilih target kolom
# -----------------------------
target_column = st.selectbox("Pilih Kolom Target", df.columns)

if target_column:
    balance_method = st.selectbox(
        "Pilih Metode Balancing",
        ["SMOTE", "RandomUnderSampler", "SMOTEENN", "SMOTETomek"]
    )
    test_size = st.slider("Ukuran Data Test (for train/test split)", 0.1, 0.5, 0.2)

    if st.button("Balance & Evaluasi Model"):
        with st.spinner("Processing..."):

            # --- Step 1: Preprocessing ---
            st.markdown("### Step 1: Data Preprocessing")
            preprocessor = Preprocessor(df, target_column)

            # cek distribusi kelas
            st.write("Distribusi kelas target sebelum split:")
            st.write(preprocessor.y.value_counts())

            class_counts = preprocessor.y.value_counts()
            if class_counts.min() < 2:
                st.warning(
                    f"Ada kelas dengan hanya {class_counts.min()} sample. "
                    "Split data tidak bisa menggunakan stratify → menggunakan random split tanpa stratify."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    preprocessor.X,
                    preprocessor.y,
                    test_size=test_size,
                    random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=test_size)

            st.success("✅ Data split successfully.")

            # --- Step 2: Initial class distribution ---
            st.markdown("### Step 2: Initial Class Distribution")
            balancer = Balancer(df, target_column)
            st.plotly_chart(
                balancer.plot_class_distribution("Initial Class Distribution"),
                use_container_width=True
            )

            # --- Step 3: Model Evaluation (Imbalanced Data) ---
            st.markdown("### Step 3: Evaluasi Model terhadap Data tak Imbang")
            evaluator = Evaluator()
            imbalanced_results = evaluator.evaluate_models(X_train, X_test, y_train, y_test)

            for model_name, metrics in imbalanced_results.items():
                st.markdown(f"#### Results for {model_name} (Data tak Imbang)")
                df_metrics = pd.DataFrame(metrics).transpose()
                st.dataframe(df_metrics.style.background_gradient(cmap="Blues"))

            # --- Step 4: Balancing Dataset ---
            st.markdown("### Step 4: Balancing Dataset")
            X_res, y_res = balancer.balance_data(method=balance_method)
            balanced_df = pd.DataFrame(X_res, columns=df.drop(columns=[target_column]).columns)
            balanced_df[target_column] = y_res
            st.success("✅ Dataset telah diseimbangkan.")

            # Tombol download
            csv_download = balanced_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Dataset yang telah diseimbangkan (CSV)",
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

            # --- Step 6: Model Evaluation (Balanced Data) ---
            st.markdown("### Step 6: Evaluasi Model terhadap Dataset Seimbang")
            preprocessor_bal = Preprocessor(balanced_df, target_column)

            # fallback juga untuk data balanced
            class_counts_bal = preprocessor_bal.y.value_counts()
            if class_counts_bal.min() < 2:
                X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
                    preprocessor_bal.X,
                    preprocessor_bal.y,
                    test_size=test_size,
                    random_state=42
                )
            else:
                X_train_res, X_test_res, y_train_res, y_test_res = preprocessor_bal.split_data(test_size=test_size)

            balanced_results = evaluator.evaluate_models(X_train_res, X_test_res, y_train_res, y_test_res)

            for model_name, metrics in balanced_results.items():
                st.markdown(f"#### Results for {model_name} (Balanced Data)")
                df_metrics = pd.DataFrame(metrics).transpose()
                st.dataframe(df_metrics.style.background_gradient(cmap="Greens"))

        st.success("Done! Dataset processed, balanced, evaluated, and available for download.")