import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import shap
#-------------------------------------------------pagecong____________________
st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")
st.title("FINANCIAL TRANSACTION FRAUD DETECTION")
st.markdown("""
THERE ARE  SOME FEATURES:
- **USER MODE**:for user
- **DEVELOPER MODE**:for developers
- **REAL TIME STIMULATION**
""")
#===-------------------------------------------model loading---------------------------------------
MODEL_P= "model_forest.pkl"
SCALER_P= "scaler (1).pkl"

def load_models():
    if os.path.exists(MODEL_P):
        try:
            model = joblib.load(MODEL_P)
            st.success(f" Model is loaded successfully from: `{MODEL_P}`")
            return model
        except Exception as e:
            st.error(f" Error in loading model: {e}")
            return None
    else:
        st.error(f"Model file not found: `{MODEL_P}`")
        return None

def load_scalerfile():
    if os.path.exists(SCALER_P):
        try:
            scaler = joblib.load(SCALER_P)
            st.info(f" successfully Scaler loaded from: `{SCALER_P}`")
            return scaler
        except Exception as e:
            st.warning(f" Error in loading scaler: {e}")
            return None
    else:
        st.warning(f"⚠ Scaler file not found: `{SCALER_P}`")
        return None

model = load_models()
scaler = load_scalerfile()

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

sample = None
if os.path.exists("data/X_test.csv"):
    try:
        sample = pd.read_csv("data/X_test.csv")
        st.info("a sample csv file for test pca.")
    except Exception:
        pass

with st.sidebar:
    st.markdown("---")
    st.write("**Model Info:**")
    st.write(f"- Model: {type(model).__name__ if model else 'None'}")
    st.write(f"- Scaler loaded: {'Yes' if scaler else 'No'}")
    st.markdown("---")

Log_File="prediction.csv"
def save_predictionshistory(record: dict):
    df = pd.DataFrame([record])
    if os.path.exists(Log_File):
        df.to_csv(Log_File, mode="a", header=False, index=False)
    else:
        df.to_csv(Log_File, index=False)

def load_predictionsunit():
    if os.path.exists(Log_File):
        try:
            return pd.read_csv(Log_File, on_bad_lines='skip')
        except Exception:
            st.warning("⚠ Error reading CSV — skipping corrupted lines.")
            return pd.DataFrame(columns=["Mode", "Amount", "Time", "Prediction", "Score"])
    return pd.DataFrame(columns=["Mode", "Amount", "Time", "Prediction", "Score"])


def delete_prediction_history():
    if os.path.exists(Log_File):
        os.remove(Log_File)
        return True
    return False

def get_nearest_sample(amount, time_val):
    if sample is None:
        return None
    df = sample.copy()
    if "Time" not in df or "Amount" not in df:
        return None
    df["dist"] = ((df["Time"] - time_val)**2 + (df["Amount"] - amount)**2)**0.5
    row = df.loc[df["dist"].idxmin()]
    return {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29) if f"V{i}" in row}

def build_input_row(amount, time_val):
    vals = get_nearest_sample(amount, time_val)
    if vals is None:
        vals = {f"V{i}": float(np.random.normal(0, 0.1)) for i in range(1, 29)}
    row = {"Time": float(time_val), **vals, "Amount": float(amount)}
    return row
#-----------------------------------------------tab-----------------------------------------------------
tabs = st.tabs([" USER MODE", "DEVELOPER MODE", " REAL STIMULATION MODE", "EXPLANABILITY"])
#-----------------------------------user---------------------------------------------------
with tabs[0]:
    st.subheader("User Mode — Simple Transaction Input")

    c1, c2 = st.columns([1, 1])
    with c1:
        amount = st.number_input("TRANSACTION AMOUNT")
        time_val = st.number_input("TRANSACTION TIME ")
        card_type = st.selectbox("CARD TYPE", ["Credit", "Debit", "Prepaid", "Virtual"])
        merchant = st.selectbox("MERCHANT TYPE", ["Shopping", "Food", "Travel", "Fuel", "Entertainment", "Other"])
    with c2:
        country = st.selectbox("COUNTRY", ["India", "USA", "UK", "Germany", "France","China", "Other"])
        channel = st.selectbox("TRANSACTION TYPE", ["Online", "In-person"])
        recent_tx = st.slider("RECENT TRANSACTION IN 24 HOURS", 0, 200, 3)

    if st.button("PRESS"):
        input_row = build_input_row(amount, time_val)
        input_df = pd.DataFrame([input_row], columns=FEATURES)

        if scaler is not None:
            try:
                input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])
            except Exception:
                st.warning("Scaler failed — skipping normalization.")

        if model is not None:
            prediction = model.predict(input_df)
            pred_label = "FRAUD" if prediction[0] == -1 else "LEGIT"
            try:
                score = float(-model.decision_function(input_df)[0])
            except Exception:
                score = None

            if pred_label == "FRAUD":
                st.error(f" Fraud Detected (score: {score:.4f})")
            else:
                st.success(f" Legitimate Transaction (score: {score:.4f})")

            record = {
                "Mode": "User",
                "Amount": amount,
                "Time": time_val,
                "CardType": card_type,
                "Merchant": merchant,
                "Country": country,
                "Channel": channel,
                "RecentTx": recent_tx,
                "Prediction": pred_label,
                "Score": score
            }
            save_predictionshistory(record)
    st.markdown("---")
    st.subheader(" HISTORY")
    if st.button(" CLEAR HISTORY", key="delete_user_history"):
        if delete_prediction_history():
            st.success("History cleared.")
            st.rerun()
        else:
            st.warning("No history found.")

    history = load_predictionsunit()
    user_records = history[history["Mode"] == "User"]

    if not user_records.empty:
        st.write(" Recently Predictions")
        st.dataframe(user_records.tail(10))
        st.write("Prediction Overview")
        fig, ax = plt.subplots()
        sns.countplot(data=user_records, x="Prediction", palette="coolwarm", ax=ax)
        ax.set_title("Distribution of User Predictions")
        ax.set_xlabel("Prediction Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No prediction data available for User Mode yet.")

with tabs[1]:
    st.subheader("Developer Mode – Analyze Transactions with CSV or Manual Input")

    mode_choice = st.radio("Select Input Type", ["Batch CSV Upload", "Manual Entry"], horizontal=True)

    if mode_choice == "Batch CSV Upload":
        uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully — {batch_df.shape[0]} records found.")
                st.dataframe(batch_df.head())

                if st.button("PRESS", key="batch_predict_btn"):

                    if scaler is not None:
                        try:
                            batch_df[["Time", "Amount"]] = scaler.transform(batch_df[["Time", "Amount"]])
                        except Exception:
                            st.warning("Scaling failed.")

                    predictions = model.predict(batch_df)
                    risk_scores = -model.decision_function(batch_df)

                    batch_df["Prediction"] = np.where(predictions == -1, "🚨 FRAUD", "✅ LEGIT")
                    batch_df["Score"] = risk_scores
                    st.session_state["last_input_df"] = batch_df

                    st.subheader("Batch Prediction Results")
                    st.dataframe(batch_df.head(20))

                    fraud_percent = (batch_df["Prediction"] == "🚨 FRAUD").mean() * 100
                    st.metric("Detected Fraud Percentage", f"{fraud_percent:.2f}%")

                    fig, ax = plt.subplots()
                    sns.countplot(x="Prediction", data=batch_df, palette="coolwarm", ax=ax)
                    plt.title("Fraud vs Legit Transactions")
                    st.pyplot(fig)

                    for _, record in batch_df.iterrows():
                        save_predictionshistory({
                            "Mode": "Developer",
                            "Amount": record["Amount"],
                            "Time": record["Time"],
                            "Prediction": record["Prediction"],
                            "Score": record["Score"]
                        })
            except Exception as e:
                st.error(f" Unable to read CSV file: {e}")
        else:
            st.info("Upload a CSV file to start batch prediction analysis.")

    elif mode_choice == "Manual Entry":
        st.markdown("### Enter Transaction Details Manually")

        col1, col2 = st.columns(2)
        with col1:
            t_val = st.number_input("Transaction Time", value=0.0, step=1.0)
        with col2:
            amt = st.number_input("Transaction Amount", value=0.0, step=0.1)

        cols = st.columns(4)
        features_input = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            features_input[f"V{i}"] = cols[col_idx].number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}_input")

        if st.button("Predict", key="manual_predict_btn"):
            row_data = {"Time": float(t_val), **features_input, "Amount": float(amt)}
            input_df = pd.DataFrame([row_data], columns=FEATURES)
            if scaler is not None:
                try:
                    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])
                except Exception:
                    st.warning("Scaler could not be applied .")

            outcome = model.predict(input_df)[0]
            score = float(-model.decision_function(input_df)[0])
            label = "🚨 FRAUD" if outcome == -1 else "✅ LEGIT"
            st.session_state["last_input_df"] = input_df
            st.session_state["last_prediction"] = label

            if label == "🚨 FRAUD":
                  st.error(f"🚨 Potential Fraud Detected (Score: {score:.4f})")
            else:
                st.success(f"Transaction Appears Legitimate (Score: {score:.4f})")

            save_predictionshistory({
                "Mode": "Developer",
                "Amount": amt,
                "Time": t_val,
                "Prediction": label,
                "Score": score
            })

    st.markdown("---")
    st.subheader("Developer Prediction History")

    if st.button("🗑 Clear History", key="clear_dev_history"):
        if delete_prediction_history():
            st.success(" All developer predictions removed.")
            st.rerun()
        else:
            st.warning("No prediction history to delete.")

    history_df = load_predictionsunit()
    dev_history = history_df[history_df["Mode"] == "Developer"]

    if not dev_history.empty:
        st.dataframe(dev_history.tail(10))
        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=dev_history, palette="coolwarm", ax=ax)
        plt.title("Developer Predictions Overview")
        st.pyplot(fig)
    else:
        st.info("No records found in developer prediction history.")
#----------------------------------------------------stimulation---------------------------------------------
with tabs[2]:
    st.header(" Real-Time Transaction Stream Simulation")

    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    simulate = st.checkbox("Enable Real-Time Simulation", value=False)

    if simulate:
        stream_speed = st.slider("Transaction speed (seconds per record)", 0.5, 5.0, 1.5)
        num_tx = st.slider("Number of simulated transactions", 5, 100, 20)

        if csv_file is not None:
            data_source = pd.read_csv(csv_file)
            if len(data_source) > num_tx:
                data_source = data_source.sample(n=num_tx).reset_index(drop=True)

            st.success(f"Streaming {len(data_source)} transactions...")

            placeholder_main = st.empty()
            chart_placeholder = st.empty()
            progress = st.progress(0)
            fraud_count = legit_count = 0
            results = []

            for i, row in data_source.iterrows():
                row_df = pd.DataFrame([row], columns=FEATURES)

                if scaler is not None:
                    try:
                        row_df[["Time", "Amount"]] = scaler.transform(row_df[["Time", "Amount"]])
                    except Exception:
                        pass

                pred = model.predict(row_df)[0]
                label = "🚨 FRAUD" if pred == -1 else "✅ LEGIT"
                score = float(-model.decision_function(row_df)[0])

                if pred == -1:
                    fraud_count += 1
                else:
                    legit_count += 1

                results.append({
                    "Transaction #": i + 1,
                    "Amount": float(row["Amount"]),
                    "Time": float(row["Time"]),
                    "Prediction": label,
                    "Score": score
                })

                with placeholder_main.container():
                    st.subheader(f"Transaction #{i + 1}")
                    st.metric("Amount", f"{row['Amount']:.2f}")
                    st.metric("Score", f"{score:.3f}")
                    st.markdown(f"### Result: {label}")

                    c1, c2 = st.columns(2)
                    c1.metric("✅ Legit Count", legit_count)
                    c2.metric("🚨 Fraud Count", fraud_count)

                chart_data = pd.DataFrame(results)["Prediction"].value_counts().rename_axis("Type").reset_index(name="Count")
                chart_placeholder.bar_chart(chart_data.set_index("Type"))
                progress.progress((i + 1) / len(data_source))
                time.sleep(stream_speed)

            st.success("✅ Stream complete!")
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("Please upload a CSV file to start streaming.")
#----------------------------------------------------explanabitity==============================
with tabs[3]:
    st.header("Model Explainability (SHAP)")
    st.write("View how each feature affects the prediction.")

    if "last_input_df" not in st.session_state:
        st.warning("Run a Developer Mode.")
    else:
        try:
            df = st.session_state["last_input_df"]
            numeric_df = df.select_dtypes(include=[np.number])

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.Explainer(model, numeric_df)
                shap_values = explainer(numeric_df)

            st.subheader("Feature Importance")
            fig1, ax1 = plt.subplots()
            shap.summary_plot(shap_values, numeric_df, show=False)
            st.pyplot(fig1, bbox_inches='tight', clear_figure=True)

            st.subheader(" Local Explanation")
            fig2, ax2 = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig2, bbox_inches='tight', clear_figure=True)

        except Exception as e:
            st.error(f"Error in SHAP analysis: {e}")


