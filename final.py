import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import shap

#---------------------page confi_____________________________________________

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")
st.title("FINANCIAL TRANSACTION FRAUD DETECTION")
st.markdown("""
THERE ARE  SOME FEATURES:
- **USER MODE**:for user
- **DEVELOPER MODE**:for developers
- **REAL TIME STIMULATION**
""")
#------------------------------------models loading _______________________________________

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

#-----------------------------------------for optional dataset__________________________
sample = None
if os.path.exists("data/X_test.csv"):
    try:
        sample = pd.read_csv("data/X_test.csv")
        st.info("a sample csv file for test pca.")
    except Exception:
        pass
#-----------------------------------------sidebar_________________________________________
with st.sidebar:
    st.markdown("---")
    st.write("**Model Info:**")
    st.write(f"- Model: {type(model).__name__ if model else 'None'}")
    st.write(f"- Scaler loaded: {'Yes' if scaler else 'No'}")
    st.markdown("---")
#--------------------------------------Utility--------------------------------------------------
Log_File="prediction.csv"
def save_predictionshistory(record: dict):
    df = pd.DataFrame([record])
    if os.path.exists(Log_File):
        df.to_csv(Log_File, mode="a", header=False, index=False)
    else:
        df.to_csv(Log_File, index=False)

def load_predictionsunit():
    if os.path.exists(Log_File):
        return pd.read_csv(Log_File)
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
#------------------------------------------------tab_____________________________________________________
tabs = st.tabs([" USER MODE", "DEVELOPER MODE", " REAL STIMULATION MODE", "EXPLANABILITY"])
#---------------------------------------------------USER MODE_____________________________________________
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


#--------------------------------------------------DEVELOPER MODE_____________________________________________________
with tabs[1]:
    st.subheader(" Developer Mode")
    mode_choice = st.radio(
        "Select Input Mode",
        [" Batch CSV Upload", "Manual Entry"],
        horizontal=True
    )

  #----------------------------------------------------------------batch----
    if mode_choice == "Batch CSV Upload":
        uploaded_batch = st.file_uploader("Upload CSV ", type=["csv"])

        if uploaded_batch is not None:
            try:
                csv_df = pd.read_csv(uploaded_batch)
                st.success(f"File load successfully with {csv_df.shape[0]} records.")
                st.dataframe(csv_df.head())

                if st.button("PRESS", key="predict_batch"):
                    if scaler is not None:
                        try:
                            csv_df[["Time", "Amount"]] = scaler.transform(csv_df[["Time", "Amount"]])
                        except Exception:
                            st.warning("failed.")

                    prediction = model.predict(csv_df)
                    scores = -model.decision_function(csv_df)
                    csv_df["Prediction"] = np.where(prediction== -1, " FRAUD", " LEGIT")
                    csv_df["Score"] = scores

                    st.subheader(" Prediction outcomes")
                    st.dataframe(csv_df.head(20))

                    fraud_rate = (csv_df["Prediction"] == "🚨 FRAUD").mean() * 100
                    st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

                    fig, ax = plt.subplots()
                    sns.countplot(x="Prediction", data=csv_df, palette="coolwarm", ax=ax)
                    plt.title("Fraud vs Legit Transactions")
                    st.pyplot(fig)

                    for _, row in csv_df.iterrows():
                        record = {
                            "Mode": "Developer",
                            "Amount": row["Amount"],
                            "Time": row["Time"],
                            "Prediction": row["Prediction"],
                            "Score": row["Score"]
                        }
                        save_predictionshistory(record)
                    st.session_state["last_input_df"] = csv_df

            except Exception as e:
                st.error(f" Failed to read CSV: {e}")

        else:
            st.info("Please upload a CSV file.")
            #-----------------------manual_____________________________________________
    else:
        st.markdown("### Enter Transaction Details ")

        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Transaction Time", min_value=0.0, value=0.0, step=1.0)
        with c2:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.1)
        columns = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            idx = (i - 1) % 4
            v_values[f"V{i}"] = columns[idx].number_input(f"V{i}", value=0.0, format="%.6f", key=f"V{i}")

        if st.button(" PRESS", key="predict_manual"):
            input_row = {"Time": float(time_val), **v_values, "Amount": float(amount)}
            input_df = pd.DataFrame([input_row], columns=FEATURES)

            if scaler is not None:
                try:
                    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])
                except Exception:
                    st.warning(" Scaler transformation failed — using raw values.")

            pred = model.predict(input_df)
            pred_label = "FRAUD" if pred[0] == -1 else "LEGIT"
            score = float(-model.decision_function(input_df)[0])

            # Display result
            if pred_label == "FRAUD":
                st.error(f"🚨 Fraud Detected (score: {score:.4f})")
            else:
                st.success(f"Legitimate Transaction (score: {score:.4f})")

            # Save and store in session
            record = {
                "Mode": "Developer", "Amount": amount, "Time": time_val,
                "Prediction": pred_label, "Score": score
            }
            save_predictionshistory(record)
            st.session_state["last_input_df"] = input_df

  #_________________________________
    st.subheader("Developer Prediction History")


    if st.button("CLEAR", key="delete_dev_history"):
        if os.path.exists(Log_File):
            os.remove(Log_File)
            st.success(" prediction history deleted successfully.")
            st.rerun()
        else:
            st.info("No prediction history found to delete.")

    history = load_predictionsunit()
    developer_records = history[history["Mode"] == "Developer"]

    if not developer_records.empty:
        st.write(" Recent Developer Predictions")
        st.dataframe(developer_records.tail(10))

        st.write(" Fraud vs Legit Prediction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=developer_records, x="Prediction", palette="coolwarm", ax=ax)
        ax.set_title("Developer Predictions Overview")
        st.pyplot(fig)
    else:
        st.info("No developer predictions recorded yet.")
#__________________________________________live Streaming________________________________________________
with tabs[2]:
    st.header(" Real-Time Transaction stream")

    file_upload = st.file_uploader("Upload CSV file ", type=["csv"])
    real_simulate = st.checkbox("Enable Real-Time Simulation", value=False)

    if  real_simulate:
        speed = st.slider("Transaction speed (seconds per record)", 0.2, 5.0, 1.5)
        n_tx = st.slider("Number of simulated transactions", 5, 100, 20)
        source_choice = st.radio("Data Source", ["Random", "Use Uploaded CSV "], index=0)

        if source_choice == "Use Uploaded CSV if available" and  file_upload is not None:
            data = pd.read_csv( file_upload)
            if len(data) > n_tx:
                data_source = data.sample(n=n_tx).reset_index(drop=True)
        else:
            data = pd.DataFrame({
                "Time": np.random.randint(0, 500000, n_tx),
                "Amount": np.random.uniform(1, 15000, n_tx),
            })
            for i in range(1, 29):
                data[f"V{i}"] = np.random.normal(0, 1, n_tx)
            data_source = data[FEATURES]

        st.success(f"Streaming {len(data_source)} transactions...")

        placeholder = st.empty()
        chart_placeholder = st.empty()
        progress = st.progress(0)

        fraud_count = legit_count = 0
        results = []

        for i, row in data_source.iterrows():
            row = pd.DataFrame([row], columns=FEATURES)

            if scaler is not None:
                try:
                    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
                except Exception:
                    pass

            prediction = model.predict(row)[0]
            label = "🚨 FRAUD" if prediction == -1 else "✅ LEGIT"
            score = float(-model.decision_function(row)[0])

            if prediction == -1:
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

            with placeholder.container():
                st.subheader(f"Transaction #{i + 1}")
                st.metric("Amount", f"{row['Amount']:.2f}")
                st.metric("Score", f"{score:.3f}")
                st.markdown(f"### Result: {label}")

                c1, c2 = st.columns(2)
                c1.metric("✅ Legit Count", legit_count)
                c2.metric("🚨 Fraud Count", fraud_count)

            chart = pd.DataFrame(results)["Prediction"].value_counts().rename_axis("Type").reset_index(name="Count")
            chart_placeholder.bar_chart(chart.set_index("Type"))

            progress.progress((i + 1) / len(data_source))
            time.sleep(speed)

        st.success("✅ Stream complete!")
        st.dataframe(pd.DataFrame(results))