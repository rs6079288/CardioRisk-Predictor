import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from utils import (
    predict_single, 
    get_advice, 
    process_batch_data, 
    generate_shap_explanation, 
    generate_medical_report_pdf, 
    generate_batch_pdf_report
)

def init_db():
    conn = sqlite3.connect('patient_history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_name TEXT,
                        age INTEGER, 
                        risk_score REAL,
                        risk_status TEXT,
                        timestamp TEXT)''')
    conn.commit()
    conn.close()

# Function for saving the record
def save_patient_record(name, age, risk_score, risk_status):
    conn = sqlite3.connect('patient_history.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''INSERT INTO history (patient_name, age, risk_score, risk_status, timestamp)
                      VALUES (?,?,?,?,?)''', (name, age, risk_score, risk_status, timestamp))
    conn.commit()
    conn.close()



# Function for fetching the old record
def get_patient_history(name):
    conn = sqlite3.connect('patient_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, age, risk_score, risk_status FROM history WHERE patient_name = ? ORDER BY id DESC', (name,))
    data = cursor.fetchall()
    conn.close()
    return data

# Function for fetching ALL saved patient records
def get_all_patients():
    conn = sqlite3.connect('patient_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, patient_name, age, risk_score, risk_status, timestamp FROM history ORDER BY id DESC')
    data = cursor.fetchall()
    conn.close()
    return data

# Function to delete a specific record by its ID
def delete_patient_record(record_id):
    conn = sqlite3.connect('patient_history.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()

# Initialize db after the app load
init_db()

# Page Configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Sidebar Navigation for Multiple Pages
st.sidebar.title("🏥 Navigation Panel")
app_mode = st.sidebar.radio(
    "Choose Mode", 
    ["Single Prediction", "Batch Prediction (CSV)", "Saved Patient Records"]
)

# --- PAGE 1: SINGLE PREDICTION ---
if app_mode == "Single Prediction":
    st.title("❤️ Cardiovascular Disease Risk Predictor")
    st.write("Enter Your Health Parameters Below To Check Your Health Disease Risk Score Using Our AI Model.")

    st.subheader("Patient Health Details")
    patient_name = st.text_input("Patient Name", value="Patient 1")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Years)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Serum Cholestrol (chol in mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
        restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope Peak Exercise ST segment (slope)", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalessemia (thal)", options=[0, 1, 2, 3])

    if st.button("Predict Risk"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        predictions, risk_percentage, scaled_data = predict_single(input_data)
        status, advice = get_advice(risk_percentage)
        
        st.session_state['scaled_data'] = scaled_data
        st.session_state['risk_percentage'] = risk_percentage
        st.session_state['status'] = status
        st.session_state['advice'] = advice
        st.session_state['patient_name'] = patient_name
        
        fig_shap = generate_shap_explanation(scaled_data)
        st.session_state['fig_shap'] = fig_shap
        
        # save_patient_record(patient_name, age, risk_percentage, status)

    if 'risk_percentage' in st.session_state:
        risk_percentage = st.session_state['risk_percentage']
        status = st.session_state['status']
        advice = st.session_state['advice']

        st.markdown("---")
        st.subheader("Prediction Result")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Score (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if risk_percentage > 50 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        if risk_percentage > 50:
            st.error(f"⚠️ **{status} Detected!** (Score: {risk_percentage:.2f}%)")
        else:
            st.success(f"✅ **{status} Detected!** (Score: {risk_percentage:.2f}%)")
        st.info(f"**Medical Advice:** {advice}")

        st.markdown("---")
        st.subheader("📊 Why this prediction?")
        if st.button("Generate Explanation Graph"):
            with st.spinner("Generating SHAP explanation..."):
                try:
                    fig_shap = generate_shap_explanation(st.session_state['scaled_data'])
                    st.pyplot(fig_shap)
                    plt.clf()
                    st.info("**How to read this:** Red bars push the risk score higher, while blue bars push the risk score lower based on individual health parameters.")
                except Exception as e:
                    st.error(f"SHAP Error: {e}")

        st.markdown("---")
        st.subheader("📥 Download Medical Report")
        
        
        st.markdown("---")
        if st.button("💾 Save This Record to Database"):
            save_patient_record(patient_name, age, risk_percentage, status)
            st.success("Record has succesfully saved")
       
       
        try:
            patient_inputs = {
                "Age": age,
                "Sex": sex,
                "Chest Pain Type (cp)": cp,
                "Resting Blood Pressure": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar": fbs,
                "Resting ECG": restecg,
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": exang,
                "Oldpeak": oldpeak,
                "Slope": slope,
                "Major Vessels (ca)": ca,
                "Thal": thal
            }
            
            pdf_bytes = generate_medical_report_pdf(
                patient_data=patient_inputs,
                prediction_result=st.session_state.get('status', 'Unknown'),
                risk_probability=st.session_state.get('risk_percentage', 0.0),
                fig_shap=st.session_state.get('fig_shap', None)
            )
            
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)
            elif isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode('latin1')

            st.download_button(
                label="📄 Click here to download PDF Report",
                data=pdf_bytes,
                file_name="Medical_Report_Card.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Generation Error: {e}")


# --- PAGE 2: BATCH PREDICTION (CSV) ---
elif app_mode == "Batch Prediction (CSV)":
    st.title("📂 Batch Patient Risk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file containing patient data", type=["csv"])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", input_df.head())
        
        if st.button("Process Batch Data"):
            with st.spinner("Processing batch data..."):
                st.session_state['result_df'] = process_batch_data(input_df)
                st.success("Batch Processing Successful!")

        if 'result_df' in st.session_state:
            result_df = st.session_state['result_df']
            st.dataframe(result_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Batch Analytics Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🥧 Risk Level Distribution")
                status_col = None
                for col in result_df.columns:
                    if 'status' in col.lower() or 'risk' in col.lower():
                        status_col = col
                        break
                        
                if status_col:
                    risk_counts = result_df[status_col].value_counts().reset_index()
                    risk_counts.columns = ['Risk_Status', 'Count']
                    
                    fig_pie = px.pie(
                        risk_counts, 
                        names='Risk_Status', 
                        values='Count', 
                        hole=0.4,
                        color_discrete_sequence=['#2eb82e', '#ff4d4d', '#ff9900', '#ffc000']
                    )
                    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Risk status column not found in data.")
                    
            with col2:
                st.markdown("##### 📈 Risk by Age Groups")
                age_col = None
                score_col = None
                
                for col in result_df.columns:
                    if col.strip().lower() == 'age':
                        age_col = col
                    if any(term in col.lower() for term in ['percentage', 'score', 'probability', 'confidence']):
                        score_col = col
                        
                if age_col and score_col:
                    result_df['Age_Group'] = pd.cut(
                        result_df[age_col], 
                        bins=[0, 30, 45, 60, 100], 
                        labels=['<30', '30-45', '46-60', '60+']
                    )
                    age_risk = result_df.groupby('Age_Group', observed=False)[score_col].mean().reset_index()
                    
                    fig_bar = px.bar(
                        age_risk, 
                        x='Age_Group', 
                        y=score_col,
                        text_auto='.1f',
                        color=score_col,
                        color_continuous_scale='Reds'
                    )
                    fig_bar.update_layout(
                        xaxis_title="Age Group", 
                        yaxis_title="Avg Risk Score (%)",
                        margin=dict(t=10, b=10, l=10, r=10), 
                        height=300,
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(f"Could not find Age or Score column. Available columns: {list(result_df.columns)}")
            
            try:
                pdf_bytes_batch = generate_batch_pdf_report(result_df)
                if isinstance(pdf_bytes_batch, bytearray):
                    pdf_bytes_batch = bytes(pdf_bytes_batch)
                elif isinstance(pdf_bytes_batch, str):
                    pdf_bytes_batch = pdf_bytes_batch.encode('latin1')

                st.download_button(
                    label="📥 Download Batch PDF Report",
                    data=pdf_bytes_batch,
                    file_name="batch_medical_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Batch PDF Error: {e}")


# --- PAGE 3: SAVED PATIENT RECORDS & HISTORY TRACKER ---
elif app_mode == "Saved Patient Records":
    st.title("📋Saved Patient Records")
    st.write("Here you can see all the saved Patient Records.")

    all_records = get_all_patients()
    if all_records:
        # Data ko DataFrame mein convert karein
        all_df = pd.DataFrame(all_records, columns=['ID', 'Patient Name', 'Age', 'Risk Score (%)', 'Status', 'Timestamp'])
        
        # Ek naya column 'Delete' (checkbox) add kar rahe hain
        all_df.insert(0, "Delete", False)
        
        # st.data_editor ka use karke interactive table bana rahe hain
        edited_df = st.data_editor(
            all_df,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Select this checkbox to delete the record",
                    default=False,
                )
            },
            disabled=['ID', 'Patient Name', 'Age', 'Risk Score (%)', 'Status', 'Timestamp'],
            hide_index=True,
            use_container_width=True
        )
        
        # Check karein ki user ne kisi row ko delete ke liye tick kiya hai ya nahi
        rows_to_delete = edited_df[edited_df["Delete"] == True]
        
        if not rows_to_delete.empty:
            if st.button("🗑️ Delete Selected Records"):
                for idx, row in rows_to_delete.iterrows():
                    delete_patient_record(row['ID'])
                st.success("Selected records has successfully deleted")
                st.rerun()
    else:
        st.info("There is no Saved Patient Record.")