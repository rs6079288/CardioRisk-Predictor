import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile


# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Load Saved model and scaler 
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict_single(input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    prediction_proba = model.predict_proba(scaled_data)[0]
    risk_percentage = prediction_proba[1] * 100
    return prediction, risk_percentage, scaled_data

def get_advice(risk_percentrage):
    if risk_percentrage <= 25:
        status = "Very Low Risk"
        advice = "Do: maintain Healthy diet, Stay active. Don't: Avoid smoking and Junk food"
    elif risk_percentrage <= 50:
        status = "Low-Moderate Risk"
        advice = "Do: Do regular cardio, Monitor Cholestrol. Don't: Avoid Sedentary lifestyle & high sugar."
    elif risk_percentrage <= 75:
        status = "Moderate-High Risk"
        advice = "Do: Consult Doctor, Monitor BP weekly. Don't: Avoid High sodium diet & heavy stress."
    else:
        status = "Critical High Risk"
        advice = "Do: Consult Cardiologist immediately, take medication strictly. Don't: Avoid heavy exercise & smoking"
    return status, advice

def process_batch_data(input_df):
    if 'Name' in input_df.columns:
        model_input_df = input_df.drop(columns=['Name'])
    else:
        model_input_df = input_df

    scaled_data = scaler.transform(model_input_df)
    probabilities = model.predict_proba(scaled_data)[:, 1] * 100

    result_df = input_df.copy()
    statuses = []
    advice_list = []

    for p in probabilities:
        status, advice = get_advice(p)
        statuses.append(status)
        advice_list.append(advice)

    result_df['Risk Status'] = statuses
    result_df['Confidence Score (%)'] = probabilities.round(2)
    result_df['Advices'] = advice_list

    return result_df

def generate_shap_explanation(scaled_data):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(scaled_data)

    if isinstance(shap_vals, list):
        s_vals = shap_vals[1][0]
    elif len(np.array(shap_vals).shape) == 3:
        s_vals = shap_vals[0, :, 1]
    else:
        s_vals = shap_vals[0]

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    fig_shap, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if val > 0 else 'blue' for val in s_vals]
    ax.barh(features, s_vals, color=colors)
    ax.set_xlabel("Impact on Model Output")
    ax.set_title("Feature Impact on Risk Prediction")
    ax.axvline(0, color='grey', linewidth=0.8)

    return fig_shap

def generate_medical_report_pdf(patient_data, prediction_result, risk_probability, fig_shap):
    pdf = FPDF()
    pdf.add_page()

    # Title / Header
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(30, 144, 255)
    pdf.cell(0, 10, "Cardiovascular AI - Medical Report Card", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "AI-Powered Heart Disease Risk Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Patient Prediction Summary Box
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Assessment Result:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 12)
    if "High" in str(prediction_result):
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(0, 150, 0)
    
    pdf.cell(0, 8, f"Status: {prediction_result} (Probability: {risk_probability:.2f})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Input parameters Section
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Patient Health Parameters:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    col_width = 95
    row_height = 6

    # Mapping dictionaries for text conversion
    sex_map={1:'Male',0:'Female','1':'Male','0':'Female'}
    exang_map={1:'Yes',0:'No','1':'Yes','0':'No'}

    for key, value in patient_data.items():
        display_val=value
        if key.lower()=='sex' and value in sex_map:
            display_val=sex_map[value]
        elif (key.lower()=='exang' or 'angina' in key.lower()) and value in exang_map:
            display_val=exang_map[value]
        pdf.cell(col_width, row_height, f"{key}:", border=1)
        pdf.cell(col_width, row_height, f"{display_val}", border=1, new_x='LMARGIN', new_y="NEXT")

    pdf.ln(10)

    # Attaching SHAP Graph
    if fig_shap is not None:


        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            fig_shap.savefig(tmp_file.name, bbox_inches='tight', dpi=150)
            tmp_path = tmp_file.name

        pdf.image(tmp_path, x=15, w=180)
        os.unlink(tmp_path)

    # Footer disclaimer
    pdf.set_y(-25)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, "Disclaimer: This report is generated by an AI model (CardioCare AI) for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.", align="C")

    pdf_output = pdf.output()
    if isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return pdf_output

def generate_batch_pdf_report(result_df):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    # Title Section
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(30, 144, 255)
    pdf.cell(0, 8, "CardioCare AI - Batch Patients Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Multiple Patients Risk Analysis Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # Sirf zaroori columns select karein taaki text overlap na ho
    preferred_columns = ['Name', 'age', 'sex', 'cp', 'trestbps', 'chol', 'Risk Status', 'Confidence Score (%)', 'Advices']
    columns = [col for col in preferred_columns if col in result_df.columns]
    if not columns:
        columns = list(result_df.columns)
        
    # Har column ke liye proper fix width define karein
    col_widths = {
        'Name': 30, 'age': 12, 'sex': 12, 'cp': 12, 'trestbps': 22, 
        'chol': 22, 'Risk Status': 38, 'Confidence Score (%)': 25, 'Advices': 95
    }
    
    # Table Header
    pdf.set_font("helvetica", "B", 8)
    pdf.set_fill_color(30, 144, 255)
    pdf.set_text_color(255, 255, 255)
    
    for col in columns:
        w = col_widths.get(col, 20)
        pdf.cell(w, 7, str(col)[:18], border=1, fill=True, align="C")
    pdf.ln()
    
    # Table Rows
    pdf.set_font("helvetica", "", 7)
    pdf.set_text_color(0, 0, 0)
    
    for index, row in result_df.iterrows():
        for col in columns:
            w = col_widths.get(col, 20)
            val = str(row[col])
            # Text ko limit karein taaki cell ke bahar na jaye
            pdf.cell(w, 6, val[:40], border=1, align="C")
        pdf.ln()
        
    pdf_output = pdf.output()
    if isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    return pdf_output