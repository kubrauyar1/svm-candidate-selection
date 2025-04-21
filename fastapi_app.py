from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from job_applicant_svm import model, scaler

app = FastAPI(title="Job Application Evaluation API",
             description="SVM-based job application evaluation system",
             version="1.0.0")

class ApplicantData(BaseModel):
    experience_years: float
    technical_score: float

@app.post("/predict")
async def predict_applicant(data: ApplicantData):
    try:
        # Scale the input
        input_data = np.array([[data.experience_years, data.technical_score]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Check if the prediction matches our criteria
        if data.experience_years < 2 and data.technical_score < 60:
            result = "Not hired"
            prediction_code = 1
        else:
            result = "Hired"
            prediction_code = 0
        
        return {
            "experience_years": data.experience_years,
            "technical_score": data.technical_score,
            "result": result,
            "prediction_code": prediction_code
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Job Application Evaluation API",
        "usage": "Send experience_years and technical_score to the POST /predict endpoint"
    } 