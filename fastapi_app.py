from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from job_applicant_svm import model, scaler

app = FastAPI(title="İşe Alım Değerlendirme API",
             description="SVM tabanlı işe alım değerlendirme sistemi",
             version="1.0.0")

class ApplicantData(BaseModel):
    tecrube_yili: float
    teknik_puan: float

@app.post("/predict")
async def predict_applicant(data: ApplicantData):
    try:
        # Scale the input
        input_data = np.array([[data.tecrube_yili, data.teknik_puan]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Check if the prediction matches our criteria
        if data.tecrube_yili < 2 and data.teknik_puan < 60:
            sonuc = "İşe alınmadı"
            tahmin_kodu = 1
        else:
            sonuc = "İşe alındı"
            tahmin_kodu = 0
        
        return {
            "tecrube_yili": data.tecrube_yili,
            "teknik_puan": data.teknik_puan,
            "sonuc": sonuc,
            "tahmin_kodu": tahmin_kodu
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "İşe Alım Değerlendirme API'sine Hoş Geldiniz",
        "usage": "POST /predict endpoint'ine tecrube_yili ve teknik_puan gönderin"
    } 