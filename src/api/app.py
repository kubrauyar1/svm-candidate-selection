from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from ..models.svm_model import JobApplicantSVM

app = FastAPI(
    title="İşe Alım Değerlendirme API",
    description="SVM tabanlı işe alım değerlendirme sistemi",
    version="1.0.0"
)

# Initialize the model
model = JobApplicantSVM()
X, y = model.generate_data()
model.prepare_data(X, y)
model.train_model()

class ApplicantData(BaseModel):
    tecrube_yili: float
    teknik_puan: float

@app.post("/predict")
async def predict_applicant(data: ApplicantData):
    try:
        prediction = model.predict(data.tecrube_yili, data.teknik_puan)
        
        return {
            "tecrube_yili": data.tecrube_yili,
            "teknik_puan": data.teknik_puan,
            "sonuc": prediction,
            "tahmin_kodu": 1 if prediction == "İşe alınmadı" else 0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "İşe Alım Değerlendirme API'sine Hoş Geldiniz",
        "usage": "POST /predict endpoint'ine tecrube_yili ve teknik_puan gönderin"
    } 