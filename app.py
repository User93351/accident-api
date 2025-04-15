# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import os
from email.mime.text import MIMEText
import smtplib

app = FastAPI()

model = None
class_names = ["accident", "no_accident"]
API_KEY = os.getenv("API_KEY", "your_default_key_here")

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model("my_model.keras")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

def notify_authorities():
    sender_email = "shrinkhalshrinkhal@gmail.com"
    receiver_emails = [
        "privatehvro@gmail.com",
        "shrinkhalshrinkhal22@gmail.com",
        "adityabhatt78910@gmail.com",
        "lakshitatak1@gmail.com",
        "iishasharrma@gmail.com"
    ]
    password = os.getenv("EMAIL_PASSWORD")
    subject = "🚨 Accident Detected!"
    body = "An accident has been detected. Please check the monitoring system immediately."
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_emails)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_emails, message.as_string())
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1280, 720))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    confidence = float(np.max(preds))
    class_idx = int(np.argmax(preds))
    class_label = class_names[class_idx]
    if class_label == "accident" and confidence > 0.6:
        notify_authorities()
    return JSONResponse({
        "class": class_label,
        "confidence": round(confidence, 3)
    })
