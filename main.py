from fastapi import FastAPI
from request_bodies.general_request import GeneralRequest
from request_bodies.face_detection_request import FaceDetectionRequest
from request_bodies.speech_to_text_request import SpeechToTextRequest
from request_bodies.VQA_schema import VQA_Request
from utils import *
from models.Image_caption import *
import json
import requests

app = FastAPI()

@app.get("/")
async def root():
    return {'result': 'hello'}

@app.post("/speechtotext")
async def speechtotext(req: SpeechToTextRequest):
    if req.sound != '':
        result = speech_to_text(req.sound, req.lang)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    return str(result)

@app.post("/money")
async def money(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = predict_currency(decoded_img)
        if req.lang != 'ar':
            result = translate('ar', req.lang, result)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    return str(result)

@app.post("/checkface")
async def checkface(req: FaceDetectionRequest):
    if req.img != '' and req.user_id != '':
        database_url = "https://mopser-fc1b9-default-rtdb.firebaseio.com/"
        face_data_path = "/faces/" + str(req.user_id)
        response = requests.get(database_url + face_data_path + ".json")
        face_data = response.json()
        decoded_img = decode_img(req.img)
        result = check_face(req.lang, decoded_img, face_data)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    return str(result)

@app.post("/emotions")
async def emotions(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        new_img, emotion = emotion_finder(decoded_img)
        if (emotion == ''):
            result = "We can't find any face. Please try again "
        else:
            result = "Looks " + emotion   
    else:
        result = "Error. Please try again."
    if req.lang != 'en':
        result = translate('en', req.lang, result)
    return str(result)

@app.post("/color")
async def color(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = predict_color(decoded_img)
    else:
        result = "Error. Please try again."
    if req.lang != 'en':
        result = translate('en', req.lang, result)
    return str(result)
    
@app.post("/vqa")
async def VQA(req:VQA_Request):
    if req.img != '' or req.question != '':
        if req.lang != "en":
            req.question = translate(req.lang, "en", req.question)
        decoded_img = decode_img(req.img)
        result = VQA_Predict(decoded_img, req.question)
    else:
        result = "Error. Please try again."
    if req.lang != 'en':
        result = translate('en', req.lang, result)
    return str(result)
       
@app.post("/imagecaption")
async def image_caption(req:GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = imageCaption_predict(decoded_img)
    else:
        result = "Error. Please try again."
    if req.lang != 'en':
        result = translate('en', req.lang, result)
    return str(result)
            
@app.post("/scan")
async def scanning(req:GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = detect(decoded_img)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    return str(result)  