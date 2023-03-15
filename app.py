from fastapi import FastAPI
from request_bodies.general_request import GeneralRequest
from request_bodies.speech_to_text_request import SpeechToTextRequest
from request_bodies.VQA_schema import VQA_Request
from utils import *
from models.Image_caption import *
import json

app = FastAPI()

@app.get("/")
async def root():
    return {'result': 'hello'}

@app.post("/speechtotext")
async def speech_to_text(req: SpeechToTextRequest):
    if req.sound != '':
        result = speech_to_text(req.sound, req.lang)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
        result = {
            "lang": req.lang,
            "result": result
        }
    return json.dumps(result)

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
    result = text_to_speech(result, req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)

@app.post("/checkface")
async def check_face():
    return {'msg': 'hello'}

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
    result = text_to_speech(result, req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)

@app.post("/color")
async def color(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = predict_color(decoded_img)
    else:
        result = "Error. Please try again."
    if req.lang != 'en':
        result = translate('en', req.lang, result)
    result = text_to_speech(result,req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)
    


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
    result = text_to_speech(result, req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)
      
    
    
@app.post("/imagecaption")
async def image_caption(req:GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = imageCaption_predict(decoded_img)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    result = text_to_speech(result,req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)
        
    
    
@app.post("/scan")
async def scanning(req:GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = scanning_predict(decoded_img, req.lang)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
    result = text_to_speech(result,req.lang)
    output = {
        "lang": req.lang,
        "result": str(result)
    }
    return json.dumps(output)
    
