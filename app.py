from fastapi import FastAPI
from request_bodies.general_request import GeneralRequest
from request_bodies.speech_to_text_request import SpeechToTextRequest
from utils import *

app = FastAPI()

@app.get("/")
async def root():
    return {'msg': 'hello'}

@app.post("/speechtotext")
async def speech_to_text(req: SpeechToTextRequest):
    if req.sound != '':
        result = predict_speech_to_text(req.sound)
        return result
    else:
        return {
                'text': 'Error. Please try again',
                }

@app.post("/money")
async def money():
    return {'msg': 'hello'}

@app.post("/ocr")
async def ocr():
    return {'msg': 'hello'}

@app.post("/addface")
async def add_face():
    return {'msg': 'hello'}

@app.post("/checkface")
async def check_face():
    return {'msg': 'hello'}

@app.post("/emotions")
async def emotions():
    return {'msg': 'hello'}

@app.post("/imagecaption")
async def image_caption():
    return {'msg': 'hello'}

@app.post("/color")
async def color(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = predict_color(decoded_img)
        if(req.lang != 'en'):
            result = translate('en', req.lang, result)
        return result
    else:
        return "Error. Please Try Again"

@app.post("/questions")
async def questions():
    return {'msg': 'hello'}