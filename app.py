from fastapi import FastAPI
from request_bodies.general_request import GeneralRequest
from request_bodies.speech_to_text_request import SpeechToTextRequest
from request_bodies.VQA_schema import VQA_Request
from utils import *
import base64


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
        result = "Error. Please try again."
        return result

@app.post("/money")
async def money(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = Curr_Pred(decoded_img)
        return predict_text_to_speech(result,"ar")
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
        return predict_text_to_speech(result, req.lang)

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
async def emotions(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        new_img, emotion = emotion_finder(decoded_img)
        if (emotion == []):
            emotion = "عفوا لا نستطيع اكتشاف وجوه، حاول مرة أخرى "
        else:
            emotion = " و ".join(emotion)
            emotion = "يبدو " + emotion
        return predict_text_to_speech(emotion,"ar")    
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
        return predict_text_to_speech(result, req.lang)


    
@app.post("/color")
async def color(req: GeneralRequest):
    if req.img != '':
        decoded_img = decode_img(req.img)
        result = predict_color(decoded_img)
        if(req.lang != 'en'):
            result = translate('en', req.lang, result)
        return predict_text_to_speech(result,req.lang)
    else:
        result = "Error. Please try again."
        if req.lang != 'en':
            result = translate('en', req.lang, result)
        return predict_text_to_speech(result, req.lang)

    
    
    
@app.post("/VQA")
async def VQA(req:VQA_Request):
    if req.img != '' or req.Question != '':
        #decoded_img = decode_img(req.img)
        answerOfQuestion = VQA_Predict(req.img, req.Question)
        return {'msg': answerOfQuestion}
    else:
        return {'msg': "عفوًا لا نستطيع اجابة سؤالك , حاول مرة أخرى"}
    
    
    
    
    
@app.post("/imagecaption")
async def image_caption(req:GeneralRequest):
    if req.img != '':
        #decoded_img = decode_img(req.img)
        resultOfImage = imageCaption_predict(req.img)
        return {'msg': resultOfImage}
    else:
        return {'msg': "أعد إدخال الصورة من فضلك"}
    
    
    
    
@app.post("/scan")
async def scanning(req:GeneralRequest):
    if req.img != '':
        #decoded_img = decode_img(req.img)
        answerOfscaning = scanning_predict(req.img, req.lang)
        return {'msg': answerOfscaning}
    else:
        return {'msg': "أعد إدخال الصورة من فضلك"}
