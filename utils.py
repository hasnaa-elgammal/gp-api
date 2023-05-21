import tensorflow as tf
import numpy as np
import io, os, base64
from PIL import Image
from tensorflow import keras
from keras.models import load_model, model_from_json
import cv2
import torch
from keras.preprocessing.image import img_to_array
import imutils
from colorthief import ColorThief
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
from transformers import ViltProcessor
from transformers import ViltForQuestionAnswering
import easyocr
import face_recognition
import requests

def encode(path):
    with open(path, "rb") as file:
        result= base64.b64encode(file.read())
    return result

def decode_img(string):
    if string[0:2] == "b'":
        string = string[2:]
    if string[-1] == "'":
        string = string[:-1]
    decoded = Image.open(io.BytesIO(base64.b64decode(string)))
    return decoded

def predict_currency(img):
    y = ['100', '100', '10', '10', '200', '200', '20', '20', '50', '50', '5', '5']
    # import model
    model = tf.keras.models.load_model("models/money.h5", compile=False)
    img = img.convert("RGB")
    img = np.array(img)
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    pred_probab = model.predict(img)[0]
    pred = ""
    if max(pred_probab) < 0.45:
        pred = "من فضلك أعد تصوير العملة"
    else:
        text = y[list(pred_probab).index(max(pred_probab))]
        if (int(text) < 20):
            pred = text + " جُنَيهات "
        else:
            pred = text + " جُنَيهًا "
    return pred

def predict_color(img):
    try:
        model = tf.keras.models.load_model("models/colors.h5")
        img = img.convert("RGB")
        img = img.save("temp_files/color_img.jpg")
        color_thief = ColorThief("temp_files/color_img.jpg")
        dominant_color = color_thief.get_palette(color_count=5)
        os.remove("temp_files/color_img.jpg")
        categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
        predictions = model.predict(np.array(dominant_color))
        predictions_set = set([categories[x.argmax(axis=0)] for x in predictions])
        if len(predictions_set)>1:
            return "mixed"
        else:
            return next(iter(predictions_set))
    except:
        return "Error. Please try again."
    
def speech_to_text(speech, lang):
    try:
        if speech[0:2] == "b'":
            speech = speech[2:]
        if speech[-1] == "'":
            speech = speech[:-1]
        model = whisper.load_model("small")
        wav_file = open("temp_files/speech_temp.wav", "wb")
        decode_string = base64.b64decode(speech)
        wav_file.write(decode_string)
        result = model.transcribe("temp_files/speech_temp.wav")
        result = {
            'result': result['text'],
            'lang': result['language']
        }
        os.remove("temp_files/speech_temp.wav")
        return result
    except:
        result = {
            'result': "Error. Please try again.",
            'lang': lang
        }
        
def translate(src, target, text):
    return GoogleTranslator(source=src, target=target).translate(text)

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgusted",
                    "Scared", "Happy",
                    "Neutral", "Sad",
                    "Surprised"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

def emotion_finder(img):
    detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    emotion_model_path = FacialExpressionModel("models/model.json", 'models/emotion.h5')
    face_detection = cv2.CascadeClassifier(detection_model_path)
    img = np.array(img.convert('RGB'), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = imutils.resize(img,width=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    imgClone = img.copy()
    if len(faces) > 0:
        for face in faces: 
            (fX, fY, fW, fH) = face
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            preds = emotion_model_path.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(canvas, "ss", (fX, fY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(canvas,(fX,fY),(fX+fW,fY+fH),(255,0,0),2)

            cv2.putText(imgClone, "ss", (fX, fY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(imgClone,(fX,fY),(fX+fW,fY+fH),(255,0,0),2)
    else:
        preds = ''
    return imgClone,  preds

def text_to_speech(text,lang):
    tts = gTTS(text=text, lang=lang)
    filename = "temp_files/hello2.mp3"
    tts.save(filename)
    result = encode(filename)
    os.remove(filename)
    return result

def check_face(lang, img, face_data):
    image = np.array(img)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    match_found = False
    matched_name = None

    for face_id, face_details in face_data.items():
        known_face_url = face_details['image']
        known_face_name = face_details['name']
        response = requests.get(known_face_url)
        known_face_image = np.array(Image.open(io.BytesIO(response.content)))
        known_face_rgb = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2RGB)
        known_face_encoding = face_recognition.face_encodings(known_face_rgb)[0]
        matches = face_recognition.compare_faces([known_face_encoding], face_encodings[0])
        if matches[0]:
            match_found = True
            matched_name = known_face_name
            break

    if match_found:
        result = f"{matched_name}"
    else:
        result = "No match found: Unknown"
        if lang != 'en':
            result = translate('en', lang, result)

    return result


    
def VQA_Predict(image, text:str):
    img = np.array(image)
    Quest = text
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # prepare inputs
    encoding = processor(img, Quest, return_tensors="pt")
    # load the ViLT model
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = torch.sigmoid(logits).argmax(-1).item()
    resultOfPredect = model.config.id2label[idx]
    return resultOfPredect  
        
#Scanning funcations
def recognize_text(img_path, lang):
    reader = easyocr.Reader([lang])
    return reader.readtext(img_path)

def scanning_predict(img, lang):
    # try:
    img = img.convert("RGB")
    img = img.save("temp_files/scan_img.jpg")
    sentence = ''
    result = recognize_text("temp_files/scan_img.jpg", lang)
    for (bbox, text, prob) in result:
        if prob >= 0.1:
            sentence += f'{text}'
    os.remove("temp_files/scan_img.jpg")
    return sentence
    # except:
    #     return "Error. Please try again."
