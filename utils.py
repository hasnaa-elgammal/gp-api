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


def decode_img(string):
    decoded = Image.open(io.BytesIO(base64.b64decode(string)))
    return decoded

def Curr_Pred(img):
    y = ['100', '100', '10', '10', '200', '200', '20', '20', '50', '50', '5', '5']
    # import model
    model = tf.keras.models.load_model("models/money.h5", compile=False)
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
    
def predict_speech_to_text(speech):
    try:
        model = whisper.load_model("small")
        wav_file = open("temp_files/speech_temp.wav", "wb")
        decode_string = base64.b64decode(speech)
        wav_file.write(decode_string)
        result = model.transcribe("speech_temp.wav")
        result = {
            'text': result['text'],
            'lang': result['language']
        }
        os.remove("temp_files/speech_temp.wav")
        return result
    except:
        result = {
            'text': "Error. Please try again."
        }
        
def translate(src, target, text):
    return GoogleTranslator(source=src, target=target).translate(text)

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    EMOTIONS_ar = ["غاضب" ,"مشمئز","خائف", "سعيد", "حزين", "متفاجيء","طبيعي"]
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
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)],FacialExpressionModel.EMOTIONS_ar[np.argmax(self.preds)]   

def emotion_finder(img):
    detection_model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    emotion_model_path = FacialExpressionModel("models/model.json", 'models/emotion.h5')
    face_detection = cv2.CascadeClassifier(detection_model_path)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
   # emotion_classifier = load_model(emotion_model_path, compile=False)
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

    return imgClone,  preds

def predict_text_to_speech(text,lang):
    tts = gTTS(text=text, lang=lang)
    filename = "hello2.mp3"
    tts.save(filename)
    os.system(f"start {filename}")

def add_familiar_face(img):
    pass

def check_face(img):
    pass