import tensorflow as tf
import numpy as np
import io, os, base64
from PIL import Image
from colorthief import ColorThief
import whisper
from deep_translator import GoogleTranslator

def decode_img(string):
    decoded = Image.open(io.BytesIO(base64.b64decode(string)))
    return decoded

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
   
def add_familiar_face(img):
    pass

def check_face(img):
    pass