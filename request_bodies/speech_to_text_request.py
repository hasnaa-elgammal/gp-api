from pydantic import BaseModel

class SpeechToTextRequest(BaseModel):
    lang: str
    sound: str