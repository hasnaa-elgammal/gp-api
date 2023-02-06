from pydantic import BaseModel

class SpeechToTextRequest(BaseModel):
    sound: str