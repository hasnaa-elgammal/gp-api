from pydantic import BaseModel

class FaceDetectionRequest(BaseModel):
    user_id: str
    lang: str
    img: str