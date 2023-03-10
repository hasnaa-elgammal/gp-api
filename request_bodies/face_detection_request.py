from pydantic import BaseModel

class FaceDetectionRequest(BaseModel):
    user_id: int
    lang: str
    img: str