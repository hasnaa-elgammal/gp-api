from pydantic import BaseModel

class GeneralRequest(BaseModel):
    lang: str
    img: str