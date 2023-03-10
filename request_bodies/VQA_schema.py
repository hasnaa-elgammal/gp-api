from pydantic import BaseModel

class VQA_Request(BaseModel):
    lang: str
    question : str
    img : str
