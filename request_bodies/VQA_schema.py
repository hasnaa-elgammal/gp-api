from pydantic import BaseModel

class VQA_Request(BaseModel):
    Question : str
    img : str
