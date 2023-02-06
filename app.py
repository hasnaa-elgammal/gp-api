from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {'msg': 'hello'}

@app.post("/money")
async def money():
    return {'msg': 'hello'}

@app.post("/ocr")
async def ocr():
    return {'msg': 'hello'}

@app.post("/addface")
async def add_face():
    return {'msg': 'hello'}

@app.post("/checkface")
async def check_face():
    return {'msg': 'hello'}

@app.post("/emotions")
async def emotions():
    return {'msg': 'hello'}

@app.post("/imagecaption")
async def image_caption():
    return {'msg': 'hello'}

@app.post("/color")
async def color():
    return {'msg': 'hello'}

@app.post("/questions")
async def questions():
    return {'msg': 'hello'}