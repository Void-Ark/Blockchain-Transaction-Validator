# MODEL_INPUT===============================================================================================
from re import template
from typing import Annotated
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import Input 

# Predictor=================================================================================================
from predictor import data_predictor
    
# API=======================================================================================================
from fastapi import FastAPI, Form, Request
app = FastAPI()  

@app.get('/')
async def hello(): 
    return "API working"    



@app.post('/predict') 
async def predict_function(request: Request) : 
    data = await request.form()
    data = data._dict.values()
    val = int(data_predictor.predict(data)[0])
    result = "shi h" if val == 0 else "glt h"
    result = {'result': result}           # type: ignore
    return result

templates = Jinja2Templates(directory="template")
@app.get("/predict", response_class=HTMLResponse)
def form_post(request: Request):
    return templates.TemplateResponse('web.html', {'request': request}) 



