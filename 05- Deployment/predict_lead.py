import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
async def predict(request: Request):
    client = await request.json()
    prob = pipeline.predict_proba([client])[0, 1]
    converted = prob >= 0.5
    return JSONResponse({"converted_probability": float(prob), "converted": bool(converted)})

#uvicorn predict_lead:app --host 0.0.0.0 --port 9697
#command line to run web service using FastAPI uvicorn 
#FastAPI is the web framework that you use to build the API or web service itself.
#Uvicorn serves as the interface between the web and your application code.



