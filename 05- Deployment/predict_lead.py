import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
async def predict(request: Request):
    client = await request.json()
    probability = pipeline.predict_proba([client])[0, 1]
    converted = probability >= 0.5

    return JSONResponse({
        "converted_probability": float(probability),
        "converted": bool(converted)
    })


