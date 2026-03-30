from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from model_utils import predict_toxicity

app = FastAPI(title="Hepatotoxicity Predictor", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    smiles: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    try:
        if smiles:
            result = predict_toxicity(smiles=smiles)
        elif file:
            content = await file.read()
            result = predict_toxicity(file_bytes=content, filename=file.filename)
        else:
            raise HTTPException(status_code=400, detail="Provide either SMILES or a MOL/SDF file.")

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")