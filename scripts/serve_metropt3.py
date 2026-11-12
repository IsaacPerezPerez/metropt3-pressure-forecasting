"""HTTP facade for a locally exported MLflow model; no tracking server needed."""
import os
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


@asynccontextmanager
async def lifespan(app):
    app.state.model = mlflow.pyfunc.load_model(os.environ.get('MODEL_PATH', '/model'))
    yield


app = FastAPI(lifespan=lifespan)


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    readings: list[dict]


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(request: Request):
    try:
        return app.state.model.predict(pd.DataFrame({'readings': [request.readings]}))[0]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
