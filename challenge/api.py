import fastapi
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from .model import DelayModel
from sklearn.model_selection import train_test_split
from fastapi import HTTPException

"""
    preprocess data and fit model to serving
"""
app = fastapi.FastAPI()
DATA_PATH = Path(__file__).resolve().parents[1]/"data"/"data.csv"

data  = pd.read_csv(DATA_PATH, low_memory=False)
model = DelayModel()
features, target = model.preprocess(
                        data          = data,
                        target_column = "delay",
                    )
x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.33, random_state=42)

model.fit(
    features = x_train, 
    target   = y_train
)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return