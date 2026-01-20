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

"""
    Define the schemas feature and boy
"""
class Feature(BaseModel):
    OPERA     : str
    TIPOVUELO : Optional[str]
    MES       : Optional[int]

class Body(BaseModel):
    flights : List[Feature]


"""
    Define the endpoints
"""
@app.get("/")
def root():
    return {"message": "API running. Try /docs or /health"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(body: Body) -> dict:
    try:
        data_entry = pd.DataFrame(0, index=np.arange(len(body.flights)), columns=model.top_10_features)
        for i, flight in enumerate(body.flights):
            # OPERA
            if flight.OPERA not in model.airlines:
                raise HTTPException(
                    status_code=400,
                    detail=f"La propiedad OPERA en el indice {i} debe ser una de las aerolineas usadas en el modelo",
                )
            else:
                if flight.OPERA in model.top_10_features:
                    data_entry.loc[i,'OPERA_' + flight.OPERA] = 1
            # TIPOVUELO
            if flight.TIPOVUELO is not None:
                if flight.TIPOVUELO not in ["N", "I"]:
                    raise HTTPException(
                        status_code=400,
                        detail="La propiedad TIPOVUELO debe ser 'N' o 'I'",
                    )
                data_entry.loc[i,'TIPOVUELO_I'] = int(flight.TIPOVUELO == 'I')
            # MES
            if flight.MES in range(1, 13):
                month = 'MES_' + str(flight.MES)
                if month in model.top_10_features:
                    data_entry.loc[i,month] = 1
            else:
                raise HTTPException(
                    status_code=400,
                    detail="La propiedad MES debe estar entre 1 y 12",
                )
        pred = model.predict(data_entry)
        # JSON serializable 
        pred_list = [int(x) for x in pred]
        response = {"predict": pred_list}
        return response
    except HTTPException:
        raise
    except Exception as exception:
        # Errores inesperados 500
        raise HTTPException(status_code=500, detail=str(exception))
