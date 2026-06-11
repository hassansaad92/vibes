from fastapi import APIRouter
from pydantic import BaseModel

from core.model import predict
from db.database import save_prediction

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    machine_id: str
    filename: str
    features: dict[str, float]


@router.post("/predict")
async def predict_endpoint(req: PredictRequest):
    """
    Production endpoint — called by the Pico board (or any client) with a
    pre-extracted feature dict.  The server runs inference and persists the
    result to Supabase.
    """
    result = predict(req.features)

    record = save_prediction(
        machine_id=req.machine_id,
        filename=req.filename,
        features=req.features,
        predicted_label=result["prediction"],
        probability_faulty=result["probability_faulty"],
        probability_healthy=result["probability_healthy"],
    )

    return {
        "filename": req.filename,
        "prediction": result["prediction"],
        "probability_faulty": result["probability_faulty"],
        "probability_healthy": result["probability_healthy"],
        "record_id": record.get("id"),
    }
