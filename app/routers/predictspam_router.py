from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from config import get_db
from app.schemas.predictspam_schema import (
    SmsPredictionCreate,
    SmsPredictionResponse
)
from app.services import predictspam_service as service

router = APIRouter(tags=["SMS Spam Prediction"])

@router.post(
    "/predict",
    response_model=SmsPredictionResponse
)
def predict_sms(
    request: SmsPredictionCreate,
    db: Session = Depends(get_db)
):
    message = request.message_text.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Teks SMS tidak boleh kosong."
        )
    return service.create_prediction(db, message)

@router.get(
    "/history",
    response_model=list[SmsPredictionResponse]
)
def get_history(
    db: Session = Depends(get_db)
):
    return service.get_predictions(db)

@router.get(
    "/history/{prediction_id}",
    response_model=SmsPredictionResponse
)
def get_history_detail(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    record = service.get_prediction(db, prediction_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Data prediksi tidak ditemukan."
        )
    return record


@router.put(
    "/update/{prediction_id}",
    response_model=SmsPredictionResponse
)
def update_sms(
    prediction_id: int,
    request: SmsPredictionCreate,
    db: Session = Depends(get_db)
):
    message = request.message_text.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Teks SMS tidak boleh kosong."
        )

    record = service.update_prediction(
        db,
        prediction_id,
        message
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Data prediksi tidak ditemukan."
        )
    return record

@router.delete("/delete/{prediction_id}")
def delete_sms(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    deleted = service.delete_prediction(
        db,
        prediction_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Data prediksi tidak ditemukan."
        )

    return {
        "status": "success",
        "message": "Data prediksi berhasil dihapus."
    }
