from pathlib import Path
import joblib
from sqlalchemy.orm import Session
from app.models.predictspam_model import SmsPrediction

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "app" / "model_ai"

MODEL_PATH = MODEL_DIR / "spam_model.joblib"

model = joblib.load(MODEL_PATH)

def predict_sms(message_text: str) -> str:
    prediction = model.predict([message_text])[0]
    return str(prediction)

def create_prediction(
    db: Session,
    message_text: str
) -> SmsPrediction:
    prediction_label = predict_sms(message_text)

    record = SmsPrediction(
        message_text=message_text,
        prediction_label=prediction_label
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_predictions(db: Session):
    return (
        db.query(SmsPrediction)
        .order_by(SmsPrediction.created_at.desc())
        .all()
    )

def get_prediction(
    db: Session,
    prediction_id: int
):
    return (
        db.query(SmsPrediction)
        .filter(SmsPrediction.id == prediction_id)
        .first()
    )

def update_prediction(
    db: Session,
    prediction_id: int,
    message_text: str
):
    record = get_prediction(db, prediction_id)

    if record is None:
        return None

    record.message_text = message_text
    record.prediction_label = predict_sms(message_text)

    db.commit()
    db.refresh(record)
    return record

def delete_prediction(
    db: Session,
    prediction_id: int
):
    record = get_prediction(db, prediction_id)

    if record is None:
        return False

    db.delete(record)
    db.commit()
    return True
