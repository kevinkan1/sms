from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

class SmsPredictionCreate(BaseModel):
    message_text: str = Field(
        min_length=1,
        max_length=10000,
        description="Teks SMS yang akan dianalisis"
    )

class SmsPredictionResponse(BaseModel):
    id: int
    message_text: str
    prediction_label: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
