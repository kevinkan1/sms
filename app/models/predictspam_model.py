from sqlalchemy import Column, Integer, Text, String, DateTime
from sqlalchemy.sql import func
from config import Base

class SmsPrediction(Base):
    __tablename__ = "sms_predictions"

    id = Column(Integer, primary_key=True, index=True)
    message_text = Column(Text, nullable=False)
    prediction_label = Column(String(10), nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp()
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
