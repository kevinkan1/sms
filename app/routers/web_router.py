from collections import Counter
import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from config import get_db
from app.services import predictspam_service as service

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def dashboard_data(db: Session):
    records = service.get_predictions(db)
    total = len(records)
    spam_count = sum(1 for row in records if row.prediction_label == "spam")
    ham_count = sum(1 for row in records if row.prediction_label == "ham")

    spam_words = Counter()
    stop_words = {
        "the", "to", "a", "and", "is", "in", "you", "for", "your", "of",
        "on", "now", "it", "i", "me", "my", "we", "are", "be", "this",
        "that", "with", "have", "can", "call", "from", "at", "or", "u",
        "ur", "will", "if", "do", "so", "not", "but", "our", "get"
    }
    for row in records:
        if row.prediction_label == "spam":
            words = re.findall(r"[a-zA-Z]{3,}", row.message_text.lower())
            spam_words.update(word for word in words if word not in stop_words)

    return {
        "total": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_rate": round((spam_count / total) * 100, 1) if total else 0,
        "top_spam_words": spam_words.most_common(10),
        "recent_records": records[:5],
    }

@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    data = dashboard_data(db)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"active_page": "dashboard", **data},
    )

@router.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="analyze.html",
        context={"active_page": "analyze", "result": None, "message_text": ""},
    )

@router.post("/analyze", response_class=HTMLResponse)
def analyze_submit(
    request: Request,
    message_text: str = Form(...),
    db: Session = Depends(get_db),
):
    message_text = message_text.strip()
    if not message_text:
        return templates.TemplateResponse(
            request=request,
            name="analyze.html",
            context={
                "active_page": "analyze",
                "result": None,
                "message_text": "",
                "error": "Silakan masukkan teks SMS terlebih dahulu.",
            },
            status_code=422,
        )

    record = service.create_prediction(db, message_text)
    return templates.TemplateResponse(
        request=request,
        name="analyze.html",
        context={
            "active_page": "analyze",
            "result": record,
            "message_text": message_text,
        },
    )

@router.get("/history-page", response_class=HTMLResponse)
def history_page(
    request: Request,
    q: str = "",
    db: Session = Depends(get_db),
):
    records = service.get_predictions(db)
    query = q.strip().lower()
    if query:
        records = [
            row for row in records
            if query in row.message_text.lower()
            or query in row.prediction_label.lower()
            or query == str(row.id)
        ]
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "active_page": "history",
            "records": records,
            "q": q,
        },
    )

@router.get("/edit/{prediction_id}", response_class=HTMLResponse)
def edit_page(
    prediction_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    record = service.get_prediction(db, prediction_id)
    if record is None:
        return RedirectResponse(url="/history-page?notice=notfound", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="edit.html",
        context={"active_page": "history", "record": record},
    )

@router.post("/edit/{prediction_id}", response_class=HTMLResponse)
def edit_submit(
    prediction_id: int,
    message_text: str = Form(...),
    db: Session = Depends(get_db),
):
    message_text = message_text.strip()
    if not message_text:
        return RedirectResponse(url=f"/edit/{prediction_id}?notice=empty", status_code=303)
    record = service.update_prediction(db, prediction_id, message_text)
    if record is None:
        return RedirectResponse(url="/history-page?notice=notfound", status_code=303)
    return RedirectResponse(url="/history-page?notice=updated", status_code=303)

@router.post("/remove/{prediction_id}")
def remove_record(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    service.delete_prediction(db, prediction_id)
    return RedirectResponse(url="/history-page?notice=deleted", status_code=303)
