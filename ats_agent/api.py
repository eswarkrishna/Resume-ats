"""Small web API: uvicorn ats_agent.api:app --reload, then POST a resume to /score."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .cli import run

app = FastAPI(title="ATS Resume Scoring Agent")


@app.post("/score")
async def score(
    resume: UploadFile = File(...),
    job_description: str = Form(""),
    ai: bool = Form(False),
):
    suffix = Path(resume.filename or "resume.txt").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "Upload a .pdf, .docx, or .txt resume.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await resume.read())
        path = tmp.name
    try:
        return run(path, job_description or None, ai).to_dict()
    finally:
        Path(path).unlink(missing_ok=True)
