from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .ranker import compute_similarity
from .utils import extract_text_from_upload

app = FastAPI(title="Resume Ranker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/rank")
async def rank_resume(
    jd_text: Optional[str] = Form(None),
    resume_text: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None),
    resume_file: Optional[UploadFile] = File(None),
):
    """
    Accepts job description and resume as either raw text or file upload.
    At least one of (jd_text, jd_file) and one of (resume_text, resume_file)
    must be provided.
    """
    # resolve job description
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_content = extract_text_from_upload(jd_bytes, jd_file.filename)
        if not jd_content.strip():
            raise HTTPException(400, "Could not extract text from the uploaded JD file.")
    elif jd_text and jd_text.strip():
        jd_content = jd_text.strip()
    else:
        raise HTTPException(400, "Provide a job description (text or file).")

    # resolve resume
    if resume_file and resume_file.filename:
        resume_bytes = await resume_file.read()
        resume_content = extract_text_from_upload(resume_bytes, resume_file.filename)
        if not resume_content.strip():
            raise HTTPException(400, "Could not extract text from the uploaded resume file.")
    elif resume_text and resume_text.strip():
        resume_content = resume_text.strip()
    else:
        raise HTTPException(400, "Provide a resume (text or file).")

    score = compute_similarity(jd_content, resume_content)

    return {
        "score": score,
        "percentage": round(score * 100, 2),
        "jd_length": len(jd_content),
        "resume_length": len(resume_content),
    }
