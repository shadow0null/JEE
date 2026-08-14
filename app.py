"""StudyDesk Local JEE Engine HTTP API for Render.

The deterministic engine remains completely separate from Gemini and the
existing StudyDesk reference providers. This service exposes the local JEE
capabilities over HTTPS so an InfinityFree PHP frontend can call it.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import jee_main
from jee_engine import pdf_engine

MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024
API_KEY = os.getenv("JEE_API_KEY", "").strip()

app = FastAPI(
    title="StudyDesk Local JEE Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# InfinityFree serves the browser-facing PHP site. The API itself does not
# expose credentials, and the optional API key provides an additional server-
# to-server guard when configured in Render and the PHP application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"success": False, "error": "Request body is too large."},
                )
        except ValueError:
            pass
    return await call_next(request)


def _check_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert locally generated graph files into inline base64 data."""
    if result.get("success") and result.get("type") == "GRAPH":
        path = result.pop("file_path", None)
        if path:
            try:
                with open(path, "rb") as fh:
                    result["image_base64"] = base64.b64encode(fh.read()).decode("ascii")
            finally:
                try:
                    Path(path).unlink()
                except OSError:
                    pass
    return result


@app.get("/")
def root():
    return {"service": "StudyDesk Local JEE Engine", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "local-jee-engine"}


@app.post("/api/jee/question")
def question(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    text = payload.get("question")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if len(text) > 300:
        raise HTTPException(status_code=400, detail="Question is too long.")
    try:
        result = jee_main.process_text(text.strip())
        return _json_safe_result(result)
    except Exception:
        return {"success": False, "error": "Local calculation unavailable."}


@app.post("/api/jee/json")
def structured(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    try:
        return _json_safe_result(jee_main.process_json(payload))
    except Exception:
        return {"success": False, "error": "Local calculation unavailable."}


@app.post("/api/jee/pdf")
def pdf(payload: dict[str, Any], x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    data = payload.get("data_base64")
    if not isinstance(data, str) or not data:
        raise HTTPException(status_code=400, detail="PDF data is required.")
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF payload.")
    try:
        return pdf_engine.extract_questions(raw)
    except Exception:
        return {"success": False, "error": "PDF processing unavailable."}
