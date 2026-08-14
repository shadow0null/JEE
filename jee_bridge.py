#!/usr/bin/env python3
"""StudyDesk production bridge for the deterministic Local JEE Engine.

Reads one JSON object from stdin:
    {"question": "..."}
Writes one JSON object to stdout.

This process never calls Gemini or any external service.
"""
from __future__ import annotations

import json
import sys
import re
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jee_main
from jee_engine import pdf_engine


_SUPERSCRIPTS = str.maketrans({
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
})


def normalize_question(text: str) -> str:
    """Normalize common student Unicode notation into the engine's safe syntax."""
    text = str(text).translate(_SUPERSCRIPTS)
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    text = text.replace("π", "pi")
    return text


def main() -> int:
    try:
        raw = sys.stdin.read(64 * 1024)
        payload = json.loads(raw)
        action = str(payload.get("action", "question")).lower()
        if action == "pdf":
            data = payload.get("data_base64")
            if not isinstance(data, str) or not data:
                print(json.dumps({"success": False, "error": "PDF data is required."}))
                return 0
            try:
                pdf_bytes = base64.b64decode(data, validate=True)
            except Exception:
                print(json.dumps({"success": False, "error": "Invalid PDF payload."}))
                return 0
            result = pdf_engine.extract_questions(pdf_bytes)
            print(json.dumps(result, default=str, separators=(",", ":")))
            return 0

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            print(json.dumps({"success": False, "error": "Question is required."}))
            return 0

        question = normalize_question(question.strip())
        result = jee_main.process_text(question)

        # Graphs are generated locally. Never expose a server filesystem path.
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

        print(json.dumps(result, default=str, separators=(",", ":")))
        return 0
    except Exception:
        # Never leak tracebacks or host details through the production bridge.
        print(json.dumps({"success": False, "error": "Local calculation unavailable."}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
