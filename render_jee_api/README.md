# StudyDesk Local JEE Engine — Render API

This is the Python service that should be hosted on Render. It is intentionally
separate from the InfinityFree PHP website.

## Render settings

- **Service type:** Web Service
- **Root Directory:** leave blank
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment:** Python 3

Optional security setting:

- Add environment variable `JEE_API_KEY` with a long random secret. The PHP
  StudyDesk integration can then send the same value as `X-API-Key`.

## Endpoints

- `GET /health` — health check
- `POST /api/jee/question` — natural-language deterministic routing
- `POST /api/jee/json` — structured math/physics/unit/numerical/graph calls
- `POST /api/jee/pdf` — local PDF question extraction

The service does not call Gemini or the existing Wikipedia/arXiv/OpenAlex/
Crossref providers. Those remain part of the existing StudyDesk application.
