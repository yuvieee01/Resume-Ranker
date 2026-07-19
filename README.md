<div align="center">

# Resume Ranker

**Score how well a resume matches a job description — powered by NLP.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What It Does

Resume Ranker takes a **job description** and a **resume** (as text or PDF), preprocesses both with NLP, and returns a **similarity score** between 0 – 100%.

Under the hood it uses **TF-IDF vectorization** with bigrams and **cosine similarity** to quantify the match — no LLM calls, no API keys, everything runs locally.

---

## Architecture

```
Resume Ranker/
├── backend/                 # Python API
│   ├── app/
│   │   ├── main.py          # FastAPI – POST /api/rank
│   │   ├── ranker.py        # TF-IDF + cosine similarity logic
│   │   └── utils.py         # PDF / TXT text extraction
│   └── requirements.txt
│
├── frontend/                # Next.js web UI
│   ├── src/app/
│   │   ├── page.js          # Main page with form + results
│   │   ├── layout.js        # Root layout + SEO metadata
│   │   ├── globals.css      # Design system (light/dark tokens)
│   │   └── components/
│   │       ├── InputSection  # Text / PDF toggle input
│   │       ├── ResultCard    # Animated gauge + score
│   │       └── ThemeToggle   # Light ↔ Dark mode switch
│   └── package.json
│
└── README.md
```

---

## How It Works

| Step | What Happens |
|------|-------------|
| **1. Input** | User provides a JD and resume via text or PDF upload |
| **2. Extract** | `pdfminer` converts PDF to plaintext |
| **3. Preprocess** | `spaCy` tokenizes, lemmatizes, removes stop words |
| **4. Vectorize** | `TfidfVectorizer` builds uni+bigram TF-IDF matrix |
| **5. Score** | Cosine similarity between JD and resume vectors |
| **6. Display** | Frontend shows score gauge + match grade |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### Backend

```bash
cd backend

# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# download spaCy model
python -m spacy download en_core_web_sm

# start the server
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Test it:

```bash
curl http://localhost:8000/api/health
# → {"status":"ok"}
```

### Frontend

```bash
cd frontend

# install dependencies
npm install

# start dev server
npm run dev
```

Open `http://localhost:3000` in your browser.

> **Note:** The frontend expects the backend at `http://localhost:8000` by default.  
> Set `NEXT_PUBLIC_API_URL` in a `.env.local` file to override.

---

## API Reference

### `POST /api/rank`

Accepts `multipart/form-data` with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `jd_text` | string | Either this or `jd_file` | Job description as raw text |
| `jd_file` | file | Either this or `jd_text` | Job description as PDF/TXT |
| `resume_text` | string | Either this or `resume_file` | Resume as raw text |
| `resume_file` | file | Either this or `resume_text` | Resume as PDF/TXT |

**Response:**

```json
{
  "score": 0.4523,
  "percentage": 45.23,
  "jd_length": 847,
  "resume_length": 1293
}
```

### `GET /api/health`

Returns `{"status": "ok"}` — useful for uptime checks.

---

## Deploying

### Frontend → Vercel

1. Push this repo to GitHub
2. Import the repo in [Vercel](https://vercel.com)
3. Set the **Root Directory** to `frontend`
4. Set environment variable `NEXT_PUBLIC_API_URL` to your backend URL
5. Deploy

### Backend → Any Cloud

The backend is a standard FastAPI app. Deploy it wherever you run Python:

- **Render** / **Railway** — point to `backend/`, set start command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Docker** — write a simple Dockerfile with the `requirements.txt` and `uvicorn` entrypoint
- **AWS / GCP** — use App Engine, Cloud Run, or EC2

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP | spaCy, scikit-learn (TF-IDF + Cosine Similarity) |
| Backend | FastAPI, Uvicorn, pdfminer.six |
| Frontend | Next.js 16, CSS Modules, Inter font |
| Deployment | Vercel (frontend), Render (backend) |

---

## License

MIT — use it however you want.
