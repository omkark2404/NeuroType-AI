# NeuroType AI — Cognitive Typing Intelligence Engine
> **v1.1.0 — A real-time cognitive behavior modeling system for human-computer interaction.**

---

##  What This Is

NeuroType AI is **not** a typing app.

It is a **system-level AI product** — a real-time inference engine that models human cognitive state from typing behavior, predicts fatigue and error probability, dynamically adapts difficulty, and **learns from every session** using online gradient descent.

**Resume keywords this project unlocks:**
- Cognitive Modeling
- Real-time Inference System
- Online Learning / Adaptive ML
- Behavior Prediction Engine
- Adaptive Systems
- Human-Computer Interaction AI

---

## System Architecture

```
Client (keystroke stream)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│                       FastAPI Layer                         │
│  POST /typing/keystroke    POST /typing/session             │
│  POST /ai/predict          POST /ai/adapt                   │
│  POST /ai/stream-predict   GET  /typing/analytics           │
│  POST /auth/register|login                                  │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │keystroke_service│  ← stores raw keystroke stream
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │feature_extractor│  ← avg_interval, variance, hold_time,
       └───────┬────────┘     error_rate, burst_speed
               │
       ┌───────▼────────┐
       │   ml_model      │  ← Cognitive Brain Model (sigmoid)
       │  + update_weights│     fatigue | error_prob | consistency
       └───────┬────────┘     ↑ learns via online gradient descent
               │
       ┌───────▼────────┐
       │adaptive_engine  │  ← directive + feedback + weak_patterns
       └───────┬────────┘
               │
    ┌──────────▼─────────┐     ┌───────────────┐
    │   SQLite / MemoryDB │     │   TTL Cache   │
    │  (session_stats,    │     │  (predictions │
    │   keystrokes, users)│     │   30s TTL)    │
    └─────────────────────┘     └───────────────┘
```

---

##  Cognitive Model Explanation

The **Cognitive Typing Brain Model** (`services/ml_model.py`) uses a sigmoid activation function to map raw behavioral features to normalized cognitive predictions.

### Features (extracted by `utils/feature_extractor.py`)

| Feature             | Description                                      | Unit   |
|---------------------|--------------------------------------------------|--------|
| `avg_interval`      | Mean time between consecutive keystrokes         | ms     |
| `variance_interval` | Variance of inter-keystroke intervals            | ms²    |
| `avg_hold_time`     | Mean duration a key is held down                 | ms     |
| `error_rate`        | Fraction of keystrokes flagged as errors         | 0–1    |
| `burst_speed`       | Peak typing speed in any 2-second sliding window | keys/s |

### Sigmoid Activation

```
sigmoid(x) = 1 / (1 + e^(-x))
```

Maps any real-valued linear combination to (0, 1) — ideal for probability-like outputs.

### Fatigue Model

```
fatigue_input = W_VAR  × (variance_interval / 10,000)
              + W_HOLD × (avg_hold_time / 300)

fatigue = sigmoid(fatigue_input)
```

- High **variance** → irregular rhythm → rising fatigue
- High **hold time** → sluggish key presses → rising fatigue

### Error Probability Model

```
error_input = W_ERR    × error_rate
            + W_FATIGUE × fatigue

error_prob = sigmoid(error_input)
```

Fatigued users make more errors — fatigue feeds directly into the error model.

### Consistency Score

```
consistency = max(0, 100 − (std_dev(intervals) / 100) × 10)
```

Zero variance = perfectly consistent = score of 100. Wider spread → lower score.

### Default Weight Configuration (`config.py`)

| Weight                    | Default | Controls                              |
|---------------------------|---------|---------------------------------------|
| `WEIGHT_VARIANCE_INTERVAL`| 0.4     | Sensitivity of fatigue to rhythm      |
| `WEIGHT_AVG_HOLD_TIME`    | 0.3     | Sensitivity of fatigue to sluggishness|
| `WEIGHT_ERROR_RATE`       | 0.5     | Error-rate contribution to error_prob |
| `WEIGHT_FATIGUE_FEEDBACK` | 0.6     | Fatigue contribution to error_prob    |
| `LEARNING_RATE`           | 0.05    | Online learning step size             |

---

##  Online Learning Layer (v1.1.0)

After every completed session, the model updates its weights using **online stochastic gradient descent** on the error-probability sub-model:

```
predicted  = model's current error_prob output
residual   = actual_error_rate − predicted
Δw_i       = LEARNING_RATE × residual × feature_i
w_i        = clamp(w_i + Δw_i, 0.01, 2.0)
```

This means the model **adapts to each user's individual typing patterns** over time without any manual retuning. Weights are clamped to `[0.01, 2.0]` to prevent numerical explosion.

Triggered automatically at the end of every `POST /typing/session`.

---

##  How AI Adapts Difficulty

The **Adaptive Difficulty Engine** (`services/adaptive_engine.py`) uses a priority-ordered threshold decision tree on the model's predictions:

```
if fatigue > 0.7:
    → directive: "reduce_difficulty"
    → feedback:  "Take a short break before continuing."

elif error_prob > 0.5:
    → directive: "focus_accuracy_exercises"
    → feedback:  "Focus on accuracy over speed."

elif consistency > 80.0:
    → directive: "increase_difficulty"
    → feedback:  "Excellent consistency! Increase your pace."

else:
    → directive: "maintain_difficulty"
    → feedback:  "Good performance. Maintain your rhythm."
```

Thresholds are fully configurable via environment variables.

---

##  Trend Analysis (v1.1.0)

`analytics_service.compute_trend(user_id)` analyses WPM slope across the last 10 sessions using **ordinary least-squares regression** (no NumPy — pure stdlib):

```
slope = (n⋅Σxy − Σx⋅Σy) / (n⋅Σx² − (Σx)²)

slope > +0.5 → "improving"
slope < −0.5 → "declining"
else         → "stable"
```

Returned in every `GET /typing/analytics` response.

---

##  Weak Pattern Detection (v1.1.0)

`analytics_service.detect_weak_patterns(session_id)` identifies the **top 3 character bigrams** where the user's errors cluster:

```python
# For each error keystroke, form bigram = prev_key + error_key
# Count bigram frequencies → return top 3
```

Example: `["he", "re", "th"]` means the user most often errors on these digraphs.
Returned in every `POST /ai/adapt` response for personalised drill targeting.

---

##  Real-time Stream Inference (v1.1.0)

`POST /ai/stream-predict` accepts raw keystrokes directly in the request body — **no session storage round-trip**:

```
keystrokes (in request) → feature_extractor → ml_model → adaptive_engine → response
```

Results are cached with a 30-second TTL to prevent redundant recomputes on repeated calls with the same last-keystroke timestamp.

---

##  API Endpoints

### Auth

| Method | Path             | Description                        |
|--------|------------------|------------------------------------|
| POST   | `/auth/register` | Register a new user account        |
| POST   | `/auth/login`    | Login and receive a JWT token      |

### Typing

| Method | Path                 | Description                                                   |
|--------|----------------------|---------------------------------------------------------------|
| POST   | `/typing/keystroke`  | Submit a single keystroke event                               |
| POST   | `/typing/session`    | Submit full session → triggers learning + stats computation  |
| GET    | `/typing/analytics`  | Get aggregate analytics + performance trend                   |

### AI

| Method | Path                   | Description                                              |
|--------|------------------------|----------------------------------------------------------|
| POST   | `/ai/predict`          | Run cognitive prediction (cached, session-based)         |
| POST   | `/ai/adapt`            | Get directive + feedback + weak_patterns                 |
| POST   | `/ai/stream-predict`   | **Real-time** inference on raw keystroke list (no DB)   |

Full interactive docs available at **`/docs`** (Swagger UI) and **`/redoc`** after startup.

---

##  Sample Test Data

### Step 1 — Register

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secure123"}'
```

### Step 2 — Submit a Session (triggers online learning)

```bash
curl -X POST http://localhost:8000/typing/session \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":    "alice",
    "session_id": "sess-001",
    "keystrokes": [
      {"user_id":"alice","session_id":"sess-001","key":"t","timestamp":1000,"is_error":false,"hold_duration":80},
      {"user_id":"alice","session_id":"sess-001","key":"h","timestamp":1120,"is_error":false,"hold_duration":75},
      {"user_id":"alice","session_id":"sess-001","key":"e","timestamp":1260,"is_error":true, "hold_duration":70},
      {"user_id":"alice","session_id":"sess-001","key":"r","timestamp":1500,"is_error":false,"hold_duration":60},
      {"user_id":"alice","session_id":"sess-001","key":"Space","timestamp":1700,"is_error":false,"hold_duration":50},
      {"user_id":"alice","session_id":"sess-001","key":"f","timestamp":2000,"is_error":false,"hold_duration":80},
      {"user_id":"alice","session_id":"sess-001","key":"a","timestamp":2300,"is_error":false,"hold_duration":75},
      {"user_id":"alice","session_id":"sess-001","key":"t","timestamp":2600,"is_error":false,"hold_duration":80}
    ]
  }'
# Logs will show: "WeightUpdate — residual=0.xxxx | w_var=... w_hold=... w_err=... w_fat=..."
```

### Step 3 — Real-time Stream Predict (no session storage needed)

```bash
curl -X POST http://localhost:8000/ai/stream-predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "keystrokes": [
      {"user_id":"alice","session_id":"live","key":"t","timestamp":1000,"is_error":false,"hold_duration":80},
      {"user_id":"alice","session_id":"live","key":"h","timestamp":1120,"is_error":true, "hold_duration":95},
      {"user_id":"alice","session_id":"live","key":"e","timestamp":1230,"is_error":false,"hold_duration":75}
    ]
  }'
```

**Expected Response:**
```json
{
  "fatigue":     0.38,
  "error_prob":  0.58,
  "consistency": 95.1,
  "directive":   "focus_accuracy_exercises",
  "feedback":    "Your error rate is climbing. Focus on accuracy over speed.",
  "mode":        "realtime"
}
```

### Step 4 — Adapt (with weak patterns)

```bash
curl -X POST http://localhost:8000/ai/adapt \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "session_id": "sess-001"}'
```

**Expected Response:**
```json
{
  "directive":     "focus_accuracy_exercises",
  "feedback":      "Your error rate is climbing. Focus on accuracy over speed.",
  "weak_patterns": ["he", "th"],
  "predictions": {
    "fatigue":     0.559,
    "error_prob":  0.607,
    "consistency": 93.6
  }
}
```

### Step 5 — Analytics with trend (after 3+ sessions)

```bash
curl "http://localhost:8000/typing/analytics?user_id=alice"
```

**Expected Response:**
```json
{
  "user_id":        "alice",
  "total_sessions": 3,
  "avg_wpm":        67.3,
  "avg_accuracy":   91.2,
  "trend":          "improving",
  "sessions":       [...]
}
```

---

##  How to Run

### 1. Install dependencies

```bash
cd "NeuroType AI"
python3 -m pip install -r requirements.txt
```

### 2. Start the server

```bash
uvicorn app:app --reload
```

The engine starts at **`http://localhost:8000`**.
Interactive API docs: **`http://localhost:8000/docs`**

### 3. Configuration (optional)

Override any setting via environment variables:

```bash
export DB_TYPE=memory          # "sqlite" (default) or "memory"
export LOG_LEVEL=DEBUG
export LEARNING_RATE=0.05      # online learning step size
export CACHE_TTL_SECONDS=60    # prediction cache TTL
export FATIGUE_THRESHOLD=0.65
uvicorn app:app --reload
```

---

##  File Structure

```
NeuroType AI/          (17 files — well under 20)
│
├── app.py             ← FastAPI entry point, lifespan, init_cache()
├── config.py          ← Settings, ML weights, LEARNING_RATE, CACHE_TTL
├── requirements.txt   ← Python dependencies (5 packages)
├── README.md
│
├── routes/
│   ├── __init__.py
│   ├── typing.py      ← /typing endpoints + online learning trigger
│   ├── ai.py          ← /ai/predict, /ai/adapt, /ai/stream-predict
│   └── auth.py        ← /auth endpoints
│
├── services/
│   ├── __init__.py
│   ├── keystroke_service.py  ← keystroke persistence + retrieval
│   ├── analytics_service.py  ← WPM, accuracy, trend, weak_patterns
│   ├── ai_engine.py          ← pipeline orchestrator
│   ├── ml_model.py           ← Cognitive Brain Model + update_weights()
│   └── adaptive_engine.py    ← difficulty directives + feedback
│
├── models/
│   ├── __init__.py
│   ├── schemas.py     ← Pydantic schemas (StreamPredictRequest added)
│   └── storage.py     ← SQLite + in-memory DB + get_last_n_sessions()
│
└── utils/
    ├── __init__.py
    ├── feature_extractor.py  ← raw keystrokes → feature vector
    ├── helpers.py            ← JWT, bcrypt, logging setup
    └── cache.py              ← NEW: TTLCache singleton (30s default)
```

---

##  Technology Stack

| Layer          | Technology                          |
|----------------|-------------------------------------|
| Framework      | FastAPI + Uvicorn                   |
| Validation     | Pydantic v2                         |
| Storage        | SQLite (default) / In-memory dict   |
| Auth           | JWT (HS256) via python-jose         |
| Passwords      | bcrypt (direct library)             |
| ML             | Pure Python math (stdlib only)      |
| Online Learning| Custom SGD (no external ML library) |
| Caching        | Custom TTLCache (stdlib threading)  |
| Language       | Python 3.10+                        |

---

*NeuroType AI v1.1.0 — Built to model cognition, not just keystrokes.*
