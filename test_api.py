"""
test_api.py -- NeuroType AI Comprehensive API Test Suite
Runs all endpoints in sequence: auth -> typing -> AI
Uses httpx for synchronous HTTP requests against a running server.

Usage:
    # 1. Start the server first:
    #    uvicorn app:app --reload
    # 2. Run the tests:
    #    python test_api.py
"""

import sys
import io
import json
import time
import httpx
# Force UTF-8 output on Windows so box-drawing/emoji chars print correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000"
TIMEOUT  = 30.0   # Increased: bcrypt + SQLite + online learning can be slow

# ── ANSI colour codes ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(label: str, data=None):
    global passed
    passed += 1
    print(f"  {GREEN}[PASS]{RESET} {label}")
    if data:
        print(f"    {CYAN}{json.dumps(data, indent=2)[:300]}{RESET}")


def fail(label: str, reason: str = ""):
    global failed
    failed += 1
    print(f"  {RED}FAIL{RESET} {label}")
    if reason:
        print(f"    {RED}{reason}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{YELLOW}===  {title}  ==={RESET}")


def assert_status(resp: httpx.Response, expected: int, label: str):
    if resp.status_code == expected:
        ok(label, resp.json())
    else:
        fail(label, f"Expected HTTP {expected}, got {resp.status_code}: {resp.text[:200]}")


# ── Sample data ───────────────────────────────────────────────────────────────

USER = "test_alice"
PASS = "secure123"
SESSION_ID = "sess-test-001"

KEYSTROKES = [
    {"user_id": USER, "session_id": SESSION_ID, "key": "t", "timestamp": 1000,  "is_error": False, "hold_duration": 80},
    {"user_id": USER, "session_id": SESSION_ID, "key": "h", "timestamp": 1120,  "is_error": False, "hold_duration": 75},
    {"user_id": USER, "session_id": SESSION_ID, "key": "e", "timestamp": 1260,  "is_error": True,  "hold_duration": 70},
    {"user_id": USER, "session_id": SESSION_ID, "key": "r", "timestamp": 1500,  "is_error": False, "hold_duration": 60},
    {"user_id": USER, "session_id": SESSION_ID, "key": " ", "timestamp": 1700,  "is_error": False, "hold_duration": 50},
    {"user_id": USER, "session_id": SESSION_ID, "key": "f", "timestamp": 2000,  "is_error": False, "hold_duration": 80},
    {"user_id": USER, "session_id": SESSION_ID, "key": "a", "timestamp": 2300,  "is_error": False, "hold_duration": 75},
    {"user_id": USER, "session_id": SESSION_ID, "key": "t", "timestamp": 2600,  "is_error": False, "hold_duration": 80},
    {"user_id": USER, "session_id": SESSION_ID, "key": "i", "timestamp": 2900,  "is_error": False, "hold_duration": 65},
    {"user_id": USER, "session_id": SESSION_ID, "key": "g", "timestamp": 3100,  "is_error": True,  "hold_duration": 90},
    {"user_id": USER, "session_id": SESSION_ID, "key": "u", "timestamp": 3350,  "is_error": False, "hold_duration": 70},
    {"user_id": USER, "session_id": SESSION_ID, "key": "e", "timestamp": 3600,  "is_error": False, "hold_duration": 75},
]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client: httpx.Client):
    section("1. Health Check")
    # GET / now serves the HTML frontend
    resp = client.get("/")
    if resp.status_code == 200:
        ok("GET / returns 200")
    else:
        fail("GET / should return 200", f"Got {resp.status_code}")

    ct = resp.headers.get("content-type", "")
    if "text/html" in ct:
        ok("GET / serves HTML frontend (content-type: text/html)")
    elif "application/json" in ct:
        ok("GET / serves JSON health check (content-type: application/json)")
    else:
        fail("Unexpected content-type", ct)

    # Verify docs endpoint is accessible
    resp2 = client.get("/docs")
    if resp2.status_code == 200:
        ok("GET /docs Swagger UI is accessible")
    else:
        fail("GET /docs should return 200", f"Got {resp2.status_code}")

    # Verify redoc is accessible
    resp3 = client.get("/redoc")
    if resp3.status_code == 200:
        ok("GET /redoc ReDoc UI is accessible")
    else:
        fail("GET /redoc should return 200", f"Got {resp3.status_code}")



def test_auth(client: httpx.Client):
    section("2. Authentication")

    # Register
    resp = client.post("/auth/register", json={"username": USER, "password": PASS})
    if resp.status_code == 200:
        ok("POST /auth/register — user created", resp.json())
    elif resp.status_code == 400 and "already taken" in resp.text:
        ok("POST /auth/register — user already exists (re-run), continuing")
    else:
        fail("POST /auth/register", resp.text)

    # Duplicate register → should 400
    resp2 = client.post("/auth/register", json={"username": USER, "password": PASS})
    if resp2.status_code == 400:
        ok("POST /auth/register — duplicate username correctly rejected (400)")
    else:
        fail("POST /auth/register — duplicate should return 400", f"Got {resp2.status_code}")

    # Login
    resp3 = client.post("/auth/login", json={"username": USER, "password": PASS})
    assert_status(resp3, 200, "POST /auth/login — JWT token returned")
    body = resp3.json()
    if "access_token" in body and body.get("token_type") == "bearer":
        ok("Login response has access_token + token_type=bearer")
    else:
        fail("Login response shape wrong", str(body))

    # Wrong password → 401
    resp4 = client.post("/auth/login", json={"username": USER, "password": "wrongpass"})
    if resp4.status_code == 401:
        ok("POST /auth/login — wrong password correctly rejected (401)")
    else:
        fail("POST /auth/login — wrong password should return 401", f"Got {resp4.status_code}")


def test_keystroke(client: httpx.Client):
    section("3. Single Keystroke Submission")
    ks = KEYSTROKES[0].copy()
    ks["session_id"] = "single-ks-session"
    resp = client.post("/typing/keystroke", json=ks)
    assert_status(resp, 200, "POST /typing/keystroke — single keystroke recorded")


def test_session(client: httpx.Client):
    section("4. Full Session Submission (triggers online learning)")
    payload = {
        "user_id": USER,
        "session_id": SESSION_ID,
        "keystrokes": KEYSTROKES,
    }
    try:
        resp = client.post("/typing/session", json=payload)
    except httpx.ReadTimeout:
        fail("POST /typing/session timed out — server too slow"); return
    assert_status(resp, 200, "POST /typing/session — session stored + stats computed")
    body = resp.json()
    if "stats" in body:
        stats = body["stats"]
        ok(f"Session stats: WPM={stats.get('wpm')}, accuracy={stats.get('accuracy')}%, keys={stats.get('total_keys')}")
        if stats.get("total_keys") == len(KEYSTROKES):
            ok("Keystroke count matches payload length")
        else:
            fail("Keystroke count mismatch", f"Expected {len(KEYSTROKES)}, got {stats.get('total_keys')}")
    else:
        fail("Session response missing 'stats' field", str(body)[:200])


def test_analytics(client: httpx.Client):
    section("5. Analytics")
    resp = client.get(f"/typing/analytics?user_id={USER}")
    assert_status(resp, 200, f"GET /typing/analytics?user_id={USER}")
    body = resp.json()
    required_keys = ["user_id", "total_sessions", "avg_wpm", "avg_accuracy", "trend", "sessions"]
    missing = [k for k in required_keys if k not in body]
    if not missing:
        ok(f"Analytics response has all required fields: {required_keys}")
        ok(f"Trend: {body.get('trend')} | Sessions: {body.get('total_sessions')} | avg_wpm: {body.get('avg_wpm')}")
    else:
        fail("Analytics response missing fields", str(missing))

    # Unknown user → 404
    resp2 = client.get("/typing/analytics?user_id=nonexistent_user_xyz")
    if resp2.status_code == 404:
        ok("GET /typing/analytics — unknown user correctly returns 404")
    else:
        fail("Unknown user should return 404", f"Got {resp2.status_code}")


def test_ai_predict(client: httpx.Client):
    section("6. AI Predict")
    resp = client.post("/ai/predict", json={"user_id": USER, "session_id": SESSION_ID})
    assert_status(resp, 200, "POST /ai/predict — prediction served")
    body = resp.json()
    required = ["fatigue_level", "error_probability", "consistency_score"]
    missing = [k for k in required if k not in body]
    if not missing:
        ok(f"fatigue={body['fatigue_level']:.3f} | error_prob={body['error_probability']:.3f} | consistency={body['consistency_score']:.1f}")
    else:
        fail("AI predict response missing fields", str(missing))

    # Test cache hit
    resp2 = client.post("/ai/predict", json={"user_id": USER, "session_id": SESSION_ID})
    if resp2.status_code == 200:
        ok("POST /ai/predict — second call (cache HIT) also returns 200")

    # Missing session → 404
    resp3 = client.post("/ai/predict", json={"user_id": USER, "session_id": "no-such-session"})
    if resp3.status_code == 404:
        ok("POST /ai/predict — non-existent session correctly returns 404")
    else:
        fail("Non-existent session should return 404", f"Got {resp3.status_code}")


def test_ai_adapt(client: httpx.Client):
    section("7. AI Adapt (directive + feedback + weak patterns)")
    resp = client.post("/ai/adapt", json={"user_id": USER, "session_id": SESSION_ID})
    assert_status(resp, 200, "POST /ai/adapt — adapt response served")
    body = resp.json()
    required = ["directive", "feedback", "weak_patterns", "predictions"]
    missing = [k for k in required if k not in body]
    if not missing:
        ok(f"directive='{body['directive']}'")
        ok(f"feedback='{body['feedback'][:60]}...'")
        ok(f"weak_patterns={body['weak_patterns']}")
        preds = body.get("predictions", {})
        ok(f"predictions: fatigue={preds.get('fatigue')}, error_prob={preds.get('error_prob')}, consistency={preds.get('consistency')}")
    else:
        fail("AI adapt response missing fields", str(missing))

    valid_directives = {"reduce_difficulty", "focus_accuracy_exercises", "increase_difficulty", "maintain_difficulty"}
    if body.get("directive") in valid_directives:
        ok(f"Directive '{body['directive']}' is a valid value")
    else:
        fail("Invalid directive value", str(body.get("directive")))


def test_stream_predict(client: httpx.Client):
    section("8. AI Stream-Predict (real-time, no DB)")
    payload = {
        "user_id": USER,
        "keystrokes": [
            {"user_id": USER, "session_id": "live", "key": "t",  "timestamp": 5000, "is_error": False, "hold_duration": 80},
            {"user_id": USER, "session_id": "live", "key": "h",  "timestamp": 5120, "is_error": True,  "hold_duration": 95},
            {"user_id": USER, "session_id": "live", "key": "e",  "timestamp": 5230, "is_error": False, "hold_duration": 75},
            {"user_id": USER, "session_id": "live", "key": "r",  "timestamp": 5380, "is_error": False, "hold_duration": 60},
            {"user_id": USER, "session_id": "live", "key": "e",  "timestamp": 5550, "is_error": True,  "hold_duration": 85},
        ],
    }
    resp = client.post("/ai/stream-predict", json=payload)
    assert_status(resp, 200, "POST /ai/stream-predict — realtime inference served")
    body = resp.json()
    required = ["fatigue", "error_prob", "consistency", "directive", "feedback", "mode"]
    missing = [k for k in required if k not in body]
    if not missing:
        ok(f"mode='{body['mode']}' | fatigue={body['fatigue']:.3f} | directive='{body['directive']}'")
    else:
        fail("Stream predict response missing fields", str(missing))

    if body.get("mode") == "realtime":
        ok("mode field correctly set to 'realtime'")

    # Too few keystrokes → 400
    payload_bad = {"user_id": USER, "keystrokes": [payload["keystrokes"][0]]}
    resp2 = client.post("/ai/stream-predict", json=payload_bad)
    if resp2.status_code in (400, 422):
        ok("POST /ai/stream-predict — single keystroke correctly rejected (400/422)")
    else:
        fail("Single keystroke should be rejected", f"Got {resp2.status_code}")


def test_online_learning(client: httpx.Client):
    section("9. Online Learning — submit multiple sessions and verify trend")
    sessions = [
        ("sess-learn-001", list(range(1000, 5000, 200))),   # slow typing
        ("sess-learn-002", list(range(1000, 3000, 150))),   # medium
        ("sess-learn-003", list(range(1000, 2500, 120))),   # faster
    ]
    for sess_id, ts_list in sessions:
        ks_list = [
            {
                "user_id": USER,
                "session_id": sess_id,
                "key": chr(97 + (i % 26)),
                "timestamp": ts,
                "is_error": (i % 8 == 0),
                "hold_duration": 70.0,
            }
            for i, ts in enumerate(ts_list)
        ]
        payload = {"user_id": USER, "session_id": sess_id, "keystrokes": ks_list}
        resp = client.post("/typing/session", json=payload)
        if resp.status_code == 200:
            ok(f"Learning session '{sess_id}' submitted (WPM={resp.json()['stats']['wpm']})")
        else:
            fail(f"Session '{sess_id}' failed", resp.text[:100])

    # Check trend now (should have 4+ sessions for user)
    resp = client.get(f"/typing/analytics?user_id={USER}")
    if resp.status_code == 200:
        body = resp.json()
        ok(f"Final trend after multiple sessions: '{body.get('trend')}' (sessions={body.get('total_sessions')})")
    else:
        fail("Analytics after learning sessions", resp.text)


# ── Main Runner ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  NeuroType AI -- API Test Suite{RESET}")
    print(f"{BOLD}  Target: {BASE_URL}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Check server is up
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=TIMEOUT)
    except httpx.ConnectError:
        print(f"\n{RED}ERROR: Could not connect to {BASE_URL}{RESET}")
        print(f"  → Make sure the server is running: {BOLD}uvicorn app:app --reload{RESET}")
        sys.exit(1)

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        test_health(client)
        test_auth(client)
        test_keystroke(client)
        test_session(client)
        test_analytics(client)
        test_ai_predict(client)
        test_ai_adapt(client)
        test_stream_predict(client)
        test_online_learning(client)

    # Summary
    total = passed + failed
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Results: {GREEN}{passed}{RESET}/{BOLD}{total}{RESET} tests passed  |  {RED}{failed}{RESET} failed")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
