"""
面部皮肤质量检查 — 本地单文件 Web 应用（Flask）。

分析功能调用香港浸会大学 GenAI 接口（部署名已包含在 URL 中）。
环境变量 LLM_API_URL / LLM_API_KEY 若已设置，会覆盖下面的默认值。
演示账号 demo/demo 在退出前会记住本次上传，退出后只保留原来的三张演示照片。新注册的账号分开保存，看不到演示照片。
"""

import base64
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, session, url_for

# 课程提供的 GenAI 接口。部署模型是 gemini-2.5-pro，写在 URL 路径里。
LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://genai.hkbu.edu.hk/api/v0/rest/deployments/gemini-2.5-pro/chat/completions?api-version=v1",
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "13406d59-bab5-4d4e-9e9b-bc6b126f73b9")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-pro")

DEMO_USERS = {"demo": "demo"}
USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
USERNAME_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")

PROMPT = """You assess facial skin from a frontal photo for a skincare quality-check demo.
Return only a JSON object, no markdown, with these keys:
- cleanliness: number from 0 to 10 with one decimal. How clean the skin looks.
- quality: number from 0 to 10 with one decimal. Overall skin quality, including texture, tone, lines, and spots.
- estimated_age: string, an age or age range such as "55-65".
- description: 2 to 4 sentences describing the facial skin condition.
- cleanliness_review: one sentence explaining the cleanliness score.
- quality_review: one sentence explaining the quality score.
- tips_summary: 1 to 2 sentences of skincare advice for this face.
- tips: an array of exactly 2 objects. Each object has "title" (a short care topic) and "points" (an array of 2 short actionable tips).
- treatment: an array of 3 short points describing what skin treatment program can improve the visible condition.
If the image is not a clear frontal face, still return this JSON and say so in the description.
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history_data")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "skin-check-local-demo")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/analyze") or request.path.startswith("/api/"):
                return jsonify(error="Please sign in first."), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def as_score(value):
    number = float(str(value).strip().split("/")[0])
    return round(max(0.0, min(10.0, number)), 1)


def as_tips(value):
    groups = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_points = item.get("points") or []
            if isinstance(raw_points, str):
                raw_points = [raw_points]
            points = [str(point).strip() for point in raw_points if str(point).strip()]
            groups.append({"title": str(item.get("title") or "Tip").strip(), "points": points or ["—"]})
    if not groups:
        raise KeyError("tips")
    return groups


def as_points(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def safe_name(value, pattern):
    if re.fullmatch(pattern, value or ""):
        return value
    return None


def _cloud_users_path():
    # Streamlit Community Cloud keeps the app under /mount/src and restores that
    # folder from Git whenever the container sleeps or restarts. New accounts are
    # stored outside the repo so a write does not touch the deployed source.
    script = os.path.abspath(__file__).replace("\\", "/")
    if "/mount/src/" not in script:
        return None
    return os.path.join(tempfile.gettempdir(), "skin_check_users.json")


def _read_users_into(users, path):
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            for name, password in stored.items():
                if USERNAME_RE.fullmatch(str(name)) and isinstance(password, str) and str(name) != "demo":
                    users[str(name)] = password
    except (OSError, json.JSONDecodeError):
        pass


def load_users():
    users = dict(DEMO_USERS)
    _read_users_into(users, USERS_PATH)
    _read_users_into(users, _cloud_users_path())
    return users


def save_users(users):
    stored = {name: password for name, password in users.items() if name != "demo"}
    path = _cloud_users_path() or USERS_PATH
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2)
            handle.write("\n")
    except OSError:
        pass


def _remembered_password(payload, username):
    saved = payload.get("saved_users")
    if isinstance(saved, str):
        try:
            saved = json.loads(saved)
        except json.JSONDecodeError:
            return None
    if not isinstance(saved, dict):
        return None
    password = saved.get(username)
    return password if isinstance(password, str) else None


def _login_matches(username, password, remembered=None):
    users = load_users()
    if users.get(username) == password:
        return True
    # The cloud disk is wiped when the site restarts. The same browser sends the
    # account it registered, and a matching password is written back.
    if (
        username not in users
        and username != "demo"
        and remembered == password
        and len(password) >= 4
        and USERNAME_RE.fullmatch(username or "")
    ):
        users[username] = password
        save_users(users)
        return True
    return False


def history_folder(username):
    folder = os.path.join(HISTORY_DIR, username)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_history(username, image_bytes, ext, result, created_at=None, session_id=None):
    record_id = uuid.uuid4().hex
    folder = history_folder(username)
    image_name = record_id + ext
    with open(os.path.join(folder, image_name), "wb") as image_file:
        image_file.write(image_bytes)
    record = {
        "id": record_id,
        "created_at": created_at or datetime.now().isoformat(timespec="minutes"),
        "cleanliness": result["cleanliness"],
        "quality": result["quality"],
        "tips_summary": result["tips_summary"],
        "tips": result["tips"],
        "treatment": result["treatment"],
        "image": image_name,
    }
    if session_id:
        record["session_id"] = session_id
    with open(os.path.join(folder, record_id + ".json"), "w", encoding="utf-8") as record_file:
        json.dump(record, record_file)
    return record


def list_history(username, session_id=None):
    folder = history_folder(username)
    records = []
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as record_file:
            records.append(json.load(record_file))
    if username == "demo":
        records = [
            record for record in records
            if not record.get("session_id") or record.get("session_id") == session_id
        ]
    records.sort(key=lambda record: record["created_at"])
    return records


def forget_demo_session(session_id):
    folder = os.path.join(HISTORY_DIR, "demo")
    if not session_id or not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        with open(path, encoding="utf-8") as record_file:
            record = json.load(record_file)
        if record.get("session_id") != session_id:
            continue
        image_path = os.path.join(folder, record.get("image") or "")
        if os.path.isfile(image_path):
            os.remove(image_path)
        os.remove(path)


def demo_sid():
    if session.get("user") != "demo":
        return None
    if not session.get("sid"):
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def find_history(username, record_id):
    path = os.path.join(HISTORY_DIR, username, record_id + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as record_file:
        return json.load(record_file)


def call_llm(image_bytes, mime):
    if not LLM_API_URL or not LLM_API_KEY:
        raise RuntimeError(
            "LLM API is not configured. Set LLM_API_URL and LLM_API_KEY at the top of app.py."
        )

    payload = {
        "model": LLM_MODEL,
        "temperature": 0.2,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"},
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        LLM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": LLM_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the LLM API: {exc.reason}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM API response did not contain a message.") from exc


LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in · Skin Quality Check</title>
  <style>
    :root { color-scheme: light; --teal: #12998c; --ink: #134e4a; --muted: #5f7a78; --line: #d7e6e3; --bg: #f3f7f6; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--ink); font-family: "Segoe UI", sans-serif; }
    .panel { width: min(420px, calc(100% - 32px)); display: flex; flex-direction: column; gap: 16px; margin: 24px 0; }
    form { width: 100%; background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 32px 28px; box-shadow: 0 10px 30px rgba(19, 78, 74, 0.06); }
    h1, h2 { margin: 0 0 6px; font-size: 24px; }
    p.lead { margin: 0 0 22px; color: var(--muted); }
    label { display: block; margin: 14px 0 6px; font-size: 14px; font-weight: 600; }
    input { width: 100%; height: 44px; border: 1px solid var(--line); border-radius: 10px; padding: 0 12px; font: inherit; }
    input:focus { outline: 2px solid rgba(18, 153, 140, 0.25); border-color: var(--teal); }
    button { width: 100%; height: 46px; margin-top: 20px; border: 0; border-radius: 12px; background: var(--teal); color: #fff; font: inherit; font-weight: 700; cursor: pointer; }
    button:hover { background: #0e857a; }
    button.secondary { background: #fff; color: var(--ink); border: 1px solid var(--line); }
    button.secondary:hover { background: #f3f7f6; }
    form[hidden] { display: none !important; }
    .error { margin: 0 0 8px; padding: 10px 12px; border-radius: 10px; background: #fff1f1; color: #9f1239; }
    .notice { margin: 0 0 8px; padding: 10px 12px; border-radius: 10px; background: #ecfdf5; color: #065f46; }
    .hint { margin: 16px 0 0; color: var(--muted); font-size: 13px; text-align: center; }
  </style>
</head>
<body>
  <div class="panel">
    <form method="post" id="login-form" {% if register_error or created %}hidden{% endif %}>
      <h1>Skin Quality Check</h1>
      <p class="lead">Sign in to upload a frontal face photo.</p>
      {% if error %}<p class="error">{{ error }}</p>{% endif %}
      <input type="hidden" name="action" value="login">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit" name="action" value="login">Sign in</button>
      <button class="secondary" id="show-register" type="button">Create a new account</button>
      <p class="hint">Demo account: username <strong>demo</strong>, password <strong>demo</strong></p>
    </form>
    <form method="post" id="register-form" {% if not register_error and not created %}hidden{% endif %}>
      <h2>Create an account</h2>
      {% if created %}
      <p class="notice">Account created. Click Back to sign in.</p>
      {% else %}
      <p class="lead">A new account starts with an empty history. Uploads stay with this username.</p>
      {% if register_error %}<p class="error">{{ register_error }}</p>{% endif %}
      <input type="hidden" name="action" value="register">
      <label for="new-username">Username</label>
      <input id="new-username" name="username" autocomplete="username" required>
      <label for="new-password">Password</label>
      <input id="new-password" name="password" type="password" autocomplete="new-password" required>
      <label for="confirm">Confirm password</label>
      <input id="confirm" name="confirm" type="password" autocomplete="new-password" required>
      <button type="submit" name="action" value="register">Create account</button>
      {% endif %}
      <button class="secondary" id="back-login" type="button">Back</button>
    </form>
  </div>
  <script>
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");
    document.getElementById("show-register").addEventListener("click", () => {
      loginForm.hidden = true;
      registerForm.hidden = false;
      document.getElementById("new-username").focus();
    });
    document.getElementById("back-login").addEventListener("click", () => {
      if (new URLSearchParams(location.search).get("created")) {
        location.href = "/login";
        return;
      }
      registerForm.hidden = true;
      loginForm.hidden = false;
      document.getElementById("username").focus();
    });
  </script>
</body>
</html>
"""

MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Skin Quality Check</title>
  <style>
    :root { color-scheme: light; --teal: #12998c; --teal-dark: #0e857a; --ink: #134e4a; --muted: #5f7a78; --line: #d7e6e3; --bg: #f3f7f6; --soft: #f7fbfa; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", sans-serif; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 16px 24px; background: #fff; border-bottom: 1px solid var(--line); }
    header h1 { margin: 0; font-size: 20px; }
    header p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
    header form { margin: 0; }
    header button, .ghost, .primary { height: 40px; padding: 0 16px; border-radius: 10px; font: inherit; font-weight: 700; cursor: pointer; }
    header button, .ghost { background: #fff; color: var(--ink); border: 1px solid var(--line); }
    .primary { background: var(--teal); color: #fff; border: 0; }
    .primary:hover { background: var(--teal-dark); }
    .primary:disabled { opacity: 0.55; cursor: not-allowed; }
    main { width: min(1120px, calc(100% - 32px)); margin: 24px auto 40px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
    .side { display: flex; flex-direction: column; gap: 20px; }
    .block { display: flex; flex-direction: column; gap: 10px; }
    main h2 { margin: 0; font-size: 18px; font-weight: 700; line-height: 1.3; }
    section { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 20px; }
    .preview { position: relative; width: 100%; aspect-ratio: 1 / 1.2; border-radius: 16px; background: #123; overflow: hidden; cursor: pointer; }
    .preview.dragover { box-shadow: inset 0 0 0 3px var(--teal); }
    .preview img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; object-position: center; display: none; }
    .preview.has-image img { display: block; }
    .upload-btn { position: absolute; top: 16px; left: 16px; z-index: 1; display: inline-flex; align-items: center; height: 40px; padding: 0 14px; border-radius: 10px; background: var(--teal); color: #fff; font-weight: 700; cursor: pointer; }
    .preview.has-image .upload-btn { background: rgba(255,255,255,0.94); color: var(--ink); }
    .drop-hint { position: absolute; left: 20px; right: 20px; bottom: 18px; color: rgba(255,255,255,0.9); font-size: 14px; text-align: center; pointer-events: none; }
    .preview.has-image .drop-hint { display: none; }
    input[type="file"] { display: none; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 0; }
    .actions .ghost, .actions .primary { width: 100%; height: 44px; }
    .guidelines { margin: 0; color: var(--muted); font-size: 14px; }
    .guidelines strong { color: #be123c; }
    .guidelines ul { margin: 8px 0 0; padding-left: 18px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 14px; background: var(--soft); padding: 12px; }
    .metric span, .kicker { display: block; color: var(--teal-dark); font-size: 13px; font-weight: 700; }
    .metric strong, #age { display: block; margin-top: 8px; font-size: 22px; }
    .meters { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
    .meter { border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
    .meter label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 700; color: var(--teal-dark); }
    input[type="range"] { --fill: 0%; -webkit-appearance: none; appearance: none; width: 100%; height: 18px; margin-top: 10px; background: transparent; pointer-events: none; }
    input[type="range"]::-webkit-slider-runnable-track { height: 8px; border-radius: 999px; background: linear-gradient(to right, #11D9A1 var(--fill), #d7e6e3 var(--fill)); }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; margin-top: -5px; border: 0; border-radius: 50%; background: #11D9A1; }
    input[type="range"]::-moz-range-track { height: 8px; border: 0; border-radius: 999px; background: #d7e6e3; }
    input[type="range"]::-moz-range-progress { height: 8px; border-radius: 999px; background: #11D9A1; }
    input[type="range"]::-moz-range-thumb { width: 18px; height: 18px; border: 0; border-radius: 50%; background: #11D9A1; }
    .report-section .kicker { font-size: 16px; font-weight: 700; color: var(--ink); }
    .report-section h3 { margin: 16px 0 6px; font-size: 16px; font-weight: 700; color: var(--ink); }
    .report-section p { margin: 0; line-height: 1.5; }
    .report-points { margin: 0; padding-left: 18px; font-size: 14px; line-height: 1.5; }
    .report-points li + li { margin-top: 4px; }
    .report-points strong { font-weight: 700; }
    .status { display: none; margin-bottom: 12px; padding: 10px 12px; border-radius: 10px; background: #fff7ed; color: #9a3412; }
    .status.show { display: block; }
    .tabs { display: flex; gap: 8px; }
    header .tab { margin: 0; }
    header .tab.is-active { background: var(--teal); color: #fff; border-color: var(--teal); }
    main[hidden] { display: none; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin: 0 0 8px; font-size: 13px; font-weight: 700; }
    .swatch { width: 22px; height: 4px; margin-right: 6px; border-radius: 999px; display: inline-block; vertical-align: middle; }
    .swatch.clean { background: #11D9A1; }
    .swatch.quality { background: #1f6f8b; }
    #chart { width: 100%; height: auto; display: block; }
    .chart-empty { margin: 8px 0 0; color: var(--muted); font-size: 14px; }
    .history-photo { display: block; width: 100%; height: auto; border-radius: 16px; }
    .recommend-link { display: block; width: 100%; margin: 0; padding: 14px 16px; border: 0; border-radius: 14px; background: #11D9A1; color: #08352f; font: inherit; font-size: 16px; font-weight: 700; text-align: left; cursor: pointer; }
    .report-section .recommend-link { margin-top: 16px; }
    .recommend-link:hover { background: #0ec892; }
    .project-scroll, .history-scroll { max-height: calc(var(--screen, 100vh) - 180px); overflow-y: auto; }
    .project-scroll { display: flex; flex-direction: column; gap: 12px; padding-right: 4px; }
    .project-card { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 16px; cursor: pointer; }
    .project-card.is-selected { border-color: #11D9A1; box-shadow: inset 0 0 0 1px #11D9A1; }
    .project-card h3 { margin: 0 0 8px; font-size: 16px; }
    .project-card p { margin: 0 0 8px; line-height: 1.5; }
    .project-card .price { color: var(--teal-dark); font-weight: 700; }
    .project-card button { margin-top: 4px; }
    #predict-chart { width: 100%; height: auto; display: block; }
    .reserve-form { display: flex; flex-direction: column; gap: 0; }
    .reserve-form[hidden] { display: none !important; }
    .reserve-form label { display: block; margin: 12px 0 6px; font-size: 14px; font-weight: 700; }
    .reserve-form input, .reserve-form select { width: 100%; height: 44px; border: 1px solid var(--line); border-radius: 10px; padding: 0 12px; font: inherit; background: #fff; }
    .reserve-form .primary { width: 100%; margin-top: 16px; }
    #reserve-done { margin-top: 12px; color: var(--teal-dark); font-weight: 700; }
    #history-time { margin-bottom: 12px; }
    @media (max-width: 860px) { main, .metrics, .meters { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Skin Quality Check</h1>
      <p>Signed in as {{ user }}</p>
    </div>
    <nav class="tabs">
      <button class="tab is-active" id="tab-home" type="button">Home</button>
      <button class="tab" id="tab-history" type="button">History</button>
      <button class="tab" id="tab-recommend" type="button">Recommend</button>
    </nav>
    <form method="post" action="{{ url_for('logout') }}"><button type="submit">Log out</button></form>
  </header>
  <main id="home-view">
    <div class="block">
    <h2>1. Source Capture</h2>
    <section>
      <div class="preview" id="preview">
        <img id="photo" alt="Uploaded frontal face photo">
        <label class="upload-btn" for="file">Upload Frontal Face Photo</label>
        <p class="drop-hint">Click anywhere in this area, or drag a photo here.</p>
        <input id="file" type="file" accept="image/jpeg,image/png,image/webp,image/gif">
      </div>
    </section>
    <div class="actions">
      <button class="ghost" id="clear" type="button">Clear Image</button>
      <button class="primary" id="analyze" type="button" disabled>Analyze Skin</button>
    </div>
    <div class="guidelines">
      <strong>Guidelines for best results:</strong>
      <ul>
        <li>Capture the photo in bright, natural, neutral lighting.</li>
        <li>Present a clean, makeup-free face facing front directly.</li>
        <li>Ensure the photo is clear and well-focused.</li>
      </ul>
    </div>
    </div>
    <div class="side">
    <div class="block">
    <h2>2. Evaluation Results</h2>
    <section>
      <p class="status" id="status"></p>
      <div class="metrics">
        <div class="metric"><span>Skin Cleanliness</span><strong id="cleanliness">—</strong></div>
        <div class="metric"><span>Skin Quality</span><strong id="quality">—</strong></div>
        <div class="metric"><span>Estimated Age</span><strong id="age">—</strong></div>
      </div>
      <div class="meters">
        <div class="meter">
          <label>Cleanliness Gage Meter <span id="cleanliness-num">0</span></label>
          <input id="cleanliness-meter" type="range" min="0" max="10" step="0.1" value="0" tabindex="-1">
        </div>
        <div class="meter">
          <label>Quality Gage Meter <span id="quality-num">0</span></label>
          <input id="quality-meter" type="range" min="0" max="10" step="0.1" value="0" tabindex="-1">
        </div>
      </div>
    </section>
    </div>
    <div class="block">
    <h2>3. Professional Skincare Report</h2>
    <section class="report-section">
      <p class="kicker">Overall Condition Summary</p>
      <p id="description">Upload a photo and choose Analyze Skin. The result will appear here.</p>
      <h3>1. Skin Cleanliness</h3>
      <ul class="report-points">
        <li><strong>KPI Rating:</strong> <span id="cleanliness-kpi">—</span></li>
        <li><strong>Aesthetician Review:</strong> <span id="cleanliness-review">—</span></li>
      </ul>
      <h3>2. Skin Quality</h3>
      <ul class="report-points">
        <li><strong>KPI Rating:</strong> <span id="quality-kpi">—</span></li>
        <li><strong>Aesthetician Review:</strong> <span id="quality-review">—</span></li>
      </ul>
    </section>
    </div>
    <div class="block">
    <h2>4. Skincare Tips</h2>
    <section class="report-section" id="tips">
      <p class="kicker">Recommended Routine</p>
      <p id="tips-summary">Upload a photo and choose Analyze Skin. Tips will appear here.</p>
      <h3>1. —</h3>
      <ul class="report-points"><li>—</li></ul>
      <h3>2. —</h3>
      <ul class="report-points"><li>—</li></ul>
      <button class="recommend-link" type="button">What skin treatment program can improve...</button>
    </section>
    </div>
    </div>
  </main>
  <main id="history-view" hidden>
    <div class="block">
      <h2>Skin History</h2>
      <section>
        <p class="legend"><span><i class="swatch clean"></i>Skin Cleanliness</span><span><i class="swatch quality"></i>Skin Quality</span></p>
        <svg id="chart" role="img" aria-label="Cleanliness and quality over time"></svg>
        <p class="chart-empty" id="chart-empty">Each analyzed photo on Home becomes a point on this chart.</p>
      </section>
      <button class="recommend-link" id="open-recommend" type="button">What skin treatment program can improve...</button>
    </div>
    <div class="block">
      <h2>Selected Record</h2>
      <section class="history-scroll">
        <p id="history-placeholder">Click a point to see the photo from that time and its skincare tips.</p>
        <div id="history-record" hidden>
          <p class="kicker" id="history-time"></p>
          <img class="history-photo" id="history-photo" alt="Uploaded photo from the selected time">
          <div id="history-tips"></div>
        </div>
      </section>
    </div>
  </main>
  <main id="recommend-view" hidden>
    <div class="block">
      <h2>Medical Beauty Projects</h2>
      <p class="chart-empty" id="recommend-context">Choose a time on History first.</p>
      <div class="project-scroll" id="project-list"></div>
    </div>
    <div class="block">
      <h2>Predicted Effect</h2>
      <section>
        <p class="legend"><span><i class="swatch clean"></i>Skin Cleanliness</span><span><i class="swatch quality"></i>Skin Quality</span></p>
        <svg id="predict-chart" role="img" aria-label="Predicted cleanliness and quality"></svg>
        <p class="chart-empty" id="predict-note"></p>
      </section>
      <form class="reserve-form" id="reserve-form" hidden>
        <h2>Reservation</h2>
        <section>
          <p class="kicker" id="reserve-project"></p>
          <label for="reserve-date">Date</label>
          <input id="reserve-date" name="date" type="date" required>
          <label for="reserve-time">Time</label>
          <select id="reserve-time" name="time" required>
            <option value="10:00">10:00</option>
            <option value="11:30">11:30</option>
            <option value="14:00">14:00</option>
            <option value="15:30">15:30</option>
            <option value="17:00">17:00</option>
          </select>
          <p class="price" id="reserve-price"></p>
          <button class="primary" type="submit">Pay</button>
          <p id="reserve-done" hidden></p>
        </section>
      </form>
    </div>
  </main>
  <script>
    const fileInput = document.getElementById("file");
    const photo = document.getElementById("photo");
    const preview = document.getElementById("preview");
    const analyze = document.getElementById("analyze");
    const status = document.getElementById("status");
    let objectUrl = "";

    function showStatus(message) {
      status.textContent = message;
      status.classList.toggle("show", Boolean(message));
    }

    function setScore(name, value) {
      const score = Number(value);
      document.getElementById(name).textContent = score.toFixed(1) + " / 10";
      document.getElementById(name + "-num").textContent = score.toFixed(1);
      const meter = document.getElementById(name + "-meter");
      meter.value = score;
      meter.style.setProperty("--fill", (score / 10) * 100 + "%");
    }

    function renderTips(summary, groups) {
      const box = document.getElementById("tips");
      const items = Array.isArray(groups) && groups.length ? groups : [{ title: "—", points: ["—"] }, { title: "—", points: ["—"] }];
      box.replaceChildren();
      const kicker = document.createElement("p");
      kicker.className = "kicker";
      kicker.textContent = "Recommended Routine";
      const lead = document.createElement("p");
      lead.id = "tips-summary";
      lead.textContent = summary || "Upload a photo and choose Analyze Skin. Tips will appear here.";
      box.append(kicker, lead);
      items.forEach((group, index) => {
        const heading = document.createElement("h3");
        heading.textContent = (index + 1) + ". " + (group.title || "—");
        const list = document.createElement("ul");
        list.className = "report-points";
        const points = Array.isArray(group.points) && group.points.length ? group.points : ["—"];
        points.forEach((point) => {
          const item = document.createElement("li");
          item.textContent = point;
          list.append(item);
        });
        box.append(heading, list);
      });
      const button = document.createElement("button");
      button.type = "button";
      button.className = "recommend-link";
      button.textContent = "What skin treatment program can improve...";
      box.append(button);
    }

    function resetResults() {
      document.getElementById("cleanliness").textContent = "—";
      document.getElementById("quality").textContent = "—";
      document.getElementById("age").textContent = "—";
      document.getElementById("description").textContent = "Upload a photo and choose Analyze Skin. The result will appear here.";
      document.getElementById("cleanliness-kpi").textContent = "—";
      document.getElementById("quality-kpi").textContent = "—";
      document.getElementById("cleanliness-review").textContent = "—";
      document.getElementById("quality-review").textContent = "—";
      renderTips();
      setScore("cleanliness", 0);
      setScore("quality", 0);
      document.getElementById("cleanliness").textContent = "—";
      document.getElementById("quality").textContent = "—";
    }

    function isImageFile(file) {
      return file.type.startsWith("image/") || /\\.(jpe?g|png|webp|gif)$/i.test(file.name);
    }

    function useFile(file) {
      if (!isImageFile(file)) {
        showStatus("Please use a JPG, PNG, WEBP, or GIF image.");
        return;
      }
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
      fileInput.dispatchEvent(new Event("change"));
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      photo.src = objectUrl;
      preview.classList.add("has-image");
      preview.classList.remove("dragover");
      analyze.disabled = false;
      showStatus("");
    });

    preview.addEventListener("click", (event) => {
      if (event.target.closest(".upload-btn") || event.target === fileInput) return;
      fileInput.click();
    });

    preview.addEventListener("dragenter", (event) => event.preventDefault());
    preview.addEventListener("dragover", (event) => {
      event.preventDefault();
      preview.classList.add("dragover");
    });
    preview.addEventListener("dragleave", (event) => {
      if (!preview.contains(event.relatedTarget)) preview.classList.remove("dragover");
    });
    preview.addEventListener("drop", (event) => {
      event.preventDefault();
      preview.classList.remove("dragover");
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) useFile(file);
    });

    document.getElementById("clear").addEventListener("click", () => {
      fileInput.value = "";
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = "";
      photo.removeAttribute("src");
      preview.classList.remove("has-image");
      analyze.disabled = true;
      showStatus("");
      resetResults();
    });

    analyze.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      analyze.disabled = true;
      showStatus("Analyzing skin…");
      const body = new FormData();
      body.append("image", file);
      try {
        const response = await fetch("/analyze", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Analysis failed.");
        setScore("cleanliness", data.cleanliness);
        setScore("quality", data.quality);
        document.getElementById("age").textContent = data.estimated_age;
        document.getElementById("description").textContent = data.description;
        document.getElementById("cleanliness-kpi").textContent = Number(data.cleanliness).toFixed(1) + " / 10";
        document.getElementById("cleanliness-review").textContent = data.cleanliness_review;
        document.getElementById("quality-kpi").textContent = Number(data.quality).toFixed(1) + " / 10";
        document.getElementById("quality-review").textContent = data.quality_review;
        renderTips(data.tips_summary, data.tips);
        showStatus("");
      } catch (error) {
        showStatus(error.message);
      } finally {
        analyze.disabled = false;
      }
    });

    const chart = document.getElementById("chart");
    let historyRecords = [];
    let selectedId = "";

    function setTab(name) {
      document.getElementById("home-view").hidden = name !== "home";
      document.getElementById("history-view").hidden = name !== "history";
      document.getElementById("recommend-view").hidden = name !== "recommend";
      document.getElementById("tab-home").classList.toggle("is-active", name === "home");
      document.getElementById("tab-history").classList.toggle("is-active", name === "history");
      document.getElementById("tab-recommend").classList.toggle("is-active", name === "recommend");
      if (name === "history") loadHistory();
      if (name === "recommend") openRecommend();
    }

    function svgNode(name, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
      return node;
    }

    function drawChart(records) {
      const width = Math.max(chart.clientWidth || 480, 320);
      const height = 320;
      const pad = { left: 36, right: 12, top: 16, bottom: 56 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      chart.setAttribute("viewBox", "0 0 " + width + " " + height);
      chart.replaceChildren();
      for (let score = 0; score <= 10; score += 2) {
        const y = pad.top + plotH - (score / 10) * plotH;
        chart.append(svgNode("line", { x1: pad.left, x2: pad.left + plotW, y1: y, y2: y, stroke: "#d7e6e3" }));
        const label = svgNode("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "#5f7a78", "font-size": "12" });
        label.textContent = String(score);
        chart.append(label);
      }
      if (!records.length) return;
      const xAt = (index) => records.length === 1 ? pad.left + plotW / 2 : pad.left + (index / (records.length - 1)) * plotW;
      const yAt = (score) => pad.top + plotH - (Number(score) / 10) * plotH;
      function addLine(key, color) {
        const d = records.map((record, index) => (index ? "L" : "M") + xAt(index) + " " + yAt(record[key])).join(" ");
        chart.append(svgNode("path", { d, fill: "none", stroke: color, "stroke-width": "2.5" }));
      }
      addLine("cleanliness", "#11D9A1");
      addLine("quality", "#1f6f8b");
      records.forEach((record, index) => {
        const x = xAt(index);
        ["cleanliness", "quality"].forEach((key) => {
          const point = svgNode("circle", {
            cx: x,
            cy: yAt(record[key]),
            r: record.id === selectedId ? 7 : 5,
            fill: key === "cleanliness" ? "#11D9A1" : "#1f6f8b",
            stroke: "#fff",
            "stroke-width": "2",
            style: "cursor:pointer",
          });
          point.addEventListener("click", () => selectRecord(record.id));
          chart.append(point);
        });
        const tick = svgNode("text", { x, y: height - 28, "text-anchor": "middle", fill: "#5f7a78", "font-size": "11" });
        tick.textContent = record.created_at.slice(5, 16).replace("T", " ");
        chart.append(tick);
      });
    }

    function fillTips(container, summary, groups) {
      container.replaceChildren();
      const kicker = document.createElement("p");
      kicker.className = "kicker";
      kicker.textContent = "Recommended Routine";
      const lead = document.createElement("p");
      lead.textContent = summary || "—";
      container.append(kicker, lead);
      (groups || []).forEach((group, index) => {
        const heading = document.createElement("h3");
        heading.textContent = (index + 1) + ". " + (group.title || "—");
        const list = document.createElement("ul");
        list.className = "report-points";
        (group.points && group.points.length ? group.points : ["—"]).forEach((point) => {
          const item = document.createElement("li");
          item.textContent = point;
          list.append(item);
        });
        container.append(heading, list);
      });
    }

    function showRecord(record) {
      document.getElementById("history-placeholder").hidden = true;
      document.getElementById("history-record").hidden = false;
      document.getElementById("history-time").textContent = record.created_at.replace("T", " ") + " · Cleanliness " + Number(record.cleanliness).toFixed(1) + " / 10 · Quality " + Number(record.quality).toFixed(1) + " / 10";
      document.getElementById("history-photo").src = record.image_url || ("/api/history/" + record.id + "/image");
      fillTips(document.getElementById("history-tips"), record.tips_summary, record.tips);
    }

    function selectRecord(id) {
      selectedId = id;
      drawChart(historyRecords);
      const record = historyRecords.find((item) => item.id === id);
      if (record) showRecord(record);
    }

    async function loadHistory() {
      const response = await fetch("/api/history");
      historyRecords = await response.json();
      document.getElementById("chart-empty").hidden = historyRecords.length > 0;
      if (selectedId && !historyRecords.some((record) => record.id === selectedId)) selectedId = "";
      drawChart(historyRecords);
    }

    const PROJECTS = [
      { id: "hydra", name: "HydraFacial Deep Cleanse", price: 980, effect: "Cleanliness jumps in the first two weeks, then levels off. Quality barely changes.", clean: [1.4, 1.7, 1.7], quality: [0.1, 0.2, 0.2] },
      { id: "acne", name: "Medical Acne Control", price: 1280, effect: "Blemishes settle week by week, so cleanliness keeps climbing.", clean: [0.4, 1.1, 2.0], quality: [0.1, 0.2, 0.4] },
      { id: "peel", name: "Clinical Chemical Peel", price: 1680, effect: "Quality dips while the skin recovers, then rises sharply.", clean: [0.1, 0.4, 0.5], quality: [-0.8, 1.2, 2.6] },
      { id: "laser", name: "Pigment Laser Toning", price: 2200, effect: "Spots stay put at first. Quality jumps late in the course.", clean: [0, 0.1, 0.2], quality: [0.1, 0.5, 3.0] },
      { id: "rf", name: "Radiofrequency Firming", price: 2580, effect: "Firmness builds slowly and is still improving at eight weeks.", clean: [0, 0.1, 0.1], quality: [0.4, 1.0, 1.8] },
      { id: "meso", name: "Skin Booster Mesotherapy", price: 1980, effect: "Hydration lifts cleanliness and quality together.", clean: [0.7, 1.2, 1.5], quality: [0.8, 1.6, 2.2] },
      { id: "led", name: "LED Repair Session", price: 480, effect: "Redness calms soon after the visit, then most of the gain fades.", clean: [0.9, 0.4, 0.1], quality: [0.2, 0.1, 0] }
    ];
    let reservedProject = null;

    function recommendProjects(record) {
      const cleanliness = Number(record.cleanliness);
      const quality = Number(record.quality);
      const priority = {
        hydra: cleanliness < 8 ? 90 : 40,
        acne: cleanliness < 7.5 ? 85 : 20,
        led: cleanliness < 8 ? 70 : 55,
        peel: quality < 6 ? 100 : quality < 7.5 ? 60 : 30,
        laser: quality < 6.5 ? 95 : 35,
        rf: quality < 7 ? 80 : 50,
        meso: quality < 8 ? 75 : 65
      };
      return PROJECTS.slice().sort((a, b) => priority[b.id] - priority[a.id] || a.price - b.price);
    }

    function forecast(start, lift) {
      const cap = (value) => Math.min(10, Math.max(0, Math.round(value * 10) / 10));
      return [cap(start), cap(start + lift[0]), cap(start + lift[1]), cap(start + lift[2])];
    }

    function drawPredict(record, project) {
      const svg = document.getElementById("predict-chart");
      const width = Math.max(svg.clientWidth || 480, 320);
      const height = 280;
      const pad = { left: 36, right: 44, top: 16, bottom: 44 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const labels = ["Now", "2 weeks", "4 weeks", "8 weeks"];
      const cleanliness = project ? forecast(Number(record.cleanliness), project.clean) : null;
      const quality = project ? forecast(Number(record.quality), project.quality) : null;
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      svg.replaceChildren();
      for (let score = 0; score <= 10; score += 2) {
        const y = pad.top + plotH - (score / 10) * plotH;
        svg.append(svgNode("line", { x1: pad.left, x2: pad.left + plotW, y1: y, y2: y, stroke: "#d7e6e3" }));
        const label = svgNode("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "#5f7a78", "font-size": "12" });
        label.textContent = String(score);
        svg.append(label);
      }
      const xAt = (index) => pad.left + (index / 3) * plotW;
      const yAt = (score) => pad.top + plotH - (score / 10) * plotH;
      function addLine(series, color) {
        const d = series.map((score, index) => (index ? "L" : "M") + xAt(index) + " " + yAt(score)).join(" ");
        svg.append(svgNode("path", { d, fill: "none", stroke: color, "stroke-width": "2.5", "stroke-dasharray": "6 4" }));
        series.forEach((score, index) => svg.append(svgNode("circle", { cx: xAt(index), cy: yAt(score), r: 4, fill: color })));
      }
      if (project) {
        addLine(cleanliness, "#11D9A1");
        addLine(quality, "#1f6f8b");
      }
      labels.forEach((text, index) => {
        const tick = svgNode("text", { x: xAt(index), y: height - 16, "text-anchor": "middle", fill: "#5f7a78", "font-size": "12" });
        tick.textContent = text;
        svg.append(tick);
      });
    }

    function renderRecommend() {
      const record = historyRecords.find((item) => item.id === selectedId);
      const list = document.getElementById("project-list");
      const form = document.getElementById("reserve-form");
      list.replaceChildren();
      form.hidden = true;
      reservedProject = null;
      document.getElementById("reserve-done").hidden = true;
      if (!record) {
        document.getElementById("recommend-context").textContent = "Choose a time on History first.";
        document.getElementById("predict-note").textContent = "";
        document.getElementById("predict-chart").replaceChildren();
        return;
      }
      document.getElementById("recommend-context").textContent = record.created_at.replace("T", " ") + " · Cleanliness " + Number(record.cleanliness).toFixed(1) + " / 10 · Quality " + Number(record.quality).toFixed(1) + " / 10";
      const projects = recommendProjects(record);
      projects.forEach((project) => {
        const card = document.createElement("article");
        card.className = "project-card";
        card.dataset.id = project.id;
        const title = document.createElement("h3");
        title.textContent = project.name;
        const effect = document.createElement("p");
        effect.textContent = project.effect;
        const price = document.createElement("p");
        price.className = "price";
        price.textContent = "HK$" + project.price;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "primary";
        button.textContent = "Reservation";
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          startReservation(record, project);
        });
        card.addEventListener("click", () => {
          showProjectChart(record, project);
          hideReservation();
        });
        card.append(title, effect, price, button);
        list.append(card);
      });
      drawPredict(record, null);
      document.getElementById("predict-note").textContent = "Click a project to see its predicted effect.";
    }

    function showProjectChart(record, project) {
      document.querySelectorAll(".project-card").forEach((card) => card.classList.toggle("is-selected", card.dataset.id === project.id));
      drawPredict(record, project);
      document.getElementById("predict-note").textContent = project.name + ". " + project.effect;
    }

    function hideReservation() {
      document.getElementById("reserve-form").hidden = true;
      reservedProject = null;
      document.getElementById("reserve-done").hidden = true;
    }

    function startReservation(record, project) {
      reservedProject = project;
      showProjectChart(record, project);
      const form = document.getElementById("reserve-form");
      form.hidden = false;
      document.getElementById("reserve-project").textContent = project.name;
      document.getElementById("reserve-price").textContent = "Amount to pay: HK$" + project.price;
      document.getElementById("reserve-done").hidden = true;
      const date = document.getElementById("reserve-date");
      if (!date.value) date.value = new Date().toISOString().slice(0, 10);
      form.scrollIntoView({ block: "nearest" });
    }

    function submitReservation(event) {
      event.preventDefault();
      if (!reservedProject) return;
      const date = document.getElementById("reserve-date").value;
      const time = document.getElementById("reserve-time").value;
      const message = "Reserved " + reservedProject.name + " on " + date + " at " + time + ". Payment of HK$" + reservedProject.price + " received.";
      alert(message);
    }

    async function openRecommend() {
      if (!historyRecords.length) {
        const response = await fetch("/api/history");
        historyRecords = await response.json();
      }
      if (!selectedId && historyRecords.length) selectedId = historyRecords[historyRecords.length - 1].id;
      renderRecommend();
    }

    document.getElementById("tab-home").addEventListener("click", () => setTab("home"));
    document.getElementById("tab-history").addEventListener("click", () => setTab("history"));
    document.getElementById("tab-recommend").addEventListener("click", () => setTab("recommend"));
    document.addEventListener("click", (event) => {
      if (event.target.closest(".recommend-link")) setTab("recommend");
    });
    document.getElementById("reserve-form").addEventListener("submit", submitReservation);
    window.addEventListener("resize", () => {
      if (!document.getElementById("history-view").hidden) drawChart(historyRecords);
    });
  </script>
</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    error = None
    register_error = None
    created = request.args.get("created") == "1"
    if request.method == "POST":
        action = request.form.get("action", "login")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()
        if action == "register":
            confirm = request.form.get("confirm", "")
            if not USERNAME_RE.fullmatch(username):
                register_error = "Username must be 1–32 characters: letters, numbers, _ or -."
            elif len(password) < 4:
                register_error = "Password must be at least 4 characters."
            elif password != confirm:
                register_error = "The two passwords do not match."
            elif username in users:
                register_error = "That username is already registered."
            else:
                users[username] = password
                save_users(users)
                return redirect(url_for("login", created=1))
        elif _login_matches(username, password):
            session.clear()
            session["user"] = username
            if username == "demo":
                session["sid"] = uuid.uuid4().hex
            return redirect(url_for("index"))
        else:
            error = "Incorrect username or password."
    return render_template_string(LOGIN_HTML, error=error, register_error=register_error, created=created)


@app.route("/logout", methods=["POST"])
def logout():
    if session.get("user") == "demo":
        forget_demo_session(session.get("sid"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(MAIN_HTML, user=session["user"])


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    image = request.files.get("image")
    if image is None or not image.filename:
        return jsonify(error="Please choose an image first."), 400

    ext = os.path.splitext(image.filename)[1].lower()
    mime = MIME_TYPES.get(ext)
    if mime is None:
        return jsonify(error="Please upload a JPG, PNG, WEBP, or GIF image."), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify(error="The selected image is empty."), 400

    try:
        raw = call_llm(image_bytes, mime)
        data = extract_json(raw)
        result = {
            "cleanliness": as_score(data["cleanliness"]),
            "quality": as_score(data["quality"]),
            "estimated_age": str(data["estimated_age"]),
            "description": str(data["description"]),
            "cleanliness_review": str(data["cleanliness_review"]),
            "quality_review": str(data["quality_review"]),
            "tips_summary": str(data["tips_summary"]),
            "tips": as_tips(data["tips"]),
            "treatment": as_points(data.get("treatment")),
        }
        save_history(session["user"], image_bytes, ext, result, session_id=demo_sid())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return jsonify(error="The model response could not be read as the expected skin report."), 502
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502

    return jsonify(result)


@app.route("/api/history")
@login_required
def history_api():
    username = safe_name(session.get("user"), r"[A-Za-z0-9_-]{1,32}")
    if username is None:
        return jsonify([])
    return jsonify(list_history(username, session.get("sid")))


@app.route("/api/history/<record_id>/image")
@login_required
def history_image(record_id):
    username = safe_name(session.get("user"), r"[A-Za-z0-9_-]{1,32}")
    record_id = safe_name(record_id, r"[0-9a-f]{32}")
    if username is None or record_id is None:
        return jsonify(error="Record not found."), 404
    record = find_history(username, record_id)
    if record is None or (record.get("session_id") and record.get("session_id") != session.get("sid")):
        return jsonify(error="Record not found."), 404
    image_path = os.path.join(HISTORY_DIR, username, record["image"])
    if not os.path.isfile(image_path):
        return jsonify(error="Record not found."), 404
    return send_file(image_path)


COMPONENT_DIR = os.path.join(tempfile.gettempdir(), "skin_check_component")
_SKIN_COMPONENT = None

SKIN_BRIDGE = r"""
<script>
(function () {
  function fitScreen() {
    let screenHeight = window.innerHeight;
    try {
      if (window.parent && window.parent.parent && window.parent.parent.innerHeight) {
        screenHeight = window.parent.parent.innerHeight;
      }
    } catch (error) {}
    document.documentElement.style.setProperty("--screen", Math.max(screenHeight, 640) + "px");
  }
  fitScreen();
  window.addEventListener("resize", fitScreen);
  const pending = {};
  const pageTag = Math.random().toString(36).slice(2);
  let seq = 0;
  function nextId() { return pageTag + "-" + (++seq); }
  const originalFetch = window.fetch.bind(window);
  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
  window.fetch = async function (url, options) {
    const raw = typeof url === "string" ? url : (url && url.url) || "";
    const path = raw.split("?")[0];
    if (path !== "/analyze" && path.indexOf("/api/") !== 0) return originalFetch(url, options);
    const id = nextId();
    let payload = {};
    const body = options && options.body;
    if (body instanceof FormData) {
      const file = body.get("image");
      payload = { filename: file.name, image: await fileToBase64(file) };
    }
    return new Promise((resolve) => {
      pending[id] = resolve;
      window.parent.postMessage({ source: "skin-app", id: id, path: path, payload: payload }, "*");
    });
  };
  window.addEventListener("message", (event) => {
    const reply = event.data && event.data.source === "skin-host" && event.data.reply;
    if (!reply || !pending[reply.id]) return;
    const resolve = pending[reply.id];
    delete pending[reply.id];
    resolve({
      ok: reply.status < 400,
      status: reply.status,
      json: async () => reply.body
    });
  });
  function readAccounts() {
    try {
      const data = JSON.parse(localStorage.getItem("skin_users") || "{}");
      return data && typeof data === "object" ? data : {};
    } catch (error) {
      return {};
    }
  }
  function keepRegisteredAccount() {
    let pendingAccount = null;
    try { pendingAccount = JSON.parse(localStorage.getItem("skin_pending_user") || "null"); } catch (error) {}
    const created = document.querySelector("#register-form .notice");
    const failed = document.querySelector("#register-form .error");
    if (pendingAccount && created && pendingAccount.username && pendingAccount.password) {
      const accounts = readAccounts();
      accounts[pendingAccount.username] = pendingAccount.password;
      try { localStorage.setItem("skin_users", JSON.stringify(accounts)); } catch (error) {}
      try { localStorage.removeItem("skin_pending_user"); } catch (error) {}
    } else if (failed) {
      try { localStorage.removeItem("skin_pending_user"); } catch (error) {}
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", keepRegisteredAccount);
  else keepRegisteredAccount();
  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!form || !form.getAttribute) return;
    const action = form.getAttribute("action") || "";
    const isAccount = form.id === "login-form" || form.id === "register-form";
    const isLogout = action.indexOf("logout") >= 0;
    if (!isAccount && !isLogout) return;
    event.preventDefault();
    const payload = {};
    new FormData(form).forEach((value, key) => { payload[key] = value; });
    if (form.id === "register-form") {
      try {
        localStorage.setItem("skin_pending_user", JSON.stringify({
          username: String(payload.username || "").trim(),
          password: String(payload.password || "")
        }));
      } catch (error) {}
    }
    if (form.id === "login-form") payload.saved_users = readAccounts();
    const id = nextId();
    window.parent.postMessage({
      source: "skin-app",
      id: id,
      path: isLogout ? "/logout" : "/login",
      payload: payload
    }, "*");
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("#back-login");
    if (!button) return;
    const form = document.getElementById("register-form");
    if (!form || !form.querySelector(".notice, .error")) return;
    window.parent.postMessage({
      source: "skin-app",
      id: nextId(),
      path: "/login-back",
      payload: {}
    }, "*");
  });
})();
</script>
"""

COMPONENT_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>html,body{margin:0;background:#f3f7f6}iframe{width:100%;border:0;display:block}</style>
</head>
<body>
  <iframe id="app" title="Skin Quality Check"></iframe>
  <script>
    function send(type, data) {
      const message = { isStreamlitMessage: true, type: type, apiVersion: 1 };
      Object.keys(data || {}).forEach((key) => { message[key] = data[key]; });
      window.parent.postMessage(message, "*");
    }
    const frame = document.getElementById("app");
    let screen = "";
    let version = "";
    let lastReply = "";
    let lastHeight = 0;
    let resizing = false;
    function viewportHeight() {
      return Math.max((window.parent && window.parent.innerHeight) || window.innerHeight || 800, 640);
    }
    function resize() {
      if (resizing) return;
      resizing = true;
      const viewport = viewportHeight();
      let height = viewport;
      try {
        const doc = frame.contentDocument;
        if (doc && doc.body && screen !== "login") {
          const previous = doc.body.style.minHeight;
          doc.body.style.minHeight = "0px";
          height = Math.max(doc.body.scrollHeight, viewport);
          doc.body.style.minHeight = previous;
        }
      } catch (error) {}
      frame.style.height = height + "px";
      if (Math.abs(height - lastHeight) >= 2) {
        lastHeight = height;
        send("streamlit:setFrameHeight", { height: height });
      }
      resizing = false;
    }
    frame.addEventListener("load", () => {
      lastHeight = 0;
      resize();
      try {
        new ResizeObserver(() => resize()).observe(frame.contentDocument.body);
      } catch (error) {}
    });
    window.addEventListener("resize", resize);
    window.addEventListener("message", (event) => {
      const data = event.data || {};
      if (data.source === "skin-app") {
        send("streamlit:setComponentValue", { value: data, dataType: "json" });
        return;
      }
      if (data.type !== "streamlit:render") return;
      const args = data.args || {};
      const pages = { login: args.login_html || "", app: args.main_html || "" };
      if (args.screen !== screen || args.version !== version) {
        screen = args.screen;
        version = args.version || "";
        lastHeight = 0;
        if (screen === "login") {
          const viewport = viewportHeight();
          frame.style.height = viewport + "px";
          send("streamlit:setFrameHeight", { height: viewport });
          lastHeight = viewport;
        }
        frame.srcdoc = pages[screen] || pages.login;
      }
      if (args.reply && args.reply.id && args.reply.id !== lastReply) {
        lastReply = args.reply.id;
        if (frame.contentWindow) frame.contentWindow.postMessage({ source: "skin-host", reply: args.reply }, "*");
      }
      setTimeout(resize, 50);
    });
    send("streamlit:componentReady", { apiVersion: 1 });
  </script>
</body>
</html>
"""


def _skin_component():
    global _SKIN_COMPONENT
    import streamlit.components.v1 as components

    os.makedirs(COMPONENT_DIR, exist_ok=True)
    target = os.path.join(COMPONENT_DIR, "index.html")
    current = ""
    if os.path.isfile(target):
        with open(target, encoding="utf-8") as handle:
            current = handle.read()
    if current != COMPONENT_PAGE:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(COMPONENT_PAGE)
    if _SKIN_COMPONENT is None:
        _SKIN_COMPONENT = components.declare_component("skin_check", path=COMPONENT_DIR)
    return _SKIN_COMPONENT


def _page_html(screen, user, login_error, register_error, created):
    with app.test_request_context():
        if screen == "app":
            html = render_template_string(MAIN_HTML, user=user)
        else:
            html = render_template_string(
                LOGIN_HTML,
                error=login_error,
                register_error=register_error,
                created=created,
            )
    return html.replace("<head>", "<head>" + SKIN_BRIDGE, 1)


def _history_payload(username, session_id):
    records = []
    for record in list_history(username, session_id):
        item = dict(record)
        image_path = os.path.join(HISTORY_DIR, username, record.get("image") or "")
        if os.path.isfile(image_path):
            ext = os.path.splitext(image_path)[1].lower()
            mime = MIME_TYPES.get(ext, "image/png")
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("ascii")
            item["image_url"] = f"data:{mime};base64,{encoded}"
        records.append(item)
    return records


def _handle_skin_event(event):
    path = event.get("path") or ""
    payload = event.get("payload") or {}
    if path == "/logout":
        if st_user() == "demo":
            forget_demo_session(st_sid())
        return {"screen": "login", "login_error": None, "register_error": None}
    if path == "/login-back":
        return {"login_error": None, "register_error": None}
    if path == "/login":
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        if payload.get("action") == "register":
            confirm = str(payload.get("confirm") or "")
            users = load_users()
            if not USERNAME_RE.fullmatch(username):
                return {"register_error": "Username must be 1–32 characters: letters, numbers, _ or -."}
            if len(password) < 4:
                return {"register_error": "Password must be at least 4 characters."}
            if password != confirm:
                return {"register_error": "The two passwords do not match."}
            if username in users:
                return {"register_error": "That username is already registered.", "login_error": None}
            users[username] = password
            save_users(users)
            return {"created": True, "register_error": None, "login_error": None}
        remembered = _remembered_password(payload, username)
        if _login_matches(username, password, remembered):
            return {"user": username, "screen": "app", "sid": uuid.uuid4().hex if username == "demo" else st_sid(), "login_error": None, "register_error": None}
        return {"login_error": "Incorrect username or password.", "register_error": None}
    if path == "/analyze":
        if not st_user():
            return {"reply": {"id": event.get("id"), "status": 401, "body": {"error": "Please sign in first."}}}
        try:
            encoded = payload.get("image") or ""
            image_bytes = base64.b64decode(encoded)
            ext = os.path.splitext(str(payload.get("filename") or ""))[1].lower()
            mime = MIME_TYPES.get(ext)
            if mime is None or not image_bytes:
                raise RuntimeError("Please upload a JPG, PNG, WEBP, or GIF image.")
            data = extract_json(call_llm(image_bytes, mime))
            result = {
                "cleanliness": as_score(data["cleanliness"]),
                "quality": as_score(data["quality"]),
                "estimated_age": str(data["estimated_age"]),
                "description": str(data["description"]),
                "cleanliness_review": str(data["cleanliness_review"]),
                "quality_review": str(data["quality_review"]),
                "tips_summary": str(data["tips_summary"]),
                "tips": as_tips(data["tips"]),
                "treatment": as_points(data.get("treatment")),
            }
            save_history(
                st_user(),
                image_bytes,
                ext,
                result,
                session_id=st_sid() if st_user() == "demo" else None,
            )
            return {"reply": {"id": event.get("id"), "status": 200, "body": result}}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"reply": {"id": event.get("id"), "status": 502, "body": {"error": "The model response could not be read as the expected skin report."}}}
        except RuntimeError as exc:
            return {"reply": {"id": event.get("id"), "status": 502, "body": {"error": str(exc)}}}
    if path == "/api/history":
        if not st_user():
            return {"reply": {"id": event.get("id"), "status": 401, "body": {"error": "Please sign in first."}}}
        return {"reply": {"id": event.get("id"), "status": 200, "body": _history_payload(st_user(), st_sid())}}
    return None


def st_user():
    import streamlit as st

    return st.session_state.get("skin_user")


def st_sid():
    import streamlit as st

    return st.session_state.get("skin_sid") or ""


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Skin Quality Check", layout="wide")
    st.markdown(
        """
        <style>
          header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
          .block-container { padding: 0 !important; max-width: 100% !important; }
          [data-testid="stAppViewContainer"] { background: #f3f7f6; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("skin_user", None)
    st.session_state.setdefault("skin_sid", uuid.uuid4().hex)
    st.session_state.setdefault("login_error", None)
    st.session_state.setdefault("register_error", None)
    st.session_state.setdefault("created", False)
    st.session_state.setdefault("handled_id", "")
    screen = "app" if st.session_state.skin_user else "login"
    version = "|".join([
        screen,
        st.session_state.skin_user or "",
        st.session_state.login_error or "",
        st.session_state.register_error or "",
        "1" if st.session_state.created else "0",
    ])
    event = _skin_component()(
        screen=screen,
        version=version,
        reply=st.session_state.pop("skin_reply", None),
        login_html=_page_html("login", "", st.session_state.login_error, st.session_state.register_error, st.session_state.created),
        main_html=_page_html("app", st.session_state.skin_user or "", None, None, False),
        key="skin",
        default=None,
    )
    if not event or event.get("id") == st.session_state.handled_id:
        return
    st.session_state.handled_id = event.get("id")
    outcome = _handle_skin_event(event) or {}
    if "user" in outcome:
        st.session_state.skin_user = outcome["user"]
        st.session_state.login_error = None
        st.session_state.created = False
    if outcome.get("screen") == "login":
        st.session_state.skin_user = None
        st.session_state.created = False
    if "sid" in outcome:
        st.session_state.skin_sid = outcome["sid"]
    if "login_error" in outcome:
        st.session_state.login_error = outcome["login_error"]
    if "register_error" in outcome:
        st.session_state.register_error = outcome["register_error"]
        st.session_state.created = False
    if outcome.get("created"):
        st.session_state.created = True
        st.session_state.register_error = None
        st.session_state.login_error = None
    if "reply" in outcome:
        st.session_state.skin_reply = outcome["reply"]
    st.rerun()


def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        try:
            context = get_script_run_ctx(suppress_warning=True)
        except TypeError:
            context = get_script_run_ctx()
        return context is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_streamlit_app()


if __name__ == "__main__":
    if not _running_in_streamlit():
        app.run(host="127.0.0.1", port=5000, debug=True)
