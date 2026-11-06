"""
面部皮肤质量检查 — 本地单文件 Web 应用（Flask）。

分析功能调用香港浸会大学 GenAI 接口（部署名已包含在 URL 中）。
环境变量 LLM_API_URL / LLM_API_KEY 若已设置，会覆盖下面的默认值。
演示账号：用户名 demo，密码 demo。
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from functools import wraps

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for

# 课程提供的 GenAI 接口。部署模型是 gemini-2.5-pro，写在 URL 路径里。
LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://genai.hkbu.edu.hk/api/v0/rest/deployments/gemini-2.5-pro/chat/completions?api-version=v1",
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "13406d59-bab5-4d4e-9e9b-bc6b126f73b9")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-pro")

DEMO_USERS = {"demo": "demo"}

PROMPT = """You assess facial skin from a frontal photo for a skincare quality-check demo.
Return only a JSON object, no markdown, with these keys:
- cleanliness: number from 0 to 10 with one decimal. How clean the skin looks.
- quality: number from 0 to 10 with one decimal. Overall skin quality, including texture, tone, lines, and spots.
- estimated_age: string, an age or age range such as "55-65".
- description: 2 to 4 sentences describing the facial skin condition.
- cleanliness_review: one sentence explaining the cleanliness score.
- quality_review: one sentence explaining the quality score.
If the image is not a clear frontal face, still return this JSON and say so in the description.
"""

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
            if request.path.startswith("/analyze"):
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


def call_llm(image_bytes, mime):
    if not LLM_API_URL or not LLM_API_KEY:
        raise RuntimeError(
            "LLM API is not configured. Set LLM_API_URL and LLM_API_KEY at the top of app.py."
        )

    payload = {
        "model": LLM_MODEL,
        "temperature": 0.2,
        "max_tokens": 800,
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
    form { width: min(420px, calc(100% - 32px)); background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 32px 28px; box-shadow: 0 10px 30px rgba(19, 78, 74, 0.06); }
    h1 { margin: 0 0 6px; font-size: 24px; }
    p.lead { margin: 0 0 22px; color: var(--muted); }
    label { display: block; margin: 14px 0 6px; font-size: 14px; font-weight: 600; }
    input { width: 100%; height: 44px; border: 1px solid var(--line); border-radius: 10px; padding: 0 12px; font: inherit; }
    input:focus { outline: 2px solid rgba(18, 153, 140, 0.25); border-color: var(--teal); }
    button { width: 100%; height: 46px; margin-top: 20px; border: 0; border-radius: 12px; background: var(--teal); color: #fff; font: inherit; font-weight: 700; cursor: pointer; }
    button:hover { background: #0e857a; }
    .error { margin: 0 0 8px; padding: 10px 12px; border-radius: 10px; background: #fff1f1; color: #9f1239; }
    .hint { margin: 16px 0 0; color: var(--muted); font-size: 13px; text-align: center; }
  </style>
</head>
<body>
  <form method="post">
    <h1>Skin Quality Check</h1>
    <p class="lead">Sign in to upload a frontal face photo.</p>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <label for="username">Username</label>
    <input id="username" name="username" autocomplete="username" required>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
    <p class="hint">Demo account: username <strong>demo</strong>, password <strong>demo</strong></p>
  </form>
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
    .preview { position: relative; height: 360px; border-radius: 16px; background: #123; overflow: hidden; cursor: pointer; }
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
    @media (max-width: 860px) { main, .metrics, .meters { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Skin Quality Check</h1>
      <p>Signed in as {{ user }}</p>
    </div>
    <form method="post" action="{{ url_for('logout') }}"><button type="submit">Log out</button></form>
  </header>
  <main>
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

    function resetResults() {
      document.getElementById("cleanliness").textContent = "—";
      document.getElementById("quality").textContent = "—";
      document.getElementById("age").textContent = "—";
      document.getElementById("description").textContent = "Upload a photo and choose Analyze Skin. The result will appear here.";
      document.getElementById("cleanliness-kpi").textContent = "—";
      document.getElementById("quality-kpi").textContent = "—";
      document.getElementById("cleanliness-review").textContent = "—";
      document.getElementById("quality-review").textContent = "—";
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
        showStatus("");
      } catch (error) {
        showStatus(error.message);
      } finally {
        analyze.disabled = false;
      }
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
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if DEMO_USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("index"))
        error = "Incorrect username or password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout", methods=["POST"])
def logout():
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
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return jsonify(error="The model response could not be read as the expected skin report."), 502
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
