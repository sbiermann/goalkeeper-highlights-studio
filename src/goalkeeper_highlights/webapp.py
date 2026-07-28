from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .config import load_config
from .pipeline import run

app = FastAPI(title="Goalkeeper Highlights Studio", version="0.2.0")
JOBS: dict[str, dict] = {}

class AnalyzeRequest(BaseModel):
    video: str
    output: str | None = None
    overwrite: bool = False
    qwen: bool = False
    live_preview: bool = False
    frame_stride: int = 2


def _execute(job_id: str, request: AnalyzeRequest):
    try:
        video = Path(request.video)
        if not video.exists(): raise FileNotFoundError(video)
        output = Path(request.output) if request.output else video.with_name(video.stem + "_goalkeeper_highlights")
        config = load_config(None)
        config["classification"]["enabled"] = request.qwen
        config["runtime"]["live_preview"] = request.live_preview
        config["yolo"]["frame_stride"] = max(1, request.frame_stride)
        def progress(value: float, message: str): JOBS[job_id].update(progress=value, message=message)
        progress(0.0, "Initialisiere Analyse")
        run(video, output, config, request.overwrite, progress_callback=progress)
        JOBS[job_id].update(status="finished", progress=1.0, output=str(output), report=str(output / "report.html"))
    except Exception as exc:
        JOBS[job_id].update(status="failed", message=str(exc))

@app.post("/api/jobs")
def create_job(request: AnalyzeRequest):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "status": "running", "progress": 0.0, "message": "Starte Analyse"}
    threading.Thread(target=_execute, args=(job_id, request), daemon=True).start()
    return JOBS[job_id]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS: raise HTTPException(404, "Job nicht gefunden")
    return JOBS[job_id]

@app.get("/api/report")
def report(path: str):
    report_path = Path(path).resolve()
    if not report_path.exists() or report_path.name != "report.html": raise HTTPException(404)
    return FileResponse(report_path)

@app.get("/", response_class=HTMLResponse)
def index():
    return r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Goalkeeper Highlights Studio</title>
  <style>
    body{font:16px system-ui;background:#0d1117;color:#e6edf3;max-width:900px;margin:40px auto;padding:20px}
    form,.card{background:#161b22;padding:24px;border:1px solid #30363d;border-radius:14px}
    label{display:block;margin:14px 0 6px}
    input[type=text]{width:100%;padding:12px;box-sizing:border-box;background:#0d1117;color:white;border:1px solid #30363d;border-radius:8px}
    .check{display:flex;align-items:center;gap:8px;margin-top:14px}
    button{margin-top:20px;padding:12px 18px;border:0;border-radius:8px;font-weight:700;cursor:pointer}
    button:disabled{opacity:.55;cursor:not-allowed}
    .card{margin-top:18px}
    .bar{height:14px;background:#30363d;border-radius:8px;overflow:hidden;margin-top:20px}
    .fill{height:100%;background:#2ea043;width:0;transition:width .3s}
    .error{color:#ff7b72;white-space:pre-wrap}
    .hint{color:#8b949e;font-size:.92rem}
  </style>
</head>
<body>
  <h1>Goalkeeper Highlights Studio</h1>
  <p>Lokale Videoanalyse mit YOLO, ByteTrack, optional Qwen und FFmpeg.</p>
  <form id="analysisForm">
    <label for="videoPath">Videopfad</label>
    <input id="videoPath" type="text" value="C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4">
    <label for="outputPath">Ausgabeordner (optional)</label>
    <input id="outputPath" type="text" placeholder="Leer lassen für automatischen Ordner neben dem Video">
    <label class="check"><input id="qwenEnabled" type="checkbox"> Qwen-Klassifikation aktivieren</label>
    <label class="check"><input id="previewEnabled" type="checkbox"> Live-Vorschau anzeigen</label>
    <label class="check"><input id="overwriteEnabled" type="checkbox"> Vorhandene Ausgabedateien überschreiben</label>
    <button id="startButton" type="submit">Analyse starten</button>
  </form>
  <section class="card" id="statusCard" hidden>
    <strong id="statusMessage">Starte Analyse ...</strong>
    <div class="bar"><div class="fill" id="progressFill"></div></div>
    <p id="progressText">0 %</p>
    <p class="hint" id="selectionHint" hidden>Es sollte sich ein separates OpenCV-Fenster öffnen. Dort den Torwart anklicken und anschließend Enter drücken.</p>
    <p id="resultText"></p>
  </section>
<script>
const form = document.getElementById('analysisForm');
const statusCard = document.getElementById('statusCard');
const startButton = document.getElementById('startButton');
const statusMessage = document.getElementById('statusMessage');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultText = document.getElementById('resultText');
const selectionHint = document.getElementById('selectionHint');
let pollTimer = null;

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (pollTimer) clearInterval(pollTimer);
  startButton.disabled = true;
  statusCard.hidden = false;
  selectionHint.hidden = !document.getElementById('previewEnabled').checked;
  statusMessage.textContent = 'Starte Analyse ...';
  progressFill.style.width = '0%';
  progressText.textContent = '0 %';
  resultText.textContent = '';
  resultText.className = '';

  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      headers: {'content-type':'application/json'},
      body: JSON.stringify({
        video: document.getElementById('videoPath').value,
        output: document.getElementById('outputPath').value || null,
        qwen: document.getElementById('qwenEnabled').checked,
        live_preview: document.getElementById('previewEnabled').checked,
        overwrite: document.getElementById('overwriteEnabled').checked,
        frame_stride: 2
      })
    });
    if (!response.ok) throw new Error(await response.text());
    const job = await response.json();

    const poll = async () => {
      try {
        const jobResponse = await fetch('/api/jobs/' + job.id);
        if (!jobResponse.ok) throw new Error(await jobResponse.text());
        const current = await jobResponse.json();
        const percent = Math.max(0, Math.min(100, Math.round((current.progress || 0) * 100)));
        progressFill.style.width = percent + '%';
        progressText.textContent = percent + ' %';
        statusMessage.textContent = current.message || current.status;
        if (current.status === 'finished') {
          clearInterval(pollTimer); pollTimer = null; startButton.disabled = false; selectionHint.hidden = true;
          resultText.textContent = 'Fertig. Ausgabe: ' + current.output;
        } else if (current.status === 'failed') {
          clearInterval(pollTimer); pollTimer = null; startButton.disabled = false; selectionHint.hidden = true;
          resultText.className = 'error';
          resultText.textContent = 'Fehler: ' + (current.message || 'Unbekannter Fehler');
        }
      } catch (error) {
        clearInterval(pollTimer); pollTimer = null; startButton.disabled = false;
        resultText.className = 'error'; resultText.textContent = 'Statusabfrage fehlgeschlagen: ' + error.message;
      }
    };
    await poll();
    pollTimer = setInterval(poll, 1000);
  } catch (error) {
    startButton.disabled = false;
    resultText.className = 'error';
    resultText.textContent = 'Start fehlgeschlagen: ' + error.message;
  }
});
</script>
</body>
</html>"""
