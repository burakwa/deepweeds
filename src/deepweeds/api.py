"""DeepWeeds FastAPI servisi.

Endpoints
---------
GET  /            → Tarayıcı demo arayüzü
GET  /health      → Model ve cihaz bilgisi
POST /predict     → Tek görüntü sınıflandırma
POST /predict/batch → Çoklu görüntü (max 16)
GET  /classes     → Sınıf adları listesi
"""
from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from deepweeds import __version__
from deepweeds.inference import Predictor
from deepweeds.utils import get_logger

log = get_logger()

BATCH_LIMIT = 16  # tek istekte kabul edilen maksimum görüntü sayısı


# ---------------------------------------------------------------------------
# Pydantic şemaları
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    filename: str
    top_class: str
    top_prob: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_latency_ms: float
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint: str
    device: str
    classes: list[str]
    version: str


class ClassesResponse(BaseModel):
    classes: list[str]
    count: int


# ---------------------------------------------------------------------------
# Servis katmanı
# ---------------------------------------------------------------------------

class ModelService:
    """Predictor üzerinde thread-güvenli ince sarmalayıcı."""

    def __init__(self, predictor: Predictor):
        self._predictor = predictor
        self._lock = threading.Lock()

    def _predict_image(self, filename: str, data: bytes) -> PredictionResponse:
        """Ham bytes'tan PredictionResponse üretir. Lock dışından çağrılır."""
        try:
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as exc:
            raise ValueError(f"'{filename}' geçerli bir görüntü değil.") from exc

        t0 = time.perf_counter()
        probs = self._predictor.predict_pil(image)
        ms = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs))
        return PredictionResponse(
            filename=filename,
            top_class=self._predictor.class_names[idx],
            top_prob=round(float(probs[idx]), 6),
            probabilities={c: round(float(p), 6)
                           for c, p in zip(self._predictor.class_names, probs)},
            latency_ms=round(ms, 2),
        )

    def predict_single(self, filename: str, data: bytes) -> PredictionResponse:
        with self._lock:
            return self._predict_image(filename, data)

    def predict_batch(self, files: list[tuple[str, bytes]]) -> BatchPredictionResponse:
        t0 = time.perf_counter()
        results = []
        with self._lock:
            for filename, data in files:
                results.append(self._predict_image(filename, data))
        return BatchPredictionResponse(
            results=results,
            total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            count=len(results),
        )


# ---------------------------------------------------------------------------
# Demo HTML (inline, bağımsız)
# ---------------------------------------------------------------------------

DEMO_HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepWeeds Demo</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container { max-width: 760px; margin: 0 auto; }
    h1 { font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin-bottom: .4rem; }
    .subtitle { color: #94a3b8; font-size: .95rem; margin-bottom: 2rem; }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .drop-zone {
      border: 2px dashed #334155;
      border-radius: 8px;
      padding: 2.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: #38bdf8;
      background: #0f172a;
    }
    .drop-zone input { display: none; }
    .drop-zone p { color: #64748b; font-size: .9rem; margin-top: .5rem; }
    #preview {
      max-width: 100%;
      max-height: 280px;
      border-radius: 8px;
      margin: 1rem auto;
      display: none;
    }
    .btn {
      display: inline-block;
      padding: .65rem 1.6rem;
      background: #0ea5e9;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background .2s, transform .1s;
    }
    .btn:hover { background: #38bdf8; }
    .btn:active { transform: scale(.97); }
    .btn:disabled { background: #334155; cursor: not-allowed; }
    #result { margin-top: 1.5rem; }
    .top-label {
      font-size: 1.3rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 1rem;
    }
    .bar-row { margin-bottom: 8px; }
    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: .85rem;
      color: #cbd5e1;
      margin-bottom: 3px;
    }
    .bar-track {
      height: 8px;
      background: #334155;
      border-radius: 4px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #0ea5e9, #38bdf8);
      border-radius: 4px;
      transition: width .5s ease;
    }
    .meta { color: #64748b; font-size: .8rem; margin-top: 1rem; }
    .spinner {
      border: 3px solid #334155;
      border-top-color: #38bdf8;
      border-radius: 50%;
      width: 28px; height: 28px;
      animation: spin .7s linear infinite;
      display: none;
      margin: 1rem auto;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div class="container">
  <h1>🌿 DeepWeeds Sınıflandırıcı</h1>
  <p class="subtitle">Tarla fotoğrafı yükleyin; model 9 yabani ot sınıfı için olasılık üretir.</p>
  <div class="card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <svg width="40" height="40" fill="none" stroke="#38bdf8" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
      </svg>
      <p>Görüntüyü buraya sürükleyin veya tıklayarak seçin</p>
      <input type="file" id="fileInput" accept="image/*">
    </div>
    <img id="preview" alt="Önizleme">
    <div style="text-align:center;margin-top:1rem">
      <button class="btn" id="predictBtn" onclick="predict()" disabled>Tahmin Et</button>
    </div>
    <div class="spinner" id="spinner"></div>
    <div id="result"></div>
  </div>
</div>
<script>
  const fileInput = document.getElementById('fileInput');
  const preview   = document.getElementById('preview');
  const predictBtn = document.getElementById('predictBtn');
  const dropZone  = document.getElementById('dropZone');
  const spinner   = document.getElementById('spinner');

  fileInput.addEventListener('change', () => loadFile(fileInput.files[0]));

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });

  function loadFile(f) {
    if (!f) return;
    const url = URL.createObjectURL(f);
    preview.src = url;
    preview.style.display = 'block';
    predictBtn.disabled = false;
    document.getElementById('result').innerHTML = '';
  }

  async function predict() {
    const f = fileInput.files[0];
    if (!f) { alert('Önce bir dosya seçin'); return; }
    predictBtn.disabled = true;
    spinner.style.display = 'block';
    document.getElementById('result').innerHTML = '';
    try {
      const fd = new FormData();
      fd.append('file', f);
      const r = await fetch('/predict', { method: 'POST', body: fd });
      if (!r.ok) { alert('Sunucu hatası: ' + r.status); return; }
      const j = await r.json();
      renderResult(j);
    } catch(e) {
      alert('İstek başarısız: ' + e.message);
    } finally {
      predictBtn.disabled = false;
      spinner.style.display = 'none';
    }
  }

  function renderResult(j) {
    const items = Object.entries(j.probabilities).sort((a, b) => b[1] - a[1]);
    let html = `<div class="top-label">🏆 ${j.top_class} &mdash; ${(j.top_prob*100).toFixed(1)}%</div>`;
    for (const [cls, p] of items) {
      const pct = (p * 100).toFixed(1);
      html += `
        <div class="bar-row">
          <div class="bar-label"><span>${cls}</span><span>${pct}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    }
    html += `<p class="meta">Gecikme: ${j.latency_ms} ms | Dosya: ${j.filename}</p>`;
    document.getElementById('result').innerHTML = html;
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Uygulama fabrikası
# ---------------------------------------------------------------------------

def create_app(model_path: str | Path, device: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("Model yükleniyor: %s", model_path)
        predictor = Predictor.from_checkpoint(model_path, device)
        app.state.service = ModelService(predictor)
        log.info("Hazır — sınıflar: %s | cihaz: %s", predictor.class_names, predictor.device)
        yield
        log.info("Uygulama kapatılıyor.")

    app = FastAPI(
        title="DeepWeeds API",
        version=__version__,
        description="Yabani ot sınıflandırma servisi. 9 sınıf, timm tabanlı.",
        lifespan=lifespan,
    )

    # CORS — geliştirme ortamı için tüm kaynaklara izin ver
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- Rotalar ---

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return DEMO_HTML

    @app.get("/health", response_model=HealthResponse, summary="Servis sağlık kontrolü")
    def health() -> HealthResponse:
        p: Predictor = app.state.service._predictor
        return HealthResponse(
            status="ok",
            model=p.meta["config"]["model"]["name"],
            checkpoint=str(p.checkpoint_path),
            device=str(p.device),
            classes=p.class_names,
            version=__version__,
        )

    @app.get("/classes", response_model=ClassesResponse, summary="Desteklenen sınıflar")
    def classes() -> ClassesResponse:
        p: Predictor = app.state.service._predictor
        return ClassesResponse(classes=p.class_names, count=len(p.class_names))

    @app.post("/predict", response_model=PredictionResponse, summary="Tek görüntü sınıflandır")
    async def predict(file: Annotated[UploadFile, File(description="Sınıflandırılacak görüntü")]) -> PredictionResponse:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Boş dosya gönderildi.")
        try:
            return app.state.service.predict_single(file.filename or "upload.jpg", data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/predict/batch", response_model=BatchPredictionResponse,
              summary=f"Çoklu görüntü sınıflandır (max {BATCH_LIMIT})")
    async def predict_batch(
        files: Annotated[list[UploadFile], File(description=f"Görüntü listesi (max {BATCH_LIMIT})")]
    ) -> BatchPredictionResponse:
        if len(files) > BATCH_LIMIT:
            raise HTTPException(400, f"En fazla {BATCH_LIMIT} görüntü gönderilebilir.")
        if not files:
            raise HTTPException(400, "En az 1 görüntü gereklidir.")
        pairs: list[tuple[str, bytes]] = []
        for uf in files:
            raw = await uf.read()
            if not raw:
                raise HTTPException(400, f"'{uf.filename}' boş dosya.")
            pairs.append((uf.filename or "upload.jpg", raw))
        try:
            return app.state.service.predict_batch(pairs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app