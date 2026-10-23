from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from deepweeds import __version__
from deepweeds.inference import Predictor
from deepweeds.utils import get_logger

log = get_logger()


class PredictionResponse(BaseModel):
    filename: str
    top_class: str
    top_prob: float
    probabilities: dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint: str
    device: str
    classes: list[str]


class ModelService:
    """Predictor üzerinde thread-güvenli ince sarmalayıcı."""

    def __init__(self, predictor: Predictor):
        self._predictor = predictor
        self._lock = threading.Lock()

    def predict(self, filename: str, data: bytes) -> PredictionResponse:
        try:
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as exc:
            raise ValueError("Dosya geçerli bir görüntü değil.") from exc
        with self._lock:
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


DEMO_HTML = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>DeepWeeds Demo</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1c2b33}
button{padding:.5rem 1rem;border:0;background:#1a7f37;color:#fff;border-radius:6px;cursor:pointer}
.bar{height:10px;background:#1a7f37;border-radius:5px;margin:2px 0 10px}
.row{margin:6px 0}.muted{color:#5a6a72;font-size:.9rem}</style></head><body>
<h1>DeepWeeds Sınıflandırıcı</h1>
<p class="muted">Tarla fotoğrafı yükleyin; model 9 sınıf için olasılık üretir (yerel, internet gerektirmez).</p>
<input type="file" id="file" accept="image/*"> <button onclick="send()">Tahmin et</button>
<div id="result"></div>
<script>
async function send(){
  const f=document.getElementById('file').files[0];
  if(!f){alert('Önce bir dosya seçin');return}
  const fd=new FormData(); fd.append('file',f);
  const r=await fetch('/predict',{method:'POST',body:fd});
  if(!r.ok){alert('Hata '+r.status);return}
  const j=await r.json();
  const items=Object.entries(j.probabilities).sort((a,b)=>b[1]-a[1]);
  document.getElementById('result').innerHTML=
    '<p><b>'+j.top_class+'</b> — '+(j.top_prob*100).toFixed(1)+'% ('+j.latency_ms+' ms)</p>'+
    items.map(([c,p])=>'<div class="row">'+c+' — '+(p*100).toFixed(1)+
      '%<div class="bar" style="width:'+(p*100)+'%"></div></div>').join('');
}
</script></body></html>"""


def create_app(model_path: str | Path, device: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("Model yükleniyor: %s", model_path)
        predictor = Predictor.from_checkpoint(model_path, device)
        app.state.service = ModelService(predictor)
        log.info("Hazır: %s", predictor.class_names)
        yield

    app = FastAPI(title="DeepWeeds API", version=__version__, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DEMO_HTML

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p: Predictor = app.state.service._predictor
        return HealthResponse(status="ok", model=p.meta["config"]["model"]["name"],
                              checkpoint=str(p.checkpoint_path), device=str(p.device),
                              classes=p.class_names)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Boş dosya.")
        try:
            return app.state.service.predict(file.filename or "upload.jpg", data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app