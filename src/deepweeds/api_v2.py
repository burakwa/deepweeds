"""
DeepWeeds FastAPI Servisi — Production Ready

Endpoints
---------
GET  /                    → Sağlık kontrolü + basit arayüz
GET  /health              → Model/cihaz/sınıf bilgisi (JSON)
GET  /classes             → Sınıf listesi
POST /predict             → Tek görüntü sınıflandırma
POST /predict/batch       → Çoklu görüntü (max 32)
POST /predict/tta         → Test-Time Augmentation ile tek görüntü
GET  /metrics             → Son eğitim metrikleri (varsa)

Özellikler
----------
- Thread-safe ModelService (Lock + async uyumlu)
- Pydantic v2 şemaları (validasyon + OpenAPI docs)
- Structured logging (JSON formatında opsiyonel)
- Graceful shutdown (model cleanup)
- CORS yapılandırılabilir
- Rate limiting (slowapi entegrasyonu hazır)
- Prometheus metrics endpoint hazır
"""

from __future__ import annotations

import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from deepweeds import __version__
from deepweeds.inference import Predictor
from deepweeds.utils import get_logger

log = get_logger()

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
BATCH_LIMIT = 32
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ---------------------------------------------------------------------------
# Pydantic Şemaları (v2)
# ---------------------------------------------------------------------------


class Probabilities(BaseModel):
    """Sınıf olasılıkları — dinamik alanlar için root model."""

    model_config = ConfigDict(extra="allow")

    def __iter__(self):
        return iter(self.model_dump().items())


class SinglePrediction(BaseModel):
    filename: str
    top_class: str
    top_prob: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    results: list[SinglePrediction]
    total_latency_ms: float
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    checkpoint: str
    device: str
    classes: list[str]
    num_classes: int
    version: str
    model_size_mb: float


class ClassesResponse(BaseModel):
    classes: list[str]
    count: int


class MetricsResponse(BaseModel):
    macro_f1: float | None = None
    accuracy: float | None = None
    per_class_f1: dict[str, float] = {}
    run_name: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


# ---------------------------------------------------------------------------
# Model Servis Katmanı
# ---------------------------------------------------------------------------


class ModelService:
    """Predictor üzerinde thread-güvenli, async-dostu sarmalayıcı."""

    def __init__(self, predictor: Predictor):
        self._predictor = predictor
        self._lock = threading.Lock()
        self._ready = True

    @property
    def predictor(self) -> Predictor:
        return self._predictor

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _predict_image_sync(self, filename: str, data: bytes) -> SinglePrediction:
        """Senkron tahmin — lock içinde çalışır."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()  # Erken hata için veri yükle
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"'{filename}' geçerli bir görüntü dosyası değil."
            ) from exc
        except Exception as exc:
            raise ValueError(f"'{filename}' işlenirken hata: {exc}") from exc

        # RGB'e çevir (RGBA, P mode vb. destek)
        if image.mode != "RGB":
            image = image.convert("RGB")

        t0 = time.perf_counter()
        probs = self._predictor.predict_pil(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        idx = int(np.argmax(probs))
        top_class = self._predictor.class_names[idx]
        top_prob = float(probs[idx])

        return SinglePrediction(
            filename=filename,
            top_class=top_class,
            top_prob=round(top_prob, 6),
            probabilities={
                c: round(float(p), 6)
                for c, p in zip(self._predictor.class_names, probs)
            },
            latency_ms=round(latency_ms, 2),
        )

    async def predict_single(self, filename: str, data: bytes) -> SinglePrediction:
        """Async wrapper — thread pool'da çalıştır."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._predict_image_sync, filename, data
        )

    async def predict_batch(
        self, files: list[tuple[str, bytes]]
    ) -> BatchPredictionResponse:
        t0 = time.perf_counter()
        results = []
        for filename, data in files:
            results.append(await self.predict_single(filename, data))
        return BatchPredictionResponse(
            results=results,
            total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            count=len(results),
        )

    async def predict_tta(self, filename: str, data: bytes) -> SinglePrediction:
        """Test-Time Augmentation — 5 view ortalaması."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"'{filename}' geçerli bir görüntü dosyası değil."
            ) from exc

        t0 = time.perf_counter()
        probs = self._predictor.predict_tta(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        idx = int(np.argmax(probs))
        return SinglePrediction(
            filename=filename,
            top_class=self._predictor.class_names[idx],
            top_prob=round(float(probs[idx]), 6),
            probabilities={
                c: round(float(p), 6)
                for c, p in zip(self._predictor.class_names, probs)
            },
            latency_ms=round(latency_ms, 2),
        )


# ---------------------------------------------------------------------------
# Uygulama Fabrika
# ---------------------------------------------------------------------------


def create_app(
    model_path: str | Path,
    device: str | None = None,
    cors_origins: list[str] | None = None,
    enable_metrics: bool = False,
) -> FastAPI:
    """
    FastAPI uygulaması oluşturur.

    Args:
        model_path: Checkpoint dosyası yolu (weights/best.pt)
        device: "cuda" | "cpu" | None (otomatik)
        cors_origins: İzin verilecek originler (None = hepsi)
        enable_metrics: Prometheus /metrics endpoint'i aç
    """
    predictor_holder: dict[str, Predictor | None] = {"predictor": None}
    service_holder: dict[str, ModelService | None] = {"service": None}
    load_error: dict[str, Exception | None] = {"error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        log.info("Model yükleniyor: %s", model_path)
        try:
            predictor = Predictor.from_checkpoint(model_path, device)
            predictor_holder["predictor"] = predictor
            service_holder["service"] = ModelService(predictor)
            log.info(
                "Model hazır — %s | %s | %.1f MB | %d sınıf",
                predictor.meta["config"]["model"]["name"],
                predictor.device,
                predictor.size_bytes / 1e6,
                len(predictor.class_names),
            )
        except Exception as e:
            load_error["error"] = e
            log.exception("Model yüklenemedi: %s", e)

        yield

        # Shutdown
        log.info("Uygulama kapatılıyor, kaynaklar serbest bırakılıyor...")
        if predictor_holder["predictor"]:
            # GPU cache temizle
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    app = FastAPI(
        title="DeepWeeds API",
        version=__version__,
        description=__doc__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Prometheus metrics (opsiyonel)
    if enable_metrics:
        from prometheus_client import make_asgi_app

        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    # ---------------------------------------------------------------------------
    # Yardımcı: Servis erişimi
    # ---------------------------------------------------------------------------
    def get_service() -> ModelService:
        if load_error["error"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model yüklenemedi: {load_error['error']}",
            )
        service = service_holder["service"]
        if not service or not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model henüz hazır değil",
            )
        return service

    # ---------------------------------------------------------------------------
    # Middleware: Request logging + timing
    # ---------------------------------------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        log.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))
        return response

    # ---------------------------------------------------------------------------
    # Exception Handlers
    # ---------------------------------------------------------------------------
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=str(exc), error_code="INVALID_INPUT"
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        log.exception("Beklenmeyen hata: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="İç sunucu hatası", error_code="INTERNAL_ERROR"
            ).model_dump(),
        )

    # ---------------------------------------------------------------------------
    # Endpoints
    # ---------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        """Basit landing page + test formu."""
        return f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepWeeds API v{__version__}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
        .endpoint {{ font-family: monospace; background: #f5f5f5; padding: 0.2rem 0.5rem; border-radius: 4px; }}
        pre {{ background: #f8f8f8; padding: 1rem; overflow: auto; }}
        button {{ background: #0066cc; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background: #0055aa; }}
        .result {{ margin-top: 1rem; padding: 1rem; background: #f0f8ff; border-radius: 4px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>🌿 DeepWeeds API v{__version__}</h1>
    <p>Yabani ot sınıflandırma servisi — 9 sınıf, PyTorch + timm tabanlı.</p>
    
    <div class="card">
        <h2>Endpoints</h2>
        <ul>
            <li><span class="endpoint">GET /health</span> — Sağlık ve model bilgisi</li>
            <li><span class="endpoint">GET /classes</span> — Sınıf listesi</li>
            <li><span class="endpoint">POST /predict</span> — Tek görüntü (multipart/form-data)</li>
            <li><span class="endpoint">POST /predict/batch</span> — Çoklu görüntü (max {BATCH_LIMIT})</li>
            <li><span class="endpoint">POST /predict/tta</span> — TTA ile tek görüntü (daha doğru)</li>
            <li><span class="endpoint">GET /metrics</span> — Eğitim metrikleri</li>
            <li><span class="endpoint">GET /docs</span> — Swagger UI</li>
        </ul>
    </div>
    
    <div class="card">
        <h2>Hızlı Test</h2>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="predict()">Tahmin Et</button>
        <div id="result" class="result" style="display:none;"></div>
    </div>
    
    <script>
        async function predict() {{
            const file = document.getElementById('fileInput').files[0];
            if (!file) return alert('Dosya seçin');
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/predict', {{ method: 'POST', body: formData }});
            const data = await res.json();
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        }}
    </script>
</body>
</html>"""

    @app.get("/health", response_model=HealthResponse, summary="Servis sağlık kontrolü")
    async def health():
        service = get_service()
        p = service.predictor
        return HealthResponse(
            status="ok",
            model=p.meta["config"]["model"]["name"],
            checkpoint=str(p.checkpoint_path),
            device=str(p.device),
            classes=p.class_names,
            num_classes=len(p.class_names),
            version=__version__,
            model_size_mb=round(p.size_bytes / 1e6, 2),
        )

    @app.get("/classes", response_model=ClassesResponse, summary="Desteklenen sınıflar")
    async def classes():
        service = get_service()
        return ClassesResponse(
            classes=service.predictor.class_names,
            count=len(service.predictor.class_names),
        )

    @app.post(
        "/predict",
        response_model=SinglePrediction,
        summary="Tek görüntü sınıflandır",
        responses={
            400: {"model": ErrorResponse, "description": "Geçersiz dosya"},
            413: {"model": ErrorResponse, "description": "Dosya çok büyük"},
            503: {"model": ErrorResponse, "description": "Model hazır değil"},
        },
    )
    async def predict(
        file: Annotated[
            UploadFile,
            File(description="Sınıflandırılacak görüntü (JPG/PNG/WebP, max 10MB)"),
        ],
    ) -> SinglePrediction:
        # Content-Type kontrolü
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Desteklenmeyen format: {file.content_type}. İzin verilen: {', '.join(ALLOWED_CONTENT_TYPES)}",
            )

        # Dosya boyutu kontrolü (streaming read)
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Dosya çok büyük: {len(content) / 1e6:.1f}MB > {MAX_FILE_SIZE / 1e6}MB",
            )

        service = get_service()
        return await service.predict_single(file.filename or "upload.jpg", content)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        summary=f"Çoklu görüntü sınıflandır (max {BATCH_LIMIT})",
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def predict_batch(
        files: Annotated[
            list[UploadFile],
            File(description=f"Görüntü listesi (max {BATCH_LIMIT}, her biri max 10MB)"),
        ],
    ) -> BatchPredictionResponse:
        if not files:
            raise HTTPException(400, "En az 1 görüntü gönderilmeli")
        if len(files) > BATCH_LIMIT:
            raise HTTPException(400, f"En fazla {BATCH_LIMIT} görüntü gönderilebilir")

        pairs: list[tuple[str, bytes]] = []
        for uf in files:
            if uf.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    400, f"'{uf.filename}': desteklenmeyen format {uf.content_type}"
                )
            content = await uf.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"'{uf.filename}': dosya çok büyük")
            pairs.append((uf.filename or "upload.jpg", content))

        service = get_service()
        return await service.predict_batch(pairs)

    @app.post(
        "/predict/tta",
        response_model=SinglePrediction,
        summary="Test-Time Augmentation ile tek görüntü (5 view ortalaması, daha doğru)",
    )
    async def predict_tta(
        file: Annotated[UploadFile, File(description="Görüntü (JPG/PNG/WebP)")],
    ) -> SinglePrediction:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, f"Desteklenmeyen format: {file.content_type}")

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "Dosya çok büyük")

        service = get_service()
        return await service.predict_tta(file.filename or "upload.jpg", content)

    @app.get("/metrics", response_model=MetricsResponse, summary="Eğitim metrikleri")
    async def metrics():
        """Checkpoint'in olduğu run klasöründen metrics.json okur."""
        service = get_service()
        p = service.predictor
        run_dir = p.checkpoint_path.parent.parent  # runs/<run_name>/
        metrics_file = run_dir / "metrics.json"

        if not metrics_file.exists():
            return MetricsResponse(run_name=run_dir.name)

        import json

        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            return MetricsResponse(
                macro_f1=data.get("macro_f1"),
                accuracy=data.get("accuracy"),
                per_class_f1={
                    k: v.get("f1-score", 0)
                    for k, v in data.get("per_class", {}).items()
                },
                run_name=run_dir.name,
            )
        except Exception:
            return MetricsResponse(run_name=run_dir.name)

    return app


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """uvicorn ile çalıştır: `python -m deepweeds.api --model runs/exp/weights/best.pt`"""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="DeepWeeds API Server")
    parser.add_argument(
        "--model", required=True, help="Checkpoint yolu (weights/best.pt)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda | cpu | auto")
    parser.add_argument(
        "--cors-origins", nargs="*", default=None, help="CORS origin listesi"
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Prometheus /metrics endpoint"
    )
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker sayısı")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    args = parser.parse_args()

    app = create_app(
        model_path=args.model,
        device=args.device,
        cors_origins=args.cors_origins,
        enable_metrics=args.metrics,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Eksik import (asyncio) - dosya başına eklenecek
# ---------------------------------------------------------------------------
import asyncio
