import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import io
from PIL import Image
import numpy as np

from deepweeds.api import (
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ClassesResponse,
    ModelService,
    create_app,
    BATCH_LIMIT,
    DEMO_HTML,
)


class TestPydanticModels:
    def test_prediction_response(self):
        resp = PredictionResponse(
            filename="test.jpg",
            top_class="Lantana",
            top_prob=0.95,
            probabilities={"Lantana": 0.95, "Negative": 0.05},
            latency_ms=12.5,
        )
        assert resp.filename == "test.jpg"
        assert resp.top_class == "Lantana"
        assert resp.top_prob == 0.95

    def test_batch_prediction_response(self):
        resp = BatchPredictionResponse(
            results=[],
            total_latency_ms=25.0,
            count=0,
        )
        assert resp.count == 0

    def test_health_response(self):
        resp = HealthResponse(
            status="ok",
            model="resnet50",
            checkpoint="runs/exp/weights/best.pt",
            device="cuda",
            classes=["A", "B"],
            version="0.1.0",
        )
        assert resp.status == "ok"

    def test_classes_response(self):
        resp = ClassesResponse(classes=["A", "B"], count=2)
        assert resp.count == 2


class TestModelService:
    @pytest.fixture
    def mock_predictor(self):
        predictor = MagicMock()
        predictor.class_names = ["A", "B", "C"]
        predictor.predict_pil.return_value = np.array([0.1, 0.7, 0.2])
        predictor.device.type = "cpu"
        return predictor

    @pytest.fixture
    def valid_image_bytes(self):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_predict_single(self, mock_predictor, valid_image_bytes):
        service = ModelService(mock_predictor)
        
        result = service.predict_single("test.jpg", valid_image_bytes)
        
        assert isinstance(result, PredictionResponse)
        assert result.filename == "test.jpg"
        assert result.top_class == "B"  # index 1 has 0.7
        assert abs(result.top_prob - 0.7) < 1e-5

    def test_predict_batch(self, mock_predictor, valid_image_bytes):
        service = ModelService(mock_predictor)
        
        files = [
            ("img1.jpg", valid_image_bytes),
            ("img2.jpg", valid_image_bytes),
        ]
        result = service.predict_batch(files)
        
        assert isinstance(result, BatchPredictionResponse)
        assert result.count == 2
        assert len(result.results) == 2
        assert result.total_latency_ms >= 0

    def test_predict_single_invalid_image(self, mock_predictor):
        service = ModelService(mock_predictor)
        
        # Mock PIL to raise UnidentifiedImageError
        from PIL import UnidentifiedImageError
        mock_predictor.predict_pil.side_effect = UnidentifiedImageError("invalid")
        
        with pytest.raises(ValueError, match="geçerli bir görüntü değil"):
            service.predict_single("bad.jpg", b"not_an_image")


class TestCreateApp:
    @patch("deepweeds.api.Predictor.from_checkpoint")
    def test_creates_fastapi_app(self, mock_from_checkpoint):
        mock_predictor = MagicMock()
        mock_predictor.class_names = ["A", "B"]
        mock_predictor.meta = {"config": {"model": {"name": "resnet18"}}}
        mock_predictor.checkpoint_path = Path("test.pt")
        mock_predictor.device = "cpu"
        mock_from_checkpoint.return_value = mock_predictor
        
        app = create_app("test.pt")
        
        assert app.title == "DeepWeeds API"
        assert app.version == "0.1.0"

    def test_demo_html_exists(self):
        assert "DeepWeeds" in DEMO_HTML
        assert "predict" in DEMO_HTML.lower()

    def test_batch_limit_constant(self):
        assert BATCH_LIMIT == 16