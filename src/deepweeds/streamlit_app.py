from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from PIL import Image

from deepweeds.config import CLASS_NAMES
from deepweeds.inference import Predictor


class ModelService:
    """Model yükleme ve çıkarım servisi - Singleton pattern ile cache yönetimi."""

    _instance: ModelService | None = None
    _predictor: Predictor | None = None
    _checkpoint_path: Path | None = None

    def __new__(cls, checkpoint_path: Path | None = None) -> ModelService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path
        if self._predictor is None:
            self._load_model()

    def _load_model(self) -> None:
        if self._checkpoint_path is None:
            self._checkpoint_path = Path("runs/exp-effb3-sqrt/weights/best.pt")
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint bulunamadı: {self._checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = Predictor.from_checkpoint(self._checkpoint_path, device=device)

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._load_model()
        return self._predictor

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES

    @property
    def device(self) -> torch.device:
        return self.predictor.device

    @property
    def model_size_mb(self) -> float:
        return self.predictor.size_bytes / (1024 * 1024)

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path


class InferenceEngine:
    """Çıkarım motoru - farklı modları (normal, TTA) yönetir."""

    def __init__(self, service: ModelService) -> None:
        self.service = service

    def predict_single(self, image: Image.Image, use_tta: bool = False) -> np.ndarray:
        if use_tta:
            return self.service.predictor.predict_tta(image)
        return self.service.predictor.predict_pil(image)

    def benchmark(self, n_samples: int = 10) -> dict[str, float]:
        import pandas as pd
        test_df = pd.read_csv("data/test.csv")
        return self.service.predictor.benchmark_latency(test_df, n=n_samples)


class StreamlitUI:
    """Streamlit arayüz bileşenleri - SRP prensibi."""

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self.class_names = engine.service.class_names

    def render_header(self) -> None:
        st.set_page_config(
            page_title="DeepWeeds Sınıflandırıcı",
            page_icon="🌿",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.title("🌿 DeepWeeds Ot Sınıflandırıcı")
        st.caption("EfficientNet-B3 (sqrt weighted loss) — exp-effb3-sqrt")

    def render_sidebar(self) -> dict[str, Any]:
        with st.sidebar:
            st.header("⚙️ Ayarlar")
            use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=False, help="5 farklı augmentasyon ile tahmin alıp ortalama alır (daha yavaş, daha kararlı)")
            st.divider()
            st.subheader("📊 Model Bilgisi")
            st.metric("Cihaz", str(self.engine.service.device).upper())
            st.metric("Model Boyutu", f"{self.engine.service.model_size_mb:.1f} MB")
            st.metric("Giriş Boyutu", "256×256")
            st.metric("Sınıf Sayısı", len(self.class_names))
            st.caption(f"📁 Checkpoint: `{self.engine.service.checkpoint_path.name}`")
            st.divider()
            if st.button("🔄 Modeli Yeniden Yükle", use_container_width=True):
                ModelService._predictor = None
                st.rerun()
        return {"use_tta": use_tta}

    def render_uploader(self) -> Image.Image | None:
        uploaded = st.file_uploader(
            "Görsel Yükle (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Sınıflandırılacak ot görselini seçin",
        )
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                return image
            except Exception as e:
                st.error(f"Görsel okunamadı: {e}")
        return None

    def render_preview(self, image: Image.Image) -> None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Yüklenen Görsel", use_container_width=True)
        with col2:
            st.info(f"📐 Orijinal boyut: {image.size[0]}×{image.size[1]} px\n"
                    f"🎨 Mod: {image.mode}")

    def render_results(self, probs: np.ndarray, latency_ms: float) -> None:
        pred_idx = int(probs.argmax())
        pred_name = self.class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        st.success(f"### 🎯 Tahmin: **{pred_name}** ({confidence:.1f}%)")
        st.caption(f"⏱️ Çıkarım süresi: {latency_ms:.1f} ms")

        st.divider()
        st.subheader("📈 Sınıf Olasılıkları")

        cols = st.columns(3)
        for i, (name, prob) in enumerate(zip(self.class_names, probs)):
            with cols[i % 3]:
                st.metric(label=name, value=f"{prob*100:.1f}%")

        st.divider()
        st.subheader("📊 Olasılık Dağılımı")
        chart_data = {name: float(prob) for name, prob in zip(self.class_names, probs)}
        st.bar_chart(chart_data, height=300)

    def render_footer(self) -> None:
        st.divider()
        st.caption("🏗️ OOP prensipleriyle geliştirildi • DeepWeeds Yarışması")


class StreamlitApp:
    """Ana uygulama sınıfı - Facade pattern."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.service = ModelService(checkpoint_path)
        self.engine = InferenceEngine(self.service)
        self.ui = StreamlitUI(self.engine)

    def run(self) -> None:
        self.ui.render_header()
        settings = self.ui.render_sidebar()
        image = self.ui.render_uploader()

        if image:
            self.ui.render_preview(image)

            with st.spinner("Model çalıştırılıyor..."):
                start = time.perf_counter()
                probs = self.engine.predict_single(image, use_tta=settings["use_tta"])
                latency = (time.perf_counter() - start) * 1000

            self.ui.render_results(probs, latency)
        else:
            st.info("👈 Yan panelden ayarları yapın ve bir görsel yükleyin.")

        self.ui.render_footer()


def main(checkpoint_path: Path | None = None) -> None:
    try:
        app = StreamlitApp(checkpoint_path)
        app.run()
    except Exception as e:
        st.error(f"Uygulama hatası: {e}")
        st.exception(e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepWeeds Streamlit Uygulaması")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint yolu (varsayılan: runs/exp-effb3-sqrt/weights/best.pt)")
    args = parser.parse_args()
    main(Path(args.model) if args.model else None)