#!/usr/bin/env python
"""
DeepWeeds Streamlit Runner

Usage:
    python scripts/run_streamlit.py
    python scripts/run_streamlit.py --model runs/exp-effb3-sqrt/weights/best.pt
    python scripts/run_streamlit.py --model runs/exp01/weights/best.pt --port 8501
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DeepWeeds Streamlit Uygulamasını Başlat")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint yolu (varsayılan: runs/exp-effb3-sqrt/weights/best.pt)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port (varsayılan: 8501)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Streamlit host (varsayılan: localhost)"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Tarayıcıyı otomatik aç"
    )
    args = parser.parse_args()

    # Proje kök dizinini bul
    project_root = Path(__file__).parent.parent
    streamlit_app = project_root / "src" / "deepweeds" / "streamlit_app.py"

    if not streamlit_app.exists():
        print(f"Hata: Streamlit uygulaması bulunamadı: {streamlit_app}")
        sys.exit(1)

    # Model checkpoint kontrolü
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Uyarı: Model checkpoint bulunamadı: {model_path}")
            print("Varsayılan model kullanılacak.")

    # Streamlit komutunu hazırla
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(streamlit_app),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]

    if args.browser:
        cmd.append("--browser.gatherUsageStats")
        cmd.append("false")

    # Model argümanını ekle (streamlit_app.py kendi argparse'i ile işleyecek)
    if args.model:
        cmd.extend(["--", "--model", args.model])

    print(f"🚀 DeepWeeds Streamlit başlatılıyor...")
    print(f"   Uygulama: {streamlit_app}")
    print(f"   Adres: http://{args.host}:{args.port}")
    if args.model:
        print(f"   Model: {args.model}")
    print()

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Hata: Streamlit başlatılamadı (kod: {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n👋 Streamlit kapatıldı.")


if __name__ == "__main__":
    main()