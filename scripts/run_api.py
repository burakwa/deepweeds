#!/usr/bin/env python
"""
DeepWeeds API v2 Runner

Usage:
    python run_api.py --model runs/exp01/weights/best.pt
    python run_api.py --model runs/exp01/weights/best.pt --port 8000 --host 0.0.0.0 --metrics
"""

from __future__ import annotations

import argparse

import uvicorn

from deepweeds.api_v2 import create_app


def main():
    parser = argparse.ArgumentParser(description="DeepWeeds API v2 Server")
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
