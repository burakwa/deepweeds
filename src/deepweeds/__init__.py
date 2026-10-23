__version__ = "0.1.0"


def main() -> int:  # eski entry point uyumluluğu
    from deepweeds.cli import main as _main
    return _main()