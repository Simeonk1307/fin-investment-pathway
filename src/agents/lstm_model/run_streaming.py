"""Runner that picks Pathway streaming if available, else falls back to CSV streaming.

Usage (PowerShell):
  python src\agents\lstm_model\run_streaming.py --tickers AAPL MSFT --period 1y

Behavior:
- If the real `pathway` runtime is importable (and exposes `Schema`), runs the Pathway streaming pipeline
  from `lstm_test.run_pathway_streaming_pipeline()` (this will stream the historical CSV through Pathway).
- Otherwise runs the CSV-simulated streamer in `live_signals.run_demo()` and writes `outputs/lstm/signals/signals.csv`.

This file avoids importing modules that require Pathway unless Pathway is actually present.
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

# Allow running this file directly (not as module) by prepending the project root
# to sys.path so `import src...` works when invoked as a script.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pathway_available() -> bool:
    try:
        pw = importlib.import_module('pathway')
        # Some distributions provide a placeholder package; check expected attribute
        if hasattr(pw, 'Schema'):
            return True
        return False
    except Exception:
        return False


def run_pathway_mode(tickers, period, quiet: bool = False, force_train: bool = False):
    # Import lstm_test only when pathway is present to avoid import-time errors
    try:
        from src.agents.lstm_model import lstm_test
    except Exception as e:
        import traceback
        print("Failed to import Pathway pipeline module:")
        traceback.print_exc()
        return False

    print("Pathway available — running Pathway streaming pipeline.")
    # lstm_test.run_pathway_streaming_pipeline() will interactively ask before streaming
    try:
        # If lstm_test supports arguments we could pass them; keep interactive behavior.
        lstm_test.run_pathway_streaming_pipeline()
        return True
    except Exception as e:
        print(f"Pathway pipeline failed: {e}")
        return False


def run_csv_mode(tickers, period, quiet: bool = False, force_train: bool = False):
    print("Pathway not available — running CSV simulated streaming.")
    try:
        from src.agents.lstm_model.live_signals import run_demo
    except Exception as e:
        import traceback
        print("Failed to import CSV streamer:")
        traceback.print_exc()
        return False
    run_demo(tickers, period, quiet=quiet, force_train=force_train)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', default='1y')
    parser.add_argument('--quiet', action='store_true', help='Suppress repeated warnings')
    parser.add_argument('--force-train', action='store_true', help='Force an offline retrain on the 80% training split (requires torch and model files)')
    args = parser.parse_args(argv)

    # Ensure outputs dir exists
    Path('outputs/lstm/signals').mkdir(parents=True, exist_ok=True)

    if pathway_available():
        ok = run_pathway_mode(args.tickers, args.period, quiet=args.quiet, force_train=args.force_train)
        if not ok:
            if not args.quiet:
                print("Falling back to CSV mode due to Pathway pipeline failure.")
            run_csv_mode(args.tickers, args.period, quiet=args.quiet, force_train=args.force_train)
    else:
        run_csv_mode(args.tickers, args.period, quiet=args.quiet, force_train=args.force_train)


if __name__ == '__main__':
    main()
