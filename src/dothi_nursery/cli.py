"""Console entry point: launch the Shiny app in a local web server."""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="dothi-nursery",
        description="Launch the Dothistroma nursery Shiny app in your browser.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to bind (default: 8000)")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser automatically"
    )
    args = parser.parse_args(argv)

    import shiny

    shiny.run_app(
        "dothi_nursery.app:app",
        host=args.host,
        port=args.port,
        launch_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
