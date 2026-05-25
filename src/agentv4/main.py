from __future__ import annotations

import os

import uvicorn

from .api.app import create_app

app = create_app()


def main() -> None:
    host = os.getenv("AGENTV4_HOST", "0.0.0.0")
    port = int(os.getenv("AGENTV4_PORT", "8094"))
    uvicorn.run("agentv4.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
