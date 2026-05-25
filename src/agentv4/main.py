from __future__ import annotations

import uvicorn

from .api.app import create_app
from .config import get_settings

app = create_app()


def main() -> None:
    settings = get_settings()
    host = settings.host
    port = settings.port
    uvicorn.run("agentv4.main:app", host=host, port=port, reload=False, workers=settings.workers)


if __name__ == "__main__":
    main()
