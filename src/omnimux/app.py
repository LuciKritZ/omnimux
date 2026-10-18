import typing
from contextlib import asynccontextmanager

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from omnimux.config import settings
from omnimux.exceptions import CacheError, GeminiAPIError, OmniMuxError
from omnimux.routes.proxy import router as proxy_router
from omnimux.services.cache import ImageCache
from omnimux.services.compressor import ContextCompressor
from omnimux.services.db import (
    auto_import_from_json_if_empty,
    get_all_platforms_and_models,
    init_db,
)
from omnimux.services.gatekeeper import Gatekeeper
from omnimux.services.gemini_client import GeminiClient
from omnimux.services.multiplexer import Multiplexer
from omnimux.services.router_core import RouterCore
from omnimux.services.sync import sync_keys_from_json
from omnimux.services.unloader import ModelUnloader


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    import json
    from pathlib import Path

    models_data = {}
    models_path = Path(settings.models_json_path)
    if models_path.exists():
        try:
            with open(models_path, encoding="utf-8") as f:
                models_data = json.load(f)
        except Exception:  # pragma: no cover
            pass

    db = await aiosqlite.connect(settings.sqlite_db_path)
    await init_db(db)
    await auto_import_from_json_if_empty(db, settings.models_json_path)
    await sync_keys_from_json(db, settings.models_json_path)

    db_platforms = await get_all_platforms_and_models(db)
    models_dict = {**models_data.get("models", {}), **db_platforms}
    model_limits = {
        name: info["max_input_tokens"]
        for name, info in models_dict.items()
        if isinstance(info, dict) and "max_input_tokens" in info
    }

    client = httpx.AsyncClient(timeout=300.0)
    unloader = ModelUnloader(http_client=client, platforms_data=models_dict)
    compressor = ContextCompressor(model_limits=model_limits)
    cache = ImageCache()
    gemini_client = GeminiClient(client=client)

    router_core = RouterCore(db)
    gatekeeper = Gatekeeper(client, models_dict)
    multiplexer = Multiplexer(router_core, client, models_dict)

    app.state.http_client = client
    app.state.unloader = unloader
    app.state.compressor = compressor
    app.state.cache = cache
    app.state.gemini_client = gemini_client
    app.state.db = db
    app.state.router_core = router_core
    app.state.gatekeeper = gatekeeper
    app.state.multiplexer = multiplexer
    app.state.platforms = models_dict

    yield

    await client.aclose()
    await db.close()


def create_app() -> FastAPI:
    app = FastAPI(title="omnimux", lifespan=lifespan)
    app.include_router(proxy_router)

    @app.exception_handler(GeminiAPIError)
    async def gemini_exception_handler(request: Request, exc: GeminiAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)},
        )

    @app.exception_handler(CacheError)
    async def cache_exception_handler(request: Request, exc: CacheError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    @app.exception_handler(OmniMuxError)
    async def omnimux_exception_handler(request: Request, exc: OmniMuxError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app
