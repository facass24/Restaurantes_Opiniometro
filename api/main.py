"""Punto de entrada principal de la aplicación FastAPI.

Configura el ciclo de vida (lifespan), middlewares de CORS, controladores
de excepciones y el registro de routers bajo /api/v1.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_V1_PREFIX,
    CORS_ORIGINS,
)
from api.service import ServicioPrediccion
from api.routers import health, predict, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo seguro del ciclo de vida de la aplicación.

    Inicializa y carga los modelos y recursos una única vez en memoria
    durante el arranque, evitando sobrecostos por petición.
    """
    # 1. Inicialización en el arranque
    app.state.servicio = ServicioPrediccion()
    yield
    # 2. Limpieza al apagado
    app.state.servicio = None


def create_app() -> FastAPI:
    """Fábrica de la aplicación FastAPI."""
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configuración de CORS para permitir la conexión desde React / Vite
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Registro de routers bajo el prefijo unificado
    app.include_router(health.router, prefix=API_V1_PREFIX)
    app.include_router(predict.router, prefix=API_V1_PREFIX)
    app.include_router(metrics.router, prefix=API_V1_PREFIX)

    # Redirección amigable desde la raíz a la documentación interactiva
    @app.get("/", include_in_schema=False)
    async def raiz():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

