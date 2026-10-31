"""Inyección de dependencias para los routers de FastAPI."""

from fastapi import Request
from api.service import ServicioPrediccion


def obtener_servicio_prediccion(request: Request) -> ServicioPrediccion:
    """Extrae la instancia compartida de ServicioPrediccion del estado de la aplicación."""
    return request.app.state.servicio

