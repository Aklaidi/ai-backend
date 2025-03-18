from ninja import NinjaAPI

from prototype.api.dashboard import dashboard_router

api = NinjaAPI(
    title="AI Prototype",
)

api.add_router("dashboard", dashboard_router, tags=["Dashboard"])
