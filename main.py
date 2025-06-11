from fastapi import FastAPI, APIRouter, WebSocket
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# from routes.websocket import websocket_endpoint
from routes.id_upload import router as id_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
api_router = APIRouter(prefix="/api")
api_router.include_router(id_router, prefix="/id")

app.include_router(api_router)

# WebSocket route for video streaming
# @app.websocket("/api/stream")
# async def websocket_route(websocket: WebSocket):
#     await websocket_endpoint(websocket)

@app.get("/")
async def root():
    return {"message": "FastAPI is running"}
