from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from FINAL_CODING.add_data import uploadRouter
from FINAL_CODING.query_data import chatRouter


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(uploadRouter, prefix="/upload", tags=["upload"])
app.include_router(chatRouter, prefix="/query", tags=["query"])


@app.get("/")
async def root():
    return {"message": "Hello World"}
