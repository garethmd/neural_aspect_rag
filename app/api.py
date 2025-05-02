import index_website
import search
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from index_website import main as index_website
from pydantic import BaseModel
from search import main as search_documents

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchQuery(BaseModel):
    query: str


class IndexRequest(BaseModel):
    url: str


@app.post("/api/search")
async def search(query: SearchQuery):
    try:
        results = search_documents(query.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index")
async def index(request: IndexRequest):
    try:
        result = index_website(request.url)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
