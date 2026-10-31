# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# from ml_service import ShoppingPatternEngine

# app = FastAPI(
#     title="Smart Shopping Pattern Analyzer API",
#     description="Backend service supplying Market Basket Analysis and K-Means Customer Clustering.",
#     version="1.0.0",
# )
# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]
# # Enable CORS for cross-origin requests from frontend apps
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize ML Service Engine
# engine = ShoppingPatternEngine()


# @app.get("/")
# def root():
#     return {
#         "status": "Online",
#         "service": "Smart Shopping Pattern Analyzer API Engine",
#     }


# @app.get("/api/dashboard/summary")
# def get_summary():
#     """Returns dataset summary metrics for dashboard KPIs."""
#     return engine.get_dashboard_summary()


# @app.get("/api/market-basket/rules")
# def get_rules(
#     min_support: float = Query(0.02, ge=0.001, le=1.0),
#     min_confidence: float = Query(0.3, ge=0.1, le=1.0),
#     min_lift: float = Query(1.0, ge=0.0),
# ):
#     """Calculates product affinity rules using FP-Growth."""
#     return engine.get_association_rules(min_support, min_confidence, min_lift)


# @app.get("/api/customer-segments")
# def get_segments(n_clusters: int = Query(4, ge=2, le=6)):
#     """Segments customers based on Recency, Frequency, and Monetary spend."""
#     return engine.get_customer_clusters(n_clusters)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
import os
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from ml_service import ShoppingPatternEngine
from pydantic import BaseModel

app = FastAPI(
    title="Smart Shopping Pattern Analyzer API",
    description="Backend API serving Market Basket Analysis and Customer Segmentation",
    version="1.0.0",
)

# Enable CORS for all origins (allows AI Studio frontend to connect to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML Service Engine
engine = ShoppingPatternEngine()


class MiningRequest(BaseModel):
    algorithm: str = "apriori"
    min_support: float = 0.05
    min_confidence: float = 0.25
    min_lift: float = 1.1
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ClusteringRequest(BaseModel):
    k: int = 4


@app.get("/")
def root():
    return {
        "status": "Online",
        "service": "Smart Shopping Pattern Analyzer Backend",
    }


@app.get("/api/dashboard/summary")
@app.get("/api/dashboard-stats")
def get_summary():
    """Returns dataset summary metrics for dashboard KPIs."""
    return engine.get_dashboard_summary()


@app.get("/api/market-basket/rules")
def get_rules(
    min_support: float = Query(0.02, ge=0.001, le=1.0),
    min_confidence: float = Query(0.3, ge=0.1, le=1.0),
    min_lift: float = Query(1.0, ge=0.0),
):
    """Calculates product affinity rules using FP-Growth."""
    return engine.get_association_rules(min_support, min_confidence, min_lift)


@app.post("/api/market-basket/mine")
def mine_market_basket(req: MiningRequest):
    """Accepts POST requests for custom association mining parameters."""
    return engine.get_association_rules(
        min_support=req.min_support,
        min_confidence=req.min_confidence,
        min_lift=req.min_lift,
    )


@app.get("/api/customer-segments")
def get_segments(n_clusters: int = Query(4, ge=2, le=6)):
    """Segments customers based on RFM analytics using K-Means."""
    return engine.get_customer_clusters(n_clusters)


@app.post("/api/customer-segmentation/cluster")
def cluster_customers(req: ClusteringRequest):
    """Accepts POST requests for K-Means clustering with k clusters."""
    return engine.get_customer_clusters(n_clusters=req.k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)