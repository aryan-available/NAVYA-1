"""FastAPI Router for NAVYA-1 Optimization Engine.

Endpoint:
  POST /api/v1/engine/optimize

Single source of truth: NAVYA-1 Architecture Specification.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body

from engine.optimizer.core.optimizer import run_optimization

router = APIRouter(tags=["Optimization"])


@router.post("/api/v1/engine/optimize")
@router.post("/optimize")
async def optimize_dispatch(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Execute mathematical microgrid optimization for the provided forecast and state.

    Accepts the canonical optimization request JSON and returns the canonical
    optimization response containing schedule, outcome metrics, and reason codes.
    """
    try:
        result = run_optimization(payload)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid optimization request: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization engine error: {e}")
