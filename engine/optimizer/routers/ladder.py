"""FastAPI Router for NAVYA-1 Shortfall Ladder.

Endpoint:
  POST /api/v1/engine/ladder

Single source of truth: NAVYA-1 Architecture Specification.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body

from engine.optimizer.ladder.shortfall_ladder import evaluate_shortfall_ladder

router = APIRouter(tags=["Shortfall Ladder"])


@router.post("/api/v1/engine/ladder")
@router.post("/ladder")
async def get_shortfall_ladder(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Evaluate and return the active shortfall ladder stage and recommended actions.

    Accepts the canonical microgrid state and forecast, delegates to the deterministic
    shortfall ladder evaluation logic, and returns the canonical ladder response.
    """
    try:
        result = evaluate_shortfall_ladder(payload)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid ladder evaluation request: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shortfall ladder evaluation error: {e}")
