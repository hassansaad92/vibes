from fastapi import APIRouter

from db.database import get_all_machines_status, get_machine_history

router = APIRouter(tags=["status"])


@router.get("/machines/status")
async def machines_status():
    return get_all_machines_status()


@router.get("/machines/{machine_id}/history")
async def machine_history(machine_id: str, limit: int = 100):
    return get_machine_history(machine_id, limit)
