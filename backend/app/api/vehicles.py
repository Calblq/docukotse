from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.db.models import Vehicle
from app.db.session import get_db
from app.schemas import VehicleCreate, VehicleOut, VehicleUpdate

router = APIRouter(prefix="/vehicles", tags=["vehicles"], dependencies=[Depends(get_current_user)])


def _get_or_404(db: Session, vehicle_id: int) -> Vehicle:
    vehicle = db.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Vehicle not found")
    return vehicle


def _commit_or_409(db: Session) -> None:
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Plate already exists") from None


@router.get("", response_model=list[VehicleOut])
def list_vehicles(db: Session = Depends(get_db)) -> list[Vehicle]:
    return list(db.scalars(select(Vehicle).order_by(Vehicle.id)))


@router.post("", response_model=VehicleOut, status_code=status.HTTP_201_CREATED)
def create_vehicle(body: VehicleCreate, db: Session = Depends(get_db)) -> Vehicle:
    vehicle = Vehicle(**body.model_dump())
    db.add(vehicle)
    _commit_or_409(db)
    return vehicle


@router.get("/{vehicle_id}", response_model=VehicleOut)
def get_vehicle(vehicle_id: int, db: Session = Depends(get_db)) -> Vehicle:
    return _get_or_404(db, vehicle_id)


@router.patch("/{vehicle_id}", response_model=VehicleOut)
def update_vehicle(vehicle_id: int, body: VehicleUpdate, db: Session = Depends(get_db)) -> Vehicle:
    vehicle = _get_or_404(db, vehicle_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(vehicle, field, value)
    _commit_or_409(db)
    return vehicle


@router.delete(
    "/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)]
)
def delete_vehicle(vehicle_id: int, db: Session = Depends(get_db)) -> Response:
    """Admin only. The vehicle's documents are kept, unlinked from it."""
    db.delete(_get_or_404(db, vehicle_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
