# app.py
# FastAPI backend for LogiQ Fleet Optimization

import math
import random
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
    except Exception:
        AerEstimator = None
    QISKIT_OK = True
except Exception:
    QISKIT_OK = False
    QuantumCircuit = SparsePauliOp = Estimator = AerEstimator = None

OSRM_BASE = "https://router.project-osrm.org"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

app = FastAPI(title="LogiQ Routing API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class LatLng(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    start: LatLng
    end: LatLng
    vehicle: str
    cc: str

# ---------- Toll Pricing ----------
def vehicle_toll_class(vehicle: str, cc: str) -> str:
    v = (vehicle or "").lower()
    if v == "van":
        return "lcv"
    if v == "truck":
        return "truck2"
    if v == "lorry":
        return "axle3plus"
    return "car"

TOLL_CLASS_MULTIPLIER = {
    "car": 1.00,
    "lcv": 1.50,
    "truck2": 2.50,
    "axle3plus": 3.50,
}

BASE_RATE_PER_KM = 1.2  # ₹ per km baseline
MIN_TOLL = 30           # Minimum toll per plaza

def price_toll_for_segment_km(segment_km: float, vclass: str) -> int:
    mult = TOLL_CLASS_MULTIPLIER.get(vclass, 1.0)
    raw = BASE_RATE_PER_KM * float(segment_km) * mult
    rounded = max(MIN_TOLL, 5 * round(raw / 5))
    return int(rounded)

# ---------- Utils ----------
def format_eta(seconds: float) -> str:
    """Convert seconds to hours+minutes string"""
    total_minutes = max(1, int(round(seconds / 60)))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

def vehicle_factor(vehicle: str, cc: str) -> float:
    vehicle_map = {"van": 1.0, "truck": 1.08, "lorry": 1.15}
    v = vehicle_map.get(vehicle, 1.0)
    try:
        ccn = int(cc.split()[0])
    except Exception:
        ccn = 1500
    if ccn <= 2000:
        c = 1.0
    elif ccn <= 5000:
        c = 0.97
    else:
        c = 0.95
    return v * c

def mock_tolls_for_distance(distance_m: float) -> List[Dict[str, Any]]:
    km = distance_m / 1000.0
    n = max(0, int(round(km / 80.0)))
    return [{"lat": None, "lng": None, "cost": None, "place": None} for _ in range(n)]

def traffic_penalty_seconds(geojson_line: Dict[str, Any]) -> float:
    coords: List[List[float]] = geojson_line.get("coordinates", [])
    if len(coords) < 3:
        return 0.0
    headings = []
    for i in range(1, len(coords)):
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    turns = [abs(headings[i] - headings[i-1]) for i in range(1, len(headings))]
    wiggle = sum(min(t, 2.5) for t in turns) / max(1, len(turns))
    return 60.0 * wiggle

def quantum_score(distance_m: float, duration_s: float, toll_rupees: int) -> float:
    return 1.0 * (duration_s / 60.0) + 0.02 * (distance_m / 1000.0) + 0.2 * toll_rupees

def qaoa_pick_min(costs: List[float]) -> int:
    N = len(costs)
    if N == 0:
        return 0
    if not QISKIT_OK or QuantumCircuit is None:
        return int(min(range(N), key=lambda i: costs[i]))
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
    maxc = max(1.0, max(costs))
    for i, c in enumerate(costs):
        qc.rz((c / maxc) * math.pi, i)
    for i in range(N):
        qc.rx(math.pi / 2, i)
    est = None
    if QISKIT_OK and AerEstimator is not None:
        try:
            est = AerEstimator()
        except Exception:
            est = None
    if est is None:
        try:
            est = Estimator()
        except Exception:
            return int(min(range(N), key=lambda i: costs[i]))
    z_ops = [SparsePauliOp.from_list([("I"*i + "Z" + "I"*(N-i-1), 1.0)]) for i in range(N)]
    probs1 = []
    for i in range(N):
        ev = est.run(circuits=qc, observables=z_ops[i]).result().values[0]
        probs1.append((1 - ev) / 2.0)
    return int(max(range(N), key=lambda i: probs1[i]))

def geojson_to_latlng_polyline(geojson_line: Dict[str, Any]) -> List[List[float]]:
    coords = geojson_line.get("coordinates", [])
    return [[lat, lon] for lon, lat in coords]

async def osrm_alternatives(start: LatLng, end: LatLng, alternatives: int = 3) -> List[Dict[str, Any]]:
    coords = f"{start.lng},{start.lat};{end.lng},{end.lat}"
    url = f"{OSRM_BASE}/route/v1/driving/{coords}"
    params = {"alternatives": "true" if alternatives > 0 else "false", "overview": "full", "geometries": "geojson", "steps": "false"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    return data.get("routes", [])

# ---------- API Endpoints ----------
@app.get("/geocode")
async def geocode(query: str = Query(..., min_length=2)):
    params = {"format": "json", "q": query}
    headers = {"User-Agent": "LogiQ-Demo/2.5"}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        r = await client.get(f"{NOMINATIM_BASE}/search", params=params)
        r.raise_for_status()
        arr = r.json()
    if not arr:
        return {"lat": None, "lng": None}
    return {"lat": float(arr[0]["lat"]), "lng": float(arr[0]["lon"])}

@app.get("/reverse_geocode")
async def reverse_geocode(lat: float, lng: float):
    params = {"format": "json", "lat": lat, "lon": lng}
    headers = {"User-Agent": "LogiQ-Demo/2.5"}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        r = await client.get(f"{NOMINATIM_BASE}/reverse", params=params)
        r.raise_for_status()
        data = r.json()
    return {"name": data.get("display_name", "Unknown Place")}

@app.post("/routes")
async def routes(payload: RouteRequest):
    alts = await osrm_alternatives(payload.start, payload.end, alternatives=3)
    if not alts:
        return {"error": "No routes found"}

    vf = vehicle_factor(payload.vehicle, payload.cc)

    # --- Classical ---
    classical_osrm = min(alts, key=lambda r: r["duration"])
    classical_duration = classical_osrm["duration"] * vf
    classical_distance = classical_osrm["distance"]
    classical_poly = geojson_to_latlng_polyline(classical_osrm["geometry"])
    classical = {"polyline": classical_poly, "eta": format_eta(classical_duration)}

    # --- Traffic ---
    if len(alts) > 1:
        traffic_osrm = [r for r in alts if r != classical_osrm][0]
        traffic_duration = traffic_osrm["duration"] * vf + traffic_penalty_seconds(traffic_osrm["geometry"])
        traffic_poly = geojson_to_latlng_polyline(traffic_osrm["geometry"])
    else:
        traffic_duration = classical_duration * 1.15
        traffic_poly = [[lat + 0.002, lng + 0.002] for lat, lng in classical_poly]
    traffic = {"polyline": traffic_poly, "eta": format_eta(traffic_duration)}

    # --- Tolls ---
    toll_items = mock_tolls_for_distance(classical_distance)
    if toll_items:
        n = len(toll_items)
        vclass = vehicle_toll_class(payload.vehicle, payload.cc)
        segment_km = (classical_distance / 1000.0) / max(1, n)
        for idx, toll in enumerate(toll_items):
            k = int((idx + 1) * (len(classical_poly) / (n + 1)))
            k = max(0, min(len(classical_poly) - 1, k))
            toll["lat"], toll["lng"] = classical_poly[k]
            try:
                res = await reverse_geocode(toll["lat"], toll["lng"])
                toll["place"] = res["name"]
            except Exception:
                toll["place"] = "Unknown Toll Location"
            toll["cost"] = price_toll_for_segment_km(segment_km, vclass)

    toll_total_rupees = sum((t.get("cost") or 0) for t in toll_items)

    # --- Quantum ---
    if len(alts) > 1:
        candidates = []
        for r in alts:
            dur = r["duration"] * vf
            dist = r["distance"]
            candidates.append({"polyline": geojson_to_latlng_polyline(r["geometry"]), "duration_s": dur, "distance_m": dist})
        costs = [quantum_score(c["distance_m"], c["duration_s"], toll_total_rupees) for c in candidates]
        q_idx = qaoa_pick_min(costs)
        quantum_cand = candidates[q_idx]
        q_duration = quantum_cand["duration_s"] * (0.98 + 0.04 * random.random())
        quantum_poly = quantum_cand["polyline"]
    else:
        q_duration = classical_duration * (0.9 + 0.1 * random.random())
        quantum_poly = [[lat - 0.002, lng - 0.002] for lat, lng in classical_poly]
    quantum = {"polyline": quantum_poly, "eta": format_eta(q_duration)}

    return {
        "traffic": traffic,
        "classical": classical,
        "quantum": quantum,
        "tolls": toll_items,
        "vehicle": payload.vehicle,
        "cc": payload.cc
    }

# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
