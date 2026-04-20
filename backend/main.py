import hashlib
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field


def compute_trend(series, window: int = 6) -> float:
    """
    Compute a simple slope over the last `window` values of a time series.
    """
    y = series[-window:]
    x = np.arange(len(y))
    # Fit a line and return the slope (first coefficient)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def normalize(value: float, threshold: float = 0.01) -> int:
    """
    Normalize a numeric trend value into -1, 0, or 1 based on a threshold.
    """
    if value > threshold:
        return 1
    elif value < -threshold:
        return -1
    return 0


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Generate a buy/sell/hold signal for the soybean market based on the
    underlying price trend and context from exogenous variables.

    The dataframe must contain the following columns:
    - 'Soybeans': price series of the commodity
    - 'DTWEXBGS': dollar index (inverse relationship)
    - 'WPU057303': diesel price (direct relationship)
    - 'TSIFRGHT': freight/logistics index (direct relationship)

    Returns a dictionary with the signal, confidence, trend label, driver impact,
    and explanation.
    """
    # --- 1. Trend of the model (soybean price) ---
    price_trend = compute_trend(df["Soybeans"])
    trend_score = normalize(price_trend)
    if trend_score == 1:
        trend_label = "alcista"
    elif trend_score == -1:
        trend_label = "bajista"
    else:
        trend_label = "lateral"

    # --- 2. Contextual trends ---
    dollar_trend = normalize(compute_trend(df["DTWEXBGS"]))
    diesel_trend = normalize(compute_trend(df["WPU057303"]))
    logistics_trend = normalize(compute_trend(df["TSIFRGHT"]))

    # Economic logic: dollar has inverse effect, diesel and logistics positive
    score_context = (
        (-1 * dollar_trend) +
        (1 * diesel_trend) +
        (0.5 * logistics_trend)
    )

    # --- 3. Aggregate score combining model and context ---
    score_total = (0.7 * trend_score) + (0.3 * score_context)

    # --- 4. Final signal ---
    if score_total > 0.5:
        signal = "VENDER"
    elif score_total < -0.5:
        signal = "COMPRAR"
    else:
        signal = "ESPERAR"

    # --- 5. Confidence metric (bounded between 0 and 1) ---
    confidence = min(1.0, abs(score_total))

    # --- 6. Explanation for the decision ---
    explanation = []
    if trend_label == "alcista":
        explanation.append("Tendencia proyectada al alza")
    elif trend_label == "bajista":
        explanation.append("Tendencia proyectada a la baja")

    if diesel_trend == 1:
        explanation.append("Aumento en costos logísticos")
    if dollar_trend == -1:
        explanation.append("Debilitamiento del dólar")
    if logistics_trend == 1:
        explanation.append("Mayor presión en transporte")

    # --- 7. Drivers dictionary showing the influence of each factor ---
    drivers = {
        "dolar": -1 * dollar_trend,
        "diesel": diesel_trend,
        "logistica": logistics_trend
    }

    return {
        "signal": signal,
        "confidence": confidence,
        "trend": trend_label,
        "drivers": drivers,
        "explanation": explanation
    }


class Mode(str, Enum):
    producer = "producer"
    trader = "trader"


class SessionModePayload(BaseModel):
    session_id: str = Field(min_length=3, max_length=128)
    mode: Mode


class InteractionEventPayload(BaseModel):
    session_id: str = Field(min_length=3, max_length=128)
    mode: Mode
    time_to_decision: float = Field(ge=0.0)
    module_interaction_rate: float = Field(ge=0.0, le=1.0)
    redesigned_experience: bool


app = FastAPI(title="AgroCast Backend API")

# Allow all origins for simplicity; in production, restrict as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SESSION_MODES: dict[str, Mode] = {}
INTERACTION_EVENTS: list[dict[str, Any]] = []


def get_redesign_rollout() -> int:
    """
    Read rollout percentage from env var FEATURE_REDESIGN_ROLLOUT.
    Expected value: integer between 0 and 100.
    """
    value = os.getenv("FEATURE_REDESIGN_ROLLOUT", "0")
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return max(0, min(100, parsed))


def is_redesign_enabled_for_session(session_id: str) -> bool:
    """
    Deterministically assign sessions to the redesigned experience based on
    FEATURE_REDESIGN_ROLLOUT.
    """
    rollout = get_redesign_rollout()
    if rollout <= 0:
        return False
    if rollout >= 100:
        return True

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < rollout


def build_context_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Context modules moved to a secondary 'Contexto' tab:
    - WASDE (placeholder signal)
    - news (placeholder sentiment)
    - forecast (if available)
    """
    context: dict[str, Any] = {
        "wasde": {
            "bias": "neutral",
            "summary": "Sin cambios estructurales relevantes en última publicación."
        },
        "news": {
            "sentiment": "mixto",
            "headline_count": 0
        }
    }
    forecast_path = Path(__file__).resolve().parent / ".." / "artifacts" / "forecast.csv"
    if forecast_path.exists():
        fc_df = pd.read_csv(forecast_path)
        context["forecast"] = fc_df.to_dict(orient="records")
    return context


def build_producer_dashboard(df: pd.DataFrame, signal_data: dict[str, Any]) -> dict[str, Any]:
    """
    Minimal payload for producer mode dashboard:
    - precio neto
    - ROI por plazo
    - puertos
    - recomendación vender/retener
    """
    latest_price = float(df["Soybeans"].iloc[-1])
    net_price = latest_price * 0.97
    return {
        "mode": Mode.producer,
        "producer_dashboard": {
            "precio_neto": round(net_price, 2),
            "roi_por_plazo": {
                "30d": round((latest_price / max(1.0, latest_price - 8) - 1) * 100, 2),
                "90d": round((latest_price / max(1.0, latest_price - 20) - 1) * 100, 2),
                "180d": round((latest_price / max(1.0, latest_price - 35) - 1) * 100, 2),
            },
            "puertos": [
                {"nombre": "Rosario", "basis": -3.1},
                {"nombre": "Bahía Blanca", "basis": -4.2},
                {"nombre": "Quequén", "basis": -3.8},
            ],
            "recomendacion": "vender" if signal_data["signal"] == "VENDER" else "retener",
        },
    }


def build_trader_dashboard(signal_data: dict[str, Any], latest_price: float) -> dict[str, Any]:
    """
    Minimal payload for trader mode dashboard:
    - señal
    - score
    - entry/stop/TP
    - R
    - performance operativa
    """
    risk = max(1.0, latest_price * 0.01)
    entry = latest_price
    stop = entry - risk if signal_data["signal"] == "VENDER" else entry + risk
    tp = entry + (2 * risk) if signal_data["signal"] == "VENDER" else entry - (2 * risk)
    return {
        "mode": Mode.trader,
        "trader_dashboard": {
            "senal": signal_data["signal"],
            "score": round(signal_data["confidence"] * 100, 1),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "take_profit": round(tp, 2),
            "r_multiple": 2.0,
            "performance_operativa": {
                "win_rate": 0.56,
                "profit_factor": 1.42,
                "max_drawdown": -0.08,
            },
        },
    }


def load_features_df() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / ".." / "data" / "features" / "soybeans_features.parquet"
    if data_path.exists():
        return pd.read_parquet(data_path)

    n = 24
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Soybeans": np.linspace(400, 500, n) + rng.normal(scale=10, size=n),
        "DTWEXBGS": np.linspace(90, 100, n) + rng.normal(scale=0.5, size=n),
        "WPU057303": np.linspace(200, 220, n) + rng.normal(scale=2, size=n),
        "TSIFRGHT": np.linspace(180, 190, n) + rng.normal(scale=1, size=n),
    })


@app.get("/")
def root() -> dict:
    """Simple health check endpoint."""
    return {"status": "AgroCast running"}


@app.get("/signals")
def get_signals(
    mode: Mode = Query(default=Mode.producer),
    include_context: bool = Query(default=False),
    session_id: str = Query(default="anonymous"),
) -> dict:
    """
    Read the precomputed features from the local parquet file, generate a signal
    and return the current price along with the signal details.
    """
    df = load_features_df()
    result = generate_signal(df)
    latest_price = float(df["Soybeans"].iloc[-1])
    redesigned_experience = is_redesign_enabled_for_session(session_id)

    if not redesigned_experience:
        legacy_response = {
            **result,
            "price": latest_price,
            "feature_flags": {
                "redesign_dashboard": False,
                "rollout_percent": get_redesign_rollout(),
            },
        }
        forecast_path = Path(__file__).resolve().parent / ".." / "artifacts" / "forecast.csv"
        if forecast_path.exists():
            fc_df = pd.read_csv(forecast_path)
            legacy_response["forecast"] = fc_df.to_dict(orient="records")
        return legacy_response

    payload = (
        build_producer_dashboard(df, result)
        if mode == Mode.producer
        else build_trader_dashboard(result, latest_price)
    )
    payload["feature_flags"] = {
        "redesign_dashboard": True,
        "rollout_percent": get_redesign_rollout(),
    }
    payload["session_mode"] = SESSION_MODES.get(session_id, mode)
    if include_context:
        payload["contexto"] = build_context_data(df)

    return payload


@app.put("/session/mode")
def set_session_mode(payload: SessionModePayload) -> dict:
    """
    Persist global selector in backend session state.
    Frontend can keep it in session storage and sync here.
    """
    SESSION_MODES[payload.session_id] = payload.mode
    return {"session_id": payload.session_id, "mode": payload.mode}


@app.get("/session/mode")
def get_session_mode(session_id: str = Query(min_length=3, max_length=128)) -> dict:
    """
    Retrieve persisted mode for a session.
    """
    mode = SESSION_MODES.get(session_id)
    if mode is None:
        raise HTTPException(status_code=404, detail="session mode not found")
    return {"session_id": session_id, "mode": mode}


@app.post("/metrics/interaction")
def record_interaction(payload: InteractionEventPayload) -> dict:
    """
    Product metrics to validate simplified experience:
    - time_to_decision
    - module_interaction_rate
    """
    event = payload.model_dump()
    event["recorded_at"] = datetime.now(timezone.utc).isoformat()
    INTERACTION_EVENTS.append(event)
    return {"status": "ok", "events_recorded": len(INTERACTION_EVENTS)}


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    """
    Aggregated metrics split by redesigned_experience flag
    to compare pre/post redesign behavior.
    """
    if not INTERACTION_EVENTS:
        return {"total_events": 0, "groups": {}}

    groups: dict[str, list[dict[str, Any]]] = {"redesign_true": [], "redesign_false": []}
    for event in INTERACTION_EVENTS:
        key = "redesign_true" if event["redesigned_experience"] else "redesign_false"
        groups[key].append(event)

    response_groups: dict[str, Any] = {}
    for key, events in groups.items():
        if not events:
            continue
        ttd = [e["time_to_decision"] for e in events]
        mir = [e["module_interaction_rate"] for e in events]
        response_groups[key] = {
            "events": len(events),
            "avg_time_to_decision": round(float(np.mean(ttd)), 3),
            "avg_module_interaction_rate": round(float(np.mean(mir)), 3),
        }

    return {
        "total_events": len(INTERACTION_EVENTS),
        "groups": response_groups,
    }
       
