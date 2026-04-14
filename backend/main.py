from fastapi import FastAPI
import pandas as pd
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
import numpy as np


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


app = FastAPI(title="AgroCast Backend API")

# Allow all origins for simplicity; in production, restrict as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Simple health check endpoint."""
    return {"status": "AgroCast running"}


@app.get("/signals")
def get_signals() -> dict:
    """
    Read the precomputed features from the local parquet file, generate a signal
    and return the current price along with the signal details.
    """
    # Determine the path to the features file
    data_path = Path(__file__).resolve().parent / ".." / "data" / "features" / "soybeans_features.parquet"
    # Load features from parquet if available; otherwise generate dummy data
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        # Create a dummy DataFrame with synthetic data if file is missing
        # This is primarily for development; in production the parquet must be present.
        n = 24  # number of periods
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Soybeans": np.linspace(400, 500, n) + rng.normal(scale=10, size=n),
            "DTWEXBGS": np.linspace(90, 100, n) + rng.normal(scale=0.5, size=n),
            "WPU057303": np.linspace(200, 220, n) + rng.normal(scale=2, size=n),
            "TSIFRGHT": np.linspace(180, 190, n) + rng.normal(scale=1, size=n),
        })

    result = generate_signal(df)
    # Add the latest price to the response
    result["price"] = float(df["Soybeans"].iloc[-1])
    return result
