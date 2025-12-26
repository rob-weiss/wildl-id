# %%
"""Wildlife Activity Predictor - Real-time Predictions

This script loads trained wildlife activity prediction models and makes predictions
for the current time and specified temperature.

The script predicts activity levels for roe deer and wild boar based on:
- Current hour of day
- Current day of year
- Temperature (user-specified or default)

Output is a normalized activity score between 0 and 1.

Usage:
- Run the script to get predictions for current time
- Optionally specify a temperature (default: 10°C)
- Use --temperature flag to set custom temperature
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from tensorflow import keras

# Configuration
script_dir = Path(__file__).parent
model_dir = script_dir / "models"

# Location for weather data (Renningen, Germany)
LATITUDE = 48.7667
LONGITUDE = 8.9333

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Predict wildlife activity for current time"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Temperature in Celsius (default: fetch current temperature)",
)
parser.add_argument(
    "--hour",
    type=int,
    default=None,
    help="Hour of day (0-23), default: current hour",
)
parser.add_argument(
    "--day",
    type=int,
    default=None,
    help="Day of year (1-365), default: current day",
)
parser.add_argument(
    "--datetime",
    type=str,
    default=None,
    help="Datetime for prediction in ISO format (YYYY-MM-DD HH:MM), e.g., '2025-12-27 01:00'",
)
args = parser.parse_args()


def prepare_features(hour, day_of_year, temperature):
    """Prepare features with cyclical encoding for prediction.

    Parameters
    ----------
    hour : int
        Hour of day (0-23)
    day_of_year : int
        Day of year (1-365)
    temperature : float
        Temperature in Celsius

    Returns
    -------
    np.ndarray
        Feature array with shape (1, 5): [temp, hour_sin, hour_cos, day_sin, day_cos]
    """
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)

    return np.array([[temperature, hour_sin, hour_cos, day_sin, day_cos]])


def load_model_and_scaler(species_name):
    """Load trained model and scaler for a species.

    Parameters
    ----------
    species_name : str
        Species name (e.g., "roe deer", "wild boar")

    Returns
    -------
    tuple
        (model, scaler) or (None, None) if not found
    """
    species_file = species_name.replace(" ", "_")
    model_path = model_dir / f"{species_file}_activity_model.keras"
    scaler_path = model_dir / f"{species_file}_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print(f"⚠ Model files not found for {species_name}")
        return None, None

    model = keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_activity(model, scaler, hour, day_of_year, temperature):
    """Make activity prediction for given conditions.

    Parameters
    ----------
    model : keras.Model
        Trained neural network model
    scaler : StandardScaler
        Feature scaler
    hour : int
        Hour of day (0-23)
    day_of_year : int
        Day of year (1-365)
    temperature : float
        Temperature in Celsius

    Returns
    -------
    float
        Predicted activity level (0-1)
    """
    features = prepare_features(hour, day_of_year, temperature)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled, verbose=0)[0, 0]
    return prediction


def get_temperature_at_datetime(target_datetime=None):
    """Fetch temperature from Open-Meteo API (no API key required).

    Parameters
    ----------
    target_datetime : datetime or None
        Target datetime for forecast. If None, fetches current temperature.

    Returns
    -------
    float or None
        Temperature in Celsius at target time, or None if fetch fails
    """
    try:
        # Open-Meteo API - free, no API key required
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "temperature_unit": "celsius",
        }

        if target_datetime is None:
            # Get current temperature
            params["current"] = "temperature_2m"
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            temperature = data.get("current", {}).get("temperature_2m")
        else:
            # Get forecast temperature
            params["hourly"] = "temperature_2m"
            params["forecast_days"] = "7"
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            # Find the matching hour in forecast
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            # Format target datetime to match API format (YYYY-MM-DDTHH:00)
            target_str = target_datetime.strftime("%Y-%m-%dT%H:00")

            if target_str in times:
                idx = times.index(target_str)
                temperature = temps[idx]
            else:
                print(f"⚠ Forecast not available for {target_str}")
                return None

        return temperature
    except Exception as e:
        print(f"⚠ Could not fetch temperature: {e}")
        return None


def get_activity_description(activity_score):
    """Convert activity score to descriptive text.

    Parameters
    ----------
    activity_score : float
        Activity score between 0 and 1

    Returns
    -------
    str
        Description of activity level
    """
    if activity_score < 0.1:
        return "Very Low"
    elif activity_score < 0.3:
        return "Low"
    elif activity_score < 0.5:
        return "Moderate"
    elif activity_score < 0.7:
        return "High"
    else:
        return "Very High"


def main():
    """Main prediction function."""
    print("=" * 70)
    print("WILDLIFE ACTIVITY PREDICTION")
    print("=" * 70)

    # Parse datetime if provided
    if args.datetime:
        try:
            target_datetime = datetime.strptime(args.datetime, "%Y-%m-%d %H:%M")
            is_forecast = target_datetime > datetime.now()
        except ValueError:
            print("⚠ Invalid datetime format. Use: YYYY-MM-DD HH:MM")
            return
    else:
        target_datetime = datetime.now()
        is_forecast = False

    # Get time information
    hour = args.hour if args.hour is not None else target_datetime.hour
    day_of_year = (
        args.day if args.day is not None else target_datetime.timetuple().tm_yday
    )

    # Get temperature - fetch from API if not specified
    if args.temperature is None:
        if is_forecast:
            print(
                f"\nFetching temperature forecast for {target_datetime.strftime('%Y-%m-%d %H:%M')}..."
            )
            temperature = get_temperature_at_datetime(target_datetime)
        else:
            print("\nFetching current temperature...")
            temperature = get_temperature_at_datetime()

        if temperature is None:
            print("Using default temperature: 10°C")
            temperature = 10.0
        else:
            forecast_label = "forecast" if is_forecast else "current"
            print(f"✓ Temperature {forecast_label}: {temperature}°C")
    else:
        temperature = args.temperature

    # Display input conditions
    print("\nPrediction Conditions:")
    if is_forecast:
        print(
            f"  Date/Time: {target_datetime.strftime('%Y-%m-%d %H:%M:%S')} (forecast)"
        )
    else:
        print(f"  Date/Time: {target_datetime.strftime('%Y-%m-%d %H:%M:%S')} (current)")
    print(f"  Hour: {hour:02d}:00")
    print(f"  Day of Year: {day_of_year}")
    print(f"  Temperature: {temperature}°C")

    # Species to predict
    species_list = ["roe deer", "wild boar"]

    print("\n" + "=" * 70)
    print("ACTIVITY PREDICTIONS")
    print("=" * 70)

    predictions = {}

    for species in species_list:
        model, scaler = load_model_and_scaler(species)

        if model is None:
            continue

        activity = predict_activity(model, scaler, hour, day_of_year, temperature)
        predictions[species] = activity

        description = get_activity_description(activity)

        print(f"\n{species.title()}:")
        print(f"  Activity Score: {activity:.3f}")
        print(f"  Activity Level: {description}")
        print(
            f"  Likelihood: {'█' * int(activity * 50)}{' ' * (50 - int(activity * 50))} {activity * 100:.1f}%"
        )

    # Compare species
    if len(predictions) == 2:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        roe_deer_activity = predictions.get("roe deer", 0)
        wild_boar_activity = predictions.get("wild boar", 0)

        if roe_deer_activity > wild_boar_activity:
            diff = roe_deer_activity - wild_boar_activity
            print(f"\nRoe deer are more active by {diff * 100:.1f} percentage points")
        elif wild_boar_activity > roe_deer_activity:
            diff = wild_boar_activity - roe_deer_activity
            print(f"\nWild boar are more active by {diff * 100:.1f} percentage points")
        else:
            print("\nBoth species have similar activity levels")

    # Recommendations
    print("\n" + "=" * 70)
    print("CAMERA PLACEMENT RECOMMENDATIONS")
    print("=" * 70)

    best_species = max(predictions.items(), key=lambda x: x[1])
    print(f"\nBest time for wildlife observation: {best_species[0].title()}")
    print(f"Expected activity level: {get_activity_description(best_species[1])}")

    if best_species[1] < 0.3:
        print("\n⚠ Low activity expected at this time and temperature.")
        print("Consider checking predictions for different hours or temperatures.")
    elif best_species[1] > 0.6:
        print("\n✓ High activity expected! Good conditions for wildlife observation.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

# %%
