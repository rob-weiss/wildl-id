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


def get_current_temperature():
    """Fetch current temperature from Open-Meteo API (no API key required).

    Returns
    -------
    float or None
        Current temperature in Celsius, or None if fetch fails
    """
    try:
        # Open-Meteo API - free, no API key required
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m",
            "temperature_unit": "celsius",
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        temperature = data.get("current", {}).get("temperature_2m")
        return temperature
    except Exception as e:
        print(f"⚠ Could not fetch current temperature: {e}")
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

    # Get current time information
    now = datetime.now()
    hour = args.hour if args.hour is not None else now.hour
    day_of_year = args.day if args.day is not None else now.timetuple().tm_yday

    # Get temperature - fetch current if not specified
    if args.temperature is None:
        print("\nFetching current temperature...")
        temperature = get_current_temperature()
        if temperature is None:
            print("Using default temperature: 10°C")
            temperature = 10.0
        else:
            print(f"✓ Current temperature: {temperature}°C")
    else:
        temperature = args.temperature

    # Display input conditions
    print("\nPrediction Conditions:")
    print(f"  Date/Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
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
