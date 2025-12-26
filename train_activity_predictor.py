# %%
"""Wildlife Activity Prediction Model

This script trains a deep neural network to predict wildlife activity levels
for roe deer and wild boar based on environmental conditions (temperature and time).

The model predicts activity relative to the maximum observed activity in the dataset,
providing a normalized activity score between 0 and 1.

Features:
- Temperature (°C)
- Hour of day (0-23)
- Day of year (1-365)
- Month (1-12)

Target:
- Activity level (normalized by species maximum)

Dependencies:
- pandas, numpy, tensorflow, matplotlib, scikit-learn

Usage:
- Configure the model name and species if needed
- Run the script to train and evaluate the model
- Model will be saved to 'models/' directory
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
model_name = "qwen3-vl:235b-a22b-thinking"
image_dir = Path(os.environ["HOME"] + "/mnt/wildlife")
labels_dir = image_dir / f"labels_{model_name}"
script_dir = Path(__file__).parent
output_dir = script_dir / "models"
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("WILDLIFE ACTIVITY PREDICTION MODEL TRAINING")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

csv_path = labels_dir / f"labelling_results_{model_name}.csv"
if not csv_path.exists():
    print(f"Error: Results file not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} labelled images")

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6

# Filter for valid data
df_valid = df[df["timestamp"].notna() & df["temperature_celsius"].notna()].copy()
print(f"{len(df_valid)} images with valid timestamps and temperature")

# Filter for roe deer and wild boar only
target_species = ["roe deer", "wild boar"]
df_target = df_valid[df_valid["class"].isin(target_species)].copy()
print(f"{len(df_target)} images of target species (roe deer, wild boar)")

# Display species distribution
print("\nSpecies distribution:")
print(df_target["class"].value_counts())

# ============================================================================
# 2. CREATE ACTIVITY AGGREGATIONS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING ACTIVITY FEATURES")
print("=" * 70)

# Create bins for hour and temperature to aggregate activity
# We'll aggregate by species, hour, and temperature range to get activity counts
df_target["temp_bin"] = pd.cut(
    df_target["temperature_celsius"],
    bins=np.arange(-15, 35, 2),  # 2-degree bins
    labels=False,
)

# Aggregate activity by species, hour, day_of_year, and temperature bin
activity_groups = (
    df_target.groupby(["class", "hour", "day_of_year", "month", "temp_bin"])
    .size()
    .reset_index(name="activity_count")
)

# Get the actual temperature values for each bin
temp_bins = np.arange(-15, 35, 2)
activity_groups["temperature"] = activity_groups["temp_bin"].apply(
    lambda x: temp_bins[int(x)] + 1 if not pd.isna(x) else np.nan
)

# Remove rows with NaN temperature
activity_groups = activity_groups.dropna(subset=["temperature"])

print(f"Created {len(activity_groups)} activity aggregation records")

# Normalize activity by species maximum
species_max = activity_groups.groupby("class")["activity_count"].max()
print("\nMaximum activity counts by species:")
print(species_max)

activity_groups["activity_normalized"] = activity_groups.apply(
    lambda row: row["activity_count"] / species_max[row["class"]], axis=1
)

# Create separate datasets for each species
roe_deer_data = activity_groups[activity_groups["class"] == "roe deer"].copy()
wild_boar_data = activity_groups[activity_groups["class"] == "wild boar"].copy()

print(f"\nRoe deer records: {len(roe_deer_data)}")
print(f"Wild boar records: {len(wild_boar_data)}")

# ============================================================================
# 3. PREPARE TRAINING DATA
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)


# Also create data with all combinations of hour and temperature to cover full feature space
# This helps the model learn the full pattern even where data is sparse
def create_expanded_dataset(species_data, species_name):
    """Create expanded dataset with synthetic samples for better coverage."""
    # Create grid of all possible hour/temp/month combinations
    hours = np.arange(24)
    temps = np.arange(-10, 30, 2)
    months = np.arange(1, 13)
    days_of_year = np.arange(1, 366)

    # Sample synthetic data based on observed patterns
    synthetic_samples = []
    for hour in hours:
        for temp in temps:
            for month in months:
                # Estimate day_of_year from month (middle of month)
                day = month * 30

                # Check if we have nearby observations
                nearby = species_data[
                    (species_data["hour"] == hour)
                    & (species_data["temperature"].between(temp - 4, temp + 4))
                    & (species_data["month"] == month)
                ]

                if len(nearby) > 0:
                    # Use average of nearby observations
                    activity = nearby["activity_normalized"].mean()
                else:
                    # Use 0 for missing combinations (no activity observed)
                    activity = 0.0

                synthetic_samples.append(
                    {
                        "hour": hour,
                        "temperature": temp,
                        "month": month,
                        "day_of_year": day,
                        "activity_normalized": activity,
                    }
                )

    synthetic_df = pd.DataFrame(synthetic_samples)

    # Combine with original data
    combined_data = pd.concat(
        [
            species_data[
                ["hour", "temperature", "month", "day_of_year", "activity_normalized"]
            ],
            synthetic_df,
        ],
        ignore_index=True,
    )

    # Remove duplicates, keeping the original data when available
    combined_data = combined_data.drop_duplicates(
        subset=["hour", "temperature", "month"], keep="first"
    )

    return combined_data


# Train separate models for each species
def train_species_model(species_data, species_name):
    """Train a neural network for a specific species."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING MODEL FOR {species_name.upper()}")
    print("=" * 70)

    # Expand dataset for better coverage
    data = create_expanded_dataset(species_data, species_name)
    print(f"Expanded dataset: {len(data)} samples")

    # Prepare features and target
    features = ["hour", "temperature", "month", "day_of_year"]
    X = data[features].values
    y = data["activity_normalized"].values

    # Add cyclical encoding for hour and day_of_year
    hour_sin = np.sin(2 * np.pi * X[:, 0] / 24)
    hour_cos = np.cos(2 * np.pi * X[:, 0] / 24)
    day_sin = np.sin(2 * np.pi * X[:, 3] / 365)
    day_cos = np.cos(2 * np.pi * X[:, 3] / 365)
    month_sin = np.sin(2 * np.pi * X[:, 2] / 12)
    month_cos = np.cos(2 * np.pi * X[:, 2] / 12)

    X_enhanced = np.column_stack(
        [
            X[:, 1],  # temperature
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            month_sin,
            month_cos,
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train_scaled.shape[1]}")

    # Build neural network
    model = keras.Sequential(
        [
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Output between 0 and 1
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    print("\nModel architecture:")
    model.summary()

    # Train model
    print("\nTraining model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    # Evaluate model
    print("\nEvaluating model...")
    train_loss, train_mae, train_mse = model.evaluate(
        X_train_scaled, y_train, verbose=0
    )
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test, verbose=0)

    print(
        f"Training - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}"
    )
    print(f"Test - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")

    # Make predictions
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training history
    ax = axes[0, 0]
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predictions vs Actual
    ax = axes[0, 1]
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=2)
    ax.set_xlabel("Actual Activity")
    ax.set_ylabel("Predicted Activity")
    ax.set_title("Predictions vs Actual")
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1, 0]
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{species_name.title()} - Activity Prediction Model",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{species_name.replace(' ', '_')}_model_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved evaluation plot: {plot_path}")
    plt.close()

    # Save model
    model_path = output_dir / f"{species_name.replace(' ', '_')}_activity_model.keras"
    model.save(model_path)
    print(f"✓ Saved model: {model_path}")

    # Save scaler
    import pickle

    scaler_path = output_dir / f"{species_name.replace(' ', '_')}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")

    return model, scaler, history, (y_test, y_pred)


# Train models for each species
roe_deer_model, roe_deer_scaler, roe_deer_history, roe_deer_results = (
    train_species_model(roe_deer_data, "roe deer")
)

wild_boar_model, wild_boar_scaler, wild_boar_history, wild_boar_results = (
    train_species_model(wild_boar_data, "wild boar")
)

# ============================================================================
# 4. VISUALIZE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PREDICTION VISUALIZATIONS")
print("=" * 70)


def visualize_activity_predictions(model, scaler, species_name, temp_range=(-10, 25)):
    """Create visualization of predicted activity across temperature and time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Activity heatmap: hour vs temperature
    ax = axes[0, 0]
    hours = np.arange(24)
    temps = np.arange(temp_range[0], temp_range[1], 1)

    activity_grid = np.zeros((len(temps), len(hours)))

    for i, temp in enumerate(temps):
        for j, hour in enumerate(hours):
            # Use middle of the year (day 180, month 6)
            day = 180
            month = 6

            # Prepare features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 365)
            day_cos = np.cos(2 * np.pi * day / 365)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X = np.array(
                [[temp, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]]
            )
            X_scaled = scaler.transform(X)
            activity_grid[i, j] = model.predict(X_scaled, verbose=0)[0, 0]

    im = ax.imshow(activity_grid, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(range(0, 24, 3))
    ax.set_yticks(range(0, len(temps), 5))
    ax.set_yticklabels(temps[::5])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Predicted Activity: Hour vs Temperature")
    plt.colorbar(im, ax=ax, label="Activity (normalized)")

    # 2. Activity by hour at different temperatures
    ax = axes[0, 1]
    test_temps = [-5, 0, 5, 10, 15, 20]
    for temp in test_temps:
        activities = []
        for hour in hours:
            day = 180
            month = 6
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 365)
            day_cos = np.cos(2 * np.pi * day / 365)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X = np.array(
                [[temp, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]]
            )
            X_scaled = scaler.transform(X)
            activities.append(model.predict(X_scaled, verbose=0)[0, 0])

        ax.plot(hours, activities, marker="o", label=f"{temp}°C")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Activity")
    ax.set_title("Activity by Hour at Different Temperatures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 3))

    # 3. Activity by temperature at different hours
    ax = axes[1, 0]
    test_hours = [0, 6, 12, 18]
    temps_range = np.arange(-10, 25, 0.5)
    for hour in test_hours:
        activities = []
        for temp in temps_range:
            day = 180
            month = 6
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 365)
            day_cos = np.cos(2 * np.pi * day / 365)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X = np.array(
                [[temp, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]]
            )
            X_scaled = scaler.transform(X)
            activities.append(model.predict(X_scaled, verbose=0)[0, 0])

        ax.plot(temps_range, activities, marker=".", label=f"{hour}:00")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Predicted Activity")
    ax.set_title("Activity by Temperature at Different Hours")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Seasonal variation (activity by month)
    ax = axes[1, 1]
    months = np.arange(1, 13)
    for hour in [6, 12, 18, 21]:
        activities = []
        for month in months:
            temp = 10  # Fixed temperature
            day = month * 30
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 365)
            day_cos = np.cos(2 * np.pi * day / 365)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X = np.array(
                [[temp, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]]
            )
            X_scaled = scaler.transform(X)
            activities.append(model.predict(X_scaled, verbose=0)[0, 0])

        ax.plot(months, activities, marker="o", label=f"{hour}:00")

    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Activity")
    ax.set_title("Seasonal Activity Variation (at 10°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))

    plt.suptitle(
        f"{species_name.title()} - Activity Predictions", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{species_name.replace(' ', '_')}_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved predictions plot: {plot_path}")
    plt.close()


visualize_activity_predictions(roe_deer_model, roe_deer_scaler, "roe deer")
visualize_activity_predictions(wild_boar_model, wild_boar_scaler, "wild boar")

# ============================================================================
# 5. SAVE MODEL INFORMATION
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL INFORMATION")
print("=" * 70)

# Save model metadata
metadata = {
    "model_name": model_name,
    "target_species": target_species,
    "features": [
        "temperature",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
    ],
    "roe_deer": {
        "samples": len(roe_deer_data),
        "max_activity": float(species_max["roe deer"]),
        "test_mae": float(roe_deer_results[0].shape[0]),
    },
    "wild_boar": {
        "samples": len(wild_boar_data),
        "max_activity": float(species_max["wild boar"]),
        "test_mae": float(wild_boar_results[0].shape[0]),
    },
}

import json

metadata_path = output_dir / "model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata: {metadata_path}")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"Models and artifacts saved to: {output_dir}")
print("\nTo use the trained models:")
print("1. Load the model: model = keras.models.load_model('path/to/model.keras')")
print("2. Load the scaler: scaler = pickle.load(open('path/to/scaler.pkl', 'rb'))")
print("3. Prepare features with cyclical encoding")
print("4. Scale and predict: predictions = model.predict(scaler.transform(features))")

# %%
