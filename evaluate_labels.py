# %%
"""Wildlife Camera Activity Analysis and Visualization

This script analyzes wildlife camera labelling results and generates visualizations
similar to the SECACAM app activity center, including:
- Activity patterns by hour of day
- Species distribution charts
- Calendar heatmaps
- Location-based statistics
- Day/night activity patterns
- Temperature correlations

Dependencies:
- pandas, matplotlib, seaborn, numpy

Usage:
- Configure the image_dir and model name
- Run the script to generate comprehensive visualizations
"""

import os
import shutil
from pathlib import Path

import ephem
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astral import LocationInfo
from astral.moon import phase
from astral.sun import sun

# Configuration
model = "qwen3-vl:235b-a22b-thinking"
image_dir = Path(os.environ["HOME"] + "/mnt/wildlife")
labels_dir = image_dir / f"labels_{model}"
# Put visualizations in docs/diagrams
script_dir = Path(__file__).parent
output_dir = script_dir / "docs" / "diagrams"

# Delete old visualizations directory and recreate it
if output_dir.exists():
    print(f"Clearing old visualizations from {output_dir}")
    try:
        # Remove directory tree, ignoring errors for read-only files
        def handle_remove_error(func, path, exc):
            os.chmod(path, 0o777)
            func(path)

        shutil.rmtree(
            output_dir,
            ignore_errors=False,
            onerror=handle_remove_error,
        )
        print("✓ Cleared old visualizations")
    except Exception as e:
        print(f"⚠ Warning: Could not fully clear old visualizations: {e}")
        # Try to continue anyway
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize plot counter for sequential numbering
plot_num = 0

# Location for sunrise/sunset calculations
# Renningen, Baden-Württemberg, Germany
LATITUDE = 48.7667
LONGITUDE = 8.9333
TIMEZONE = "Europe/Berlin"

# Load the data
csv_path = labels_dir / f"labelling_results_{model}.csv"
if not csv_path.exists():
    print(f"Error: Results file not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} labelled images")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date
df["weekday"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year

# Filter out invalid data
df_valid = df[df["timestamp"].notna()].copy()
print(f"\n{len(df_valid)} images with valid timestamps")

# Filter out 'none' and 'unknown' classes
df = df[~df["class"].isin(["none", "unknown"])].copy()
df_valid = df_valid[~df_valid["class"].isin(["none", "unknown"])].copy()
print(f"{len(df_valid)} images after filtering out 'none' and 'unknown' classes")

# Data Quality Summary
print("\n" + "=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)
print(f"Total images: {len(df)}")
print(
    f"Images with valid timestamps: {len(df_valid)} ({100 * len(df_valid) / len(df):.1f}%)"
)
print(
    f"Images with temperature data: {df['temperature_celsius'].notna().sum()} ({100 * df['temperature_celsius'].notna().sum() / len(df):.1f}%)"
)
if len(df_valid) > 0:
    date_range = df_valid["date"].max() - df_valid["date"].min()
    print(
        f"Date range: {df_valid['date'].min()} to {df_valid['date'].max()} ({date_range.days} days)"
    )
    # Check for potential duplicates
    duplicates = df_valid.groupby("timestamp").size()
    if (duplicates > 1).any():
        print(
            f"⚠ Warning: {(duplicates > 1).sum()} timestamps have multiple images (burst mode?)"
        )
print("=" * 70)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

# ============================================================================
# SPECIES DISTRIBUTION
# ============================================================================
print("\n" + "=" * 70)
print("SPECIES DISTRIBUTION")
print("=" * 70)

species_counts = df["class"].value_counts()
print("\nTotal species detected:")
print(species_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = sns.color_palette("husl", len(species_counts))
species_counts.plot(kind="bar", ax=ax1, color=colors)
ax1.set_title(
    "Species Distribution - Counts (Log Scale)", fontsize=14, fontweight="bold"
)
ax1.set_xlabel("Species", fontsize=12)
ax1.set_ylabel("Number of Detections (log scale)", fontsize=12)
ax1.set_yscale("log")
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", alpha=0.3, which="both")

# Add count labels on bars
for i, v in enumerate(species_counts.values):
    ax1.text(
        i,
        v * 1.15,
        str(v),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Pie chart (exclude 'none' for better visualization)
species_with_animals = species_counts[species_counts.index != "none"]

# Group species that make up the last 10% into "other"
total_animals = species_with_animals.sum()
cumulative_pct: pd.Series = (species_with_animals.cumsum() / total_animals) * 100
threshold_idx = (cumulative_pct >= 90).idxmax()
threshold_position = species_with_animals.index.get_loc(threshold_idx)
threshold_pos_int = (
    int(threshold_position) if isinstance(threshold_position, (int, np.integer)) else 0
)

# Split into main species and "other"
main_species = species_with_animals.iloc[: threshold_pos_int + 1]
other_species = species_with_animals.iloc[threshold_pos_int + 1 :]

# Create final data for pie chart
if len(other_species) > 0:
    pie_data = pd.concat(
        [main_species, pd.Series([other_species.sum()], index=["other"])]
    )
    pie_colors = list(colors[: len(main_species)]) + ["lightgray"]
else:
    pie_data = main_species
    pie_colors = list(colors[: len(main_species)])

ax2.pie(
    pie_data.values,
    labels=pie_data.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=pie_colors,
)
ax2.set_title("Species Distribution - Percentages", fontsize=14, fontweight="bold")

try:
    plt.tight_layout()
except Exception:
    pass
plot_num += 1
plt.savefig(
    output_dir / f"{plot_num:02d}_species_distribution.svg", bbox_inches="tight"
)
print(f"✓ Saved: {plot_num:02d}_species_distribution.svg")
plt.close()

# ============================================================================
# SPECIES-SPECIFIC ACTIVITY PATTERNS
# ============================================================================
print("\n" + "=" * 70)
print("SPECIES-SPECIFIC ACTIVITY PATTERNS")
print("=" * 70)

# Get top species (excluding 'none')
top_species = species_counts[species_counts.index != "none"].head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, species in enumerate(top_species):
    if idx >= len(axes):
        break

    species_data = df_valid[df_valid["class"] == species]
    hourly_counts = species_data.groupby("hour").size()
    hours = range(24)
    counts = [hourly_counts.get(h, 0) for h in hours]

    ax = axes[idx]
    ax.bar(
        hours,
        counts,
        color=sns.color_palette("Set2")[idx % 8],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_title(f"{species.capitalize()} (n={len(species_data)})", fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00", "06", "12", "18", "23"])
    ax.grid(axis="y", alpha=0.3)

    # Shade night hours
    ax.axvspan(-0.5, 5.5, alpha=0.05, color="navy")
    ax.axvspan(19.5, 23.5, alpha=0.05, color="navy")

# Hide unused subplots
for idx in range(len(top_species), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Activity Patterns by Species", fontsize=16, fontweight="bold", y=1.00)
try:
    plt.tight_layout()
except Exception:
    pass
plot_num += 1
plt.savefig(
    output_dir / f"{plot_num:02d}_species_activity_patterns.svg", bbox_inches="tight"
)
print(f"✓ Saved: {plot_num:02d}_species_activity_patterns.svg")
plt.close()

# ============================================================================
# LIGHTING CONDITIONS
# ============================================================================
print("\n" + "=" * 70)
print("LIGHTING CONDITIONS ANALYSIS")
print("=" * 70)

lighting_counts = df["lighting"].value_counts()
print("\nLighting distribution:")
print(lighting_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Overall lighting distribution
colors_light = {"bright": "gold", "dark": "navy"}
lighting_counts.plot(
    kind="bar",
    ax=ax1,
    color=[colors_light.get(x, "gray") for x in lighting_counts.index],
)
ax1.set_title("Overall Lighting Conditions", fontsize=14, fontweight="bold")
ax1.set_xlabel("Lighting", fontsize=12)
ax1.set_ylabel("Number of Images", fontsize=12)
ax1.tick_params(axis="x", rotation=0)
ax1.grid(axis="y", alpha=0.3)

for i, v in enumerate(lighting_counts.values):
    ax1.text(
        i,
        v + max(lighting_counts.values) * 0.01,
        str(v),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Species activity by lighting
species_lighting = pd.crosstab(df["class"], df["lighting"], normalize="index") * 100
top_species_light = species_counts[species_counts.index != "none"].head(6).index
species_lighting_top = species_lighting.loc[top_species_light]

species_lighting_top.plot(
    kind="bar",
    stacked=False,
    ax=ax2,
    color=[colors_light.get(x, "gray") for x in species_lighting_top.columns],
)
ax2.set_title("Day/Night Activity by Species (%)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Species", fontsize=12)
ax2.set_ylabel("Percentage", fontsize=12)
ax2.tick_params(axis="x", rotation=45)
ax2.legend(title="Lighting", loc="upper right")
ax2.grid(axis="y", alpha=0.3)

try:
    plt.tight_layout()
except Exception:
    pass
plot_num += 1
plt.savefig(output_dir / f"{plot_num:02d}_lighting_analysis.svg", bbox_inches="tight")
print(f"✓ Saved: {plot_num:02d}_lighting_analysis.svg")
plt.close()

# ============================================================================
# BATTERY LEVEL ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("BATTERY LEVEL ANALYSIS")
print("=" * 70)

if "battery_level" in df.columns:
    df_battery = df[df["battery_level"].notna()].copy()
    print(
        f"\nImages with battery data: {len(df_battery)} ({100 * len(df_battery) / len(df):.1f}%)"
    )

    if (
        len(df_battery) > 0
        and "timestamp" in df_battery.columns
        and df_battery["timestamp"].notna().any()
    ):
        print(
            f"Battery level range: {df_battery['battery_level'].min():.1f}% to {df_battery['battery_level'].max():.1f}%"
        )
        print(f"Mean battery level: {df_battery['battery_level'].mean():.1f}%")
        print(f"Median battery level: {df_battery['battery_level'].median():.1f}%")

        # Prepare data with timestamps
        df_battery_time = df_battery[df_battery["timestamp"].notna()].copy()
        df_battery_time = df_battery_time.sort_values("timestamp")

        if len(df_battery_time) > 0 and "location_id" in df_battery_time.columns:
            # Get all unique locations
            all_battery_locations = df_battery_time["location_id"].unique()
            n_locations = len(all_battery_locations)

            print(f"Tracking battery for {n_locations} locations")

            # Create figure with battery over time for all locations
            fig, ax = plt.subplots(figsize=(18, 8))

            # Use a color palette with enough distinct colors
            if n_locations <= 10:
                colors_palette = sns.color_palette("tab10", n_locations)
            elif n_locations <= 20:
                colors_palette = sns.color_palette("tab20", n_locations)
            else:
                colors_palette = sns.color_palette("husl", n_locations)

            # Plot each location's battery level over time
            for idx, location in enumerate(sorted(all_battery_locations)):
                location_data = df_battery_time[
                    df_battery_time["location_id"] == location
                ].copy()
                location_data = location_data.sort_values("timestamp")

                # Only plot if there's meaningful data
                if len(location_data) >= 2:
                    ax.plot(
                        location_data["timestamp"],
                        location_data["battery_level"],
                        marker="o",
                        markersize=4,
                        linewidth=2,
                        alpha=0.7,
                        label=f"{location} (n={len(location_data)})",
                        color=colors_palette[idx],
                    )

                    # Add trend line for each location if enough data points
                    if len(location_data) >= 3:
                        x_numeric = np.arange(len(location_data))
                        z = np.polyfit(
                            x_numeric, location_data["battery_level"].values, 1
                        )
                        if abs(z[0]) > 0.01:  # Only show trend if meaningful
                            p = np.poly1d(z)
                            ax.plot(
                                location_data["timestamp"],
                                p(x_numeric),
                                linestyle="--",
                                linewidth=1,
                                alpha=0.4,
                                color=colors_palette[idx],
                            )

            # Add warning zones
            ax.axhspan(
                0, 20, alpha=0.15, color="red", label="Critical (<20%)", zorder=0
            )
            ax.axhspan(
                20, 40, alpha=0.08, color="orange", label="Low (20-40%)", zorder=0
            )

            ax.set_xlabel("Date", fontsize=13, fontweight="bold")
            ax.set_ylabel("Battery Level (%)", fontsize=13, fontweight="bold")
            ax.set_title(
                "Battery Levels Over Time by Location", fontsize=16, fontweight="bold"
            )
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Place legend outside plot area
            if n_locations <= 15:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    framealpha=0.95,
                    fontsize=9,
                )
            else:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    framealpha=0.95,
                    fontsize=8,
                    ncol=2,
                )

            try:
                plt.tight_layout()
            except Exception:
                pass

            plot_num += 1
            plt.savefig(
                output_dir / f"{plot_num:02d}_battery_levels_by_location.svg",
                bbox_inches="tight",
            )
            print(f"✓ Saved: {plot_num:02d}_battery_levels_by_location.svg")
            plt.close()

            # Print warnings for critical batteries
            critical_data = df_battery_time[df_battery_time["battery_level"] < 20]
            if len(critical_data) > 0:
                print(
                    f"\n⚠ WARNING: {len(critical_data)} readings with critical battery level (<20%)"
                )
                if "location_id" in critical_data.columns:
                    critical_locations = critical_data.groupby("location_id").agg(
                        {"battery_level": ["min", "mean", "count"]}
                    )
                    critical_locations.columns = ["Min %", "Avg %", "Count"]
                    critical_locations = critical_locations.sort_values("Avg %")
                    print("\n  Critical/low battery by location:")
                    for loc in critical_locations.index:
                        print(
                            f"    {loc}: {critical_locations.loc[loc, 'Count']:.0f} readings, "
                            f"avg {critical_locations.loc[loc, 'Avg %']:.1f}%, "
                            f"min {critical_locations.loc[loc, 'Min %']:.1f}%"
                        )

            # Summary table by location
            print("\n  Battery summary by location:")
            location_battery_summary = df_battery_time.groupby("location_id").agg(
                {"battery_level": ["mean", "min", "max", "count"]}
            )
            location_battery_summary.columns = ["Mean %", "Min %", "Max %", "Readings"]
            location_battery_summary = location_battery_summary.sort_values("Mean %")

            for loc in location_battery_summary.index:
                print(
                    f"    {loc}: mean={location_battery_summary.loc[loc, 'Mean %']:.1f}%, "
                    f"range={location_battery_summary.loc[loc, 'Min %']:.1f}-"
                    f"{location_battery_summary.loc[loc, 'Max %']:.1f}%, "
                    f"n={location_battery_summary.loc[loc, 'Readings']:.0f}"
                )
        else:
            print(
                "\n⚠ No location data available or insufficient timestamp data for battery timeline"
            )
    else:
        print("\n⚠ No battery level data with timestamps available in dataset")
else:
    print("\n⚠ Battery level column not found in dataset")
    print(f"Available columns: {df.columns.tolist()}")

# ============================================================================
# LOCATION COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("LOCATION COMPARISON")
print("=" * 70)

location_counts = df["location_id"].value_counts()
print("\nDetections by location:")
print(location_counts)

fig, ax = plt.subplots(figsize=(14, 8))

# Species diversity by location
location_species = pd.crosstab(df["location_id"], df["class"])

# Show all locations (sorted by total count)
all_locations = location_counts.index
location_species_all = location_species.loc[all_locations]

# Group species that make up the last 10% into "other"
species_totals = location_species_all.sum(axis=0).sort_values(ascending=False)
total_sightings = species_totals.sum()
cumulative_pct_loc: pd.Series = (species_totals.cumsum() / total_sightings) * 100
threshold_idx_loc = (cumulative_pct_loc >= 90).idxmax()
threshold_position_loc = species_totals.index.get_loc(threshold_idx_loc)
threshold_pos_loc_int = (
    int(threshold_position_loc)
    if isinstance(threshold_position_loc, (int, np.integer))
    else 0
)

# Split into main species and "other"
main_species_cols = species_totals.iloc[: threshold_pos_loc_int + 1].index
other_species_cols = species_totals.iloc[threshold_pos_loc_int + 1 :].index

# Create new dataframe with "other" category
if len(other_species_cols) > 0:
    location_species_grouped = location_species_all[main_species_cols].copy()
    location_species_grouped["other"] = location_species_all[other_species_cols].sum(
        axis=1
    )
else:
    location_species_grouped = location_species_all[main_species_cols].copy()

# Plot species distribution per location (stacked bar)
location_species_grouped.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=sns.color_palette("husl", len(location_species_grouped.columns)),
)
ax.set_title("Species Distribution by Location", fontsize=14, fontweight="bold")
ax.set_xlabel("Location ID", fontsize=12)
ax.set_ylabel("Number of Detections", fontsize=12)
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)

# Add total count labels on top of each bar
for i, location in enumerate(location_species_grouped.index):
    total = location_species_grouped.loc[location].sum()
    ax.text(
        i,
        total + location_species_grouped.sum(axis=1).max() * 0.01,
        str(int(total)),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

try:
    plt.tight_layout()
except Exception:
    pass
plot_num += 1
plt.savefig(output_dir / f"{plot_num:02d}_location_comparison.svg", bbox_inches="tight")
print(f"✓ Saved: {plot_num:02d}_location_comparison.svg")
plt.close()

# ============================================================================
# BAITING EFFECT ANALYSIS (Human Activity vs Roe Deer and Wild Boar)
# ============================================================================
print("\n" + "=" * 70)
print("BAITING EFFECT ANALYSIS (ROE DEER & WILD BOAR)")
print("=" * 70)

if len(df_valid) > 0:
    # Separate human and target wildlife sightings
    df_humans = df_valid[df_valid["class"] == "human"].copy()
    target_species = ["roe deer", "wild boar"]
    df_target_wildlife = df_valid[df_valid["class"].isin(target_species)].copy()

    print(f"\nHuman sightings: {len(df_humans)}")
    print(f"Roe deer sightings: {len(df_valid[df_valid['class'] == 'roe deer'])}")
    print(f"Wild boar sightings: {len(df_valid[df_valid['class'] == 'wild boar'])}")
    print(f"Combined target wildlife sightings: {len(df_target_wildlife)}")

    if len(df_humans) > 0 and len(df_target_wildlife) > 0:
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # Plot 1: Timeline showing human activity and target wildlife activity
        daily_humans = df_humans.groupby("date").size().reset_index(name="count")
        daily_humans["date"] = pd.to_datetime(daily_humans["date"])

        # Get daily counts for each species
        daily_roe_deer = (
            df_valid[df_valid["class"] == "roe deer"]
            .groupby("date")
            .size()
            .reset_index(name="count")
        )
        daily_roe_deer["date"] = pd.to_datetime(daily_roe_deer["date"])

        daily_wild_boar = (
            df_valid[df_valid["class"] == "wild boar"]
            .groupby("date")
            .size()
            .reset_index(name="count")
        )
        daily_wild_boar["date"] = pd.to_datetime(daily_wild_boar["date"])

        # Merge to have all dates
        date_range = pd.date_range(
            start=df_valid["date"].min(), end=df_valid["date"].max(), freq="D"
        )
        all_dates = pd.DataFrame({"date": date_range})
        daily_data = all_dates.merge(daily_roe_deer, on="date", how="left").fillna(0)
        daily_data = daily_data.merge(
            daily_wild_boar, on="date", how="left", suffixes=("_roe_deer", "_wild_boar")
        ).fillna(0)
        daily_data = daily_data.merge(daily_humans, on="date", how="left").fillna(0)
        daily_data.rename(columns={"count": "count_human"}, inplace=True)

        ax1_twin = ax1.twinx()

        # Stack roe deer and wild boar
        ax1.bar(
            daily_data["date"],
            daily_data["count_roe_deer"],
            color="steelblue",
            alpha=0.7,
            label="Roe Deer",
            width=1.0,
        )
        ax1.bar(
            daily_data["date"],
            daily_data["count_wild_boar"],
            bottom=daily_data["count_roe_deer"],
            color="darkgreen",
            alpha=0.7,
            label="Wild Boar",
            width=1.0,
        )
        ax1_twin.bar(
            daily_data["date"],
            daily_data["count_human"],
            color="orange",
            alpha=0.8,
            label="Human (Baiting)",
            width=1.0,
        )

        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel(
            "Wildlife Detections (Roe Deer + Wild Boar)", fontsize=12, color="black"
        )
        ax1_twin.set_ylabel(
            "Human Detections (Baiting Events)", fontsize=12, color="orange"
        )
        ax1.tick_params(axis="y")
        ax1_twin.tick_params(axis="y", labelcolor="orange")
        ax1.set_title(
            "Roe Deer & Wild Boar Activity vs Human Activity (Baiting) Over Time",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Plot 2: Roe Deer activity before and after human sightings
        window_days = 2  # Look at +/- 2 days around human sightings

        def calculate_before_after(df_species, species_name):
            """Calculate before/after activity for a species around baiting events."""
            before = []
            after = []

            for human_date in df_humans["date"].unique():
                human_date_pd = pd.to_datetime(human_date)
                # Before: 1-2 days before human sighting
                before_start = human_date_pd - pd.Timedelta(days=window_days)
                before_end = human_date_pd - pd.Timedelta(days=1)
                # After: 1-2 days after human sighting
                after_start = human_date_pd + pd.Timedelta(days=1)
                after_end = human_date_pd + pd.Timedelta(days=window_days)

                before_count = len(
                    df_species[
                        (df_species["timestamp"] >= before_start)
                        & (df_species["timestamp"] <= before_end)
                    ]
                )
                after_count = len(
                    df_species[
                        (df_species["timestamp"] >= after_start)
                        & (df_species["timestamp"] <= after_end)
                    ]
                )

                before.append(before_count / window_days)  # Average per day
                after.append(after_count / window_days)

            return np.mean(before), np.mean(after)

        # Calculate for roe deer
        df_roe_deer = df_valid[df_valid["class"] == "roe deer"].copy()
        roe_deer_before, roe_deer_after = calculate_before_after(
            df_roe_deer, "roe deer"
        )

        categories = [f"{window_days} days\nbefore", f"{window_days} days\nafter"]
        averages = [roe_deer_before, roe_deer_after]
        colors_bar = ["coral", "lightgreen"]

        bars = ax2.bar(
            categories, averages, color=colors_bar, edgecolor="black", linewidth=2
        )
        ax2.set_ylabel("Average Roe Deer Detections per Day", fontsize=12)
        ax2.set_title(
            f"Roe Deer Activity Before vs After Baiting Events (n={len(df_humans['date'].unique())} baiting dates)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, averages)):
            max_avg = float(np.max(averages)) if len(averages) > 0 else 1.0
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_avg * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        # Calculate and display percentage increase
        pct_change_roe = (
            ((roe_deer_after - roe_deer_before) / roe_deer_before * 100)
            if roe_deer_before > 0
            else 0
        )
        ax2.text(
            0.5,
            0.95,
            f"Change: {pct_change_roe:+.1f}%",
            transform=ax2.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            fontsize=12,
            fontweight="bold",
        )

        # Plot 3: Wild Boar activity before and after human sightings
        df_wild_boar = df_valid[df_valid["class"] == "wild boar"].copy()
        wild_boar_before, wild_boar_after = calculate_before_after(
            df_wild_boar, "wild boar"
        )

        averages_boar = [wild_boar_before, wild_boar_after]

        bars = ax3.bar(
            categories, averages_boar, color=colors_bar, edgecolor="black", linewidth=2
        )
        ax3.set_ylabel("Average Wild Boar Detections per Day", fontsize=12)
        ax3.set_title(
            f"Wild Boar Activity Before vs After Baiting Events (n={len(df_humans['date'].unique())} baiting dates)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, averages_boar)):
            max_avg = float(np.max(averages_boar)) if len(averages_boar) > 0 else 1.0
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_avg * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        # Calculate and display percentage increase
        pct_change_boar = (
            ((wild_boar_after - wild_boar_before) / wild_boar_before * 100)
            if wild_boar_before > 0
            else 0
        )
        ax3.text(
            0.5,
            0.95,
            f"Change: {pct_change_boar:+.1f}%",
            transform=ax3.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            fontsize=12,
            fontweight="bold",
        )

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout()
        except Exception:
            pass
        plot_num += 1
        plt.savefig(
            output_dir / f"{plot_num:02d}_baiting_effect_analysis.svg",
            bbox_inches="tight",
        )
        print(f"✓ Saved: {plot_num:02d}_baiting_effect_analysis.svg")

        # Print summary statistics
        print("\nBaiting Effect Summary:")
        print("  Roe Deer:")
        print(f"    Before baiting: {roe_deer_before:.2f} detections/day")
        print(f"    After baiting: {roe_deer_after:.2f} detections/day")
        print(f"    Change: {pct_change_roe:+.1f}%")
        print("  Wild Boar:")
        print(f"    Before baiting: {wild_boar_before:.2f} detections/day")
        print(f"    After baiting: {wild_boar_after:.2f} detections/day")
        print(f"    Change: {pct_change_boar:+.1f}%")

        plt.close()
    else:
        print("\nInsufficient data for baiting effect analysis")

# Store temperature range for later use in species-specific plots
df_temp = df[df["temperature_celsius"].notna()].copy()
if len(df_temp) > 0:
    df_temp["temp_range"] = pd.cut(
        df_temp["temperature_celsius"],
        bins=[-20, 0, 10, 20, 30, 40],
        labels=["<0°C", "0-10°C", "10-20°C", "20-30°C", ">30°C"],
    )
    # Correlation statistics
    print("\nTemperature-Activity Correlation Analysis:")
    print(
        f"  Temperature range: {df_temp['temperature_celsius'].min():.1f}°C to {df_temp['temperature_celsius'].max():.1f}°C"
    )
    print(
        f"  Most active temperature: {df_temp.groupby(pd.cut(df_temp['temperature_celsius'], bins=20), observed=True)['temperature_celsius'].count().idxmax()}"
    )

    # Correlation with hour of day
    temp_hour_corr = df_temp[["temperature_celsius", "hour"]].corr().iloc[0, 1]
    print(f"  Correlation (Temperature vs Hour): {temp_hour_corr:.3f}")

    # Peak activity temperature by species
    print("\n  Peak activity temperatures by species:")
    for species in df_temp["class"].value_counts().head(5).index:
        species_data = df_temp[df_temp["class"] == species]
        peak_temp = (
            species_data["temperature_celsius"].mode().values[0]
            if len(species_data["temperature_celsius"].mode()) > 0
            else species_data["temperature_celsius"].median()
        )
        print(f"    {species}: {peak_temp:.1f}°C (n={len(species_data)})")

    # Species-specific temperature-activity plots for Roe Deer and Wild Boar
    print("\nGenerating species-specific temperature-activity plots...")

    target_species = ["roe deer", "wild boar"]

    for species in target_species:
        species_data = df_temp[df_temp["class"] == species].copy()

        if len(species_data) >= 10:  # Only create plot if sufficient data
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

            # 1. Scatter: Temperature vs Hour
            ax1 = fig.add_subplot(gs[0, 0])
            scatter1 = ax1.scatter(
                species_data["temperature_celsius"],
                species_data["hour"],
                c=species_data["hour"],
                cmap="twilight",
                alpha=0.6,
                s=30,
                edgecolors="black",
                linewidth=0.5,
            )
            ax1.set_xlabel("Temperature (°C)", fontsize=10)
            ax1.set_ylabel("Hour of Day", fontsize=10)
            ax1.set_title("Temperature vs Time of Day", fontsize=11, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax1.set_yticks(range(0, 24, 4))

            # 2. Temperature distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(
                species_data["temperature_celsius"],
                bins=20,
                color="steelblue",
                edgecolor="black",
                alpha=0.7,
            )
            ax2.axvline(
                species_data["temperature_celsius"].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {species_data['temperature_celsius'].mean():.1f}°C",
            )
            ax2.axvline(
                species_data["temperature_celsius"].median(),
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {species_data['temperature_celsius'].median():.1f}°C",
            )
            ax2.set_xlabel("Temperature (°C)", fontsize=10)
            ax2.set_ylabel("Frequency", fontsize=10)
            ax2.set_title("Temperature Distribution", fontsize=11, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(axis="y", alpha=0.3)

            # 3. Activity count by temperature range
            ax3 = fig.add_subplot(gs[0, 2])
            temp_bins = np.arange(
                species_data["temperature_celsius"].min() - 1,
                species_data["temperature_celsius"].max() + 2,
                3,
            )
            hist_counts, edges = np.histogram(
                species_data["temperature_celsius"], bins=temp_bins
            )
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax3.bar(
                bin_centers,
                hist_counts,
                width=np.diff(edges),
                color="forestgreen",
                edgecolor="black",
                alpha=0.7,
            )
            ax3.set_xlabel("Temperature (°C)", fontsize=10)
            ax3.set_ylabel("Activity Count", fontsize=10)
            ax3.set_title("Activity by Temperature", fontsize=11, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

            # 4. Hourly activity by temperature range
            ax4 = fig.add_subplot(gs[1, :])
            species_data["temp_range"] = pd.cut(
                species_data["temperature_celsius"], bins=5, precision=0
            )
            temp_hour_pivot = (
                species_data.groupby(["temp_range", "hour"], observed=True)
                .size()
                .unstack(fill_value=0)
            )

            temp_hour_pivot.T.plot(
                kind="area", stacked=True, ax=ax4, alpha=0.7, colormap="RdYlBu_r"
            )
            ax4.set_xlabel("Hour of Day", fontsize=10)
            ax4.set_ylabel("Activity Count", fontsize=10)
            ax4.set_title(
                "Hourly Activity Pattern by Temperature Range",
                fontsize=11,
                fontweight="bold",
            )
            ax4.set_xticks(range(0, 24, 2))
            ax4.grid(axis="y", alpha=0.3)
            ax4.legend(
                title="Temp Range",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=8,
            )

            # 5. Temperature by month (if month data available)
            ax5 = fig.add_subplot(gs[2, 0])
            if (
                "month" in species_data.columns
                and len(species_data["month"].unique()) > 1
            ):
                monthly_data = []
                months = sorted(species_data["month"].unique())

                for m in months:
                    monthly_data.append(
                        species_data[species_data["month"] == m][
                            "temperature_celsius"
                        ].values
                    )

                bp = ax5.boxplot(
                    monthly_data,
                    tick_labels=[f"{m:02d}" for m in months],
                    patch_artist=True,
                    showfliers=False,
                )

                coolwarm_cmap = plt.colormaps["coolwarm"]
                colors = coolwarm_cmap(np.linspace(0, 1, len(months)))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)  # type: ignore
                    patch.set_alpha(0.7)

                ax5.set_xlabel("Month", fontsize=10)
                ax5.set_ylabel("Temperature (°C)", fontsize=10)
                ax5.set_title(
                    "Monthly Temperature Range", fontsize=11, fontweight="bold"
                )
                ax5.grid(axis="y", alpha=0.3)

            # 6. Density plot: Temperature vs Hour
            ax6 = fig.add_subplot(gs[2, 1])
            try:
                from scipy.stats import gaussian_kde

                # Create density plot
                xy = np.vstack(
                    [species_data["temperature_celsius"], species_data["hour"]]
                )
                z = gaussian_kde(xy)(xy)

                scatter2 = ax6.scatter(
                    species_data["temperature_celsius"],
                    species_data["hour"],
                    c=z,
                    s=20,
                    cmap="YlOrRd",
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=0.3,
                )
                ax6.set_xlabel("Temperature (°C)", fontsize=10)
                ax6.set_ylabel("Hour of Day", fontsize=10)
                ax6.set_title(
                    "Activity Density Heatmap", fontsize=11, fontweight="bold"
                )
                ax6.set_yticks(range(0, 24, 4))
                ax6.grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter2, ax=ax6)
                cbar.set_label("Density", fontsize=8)
            except Exception:
                ax6.text(
                    0.5,
                    0.5,
                    "Insufficient data\nfor density plot",
                    ha="center",
                    va="center",
                    transform=ax6.transAxes,
                )

            # 7. Statistics summary
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.axis("off")

            stats_text = f"""
STATISTICS SUMMARY

Sample Size: {len(species_data)}

Temperature (°C):
  Mean: {species_data["temperature_celsius"].mean():.1f}
  Median: {species_data["temperature_celsius"].median():.1f}
  Std Dev: {species_data["temperature_celsius"].std():.1f}
  Min: {species_data["temperature_celsius"].min():.1f}
  Max: {species_data["temperature_celsius"].max():.1f}

Most Active:
  Hour: {species_data["hour"].mode().values[0] if len(species_data["hour"].mode()) > 0 else "N/A"}:00
  Temp: {species_data["temperature_celsius"].mode().values[0] if len(species_data["temperature_celsius"].mode()) > 0 else species_data["temperature_celsius"].median():.1f}°C

Temp-Hour Correlation:
  {species_data[["temperature_celsius", "hour"]].corr().iloc[0, 1]:.3f}
"""
            ax7.text(
                0.1,
                0.9,
                stats_text,
                transform=ax7.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

            plt.suptitle(
                f"{species.title()} - Temperature-Activity Relationship Analysis",
                fontsize=14,
                fontweight="bold",
                y=0.998,
            )

            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig.tight_layout(rect=[0, 0, 1, 0.99])
            except Exception:
                pass
            filename = species.lower().replace(" ", "_")
            plot_num += 1
            plt.savefig(
                output_dir / f"{plot_num:02d}_{filename}_temperature_activity.svg",
                bbox_inches="tight",
            )
            print(f"✓ Saved: {plot_num:02d}_{filename}_temperature_activity.svg")
            plt.close()
        else:
            print(f"  Insufficient data for {species} (n={len(species_data)})")

else:
    print("\nNo temperature data available")

# ============================================================================
# INDIVIDUAL SPECIES TIMELINES
# ============================================================================
print("\n" + "=" * 70)
print("INDIVIDUAL SPECIES TIMELINES")
print("=" * 70)

if len(df_valid) > 0:
    # Get top species (excluding 'none')
    top_species = species_counts[species_counts.index != "none"].head(8).index.tolist()

    if len(top_species) > 0:
        # Prepare color palette
        colors_species = sns.color_palette("husl", len(top_species))

        # Create individual timeline plots for each species
        n_species = len(top_species)
        n_cols = 2
        n_rows = (n_species + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, species in enumerate(top_species):
            species_data = df_valid[df_valid["class"] == species]
            daily_species_activity = (
                species_data.groupby("date").size().reset_index(name="count")
            )
            daily_species_activity["date"] = pd.to_datetime(
                daily_species_activity["date"]
            )
            daily_species_activity = daily_species_activity.sort_values("date")

            # Calculate moving average (7-day window)
            window_size = 7
            daily_species_activity["moving_avg"] = (
                daily_species_activity["count"]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

            ax = axes[idx]

            # Stem plot for daily counts
            markerline, stemlines, baseline = ax.stem(
                daily_species_activity["date"],
                daily_species_activity["count"],
                linefmt=colors_species[idx],
                markerfmt="o",
                basefmt=" ",
            )
            markerline.set_markerfacecolor(colors_species[idx])
            markerline.set_markeredgecolor(colors_species[idx])
            markerline.set_markersize(4)
            stemlines.set_linewidth(1.5)
            stemlines.set_alpha(0.6)

            # Add moving average line
            ax.plot(
                daily_species_activity["date"],
                daily_species_activity["moving_avg"],
                color="darkred",
                linewidth=2,
                label=f"{window_size}-day moving avg",
                alpha=0.8,
            )

            ax.set_title(
                f"{species.capitalize()} Activity (n={len(species_data)})",
                fontweight="bold",
            )
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Detections", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for idx in range(len(top_species), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Individual Species Activity Timelines",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.98])
        except Exception:
            pass
        plot_num += 1
        plt.savefig(
            output_dir / f"{plot_num:02d}_individual_species_timelines.svg",
            bbox_inches="tight",
        )
        print(f"✓ Saved: {plot_num:02d}_individual_species_timelines.svg")
        plt.close()

# ============================================================================
# SUNRISE/SUNSET ANALYSIS FOR ROE DEER AND WILD BOAR
# ============================================================================
print("\n" + "=" * 70)
print("SUNRISE/SUNSET ANALYSIS")
print("=" * 70)

if len(df_valid) > 0:
    # Set up location for sunrise/sunset calculations
    location = LocationInfo("Camera Location", "Germany", TIMEZONE, LATITUDE, LONGITUDE)

    # Calculate sunrise and sunset for each date
    print(
        f"\nCalculating sunrise/sunset times for location: {LATITUDE}°N, {LONGITUDE}°E"
    )

    def get_sun_times(date):
        """Get sunrise and sunset times for a given date, adjusted to standard time (no DST jump).

        ⚠ NOTE: Times are in standard time (CET, UTC+1) WITHOUT daylight saving adjustments.
        This may be ~1 hour off local time during DST periods (late March to late October).
        This approach creates smooth curves but trades accuracy during summer months.
        """
        try:
            import pytz

            # Get sun times in UTC
            s = sun(location.observer, date=date)
            # Convert to standard time (CET, UTC+1) to avoid DST discontinuity in plots
            # This ensures smooth curves for sunrise/sunset times throughout the year
            standard_tz = pytz.timezone("Etc/GMT-1")  # CET (UTC+1, no DST)
            sunrise_standard = s["sunrise"].astimezone(standard_tz)
            sunset_standard = s["sunset"].astimezone(standard_tz)
            # Remove timezone info to get naive datetime in standard time
            return sunrise_standard.replace(tzinfo=None), sunset_standard.replace(
                tzinfo=None
            )
        except Exception:
            return None, None

    # Add sunrise/sunset times to dataframe
    df_valid["sunrise"] = df_valid["date"].apply(lambda d: get_sun_times(d)[0])
    df_valid["sunset"] = df_valid["date"].apply(lambda d: get_sun_times(d)[1])

    # Times are already in local time (naive datetimes) from get_sun_times

    # Filter out rows where sunrise/sunset calculation failed
    df_valid = df_valid[df_valid["sunset"].notna() & df_valid["sunrise"].notna()].copy()
    print(f"After filtering for valid sunrise/sunset: {len(df_valid)} images")

    if len(df_valid) > 0:
        # Calculate minutes relative to sunset (negative = before sunset, positive = after sunset)
        df_valid["minutes_from_sunset"] = (
            df_valid["timestamp"] - df_valid["sunset"]
        ).dt.total_seconds() / 60
        df_valid["minutes_from_sunrise"] = (
            df_valid["timestamp"] - df_valid["sunrise"]
        ).dt.total_seconds() / 60

        # Also calculate hours for plotting
        df_valid["hours_from_sunset"] = df_valid["minutes_from_sunset"] / 60
        df_valid["hours_from_sunrise"] = df_valid["minutes_from_sunrise"] / 60

        # Filter for roe deer and wild boar (configurable - edit this list to analyze other species)
        target_species = ["roe deer", "wild boar"]
        print(f"\nAnalyzing crepuscular activity for: {', '.join(target_species)}")
        df_target = df_valid[df_valid["class"].isin(target_species)].copy()

        # Check if target species exist in data
        for species in target_species:
            count = (df_valid["class"] == species).sum()
            if count == 0:
                print(f"⚠ Warning: '{species}' not found in dataset")

        if len(df_target) > 0:
            print(f"Found {len(df_target)} sightings of roe deer and wild boar")

            # ========== Activity relative to sunset throughout the year ==========
            for species in target_species:
                species_data = df_target[df_target["class"] == species]
                if len(species_data) == 0:
                    continue

                fig = plt.figure(figsize=(14, 6))
                gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)

                # Main scatter plot
                ax_main = fig.add_subplot(gs[0, 0])

                # Distribution plot on the right
                ax_dist = fig.add_subplot(gs[0, 1], sharey=ax_main)

                # Scatter plot: x=date, y=hours from sunset
                scatter = ax_main.scatter(
                    species_data["date"],
                    species_data["hours_from_sunset"],
                    c=species_data["hours_from_sunset"],
                    cmap="RdYlBu_r",
                    alpha=0.6,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                    vmin=-3,
                    vmax=3,
                )

                # Add horizontal line at sunset (0 hours)
                ax_main.axhline(
                    y=0,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label="Sunset",
                    alpha=0.7,
                )

                # Shade 1.5 hours before sunset
                ax_main.axhspan(
                    -1.5, 0, alpha=0.2, color="orange", label="1.5 hours before Sunset"
                )
                # Shade 0.5 hours after sunset
                ax_main.axhspan(
                    0, 0.5, alpha=0.3, color="orange", label="0.5 hours after Sunset"
                )

                ax_main.set_title(
                    f"{species.capitalize()} Activity Relative to Sunset (n={len(species_data)})\n⚠ Times in standard time (may be ~1h off during DST)",
                    fontsize=13,
                    fontweight="bold",
                )
                ax_main.set_xlabel("Date", fontsize=12)
                ax_main.set_ylabel("Hours from Sunset", fontsize=12)
                ax_main.set_ylim(-3, 3)
                ax_main.set_yticks(range(-3, 4))
                ax_main.grid(True, alpha=0.3)
                ax_main.legend(loc="upper right")
                plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax_main)
                cbar.set_label("Hours from Sunset", rotation=270, labelpad=20)

                # Distribution on the right
                ax_dist.hist(
                    species_data["hours_from_sunset"],
                    bins=60,
                    orientation="horizontal",
                    color="steelblue",
                    alpha=0.5,
                    edgecolor="black",
                    density=True,
                )

                # Add KDE (continuous distribution)
                from scipy import stats

                kde_data = species_data["hours_from_sunset"].dropna()
                if len(kde_data) > 1:
                    kde = stats.gaussian_kde(kde_data)
                    y_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                    kde_values = kde(y_range)
                    ax_dist.plot(
                        kde_values, y_range, "r-", linewidth=2, label="Density"
                    )

                ax_dist.axhline(
                    y=0, color="orange", linestyle="--", linewidth=2, alpha=0.7
                )
                ax_dist.axhspan(-1.5, 0, alpha=0.2, color="orange")
                ax_dist.axhspan(0, 0.5, alpha=0.3, color="orange")
                ax_dist.set_xlabel("Density", fontsize=10)
                ax_dist.tick_params(labelleft=False)
                ax_dist.grid(True, alpha=0.3, axis="x")

                # Count sightings before sunset
                before_sunset = (species_data["minutes_from_sunset"] < 0).sum()
                after_sunset = (species_data["minutes_from_sunset"] >= 0).sum()
                pct_before = 100 * before_sunset / len(species_data)

                # Count sightings in hot zone (1.5 hours before to 0.5 hours after sunset)
                hot_zone = (
                    (species_data["hours_from_sunset"] >= -1.5)
                    & (species_data["hours_from_sunset"] <= 0.5)
                ).sum()
                hot_zone_pct = 100 * hot_zone / len(species_data)

                # Add text annotation
                ax_main.text(
                    0.02,
                    0.98,
                    f"Before sunset: {before_sunset} ({pct_before:.1f}%)\nAfter sunset: {after_sunset}\nHot zone (-1.5h to +0.5h): {hot_zone} ({hot_zone_pct:.1f}%)",
                    transform=ax_main.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontsize=10,
                    fontweight="bold",
                )

                plot_num += 1
                plt.savefig(
                    output_dir
                    / f"{plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_scatter.svg",
                    bbox_inches="tight",
                )
                print(
                    f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_scatter.svg"
                )
                plt.close()

        # ========== Activity relative to sunrise throughout the year ==========
        for species in target_species:
            species_data = df_target[df_target["class"] == species]
            if len(species_data) == 0:
                continue

            fig = plt.figure(figsize=(14, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)

            # Main scatter plot
            ax_main = fig.add_subplot(gs[0, 0])

            # Distribution plot on the right
            ax_dist = fig.add_subplot(gs[0, 1], sharey=ax_main)

            # Scatter plot: x=date, y=hours from sunrise
            scatter = ax_main.scatter(
                species_data["date"],
                species_data["hours_from_sunrise"],
                c=species_data["hours_from_sunrise"],
                cmap="RdYlBu_r",
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
                vmin=-3,
                vmax=3,
            )

            # Add horizontal line at sunrise (0 hours)
            ax_main.axhline(
                y=0,
                color="gold",
                linestyle="--",
                linewidth=2,
                label="Sunrise",
                alpha=0.7,
            )

            # Shade 0.5 hours before sunrise
            ax_main.axhspan(
                -0.5, 0, alpha=0.3, color="gold", label="0.5 hours before Sunrise"
            )
            # Shade 1.5 hours after sunrise
            ax_main.axhspan(
                0, 1.5, alpha=0.2, color="gold", label="1.5 hours after Sunrise"
            )

            ax_main.set_title(
                f"{species.capitalize()} Activity Relative to Sunrise (n={len(species_data)})\n⚠ Times in standard time (may be ~1h off during DST)",
                fontsize=13,
                fontweight="bold",
            )
            ax_main.set_xlabel("Date", fontsize=12)
            ax_main.set_ylabel("Hours from Sunrise", fontsize=12)
            ax_main.set_ylim(-3, 3)
            ax_main.set_yticks(range(-3, 4))
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc="upper right")
            plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_main)
            cbar.set_label("Hours from Sunrise", rotation=270, labelpad=20)

            # Distribution on the right
            ax_dist.hist(
                species_data["hours_from_sunrise"],
                bins=60,
                orientation="horizontal",
                color="steelblue",
                alpha=0.5,
                edgecolor="black",
                density=True,
            )

            # Add KDE (continuous distribution)
            from scipy import stats

            kde_data = species_data["hours_from_sunrise"].dropna()
            if len(kde_data) > 1:
                kde = stats.gaussian_kde(kde_data)
                y_range = np.linspace(kde_data.min(), kde_data.max(), 100)
                kde_values = kde(y_range)
                ax_dist.plot(kde_values, y_range, "r-", linewidth=2, label="Density")

            ax_dist.axhline(y=0, color="gold", linestyle="--", linewidth=2, alpha=0.7)
            ax_dist.axhspan(-0.5, 0, alpha=0.3, color="gold")
            ax_dist.axhspan(0, 1.5, alpha=0.2, color="gold")
            ax_dist.set_xlabel("Density", fontsize=10)
            ax_dist.tick_params(labelleft=False)
            ax_dist.grid(True, alpha=0.3, axis="x")

            # Count sightings before sunrise
            before_sunrise = (species_data["minutes_from_sunrise"] < 0).sum()
            after_sunrise = (species_data["minutes_from_sunrise"] >= 0).sum()
            pct_before = 100 * before_sunrise / len(species_data)

            # Count sightings in hot zone (0.5 hours before to 1.5 hours after sunrise)
            hot_zone = (
                (species_data["hours_from_sunrise"] >= -0.5)
                & (species_data["hours_from_sunrise"] <= 1.5)
            ).sum()
            hot_zone_pct = 100 * hot_zone / len(species_data)

            # Add text annotation
            ax_main.text(
                0.02,
                0.98,
                f"Before sunrise: {before_sunrise} ({pct_before:.1f}%)\nAfter sunrise: {after_sunrise}\nHot zone (-0.5h to +1.5h): {hot_zone} ({hot_zone_pct:.1f}%)",
                transform=ax_main.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )

            plot_num += 1
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_sunrise_activity_scatter.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_sunrise_activity_scatter.svg"
            )
            plt.close()

        # ========== Seasonal patterns ==========
        for species in target_species:
            species_data = df_target[df_target["class"] == species].copy()
            if len(species_data) == 0:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(14, 6))

            # Group by month and calculate statistics
            species_data["month"] = species_data["timestamp"].dt.month

            monthly_stats = []
            for month in range(1, 13):
                month_data = species_data[species_data["month"] == month]
                if len(month_data) > 0:
                    before_sunset = (month_data["minutes_from_sunset"] < 0).sum()
                    after_sunset = (month_data["minutes_from_sunset"] >= 0).sum()
                    pct_before = 100 * before_sunset / len(month_data)
                    monthly_stats.append(
                        {
                            "month": month,
                            "before_sunset": before_sunset,
                            "after_sunset": after_sunset,
                            "pct_before": pct_before,
                            "total": len(month_data),
                        }
                    )

            if monthly_stats:
                stats_df = pd.DataFrame(monthly_stats)

                # Stacked bar chart
                x = stats_df["month"]
                width = 0.7

                p1 = ax.bar(
                    x,
                    stats_df["before_sunset"],
                    width,
                    label="Before Sunset",
                    color="gold",
                    edgecolor="black",
                )
                p2 = ax.bar(
                    x,
                    stats_df["after_sunset"],
                    width,
                    bottom=stats_df["before_sunset"],
                    label="After Sunset",
                    color="navy",
                    alpha=0.7,
                    edgecolor="black",
                )

                # Add percentage labels
                for i, (month, pct) in enumerate(
                    zip(stats_df["month"], stats_df["pct_before"])
                ):
                    total = stats_df.loc[i, "total"]
                    ax.text(
                        month,
                        total + max(stats_df["total"]) * 0.02,
                        f"{pct:.0f}%",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=8,
                    )

                ax.set_title(
                    f"{species.capitalize()} - Monthly Activity Before/After Sunset",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel("Month", fontsize=12)
                ax.set_ylabel("Number of Sightings", fontsize=12)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(
                    [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                )
                ax.legend(loc="upper right")
                ax.grid(axis="y", alpha=0.3)

            try:
                fig.tight_layout()
            except Exception:
                pass
            plot_num += 1
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_monthly_sunset_patterns.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_monthly_sunset_patterns.svg"
            )
            plt.close()

        # ========== PLOTS 19-21: Daily activity pattern over the year with sunset line ==========
        # Use gridspec to add marginal distributions
        from datetime import datetime, timedelta

        from matplotlib.gridspec import GridSpec

        today = datetime.now().date()
        twelve_months_ago = today - timedelta(days=365)

        # Create a figure for each species with distributions
        for species in target_species:
            species_data = df_target[df_target["class"] == species].copy()
            if len(species_data) == 0:
                continue

            # Filter to last 12 months
            species_data = species_data[
                species_data["date"] >= twelve_months_ago
            ].copy()
            if len(species_data) == 0:
                continue

            # Create figure for this species
            fig = plt.figure(figsize=(18, 10), constrained_layout=False)
            gs = GridSpec(
                2,
                3,
                figure=fig,
                width_ratios=[4, 1, 0.2],
                height_ratios=[1, 4],
                hspace=0.05,
                wspace=0.05,
            )

            # Top histogram (hour distribution)
            ax_top = fig.add_subplot(gs[0, 0])

            # Main scatter plot
            ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)

            # Right histogram (date distribution)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

            # Colorbar position
            ax_cbar = fig.add_subplot(gs[1, 2])

            # Extract hour and minute as decimal hour for x-axis
            species_data["hour_decimal"] = (
                species_data["timestamp"].dt.hour
                + species_data["timestamp"].dt.minute / 60
            )

            # Scatter plot: x=hour of day, y=date
            scatter = ax_main.scatter(
                species_data["hour_decimal"],
                species_data["date"],
                c=species_data["hours_from_sunset"],
                cmap="RdYlBu_r",
                alpha=0.6,
                s=30,
                edgecolors="black",
                linewidth=0.3,
                vmin=-3,
                vmax=3,
            )

            # Calculate sunset and sunrise times for all dates in range for smooth lines
            all_dates = pd.date_range(start=twelve_months_ago, end=today, freq="D")
            all_sunset_hours = []
            all_sunrise_hours = []
            all_valid_dates = []

            for date in all_dates:
                date_obj = pd.Timestamp(date).date()
                sunset_sunrise = get_sun_times(date_obj)
                if sunset_sunrise[0] is not None and sunset_sunrise[1] is not None:
                    sunrise_time, sunset_time = sunset_sunrise
                    sunset_hour = sunset_time.hour + sunset_time.minute / 60
                    sunrise_hour = sunrise_time.hour + sunrise_time.minute / 60
                    all_sunset_hours.append(sunset_hour)
                    all_sunrise_hours.append(sunrise_hour)
                    all_valid_dates.append(date_obj)

            if len(all_valid_dates) > 0:
                # Convert to numpy arrays for matplotlib compatibility
                dates_array = np.array(all_valid_dates)
                sunrise_array = np.array(all_sunrise_hours)
                sunset_array = np.array(all_sunset_hours)

                # Create shaded areas for twilight periods
                sunrise_minus_0_5 = sunrise_array - 0.5
                sunrise_plus_1_5 = sunrise_array + 1.5
                sunset_minus_1_5 = sunset_array - 1.5
                sunset_plus_0_5 = sunset_array + 0.5

                # Shade 0.5 hours before sunrise
                ax_main.fill_betweenx(  # type: ignore
                    dates_array,
                    sunrise_minus_0_5,
                    sunrise_array,
                    alpha=0.3,
                    color="gold",
                    label="0.5 hours before Sunrise",
                    zorder=5,
                )

                # Shade 1.5 hours after sunrise
                ax_main.fill_betweenx(  # type: ignore
                    dates_array,
                    sunrise_array,
                    sunrise_plus_1_5,
                    alpha=0.2,
                    color="gold",
                    label="1.5 hours after Sunrise",
                    zorder=5,
                )

                # Shade 1.5 hours before sunset
                ax_main.fill_betweenx(  # type: ignore
                    dates_array,
                    sunset_minus_1_5,
                    sunset_array,
                    alpha=0.2,
                    color="orange",
                    label="1.5 hours before Sunset",
                    zorder=5,
                )

                # Shade 0.5 hours after sunset
                ax_main.fill_betweenx(  # type: ignore
                    dates_array,
                    sunset_array,
                    sunset_plus_0_5,
                    alpha=0.3,
                    color="orange",
                    label="0.5 hours after Sunset",
                    zorder=5,
                )

                ax_main.plot(
                    sunset_array,
                    dates_array,
                    color="orange",
                    linewidth=3,
                    label="Sunset Time",
                    alpha=0.9,
                    zorder=10,
                )
                ax_main.plot(
                    sunrise_array,
                    dates_array,
                    color="gold",
                    linewidth=3,
                    label="Sunrise Time",
                    alpha=0.9,
                    zorder=10,
                )

            # Top distribution (hour of day)
            ax_top.hist(
                species_data["hour_decimal"],
                bins=48,
                color="steelblue",
                alpha=0.5,
                edgecolor="black",
                density=True,
            )

            # Add KDE for hour distribution
            from scipy import stats

            kde_hour = species_data["hour_decimal"].dropna()
            if len(kde_hour) > 1:
                kde = stats.gaussian_kde(kde_hour)
                x_range = np.linspace(0, 24, 200)
                kde_values = kde(x_range)
                ax_top.plot(x_range, kde_values, "r-", linewidth=2, label="Density")

            ax_top.set_ylabel("Density", fontsize=10)
            ax_top.tick_params(labelbottom=False)
            ax_top.grid(True, alpha=0.3, axis="y")
            ax_top.set_xlim(0, 24)
            ax_top.set_title(
                f"{species.capitalize()} Activity Throughout Year and Day (Last 12 Months, n={len(species_data)})",
                fontsize=14,
                fontweight="bold",
            )

            # Right distribution (date)
            date_values = (
                pd.to_datetime(species_data["date"]).astype("int64") / 10**9 / 86400
            )
            ax_right.hist(
                date_values,
                bins=52,
                orientation="horizontal",
                color="steelblue",
                alpha=0.5,
                edgecolor="black",
                density=True,
            )

            # Add KDE for date distribution
            if len(date_values) > 1:
                kde = stats.gaussian_kde(date_values)
                y_range = np.linspace(date_values.min(), date_values.max(), 100)
                kde_values = kde(y_range)
                ax_right.plot(kde_values, y_range, "r-", linewidth=2)

            ax_right.set_xlabel("Density", fontsize=10)
            ax_right.tick_params(labelleft=False)
            ax_right.grid(True, alpha=0.3, axis="x")

            # Main plot settings
            ax_main.set_xlabel("Hour of Day", fontsize=12)
            ax_main.set_ylabel("Date", fontsize=12)
            ax_main.set_xlim(0, 24)
            ax_main.set_xticks(range(0, 25, 2))
            ax_main.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc="upper right", fontsize=8)
            ax_main.invert_yaxis()  # Most recent dates at top

            # Right plot invert y-axis
            ax_right.invert_yaxis()

            # Add colorbar
            cbar = plt.colorbar(scatter, cax=ax_cbar)
            cbar.set_label("Hours from Sunset", rotation=270, labelpad=20, fontsize=10)

            # Count sightings in hot zones
            sunset_hot_zone = (
                (species_data["hours_from_sunset"] >= -1.5)
                & (species_data["hours_from_sunset"] <= 0.5)
            ).sum()
            sunset_hot_zone_pct = 100 * sunset_hot_zone / len(species_data)

            sunrise_hot_zone = (
                (species_data["hours_from_sunrise"] >= -0.5)
                & (species_data["hours_from_sunrise"] <= 1.5)
            ).sum()
            sunrise_hot_zone_pct = 100 * sunrise_hot_zone / len(species_data)

            # Add text annotation
            ax_main.text(
                0.02,
                0.98,
                f"Sunset hot zone (-1.5h to +0.5h): {sunset_hot_zone} ({sunset_hot_zone_pct:.1f}%)\nSunrise hot zone (-0.5h to +1.5h): {sunrise_hot_zone} ({sunrise_hot_zone_pct:.1f}%)",
                transform=ax_main.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9,
                fontweight="bold",
            )

            # Save individual figure
            plot_num += 1
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_daily_yearly_activity_pattern.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_daily_yearly_activity_pattern.svg"
            )
            plt.close()

        # Print statistics
        print("\nSunrise/Sunset Activity Statistics:")
        for species in target_species:
            species_data = df_target[df_target["class"] == species]
            if len(species_data) > 0:
                before_sunset = (species_data["minutes_from_sunset"] < 0).sum()
                after_sunset = (species_data["minutes_from_sunset"] >= 0).sum()
                pct_before = 100 * before_sunset / len(species_data)
                print(f"\n{species.capitalize()}:")
                print(f"  Total sightings: {len(species_data)}")
                print(f"  Before sunset: {before_sunset} ({pct_before:.1f}%)")
                print(f"  After sunset: {after_sunset} ({100 - pct_before:.1f}%)")
                print(
                    f"  Avg minutes from sunset: {species_data['minutes_from_sunset'].mean():.1f}"
                )
        else:
            print("\nNo data found for roe deer or wild boar")
    else:
        print("\nNo valid sunrise/sunset data available for analysis")
else:
    print("\nNo valid timestamp data for sunrise/sunset analysis")

# ============================================================================
# SOLUNAR ANALYSIS FOR ROE DEER AND WILD BOAR
# ============================================================================
print("\n" + "=" * 70)
print("SOLUNAR PERIOD ANALYSIS")
print("=" * 70)

if len(df_valid) > 0:
    print("\nCalculating moon phases and solunar periods...")

    # Set up observer location
    observer = ephem.Observer()
    observer.lat = str(LATITUDE)
    observer.lon = str(LONGITUDE)
    observer.elevation = 0

    def get_moon_data(timestamp):
        """Calculate moon phase, illumination, and position for a given timestamp."""
        try:
            observer.date = timestamp
            moon = ephem.Moon(observer)

            # Moon illumination (0-100%)
            illumination = moon.phase

            # Moon phase (0-28 day cycle, 0=new, 14=full)
            # Using moon_phase from astral (returns radians, 0=new, π=full)
            phase_radians = phase(timestamp.date())
            phase_days = (phase_radians / (2 * np.pi)) * 29.53  # Synodic month

            # Moon altitude (degrees above horizon)
            altitude = float(moon.alt) * 180 / np.pi

            # Classify moon phase
            if phase_days < 3.69 or phase_days > 25.84:
                phase_name = "New Moon"
            elif 3.69 <= phase_days < 7.38:
                phase_name = "Waxing Crescent"
            elif 7.38 <= phase_days < 11.07:
                phase_name = "First Quarter"
            elif 11.07 <= phase_days < 14.77:
                phase_name = "Waxing Gibbous"
            elif 14.77 <= phase_days < 18.46:
                phase_name = "Full Moon"
            elif 18.46 <= phase_days < 22.15:
                phase_name = "Waning Gibbous"
            elif 22.15 <= phase_days < 25.84:
                phase_name = "Last Quarter"
            else:
                phase_name = "Waning Crescent"

            # Calculate moonrise and moonset
            try:
                observer.date = timestamp.date()
                moonrise = observer.next_rising(moon).datetime()
                moonset = observer.next_setting(moon).datetime()
            except (ephem.AlwaysUpError, ephem.NeverUpError):
                moonrise = None
                moonset = None

            return {
                "illumination": illumination,
                "phase_days": phase_days,
                "phase_name": phase_name,
                "altitude": altitude,
                "moonrise": moonrise,
                "moonset": moonset,
            }
        except Exception:
            return {
                "illumination": np.nan,
                "phase_days": np.nan,
                "phase_name": "Unknown",
                "altitude": np.nan,
                "moonrise": None,
                "moonset": None,
            }

    # Add moon data to dataframe
    print("Processing moon data for each observation...")
    moon_data = df_valid["timestamp"].apply(get_moon_data)
    df_valid["moon_illumination"] = moon_data.apply(lambda x: x["illumination"])
    df_valid["moon_phase_days"] = moon_data.apply(lambda x: x["phase_days"])
    df_valid["moon_phase_name"] = moon_data.apply(lambda x: x["phase_name"])
    df_valid["moon_altitude"] = moon_data.apply(lambda x: x["altitude"])

    # Filter for roe deer and wild boar
    target_species = ["roe deer", "wild boar"]
    df_solunar = df_valid[df_valid["class"].isin(target_species)].copy()

    print(f"\nAnalyzing solunar correlations for {len(df_solunar)} observations")
    print(f"  Roe deer: {len(df_solunar[df_solunar['class'] == 'roe deer'])}")
    print(f"  Wild boar: {len(df_solunar[df_solunar['class'] == 'wild boar'])}")

    if len(df_solunar) > 0:
        # Create a combined comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for idx, species in enumerate(target_species):
            species_data = df_solunar[df_solunar["class"] == species].copy()

            if len(species_data) < 10:
                continue

            # Moon illumination distribution
            ax = axes[idx, 0]
            ax.hist(
                species_data["moon_illumination"],
                bins=20,
                color="steelblue",
                edgecolor="black",
                alpha=0.7,
            )
            ax.axvline(
                x=species_data["moon_illumination"].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {species_data['moon_illumination'].mean():.1f}%",
            )
            ax.set_xlabel("Moon Illumination (%)", fontsize=11)
            ax.set_ylabel("Number of Detections", fontsize=11)
            ax.set_title(
                f"{species.title()} - Moon Illumination Distribution",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            # Activity by moon phase
            ax = axes[idx, 1]
            phase_counts = species_data["moon_phase_name"].value_counts()
            phase_order = [
                "New Moon",
                "Waxing Crescent",
                "First Quarter",
                "Waxing Gibbous",
                "Full Moon",
                "Waning Gibbous",
                "Last Quarter",
                "Waning Crescent",
            ]
            phase_counts = phase_counts.reindex(
                [p for p in phase_order if p in phase_counts.index]
            )

            bars = ax.bar(
                range(len(phase_counts)),
                phase_counts.values,
                color=plt.cm.twilight(np.linspace(0, 1, len(phase_counts))),
                edgecolor="black",
                linewidth=1.5,
            )
            ax.set_xticks(range(len(phase_counts)))
            ax.set_xticklabels(phase_counts.index, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Number of Detections", fontsize=11)
            ax.set_title(
                f"{species.title()} - Activity by Moon Phase",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        fig.suptitle(
            "Solunar Period Comparison: Roe Deer vs Wild Boar",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.98])
        except Exception:
            pass
        plot_num += 1
        plt.savefig(
            output_dir / f"{plot_num:02d}_solunar_comparison.svg", bbox_inches="tight"
        )
        print(f"✓ Saved: {plot_num:02d}_solunar_comparison.svg")
        plt.close()

        # Print summary statistics
        print("\n" + "=" * 70)
        print("SOLUNAR CORRELATION SUMMARY")
        print("=" * 70)
        for species in target_species:
            species_data = df_solunar[df_solunar["class"] == species].copy()
            if len(species_data) > 0:
                print(f"\n{species.upper()}:")
                print(f"  Total detections: {len(species_data)}")
                print(
                    f"  Mean moon illumination: {species_data['moon_illumination'].mean():.1f}%"
                )
                print(
                    f"  Most active moon phase: {species_data['moon_phase_name'].mode().values[0] if len(species_data['moon_phase_name'].mode()) > 0 else 'N/A'}"
                )

                new_moon_pct = (
                    100
                    * len(species_data[species_data["moon_illumination"] <= 25])
                    / len(species_data)
                )
                full_moon_pct = (
                    100
                    * len(species_data[species_data["moon_illumination"] >= 75])
                    / len(species_data)
                )
                print(f"  Activity during dark moon (0-25%): {new_moon_pct:.1f}%")
                print(f"  Activity during bright moon (75-100%): {full_moon_pct:.1f}%")

                above_horizon_pct = (
                    100
                    * len(species_data[species_data["moon_altitude"] > 0])
                    / len(species_data)
                )
                print(f"  Activity when moon above horizon: {above_horizon_pct:.1f}%")
    else:
        print("\nNo data available for roe deer or wild boar")
else:
    print("\nNo valid timestamp data for solunar analysis")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

summary_stats = {
    "Total Images": len(df),
    "Images with Timestamps": len(df_valid),
    "Unique Species": df["class"].nunique(),
    "Unique Locations": df["location_id"].nunique(),
    "Date Range": f"{df_valid['date'].min()} to {df_valid['date'].max()}"
    if len(df_valid) > 0
    else "N/A",
    "Most Common Species": species_counts.index[0]
    if len(species_counts) > 0
    else "N/A",
    "Most Active Location": location_counts.index[0]
    if len(location_counts) > 0
    else "N/A",
}

print("\n")
for key, value in summary_stats.items():
    print(f"{key:.<30} {value}")

# Save summary to file
with open(output_dir / "summary_statistics.txt", "w") as f:
    f.write("WILDLIFE CAMERA ACTIVITY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")
    f.write("\n" + "=" * 70 + "\n\n")
    f.write("SPECIES COUNTS:\n")
    f.write(species_counts.to_string())
    f.write("\n\n" + "=" * 70 + "\n\n")
    f.write("LOCATION COUNTS:\n")
    f.write(location_counts.to_string())

print("\n✓ Saved: summary_statistics.txt")

print("\n" + "=" * 70)
print(f"All visualizations saved to: {output_dir}")
print("=" * 70)

# %%
