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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astral import LocationInfo
from astral.sun import sun

# Configuration
model = "qwen3-vl:235b-a22b-thinking"
image_dir = Path(os.environ["HOME"] + "/mnt/wildlife")
labels_dir = image_dir / f"labels_{model}"
# Put visualizations next to this script
script_dir = Path(__file__).parent
output_dir = script_dir / "visualizations"

# Delete old visualizations directory and recreate it
if output_dir.exists():
    shutil.rmtree(output_dir)
    print(f"Cleared old visualizations from {output_dir}")
output_dir.mkdir(exist_ok=True)

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
# 1. SPECIES DISTRIBUTION
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
cumulative_pct = (species_with_animals.cumsum() / total_animals) * 100
threshold_idx = (cumulative_pct >= 90).idxmax()
threshold_position = species_with_animals.index.get_loc(threshold_idx)

# Split into main species and "other"
main_species = species_with_animals.iloc[: threshold_position + 1]
other_species = species_with_animals.iloc[threshold_position + 1 :]

# Create final data for pie chart
if len(other_species) > 0:
    pie_data = pd.concat(
        [main_species, pd.Series([other_species.sum()], index=["other"])]
    )
    pie_colors = list(colors[: len(main_species)]) + ["lightgray"]
else:
    pie_data = main_species
    pie_colors = colors[: len(main_species)]

ax2.pie(
    pie_data.values,
    labels=pie_data.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=pie_colors,
)
ax2.set_title("Species Distribution - Percentages", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(output_dir / "01_species_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "01_species_distribution.svg", bbox_inches="tight")
print("✓ Saved: 01_species_distribution.png + .svg")
plt.close()

# ============================================================================
# 2. ACTIVITY BY HOUR OF DAY
# ============================================================================
print("\n" + "=" * 70)
print("ACTIVITY BY HOUR OF DAY")
print("=" * 70)

# Exclude humans from this analysis
df_valid_no_humans = df_valid[df_valid["class"] != "human"]
hourly_activity = df_valid_no_humans.groupby("hour").size()
print("\nActivity counts by hour (excluding humans):")
print(hourly_activity)

fig, ax = plt.subplots(figsize=(14, 6))
hours = range(24)
counts = [hourly_activity.get(h, 0) for h in hours]

# Create color gradient based on day/night
colors_hour = []
for h in hours:
    if 6 <= h < 20:  # Daytime
        colors_hour.append(sns.color_palette("YlOrRd", 24)[h])
    else:  # Nighttime
        colors_hour.append(sns.color_palette("Blues_r", 24)[h])

bars = ax.bar(hours, counts, color=colors_hour, edgecolor="black", linewidth=0.5)
ax.set_title("Wildlife Activity by Hour of Day", fontsize=16, fontweight="bold")
ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Number of Detections", fontsize=12)
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

# Add day/night indicators
ax.axvspan(-0.5, 5.5, alpha=0.1, color="navy", label="Night")
ax.axvspan(5.5, 19.5, alpha=0.1, color="gold", label="Day")
ax.axvspan(19.5, 23.5, alpha=0.1, color="navy")
ax.legend(loc="upper left")

# Add peak hours annotation
peak_hour = hourly_activity.idxmax()
peak_count = hourly_activity.max()
ax.annotate(
    f"Peak: {peak_hour:02d}:00\n({peak_count} detections)",
    xy=(peak_hour, peak_count),
    xytext=(peak_hour, peak_count * 1.15),
    ha="center",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=2),
)

plt.tight_layout()
plt.savefig(output_dir / "02_activity_by_hour.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "02_activity_by_hour.svg", bbox_inches="tight")
print("✓ Saved: 02_activity_by_hour.png + .svg")
plt.close()

# ============================================================================
# 3. SPECIES-SPECIFIC ACTIVITY PATTERNS
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
plt.tight_layout()
plt.savefig(
    output_dir / "03_species_activity_patterns.png", dpi=300, bbox_inches="tight"
)
plt.savefig(output_dir / "03_species_activity_patterns.svg", bbox_inches="tight")
print("✓ Saved: 03_species_activity_patterns.png + .svg")
plt.close()

# ============================================================================
# 5. LIGHTING CONDITIONS
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

plt.tight_layout()
plt.savefig(output_dir / "05_lighting_analysis.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "05_lighting_analysis.svg", bbox_inches="tight")
print("✓ Saved: 05_lighting_analysis.png + .svg")
plt.close()

# ============================================================================
# 6. LOCATION COMPARISON
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
cumulative_pct = (species_totals.cumsum() / total_sightings) * 100
threshold_idx = (cumulative_pct >= 90).idxmax()
threshold_position = species_totals.index.get_loc(threshold_idx)

# Split into main species and "other"
main_species_cols = species_totals.iloc[: threshold_position + 1].index
other_species_cols = species_totals.iloc[threshold_position + 1 :].index

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

plt.tight_layout()
plt.savefig(output_dir / "06_location_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "06_location_comparison.svg", bbox_inches="tight")
print("✓ Saved: 06_location_comparison.png + .svg")
plt.close()

# ============================================================================
# 7. BAITING EFFECT ANALYSIS (Human Activity vs Wildlife Activity)
# ============================================================================
print("\n" + "=" * 70)
print("BAITING EFFECT ANALYSIS")
print("=" * 70)

if len(df_valid) > 0:
    # Separate human and wildlife sightings
    df_humans = df_valid[df_valid["class"] == "human"].copy()
    df_wildlife = df_valid[df_valid["class"] != "human"].copy()
    
    print(f"\nHuman sightings: {len(df_humans)}")
    print(f"Wildlife sightings: {len(df_wildlife)}")
    
    if len(df_humans) > 0 and len(df_wildlife) > 0:
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
        
        # Plot 1: Timeline showing human activity and wildlife activity
        daily_humans = df_humans.groupby("date").size().reset_index(name="count")
        daily_humans["date"] = pd.to_datetime(daily_humans["date"])
        daily_wildlife = df_wildlife.groupby("date").size().reset_index(name="count")
        daily_wildlife["date"] = pd.to_datetime(daily_wildlife["date"])
        
        # Merge to have all dates
        date_range = pd.date_range(
            start=df_valid["date"].min(), end=df_valid["date"].max(), freq="D"
        )
        all_dates = pd.DataFrame({"date": date_range})
        daily_data = all_dates.merge(daily_wildlife, on="date", how="left").fillna(0)
        daily_data = daily_data.merge(
            daily_humans, on="date", how="left", suffixes=("_wildlife", "_human")
        ).fillna(0)
        
        ax1_twin = ax1.twinx()
        ax1.bar(
            daily_data["date"],
            daily_data["count_wildlife"],
            color="steelblue",
            alpha=0.6,
            label="Wildlife",
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
        ax1.set_ylabel("Wildlife Detections", fontsize=12, color="steelblue")
        ax1_twin.set_ylabel("Human Detections (Baiting Events)", fontsize=12, color="orange")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1_twin.tick_params(axis="y", labelcolor="orange")
        ax1.set_title("Wildlife Activity vs Human Activity (Baiting) Over Time", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        # Plot 2: Wildlife activity before and after human sightings
        window_days = 7  # Look at +/- 7 days around human sightings
        
        wildlife_before = []
        wildlife_after = []
        
        for human_date in df_humans["date"].unique():
            human_date_pd = pd.to_datetime(human_date)
            # Before: 1-7 days before human sighting
            before_start = human_date_pd - pd.Timedelta(days=window_days)
            before_end = human_date_pd - pd.Timedelta(days=1)
            # After: 1-7 days after human sighting
            after_start = human_date_pd + pd.Timedelta(days=1)
            after_end = human_date_pd + pd.Timedelta(days=window_days)
            
            before_count = len(
                df_wildlife[
                    (df_wildlife["timestamp"] >= before_start)
                    & (df_wildlife["timestamp"] <= before_end)
                ]
            )
            after_count = len(
                df_wildlife[
                    (df_wildlife["timestamp"] >= after_start)
                    & (df_wildlife["timestamp"] <= after_end)
                ]
            )
            
            wildlife_before.append(before_count / window_days)  # Average per day
            wildlife_after.append(after_count / window_days)
        
        avg_before = np.mean(wildlife_before)
        avg_after = np.mean(wildlife_after)
        
        categories = [f"{window_days} days\nbefore", f"{window_days} days\nafter"]
        averages = [avg_before, avg_after]
        colors_bar = ["coral", "lightgreen"]
        
        bars = ax2.bar(categories, averages, color=colors_bar, edgecolor="black", linewidth=2)
        ax2.set_ylabel("Average Wildlife Detections per Day", fontsize=12)
        ax2.set_title(
            f"Wildlife Activity Before vs After Baiting Events (n={len(df_humans['date'].unique())} baiting dates)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, averages)):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(averages) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        
        # Calculate and display percentage increase
        pct_change = ((avg_after - avg_before) / avg_before * 100) if avg_before > 0 else 0
        ax2.text(
            0.5,
            0.95,
            f"Change: {pct_change:+.1f}%",
            transform=ax2.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            fontsize=12,
            fontweight="bold",
        )
        
        # Plot 3: Wildlife activity by location, comparing locations with/without human activity
        location_human_counts = df_humans.groupby("location_id").size()
        location_wildlife_counts = df_wildlife.groupby("location_id").size()
        
        location_comparison = pd.DataFrame(
            {
                "wildlife": location_wildlife_counts,
                "human": location_human_counts,
            }
        ).fillna(0)
        
        # Calculate wildlife per human sighting ratio
        location_comparison["wildlife_per_human"] = location_comparison.apply(
            lambda row: row["wildlife"] / row["human"] if row["human"] > 0 else 0, axis=1
        )
        location_comparison = location_comparison.sort_values("wildlife", ascending=False)
        
        x = np.arange(len(location_comparison))
        width = 0.35
        
        bars1 = ax3.bar(
            x - width / 2,
            location_comparison["human"],
            width,
            label="Human (Baiting)",
            color="orange",
            edgecolor="black",
        )
        bars2 = ax3.bar(
            x + width / 2,
            location_comparison["wildlife"],
            width,
            label="Wildlife",
            color="steelblue",
            edgecolor="black",
        )
        
        ax3.set_xlabel("Location ID", fontsize=12)
        ax3.set_ylabel("Number of Detections", fontsize=12)
        ax3.set_title("Human Activity vs Wildlife Activity by Location", fontsize=14, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(location_comparison.index, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)
        
        # Add ratio labels above wildlife bars
        for i, (idx, row) in enumerate(location_comparison.iterrows()):
            if row["human"] > 0:
                ratio = row["wildlife_per_human"]
                ax3.text(
                    i + width / 2,
                    row["wildlife"] + location_comparison["wildlife"].max() * 0.02,
                    f"{ratio:.1f}:1",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
        
        plt.tight_layout()
        plt.savefig(output_dir / "07_baiting_effect_analysis.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "07_baiting_effect_analysis.svg", bbox_inches="tight")
        print("✓ Saved: 07_baiting_effect_analysis.png + .svg")
        plt.close()
    else:
        print("\nInsufficient data for baiting effect analysis")

# ============================================================================
# 8. TEMPERATURE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("TEMPERATURE ANALYSIS")
print("=" * 70)

df_temp = df[df["temperature_celsius"].notna()].copy()
if len(df_temp) > 0:
    print("\nTemperature statistics:")
    print(df_temp["temperature_celsius"].describe())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Temperature distribution
    ax1.hist(
        df_temp["temperature_celsius"],
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax1.set_title("Temperature Distribution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axvline(
        df_temp["temperature_celsius"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_temp['temperature_celsius'].mean():.1f}°C",
    )
    ax1.legend()

    # Activity by temperature range
    df_temp["temp_range"] = pd.cut(
        df_temp["temperature_celsius"],
        bins=[-20, 0, 10, 20, 30, 40],
        labels=["<0°C", "0-10°C", "10-20°C", "20-30°C", ">30°C"],
    )
    temp_activity = df_temp["temp_range"].value_counts().sort_index()

    temp_activity.plot(kind="bar", ax=ax2, color="coral", edgecolor="black")
    ax2.set_title("Activity by Temperature Range", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Temperature Range", fontsize=12)
    ax2.set_ylabel("Number of Detections", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    for i, v in enumerate(temp_activity.values):
        ax2.text(
            i,
            v + max(temp_activity.values) * 0.01,
            str(v),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

# ============================================================================
# 8. TEMPERATURE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("TEMPERATURE ANALYSIS")
print("=" * 70)

df_temp = df[df["temperature_celsius"].notna()].copy()
if len(df_temp) > 0:
    print("\nTemperature statistics:")
    print(df_temp["temperature_celsius"].describe())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Temperature distribution
    ax1.hist(
        df_temp["temperature_celsius"],
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax1.set_title("Temperature Distribution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axvline(
        df_temp["temperature_celsius"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_temp['temperature_celsius'].mean():.1f}°C",
    )
    ax1.legend()

    # Activity by temperature range
    df_temp["temp_range"] = pd.cut(
        df_temp["temperature_celsius"],
        bins=[-20, 0, 10, 20, 30, 40],
        labels=["<0°C", "0-10°C", "10-20°C", "20-30°C", ">30°C"],
    )
    temp_activity = df_temp["temp_range"].value_counts().sort_index()

    temp_activity.plot(kind="bar", ax=ax2, color="coral", edgecolor="black")
    ax2.set_title("Activity by Temperature Range", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Temperature Range", fontsize=12)
    ax2.set_ylabel("Number of Detections", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    for i, v in enumerate(temp_activity.values):
        ax2.text(
            i,
            v + max(temp_activity.values) * 0.01,
            str(v),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "08_temperature_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "08_temperature_analysis.svg", bbox_inches="tight")
    print("✓ Saved: 08_temperature_analysis.png + .svg")
    plt.close()
else:
    print("\nNo temperature data available")

# ============================================================================
# 9. TIMELINE VIEW
# ============================================================================
print("\n" + "=" * 70)
print("ACTIVITY TIMELINE")
print("=" * 70)

if len(df_valid) > 0:
    fig, ax = plt.subplots(figsize=(16, 6))

    # Daily activity over time
    daily_activity = df_valid.groupby("date").size().reset_index(name="count")
    daily_activity["date"] = pd.to_datetime(daily_activity["date"])
    daily_activity = daily_activity.sort_values("date")

    ax.plot(
        daily_activity["date"],
        daily_activity["count"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        daily_activity["date"], daily_activity["count"], alpha=0.3, color="steelblue"
    )

    ax.set_title("Wildlife Activity Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Detections", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha="right")

# ============================================================================
# 9. TIMELINE VIEW
# ============================================================================
print("\n" + "=" * 70)
print("ACTIVITY TIMELINE")
print("=" * 70)

if len(df_valid) > 0:
    fig, ax = plt.subplots(figsize=(16, 6))

    # Daily activity over time
    daily_activity = df_valid.groupby("date").size().reset_index(name="count")
    daily_activity["date"] = pd.to_datetime(daily_activity["date"])
    daily_activity = daily_activity.sort_values("date")

    ax.plot(
        daily_activity["date"],
        daily_activity["count"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        daily_activity["date"], daily_activity["count"], alpha=0.3, color="steelblue"
    )

    ax.set_title("Wildlife Activity Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Detections", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "09_activity_timeline.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "09_activity_timeline.svg", bbox_inches="tight")
    print("✓ Saved: 09_activity_timeline.png + .svg")
    plt.close()

# ============================================================================
# 10. SPECIES ACTIVITY TIMELINE
# ============================================================================
print("\n" + "=" * 70)
print("SPECIES ACTIVITY TIMELINE")
print("=" * 70)

if len(df_valid) > 0:
    # Get top species (excluding 'none')
    top_species = species_counts[species_counts.index != "none"].head(8).index.tolist()

    if len(top_species) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))

        # Prepare color palette
        colors_species = sns.color_palette("husl", len(top_species))

        # Plot each species over time
        for idx, species in enumerate(top_species):
            species_data = df_valid[df_valid["class"] == species]
            daily_species_activity = (
                species_data.groupby("date").size().reset_index(name="count")
            )
            daily_species_activity["date"] = pd.to_datetime(
                daily_species_activity["date"]
            )
            daily_species_activity = daily_species_activity.sort_values("date")

            ax.plot(
                daily_species_activity["date"],
                daily_species_activity["count"],
                color=colors_species[idx],
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"{species.capitalize()} (n={len(species_data)})",
                alpha=0.8,
            )

        ax.set_title(
            "Wildlife Activity Over Time by Species", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Detections", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            output_dir / "09_species_activity_timeline.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "09_species_activity_timeline.svg", bbox_inches="tight"
        )
        print("✓ Saved: 09_species_activity_timeline.png + .svg")
        plt.close()

        # Also create individual timeline plots for each species
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

            ax = axes[idx]
            ax.plot(
                daily_species_activity["date"],
                daily_species_activity["count"],
                color=colors_species[idx],
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                daily_species_activity["date"],
                daily_species_activity["count"],
                alpha=0.3,
                color=colors_species[idx],
            )

            ax.set_title(
                f"{species.capitalize()} Activity (n={len(species_data)})",
                fontweight="bold",
            )
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Detections", fontsize=10)
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
        plt.tight_layout()
        plt.savefig(
            output_dir / "10_individual_species_timelines.png",
            dpi=300,
            bbox_inches="tight",
        )
# ============================================================================
# 10. SPECIES ACTIVITY TIMELINE
# ============================================================================
print("\n" + "=" * 70)
print("SPECIES ACTIVITY TIMELINE")
print("=" * 70)

if len(df_valid) > 0:
    # Get top species (excluding 'none')
    top_species = species_counts[species_counts.index != "none"].head(8).index.tolist()

    if len(top_species) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))

        # Prepare color palette
        colors_species = sns.color_palette("husl", len(top_species))

        # Plot each species over time
        for idx, species in enumerate(top_species):
            species_data = df_valid[df_valid["class"] == species]
            daily_species_activity = (
                species_data.groupby("date").size().reset_index(name="count")
            )
            daily_species_activity["date"] = pd.to_datetime(
                daily_species_activity["date"]
            )
            daily_species_activity = daily_species_activity.sort_values("date")

            ax.plot(
                daily_species_activity["date"],
                daily_species_activity["count"],
                color=colors_species[idx],
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"{species.capitalize()} (n={len(species_data)})",
                alpha=0.8,
            )

        ax.set_title(
            "Wildlife Activity Over Time by Species", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Detections", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            output_dir / "10_species_activity_timeline.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "10_species_activity_timeline.svg", bbox_inches="tight"
        )
        print("✓ Saved: 10_species_activity_timeline.png + .svg")
        plt.close()

        # Also create individual timeline plots for each species
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

            ax = axes[idx]
            ax.plot(
                daily_species_activity["date"],
                daily_species_activity["count"],
                color=colors_species[idx],
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                daily_species_activity["date"],
                daily_species_activity["count"],
                alpha=0.3,
                color=colors_species[idx],
            )

            ax.set_title(
                f"{species.capitalize()} Activity (n={len(species_data)})",
                fontweight="bold",
            )
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Detections", fontsize=10)
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
        plt.tight_layout()
        plt.savefig(
            output_dir / "11_individual_species_timelines.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "11_individual_species_timelines.svg", bbox_inches="tight"
        )
        print("✓ Saved: 11_individual_species_timelines.png + .svg")
        plt.close()

# ============================================================================
# 12. SUNRISE/SUNSET ANALYSIS FOR ROE DEER AND WILD BOAR
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

# ============================================================================
# 12. SUNRISE/SUNSET ANALYSIS FOR ROE DEER AND WILD BOAR
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
                    bins=30,
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

                plt.savefig(
                    output_dir
                    / f"{plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_scatter.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    output_dir
                    / f"{plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_scatter.svg",
                    bbox_inches="tight",
                )
                print(
                    f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_scatter.png + .svg"
                )
                plt.close()
                plot_num += 1

        # ========== PLOTS 13-14: Activity relative to sunrise throughout the year ==========
        plot_num = 13
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
                bins=30,
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

            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_sunrise_activity_scatter.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_sunrise_activity_scatter.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_sunrise_activity_scatter.png + .svg"
            )
            plt.close()
            plot_num += 1

        # ========== PLOTS 15-16: Distribution of activity relative to sunset ==========
        plot_num = 15
        for species in target_species:
            species_data = df_target[df_target["class"] == species].copy()
            if len(species_data) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.hist(
                species_data["hours_from_sunset"],
                bins=50,
                color="steelblue",
                edgecolor="black",
                alpha=0.7,
            )
            # Shade 1.5 hours before sunset
            ax1.axvspan(
                -1.5, 0, alpha=0.2, color="orange", label="1.5 hours before Sunset"
            )
            # Shade 0.5 hours after sunset
            ax1.axvspan(
                0, 0.5, alpha=0.3, color="orange", label="0.5 hours after Sunset"
            )
            ax1.axvline(
                x=0, color="orange", linestyle="--", linewidth=2, label="Sunset"
            )
            ax1.set_title(
                f"{species.capitalize()} - Distribution Relative to Sunset",
                fontsize=12,
                fontweight="bold",
            )
            ax1.set_xlabel("Hours from Sunset", fontsize=10)
            ax1.set_ylabel("Number of Sightings", fontsize=10)
            # Add hour ticks
            ax1.set_xticks(range(-6, 7, 1))
            ax1.legend()
            ax1.grid(axis="y", alpha=0.3)

            # Hour of day with sunset overlay
            hourly_counts = species_data.groupby("hour").size()
            hours = range(24)
            counts = [hourly_counts.get(h, 0) for h in hours]

            # Calculate average sunset hour by month
            species_data["month"] = species_data["timestamp"].dt.month
            sunset_hours = (
                species_data.groupby("month")["sunset"]
                .apply(lambda x: x.dt.hour + x.dt.minute / 60)
                .mean()
            )

            ax2.bar(hours, counts, color="steelblue", alpha=0.7, edgecolor="black")

            # Plot average sunset time line
            months_range = range(1, 13)
            sunset_hours_all = []
            for month in months_range:
                month_data = species_data[species_data["month"] == month]
                if len(month_data) > 0:
                    avg_hour = (
                        month_data["sunset"].dt.hour.mean()
                        + month_data["sunset"].dt.minute.mean() / 60
                    )
                    sunset_hours_all.append(avg_hour)
                else:
                    sunset_hours_all.append(np.nan)

            # Add sunset range as shaded area
            if not all(np.isnan(sunset_hours_all)):
                valid_sunset_hours = [h for h in sunset_hours_all if not np.isnan(h)]
                if valid_sunset_hours:
                    min_sunset = min(valid_sunset_hours)
                    max_sunset = max(valid_sunset_hours)
                    ax2.axvspan(
                        min_sunset - 0.5,
                        max_sunset + 0.5,
                        alpha=0.2,
                        color="orange",
                        label=f"Sunset range ({min_sunset:.1f}h-{max_sunset:.1f}h)",
                    )

            ax2.set_title(
                f"{species.capitalize()} - Hourly Activity Pattern",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xlabel("Hour of Day", fontsize=10)
            ax2.set_ylabel("Number of Sightings", fontsize=10)
            ax2.set_xticks(hours[::2])
            ax2.legend()
            ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_distribution.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_distribution.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_sunset_activity_distribution.png + .svg"
            )
            plt.close()
            plot_num += 1

        # ========== PLOTS 17-18: Seasonal patterns ==========
        plot_num = 17
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

            plt.tight_layout()
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_monthly_sunset_patterns.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_monthly_sunset_patterns.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_monthly_sunset_patterns.png + .svg"
            )
            plt.close()
            plot_num += 1

        # ========== PLOTS 19-21: Daily activity pattern over the year with sunset line ==========
        # Use gridspec to add marginal distributions
        from datetime import datetime, timedelta

        from matplotlib.gridspec import GridSpec

        today = datetime.now().date()
        twelve_months_ago = today - timedelta(days=365)

        # Create a figure for each species with distributions
        plot_num = 19

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
                date_obj = date.date()
                sunset_sunrise = get_sun_times(date_obj)
                if sunset_sunrise[0] is not None and sunset_sunrise[1] is not None:
                    sunrise_time, sunset_time = sunset_sunrise
                    sunset_hour = sunset_time.hour + sunset_time.minute / 60
                    sunrise_hour = sunrise_time.hour + sunrise_time.minute / 60
                    all_sunset_hours.append(sunset_hour)
                    all_sunrise_hours.append(sunrise_hour)
                    all_valid_dates.append(date_obj)

            if len(all_valid_dates) > 0:
                # Create shaded areas for twilight periods
                sunrise_minus_0_5 = [h - 0.5 for h in all_sunrise_hours]
                sunrise_plus_1_5 = [h + 1.5 for h in all_sunrise_hours]
                sunset_minus_1_5 = [h - 1.5 for h in all_sunset_hours]
                sunset_plus_0_5 = [h + 0.5 for h in all_sunset_hours]

                # Shade 0.5 hours before sunrise
                ax_main.fill_betweenx(
                    all_valid_dates,
                    sunrise_minus_0_5,
                    all_sunrise_hours,
                    alpha=0.3,
                    color="gold",
                    label="0.5 hours before Sunrise",
                    zorder=5,
                )

                # Shade 1.5 hours after sunrise
                ax_main.fill_betweenx(
                    all_valid_dates,
                    all_sunrise_hours,
                    sunrise_plus_1_5,
                    alpha=0.2,
                    color="gold",
                    label="1.5 hours after Sunrise",
                    zorder=5,
                )

                # Shade 1.5 hours before sunset
                ax_main.fill_betweenx(
                    all_valid_dates,
                    sunset_minus_1_5,
                    all_sunset_hours,
                    alpha=0.2,
                    color="orange",
                    label="1.5 hours before Sunset",
                    zorder=5,
                )

                # Shade 0.5 hours after sunset
                ax_main.fill_betweenx(
                    all_valid_dates,
                    all_sunset_hours,
                    sunset_plus_0_5,
                    alpha=0.3,
                    color="orange",
                    label="0.5 hours after Sunset",
                    zorder=5,
                )

                ax_main.plot(
                    all_sunset_hours,
                    all_valid_dates,
                    color="orange",
                    linewidth=3,
                    label="Sunset Time",
                    alpha=0.9,
                    zorder=10,
                )
                ax_main.plot(
                    all_sunrise_hours,
                    all_valid_dates,
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
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_daily_yearly_activity_pattern.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_{species.replace(' ', '_')}_daily_yearly_activity_pattern.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_{species.replace(' ', '_')}_daily_yearly_activity_pattern.png + .svg"
            )
            plt.close()
            plot_num += 1

        # Save combined figure with both species
        plot_num = 21
        if True:  # Always create combined figure if we have any species
            # Combine both species into one figure
            combined_fig = plt.figure(figsize=(20, 18), constrained_layout=False)
            combined_gs = GridSpec(
                4,
                3,
                figure=combined_fig,
                width_ratios=[4, 1, 0.2],
                height_ratios=[1, 4, 1, 4],
                hspace=0.35,
                wspace=0.05,
            )

            for idx, species in enumerate(target_species):
                species_data = df_target[df_target["class"] == species].copy()
                species_data = species_data[
                    species_data["date"] >= twelve_months_ago
                ].copy()

                row_start = idx * 2
                # Top histogram (hour distribution)
                ax_top = combined_fig.add_subplot(combined_gs[row_start, 0])
                # Main scatter plot
                ax_main = combined_fig.add_subplot(
                    combined_gs[row_start + 1, 0], sharex=ax_top
                )
                # Right histogram (date distribution)
                ax_right = combined_fig.add_subplot(
                    combined_gs[row_start + 1, 1], sharey=ax_main
                )
                # Colorbar position
                ax_cbar = combined_fig.add_subplot(combined_gs[row_start + 1, 2])

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
                    date_obj = date.date()
                    sunset_sunrise = get_sun_times(date_obj)
                    if sunset_sunrise[0] is not None and sunset_sunrise[1] is not None:
                        sunrise_time, sunset_time = sunset_sunrise
                        sunset_hour = sunset_time.hour + sunset_time.minute / 60
                        sunrise_hour = sunrise_time.hour + sunrise_time.minute / 60
                        all_sunset_hours.append(sunset_hour)
                        all_sunrise_hours.append(sunrise_hour)
                        all_valid_dates.append(date_obj)

                if len(all_valid_dates) > 0:
                    # Create shaded areas for twilight periods
                    sunrise_minus_0_5 = [h - 0.5 for h in all_sunrise_hours]
                    sunrise_plus_1_5 = [h + 1.5 for h in all_sunrise_hours]
                    sunset_minus_1_5 = [h - 1.5 for h in all_sunset_hours]
                    sunset_plus_0_5 = [h + 0.5 for h in all_sunset_hours]

                    # Shade 0.5 hours before sunrise
                    ax_main.fill_betweenx(
                        all_valid_dates,
                        sunrise_minus_0_5,
                        all_sunrise_hours,
                        alpha=0.3,
                        color="gold",
                        label="0.5h before Sunrise",
                        zorder=5,
                    )
                    # Shade 1.5 hours after sunrise
                    ax_main.fill_betweenx(
                        all_valid_dates,
                        all_sunrise_hours,
                        sunrise_plus_1_5,
                        alpha=0.2,
                        color="gold",
                        label="1.5h after Sunrise",
                        zorder=5,
                    )
                    # Shade 1.5 hours before sunset
                    ax_main.fill_betweenx(
                        all_valid_dates,
                        sunset_minus_1_5,
                        all_sunset_hours,
                        alpha=0.2,
                        color="orange",
                        label="1.5h before Sunset",
                        zorder=5,
                    )
                    # Shade 0.5 hours after sunset
                    ax_main.fill_betweenx(
                        all_valid_dates,
                        all_sunset_hours,
                        sunset_plus_0_5,
                        alpha=0.3,
                        color="orange",
                        label="0.5h after Sunset",
                        zorder=5,
                    )

                    ax_main.plot(
                        all_sunset_hours,
                        all_valid_dates,
                        color="orange",
                        linewidth=3,
                        label="Sunset Time",
                        alpha=0.9,
                        zorder=10,
                    )
                    ax_main.plot(
                        all_sunrise_hours,
                        all_valid_dates,
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
                    ax_top.plot(x_range, kde_values, "r-", linewidth=2)

                ax_top.set_ylabel("Density", fontsize=10)
                ax_top.tick_params(labelbottom=False)
                ax_top.grid(True, alpha=0.3, axis="y")
                ax_top.set_xlim(0, 24)
                ax_top.set_title(
                    f"{species.capitalize()} Activity Throughout Year and Day (Last 12 Months, n={len(species_data)})",
                    fontsize=12,
                    fontweight="bold",
                )

                # Right distribution (date) - convert to numeric
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
                ax_main.legend(loc="upper right", fontsize=7)
                ax_main.invert_yaxis()
                ax_right.invert_yaxis()

                # Add colorbar
                cbar = plt.colorbar(scatter, cax=ax_cbar)
                cbar.set_label(
                    "Hours from Sunset", rotation=270, labelpad=15, fontsize=10
                )

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

            plt.savefig(
                output_dir
                / f"{plot_num:02d}_combined_daily_yearly_activity_pattern.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                output_dir
                / f"{plot_num:02d}_combined_daily_yearly_activity_pattern.svg",
                bbox_inches="tight",
            )
            print(
                f"✓ Saved: {plot_num:02d}_combined_daily_yearly_activity_pattern.png + .svg"
            )
            plt.close(combined_fig)

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
    "Most Active Hour": f"{hourly_activity.idxmax()}:00"
    if len(hourly_activity) > 0
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
