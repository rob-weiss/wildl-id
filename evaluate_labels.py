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
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration
model = "qwen3-vl:235b-a22b-thinking"
image_dir = Path(os.environ["HOME"] + "/mnt/wildlife")
labels_dir = image_dir / f"labels_{model}"
output_dir = labels_dir / "visualizations"
output_dir.mkdir(exist_ok=True)

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
ax1.set_title("Species Distribution - Counts", fontsize=14, fontweight="bold")
ax1.set_xlabel("Species", fontsize=12)
ax1.set_ylabel("Number of Detections", fontsize=12)
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", alpha=0.3)

# Add count labels on bars
for i, v in enumerate(species_counts.values):
    ax1.text(
        i,
        v + max(species_counts.values) * 0.01,
        str(v),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Pie chart (exclude 'none' for better visualization)
species_with_animals = species_counts[species_counts.index != "none"]
ax2.pie(
    species_with_animals.values,
    labels=species_with_animals.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors[: len(species_with_animals)],
)
ax2.set_title(
    "Species Distribution - Percentages (Animals Only)", fontsize=14, fontweight="bold"
)

plt.tight_layout()
plt.savefig(output_dir / "01_species_distribution.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 01_species_distribution.png")
plt.close()

# ============================================================================
# 2. ACTIVITY BY HOUR OF DAY
# ============================================================================
print("\n" + "=" * 70)
print("ACTIVITY BY HOUR OF DAY")
print("=" * 70)

hourly_activity = df_valid.groupby("hour").size()
print("\nActivity counts by hour:")
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
print("✓ Saved: 02_activity_by_hour.png")
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
print("✓ Saved: 03_species_activity_patterns.png")
plt.close()

# ============================================================================
# 4. CALENDAR HEATMAP
# ============================================================================
print("\n" + "=" * 70)
print("ACTIVITY CALENDAR HEATMAP")
print("=" * 70)

if len(df_valid) > 0:
    # Create daily counts
    daily_counts = df_valid.groupby("date").size().reset_index(name="count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for heatmap
    daily_counts["year"] = daily_counts["date"].dt.year
    daily_counts["week"] = daily_counts["date"].dt.isocalendar().week
    daily_counts["day_of_week"] = daily_counts["date"].dt.dayofweek

    # Pivot table for heatmap
    pivot_data = daily_counts.pivot_table(
        values="count", index="day_of_week", columns="week", aggfunc="sum", fill_value=0
    )

    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Detections"},
        ax=ax,
        annot=False,
    )

    ax.set_title("Wildlife Activity Calendar Heatmap", fontsize=16, fontweight="bold")
    ax.set_xlabel("Week of Year", fontsize=12)
    ax.set_ylabel("Day of Week", fontsize=12)
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir / "04_activity_calendar_heatmap.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Saved: 04_activity_calendar_heatmap.png")
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
print("✓ Saved: 05_lighting_analysis.png")
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Detections per location
location_counts.plot(
    kind="bar", ax=ax1, color=sns.color_palette("Set3", len(location_counts))
)
ax1.set_title("Total Detections by Location", fontsize=14, fontweight="bold")
ax1.set_xlabel("Location ID", fontsize=12)
ax1.set_ylabel("Number of Detections", fontsize=12)
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", alpha=0.3)

for i, v in enumerate(location_counts.values):
    ax1.text(
        i,
        v + max(location_counts.values) * 0.01,
        str(v),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Species diversity by location
location_species = pd.crosstab(df["location_id"], df["class"])
# Get top locations
top_locations = location_counts.head(10).index
location_species_top = location_species.loc[top_locations]

# Plot species distribution per location
location_species_top.plot(
    kind="bar",
    stacked=True,
    ax=ax2,
    color=sns.color_palette("husl", len(location_species_top.columns)),
)
ax2.set_title("Species Distribution by Location", fontsize=14, fontweight="bold")
ax2.set_xlabel("Location ID", fontsize=12)
ax2.set_ylabel("Number of Detections", fontsize=12)
ax2.tick_params(axis="x", rotation=45)
ax2.legend(title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "06_location_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Saved: 06_location_comparison.png")
plt.close()

# ============================================================================
# 7. TEMPERATURE ANALYSIS
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
        output_dir / "07_temperature_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Saved: 07_temperature_analysis.png")
    plt.close()
else:
    print("\nNo temperature data available")

# ============================================================================
# 8. TIMELINE VIEW
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
    plt.savefig(output_dir / "08_activity_timeline.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 08_activity_timeline.png")
    plt.close()

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
