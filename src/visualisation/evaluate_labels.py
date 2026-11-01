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

try:
    from ..labelling import label_data
    from ..labelling.species import RECENT_SPECIES
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from labelling import label_data
    from labelling.species import RECENT_SPECIES

def _apply_manual_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered = dataframe.copy()
    if "manual_label" in filtered.columns:
        manual_labels = filtered["manual_label"].replace(r"^\s*$", pd.NA, regex=True)
        filtered["class"] = manual_labels.combine_first(filtered["class"])
    return filtered


def filter_visualization_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    labeled = _apply_manual_labels(dataframe)
    return labeled.loc[labeled["class"].isin(RECENT_SPECIES)].copy()


def _create_individual_timeline_axes(n_rows, n_cols, figsize):
    figure, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    return figure, axes.flatten()


def _matplotlib_date_values(values: pd.Series) -> np.ndarray:
    return np.asarray(mdates.date2num(pd.to_datetime(values).to_numpy()))


def _create_daily_yearly_axes():
    from matplotlib.gridspec import GridSpec

    figure = plt.figure(figsize=(18, 10), constrained_layout=False)
    grid = GridSpec(
        2,
        3,
        figure=figure,
        width_ratios=[4, 1, 0.2],
        height_ratios=[1, 4],
        hspace=0.05,
        wspace=0.2,
    )
    return (
        figure,
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    )


def main(csv_path: Path | None = None, output_dir: Path | None = None) -> bool:
    """Main function to generate all visualizations."""
    # Configuration
    model = "MegaDetectorV6_MDV6-yolov10-e_classified"
    image_dir = Path(__file__).parent.parent.parent / "images"
    repo_root = Path(__file__).parent.parent.parent
    labels_dir = repo_root / "labels" / f"labels_{model}"
    csv_path = (
        Path(csv_path)
        if csv_path is not None
        else repo_root / "labels/human_labels.csv"
    )
    output_dir = (
        Path(output_dir) if output_dir is not None else repo_root / "docs" / "diagrams"
    )

    # Initialize plot counter for sequential numbering
    plot_num = 0

    # Location for sunrise/sunset calculations
    # Renningen, Baden-Württemberg, Germany
    LATITUDE = 48.7667
    LONGITUDE = 8.9333
    TIMEZONE = "Europe/Berlin"

    # Load the data
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        return False

    try:
        df = (
            pd.DataFrame(label_data.load_rows(csv_path))
            if csv_path.name == "human_labels.csv"
            else pd.read_csv(csv_path)
        )
        if "temperature_celsius" in df:
            df["temperature_celsius"] = pd.to_numeric(
                df["temperature_celsius"], errors="coerce"
            )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: Could not read results file at {csv_path}: {error}")
        return False

    required_columns = {
        "class",
        "location_id",
        "lighting",
        "temperature_celsius",
    }
    missing_columns = required_columns.difference(df.columns)
    if not {"timestamp", "timestamp_exif"}.intersection(df.columns):
        missing_columns.add("timestamp or timestamp_exif")
    if df.empty:
        print(f"Error: Results file contains no rows: {csv_path}")
        return False
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        print(f"Error: Results file is missing required columns: {missing}")
        return False

    if output_dir.exists():
        print(f"Clearing old visualizations from {output_dir}")
        try:

            def handle_remove_error(func, path, exc):
                os.chmod(path, 0o777)
                func(path)

            shutil.rmtree(
                output_dir,
                ignore_errors=False,
                onerror=handle_remove_error,
            )
            print("✓ Cleared old visualizations")
        except Exception as error:
            print(f"⚠ Warning: Could not fully clear old visualizations: {error}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} labelled images")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Parse timestamps (EXIF preferred, OCR as fallback where EXIF is missing)
    if {"timestamp_exif", "timestamp"}.issubset(df.columns):
        timestamp_source = df["timestamp_exif"].fillna(df["timestamp"])
    else:
        timestamp_source = (
            df["timestamp_exif"] if "timestamp_exif" in df else df["timestamp"]
        )
    df["timestamp"] = pd.to_datetime(timestamp_source, errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    # Filter out invalid data
    df_valid = df[df["timestamp"].notna()].copy()
    print(f"\n{len(df_valid)} images with valid timestamps")

    # Keep only requested wildlife classes
    df = filter_visualization_classes(df)
    df_valid = filter_visualization_classes(df_valid)
    print(f"{len(df_valid)} images after filtering to requested wildlife classes")

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
            f"Date range: {df_valid['date'].min():{label_data.DATE_FORMAT}} to "
            f"{df_valid['date'].max():{label_data.DATE_FORMAT}} ({date_range.days} days)"
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
        int(threshold_position)
        if isinstance(threshold_position, (int, np.integer))
        else 0
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
        ax.set_title(
            f"{species.capitalize()} (n={len(species_data)})", fontweight="bold"
        )
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
        output_dir / f"{plot_num:02d}_species_activity_patterns.svg",
        bbox_inches="tight",
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
    print("Skipping lighting analysis plot")
    plt.close()

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

    # Relevant species only, from config/relevant_species.json
    location_species = pd.crosstab(
        df["location_id"], df.loc[df["class"].isin(RECENT_SPECIES), "class"]
    )
    location_species = location_species.loc[
        [loc for loc in location_counts.index if loc in location_species.index]
    ]
    location_species_grouped = location_species[
        location_species.sum(axis=0).sort_values(ascending=False).index
    ]

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
    plt.savefig(
        output_dir / f"{plot_num:02d}_location_comparison.svg", bbox_inches="tight"
    )
    print(f"✓ Saved: {plot_num:02d}_location_comparison.svg")
    plt.close()

    # ============================================================================
    # INDIVIDUAL SPECIES TIMELINES
    # ============================================================================
    print("\n" + "=" * 70)
    print("INDIVIDUAL SPECIES TIMELINES")
    print("=" * 70)

    if len(df_valid) > 0:
        # Get top species (excluding 'none')
        top_species = (
            species_counts[species_counts.index != "none"].head(8).index.tolist()
        )

        if len(top_species) > 0:
            # Prepare color palette
            colors_species = sns.color_palette("husl", len(top_species))

            # Create individual timeline plots for each species
            n_species = len(top_species)
            n_cols = 2
            n_rows = (n_species + 1) // 2

            fig, axes = _create_individual_timeline_axes(
                n_rows, n_cols, figsize=(16, 4 * n_rows)
            )

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
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter(label_data.DATE_FORMAT)
                )
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
        location = LocationInfo(
            "Camera Location", "Germany", TIMEZONE, LATITUDE, LONGITUDE
        )

        # Calculate sunrise and sunset for each date
        print(
            f"\nCalculating sunrise/sunset times for location: {LATITUDE}°N, {LONGITUDE}°E"
        )

        def get_sun_times(date):
            """Get sunrise and sunset times in standard time (CET, UTC+1)."""
            try:
                import pytz

                s = sun(location.observer, date=date)
                standard_tz = pytz.timezone("Etc/GMT-1")
                sunrise_standard = s["sunrise"].astimezone(standard_tz)
                sunset_standard = s["sunset"].astimezone(standard_tz)
                return sunrise_standard.replace(tzinfo=None), sunset_standard.replace(
                    tzinfo=None
                )
            except Exception:
                return None, None

        # Add sunrise/sunset times to dataframe
        df_valid["sunrise"] = df_valid["date"].apply(lambda d: get_sun_times(d)[0])
        df_valid["sunset"] = df_valid["date"].apply(lambda d: get_sun_times(d)[1])

        # Filter out rows where sunrise/sunset calculation failed
        df_valid = df_valid[
            df_valid["sunset"].notna() & df_valid["sunrise"].notna()
        ].copy()
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
                # Only generate scatter plots for roe deer
                for species in [s for s in target_species if s == "roe deer"]:
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
                        -1.5,
                        0,
                        alpha=0.2,
                        color="orange",
                        label="1.5 hours before Sunset",
                    )
                    # Shade 0.5 hours after sunset
                    ax_main.axhspan(
                        0,
                        0.5,
                        alpha=0.3,
                        color="orange",
                        label="0.5 hours after Sunset",
                    )

                    ax_main.set_title(
                        f"{species.capitalize()} Activity Relative to Sunset (n={len(species_data)})\nNote: Times in standard time (may be ~1h off during DST)",
                        fontsize=13,
                        fontweight="bold",
                    )
                    ax_main.set_xlabel("Date", fontsize=12)
                    ax_main.set_ylabel("Hours from Sunset", fontsize=12)
                    ax_main.set_ylim(-3, 3)
                    ax_main.set_yticks(range(-3, 4))
                    ax_main.grid(True, alpha=0.3)
                    ax_main.legend(loc="upper right")
                    plt.setp(
                        ax_main.xaxis.get_majorticklabels(), rotation=45, ha="right"
                    )

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
            # Only generate scatter plots for roe deer
            for species in [s for s in target_species if s == "roe deer"]:
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
                    f"{species.capitalize()} Activity Relative to Sunrise (n={len(species_data)})\nNote: Times in standard time (may be ~1h off during DST)",
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
                    ax_dist.plot(
                        kde_values, y_range, "r-", linewidth=2, label="Density"
                    )

                ax_dist.axhline(
                    y=0, color="gold", linestyle="--", linewidth=2, alpha=0.7
                )
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
                if species in {"roe deer", "wild boar"}:
                    plot_num += 1
                    print(f"Skipping {species} monthly sunset plot")
                    continue

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

                # Create figure for this species with room between marginal axes.
                fig, ax_top, ax_main, ax_right, ax_cbar = _create_daily_yearly_axes()
                ax_main.sharex(ax_top)
                ax_right.sharey(ax_main)

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
                date_values = _matplotlib_date_values(species_data["date"])
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
                cbar.set_label(
                    "Hours from Sunset", rotation=270, labelpad=20, fontsize=10
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
        "Date Range": f"{df_valid['date'].min():{label_data.DATE_FORMAT}} to "
        f"{df_valid['date'].max():{label_data.DATE_FORMAT}}"
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
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

# %%
