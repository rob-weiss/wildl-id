"""Independent site/species visits derived from EXIF capture times."""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unicodedata import normalize
from zoneinfo import ZoneInfo

from src.labelling.label_data import image_key, write_records

CAMERA_TIMEZONE = ZoneInfo("Europe/Berlin")
INDEPENDENCE_GAP = timedelta(minutes=30)


def exif_datetime(row):
    value = str(row.get("timestamp_exif") or "").strip()
    if not value:
        return None
    try:
        taken = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except ValueError:
        return None
    if taken.tzinfo is not None:
        return taken
    first = taken.replace(tzinfo=CAMERA_TIMEZONE, fold=0)
    second = taken.replace(tzinfo=CAMERA_TIMEZONE, fold=1)
    if first.utcoffset() != second.utcoffset():
        return None
    return first


@dataclass(frozen=True)
class Visit:
    visit_id: str
    location_id: str
    species: str
    started_at: datetime
    ended_at: datetime
    image_keys: tuple[tuple[str, str], ...]

    @property
    def image_count(self):
        return len(self.image_keys)

    @property
    def observed_span_seconds(self):
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class VisitGrouping:
    visits: list[Visit]
    image_visits: dict[tuple[str, str], Visit]
    excluded_count: int


def _effective_species(row):
    return normalize(
        "NFC",
        str(row.get("manual_label") or "").strip()
        or str(row.get("class") or "").strip(),
    ).casefold()


def group_visits(rows):
    grouped = {}
    seen = set()
    signatures = {}
    conflicts = set()
    candidates = []
    excluded_count = 0
    for row in rows:
        taken = exif_datetime(row)
        location = normalize("NFC", str(row.get("location_id") or ""))
        name = row.get("image_file") or ""
        species = _effective_species(row)
        if (
            taken is None
            or not location.strip()
            or not name
            or species in {"", "unknown", "none"}
        ):
            excluded_count += 1
            continue
        key = image_key(row)
        taken = taken.astimezone(UTC)
        signature = (species, taken)
        if key in signatures and signatures[key] != signature:
            conflicts.add(key)
        signatures[key] = signature
        candidates.append((key, location, species, taken))

    for key, location, species, taken in candidates:
        if key in conflicts:
            excluded_count += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault((location, species), []).append((taken, key))

    visits = []

    def finish(location, species, members):
        started_at = members[0][0]
        identity = json.dumps(
            [location, species, started_at.isoformat(), members[0][1]],
            ensure_ascii=True,
        )
        visits.append(
            Visit(
                hashlib.sha256(identity.encode()).hexdigest()[:20],
                location,
                species,
                started_at,
                members[-1][0],
                tuple(key for _, key in members),
            )
        )

    for (location, species), detections in sorted(grouped.items()):
        members = []
        for taken, key in sorted(detections):
            if members and taken - members[-1][0] >= INDEPENDENCE_GAP:
                finish(location, species, members)
                members = []
            members.append((taken, key))
        if members:
            finish(location, species, members)
    visits.sort(
        key=lambda visit: (
            visit.started_at,
            visit.location_id,
            visit.species,
            visit.visit_id,
        )
    )
    return VisitGrouping(
        visits,
        {key: visit for visit in visits for key in visit.image_keys},
        excluded_count,
    )


def _recent_positive_visits(rows, now):
    rows = list(rows)
    grouping = group_visits(rows)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CAMERA_TIMEZONE)
    now = now.astimezone(UTC)
    cutoff = now - timedelta(hours=24)
    recent_times = {}
    for row in rows:
        taken = exif_datetime(row)
        if taken is None or not row.get("location_id") or not row.get("image_file"):
            continue
        key = image_key(row)
        visit = grouping.image_visits.get(key)
        if visit is None or _effective_species(row) != visit.species:
            continue
        taken = taken.astimezone(UTC)
        if cutoff <= taken <= now:
            recent_times[key] = taken
    candidates = []
    for visit in grouping.visits:
        if visit.species in {"human", "vehicle"}:
            continue
        keys = tuple(key for key in visit.image_keys if key in recent_times)
        if len(keys) < 2:
            continue
        candidate = replace(
            visit,
            started_at=recent_times[keys[0]],
            ended_at=recent_times[keys[-1]],
            image_keys=keys,
        )
        if candidate.observed_span_seconds <= 0:
            continue
        candidates.append(candidate)
    return candidates


def longest_recent_visits(rows, *, now=None):
    best = {}
    for candidate in _recent_positive_visits(rows, now):
        previous = best.get(candidate.species)
        if previous is None or (
            candidate.observed_span_seconds,
            candidate.ended_at,
            candidate.visit_id,
        ) > (previous.observed_span_seconds, previous.ended_at, previous.visit_id):
            best[candidate.species] = candidate
    return [best[species] for species in sorted(best)]


def top_recent_visits(rows, *, now=None, limit=3):
    if limit <= 0:
        return []
    return sorted(
        _recent_positive_visits(rows, now),
        key=lambda visit: (
            visit.observed_span_seconds,
            visit.ended_at,
            visit.visit_id,
        ),
        reverse=True,
    )[:limit]


def export_visits(grouping, output_dir):
    output_dir = Path(output_dir)
    summary_path = output_dir / "visits.csv"
    members_path = output_dir / "visit_images.csv"
    write_records(
        summary_path,
        (
            {
                "visit_id": visit.visit_id,
                "location_id": visit.location_id,
                "species": visit.species,
                "started_at": visit.started_at.isoformat(),
                "ended_at": visit.ended_at.isoformat(),
                "observed_span_seconds": visit.observed_span_seconds,
                "image_count": visit.image_count,
                "independence_minutes": 30,
            }
            for visit in grouping.visits
        ),
        (
            "visit_id",
            "location_id",
            "species",
            "started_at",
            "ended_at",
            "observed_span_seconds",
            "image_count",
            "independence_minutes",
        ),
    )
    write_records(
        members_path,
        (
            {"visit_id": visit.visit_id, "location_id": location, "image_file": name}
            for visit in grouping.visits
            for location, name in visit.image_keys
        ),
        ("visit_id", "location_id", "image_file"),
    )
    return summary_path, members_path
