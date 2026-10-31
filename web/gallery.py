"""Image gallery with a smart filter that turns prompts into CSV row filters locally."""

import calendar
import csv
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from nicegui import ui
from PIL import Image
from src.labelling.organize_images import (
    capture_time_from_exif,
    image_paths,
    nfc,
    parse_camera_filename_time,
)
from src.labelling.review_app import (
    IMAGES_DIR,
    LABEL_EMOJIS,
    box_overlay_svg,
    cloud_image_url,
    display_label,
)
from src.labelling.review_labels import SHORTCUTS

LOCAL_IMAGE_ROUTE = "/local-images"
PAGE_SIZE = 48
RECENT_SPECIES = {"roe deer", "wild boar", "badger", "marten", "fox", "hare", "raccoon"}
EXAMPLE_PROMPT = "Show only wild boar of the last two weeks"
FILTER_EXAMPLES = {
    "Dates": {
        "last 7 days": "Last 7 days",
        "last 30 days": "Last 30 days",
        "last 90 days": "Last 90 days",
        "yesterday": "Yesterday",
        "this month": "This month",
        "this year": "This year",
        "in winter": "Winter",
        "in spring": "Spring",
        "in summer": "Summer",
        "in autumn": "Autumn",
    },
    "Lighting": {
        "at night": "Night",
        "during the day": "Day",
    },
    "Temperature": {
        "temperature below 0 °C": "Below 0 °C",
        "temperature above 10 C": "Above 10 °C",
    },
    "Review status": {
        "unlabelled": "Not yet labelled",
        "manually labelled": "Manually labelled",
        "misclassified": "Misclassified",
    },
    "Order": {
        "latest 20": "Newest 20",
        "oldest 20": "Oldest 20",
        "oldest first": "Oldest first",
    },
}
FILTER_GUIDE = """### Species and locations

The effective species label is `manual_label` when set; otherwise it is `class`.
Matching ignores case and accepts simple English plurals. `deer` matches roe and
red deer; `bird` or `vogel` matches pigeon, jay, crow, song bird, and owl. Other
aliases include `boar`, `pig`, and `wildschwein` for wild boar; `reh`/`rehe` for
roe deer; `fuchs`/`füchse`, `dachs`, `hase`/`hasen`, and `eichhörnchen`;
`songbird`; `person`/`persons`, `people`, and `mensch`/`menschen` for human;
`car`/`cars` for vehicle; and `empty` or `nothing` for none. Standard labels
include roe deer, red deer, wild boar, pigeon, badger, marten, dog, fox, hare,
squirrel, jay, raccoon, crow, human, none, unknown, owl, cat, vehicle, and song
bird. Labels present in the CSV can also be used directly.

Location names come from `location_id`. Matching ignores case and accents.
Using the first word selects every location that starts with it (for example,
`Sauloch` matches both Sauloch locations).

### Include and exclude

Use `not`, `no`, `except`, `excluding`, `exclude`, `without`, `other than`,
`apart from`, or `but not` before a species or location to exclude it. Multiple
species are alternatives; multiple locations are alternatives. Species,
location, date, lighting, temperature, and review conditions combine together.

### Dates and seasons

Supported relative dates: `today`, `yesterday`, `this week`, `this month`,
`this year`, and `last`, `past`, `previous`, or `recent` followed by an optional
count and `hour`, `day`, `week`, `month`, or `year`. Hours also accept `h`
or `hr`, such as `last 24h` or `last 24 hours`. Hour filters use a rolling
timestamp window ending now, not a calendar-day range or image count.
Undated and future images are excluded; timestamps without an offset use
the server's local time zone. Counts accept digits, `a`/`an`,
`one` through `twelve`, `couple of`, `few`, and `several`.

Absolute dates accept `YYYY-MM-DD`, `D.M.YYYY`, `D Month YYYY`, `Month D YYYY`,
`Month YYYY`, or a year from 1900 to 2099. Day numbers may have `st`, `nd`,
`rd`, or `th`; English month names or abbreviations are accepted. `since`,
`from`, and `starting` include the given date; `after` starts the next day.
`before` ends the previous day; `until`, `till`, and `up to` include the given
date. `between DATE and DATE` and `from DATE to DATE` include both endpoints.
`in March` selects that month in any year; `in winter`, `spring`, `summer`,
`autumn`, or `fall` selects those calendar months in any year. Relative weeks
start on Monday. Date filters skip rows without a valid timestamp; undated rows
also sort after dated rows.

### Lighting and temperature

Night terms: `at night`, `night`, `nighttime`, `nocturnal`, `in the dark`, and
`dark`. Day terms: `during the day`, `by day`, `daylight`, `daytime`, and
`bright`. `not at night` selects bright rows; `not during the day` selects dark
rows.

Temperature comparisons support `below`, `under`, `less than`, `lower than`,
`colder than`, `<`, `above`, `over`, `more than`, `higher than`, `warmer than`,
and `>`. Add a number and optional `C`, `°C`, `°`, `Celsius`, or `degrees` unit;
or say `temperature below 5`. Comparisons are strict. `freezing`, `frost`,
`frosty`, `below zero`, and `below freezing` mean below 0 °C. Temperature
filters skip rows without a valid temperature.

### Review status

Unreviewed terms include `unreviewed`, `unlabelled`/`unlabeled`, `unchecked`,
`not [yet] [been] reviewed/labelled/checked`, and `without [a] manual label(s)`.
Reviewed terms include `reviewed`, `labelled`/`labeled`, `manually labelled`,
`verified`, `checked`, and `with [a] manual label(s)`.

`misclassified`, `corrected`, or a `wrong`/`incorrect` prediction, classification,
or label selects rows with a manual label different from the model's `class`.
Forms such as `wrongly predicted` and `incorrectly classified` are also accepted.

### Order and result count

Results default to newest first. `oldest first`, `earliest first`,
`chronological`, or `sorted by oldest` reverses the order. `latest N`,
`last N` (when N is not followed by a time unit), `newest N`, `recent N`, or
`most recent N` keeps the newest N results. `first N`, `oldest N`, or
`earliest N` keeps the oldest N and sorts oldest first.

Unrecognized words are ignored. If no supported filter is found, the gallery
shows all images and reports that it could not understand the prompt. Prompts
are parsed locally with these rules; generated code is never executed."""

LABEL_ALIASES = {
    "boar": ("wild boar",),
    "boars": ("wild boar",),
    "pig": ("wild boar",),
    "pigs": ("wild boar",),
    "wildschwein": ("wild boar",),
    "wildschweine": ("wild boar",),
    "deer": ("roe deer", "red deer"),
    "reh": ("roe deer",),
    "rehe": ("roe deer",),
    "fuchs": ("fox",),
    "füchse": ("fox",),
    "dachs": ("badger",),
    "hase": ("hare",),
    "hasen": ("hare",),
    "eichhörnchen": ("squirrel",),
    "bird": ("pigeon", "jay", "crow", "song bird", "owl"),
    "birds": ("pigeon", "jay", "crow", "song bird", "owl"),
    "vogel": ("pigeon", "jay", "crow", "song bird", "owl"),
    "vögel": ("pigeon", "jay", "crow", "song bird", "owl"),
    "songbird": ("song bird",),
    "songbirds": ("song bird",),
    "person": ("human",),
    "persons": ("human",),
    "people": ("human",),
    "mensch": ("human",),
    "menschen": ("human",),
    "car": ("vehicle",),
    "cars": ("vehicle",),
    "empty": ("none",),
    "nothing": ("none",),
}
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "couple of": 2,
    "few": 3,
    "several": 3,
}
MONTHS = {
    name.casefold(): number
    for number in range(1, 13)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
} | {"sept": 9}
SEASONS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "autumn": {9, 10, 11},
    "fall": {9, 10, 11},
}

_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_YEAR = r"(?:19|20)\d{2}"
_DATE = (
    r"(?:\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}\.\d{1,2}\.\d{4}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})\.?,?\s+\d{{4}}"
    rf"|(?:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|(?:{_MONTH})\.?\s+\d{{4}}"
    rf"|{_YEAR})"
)
_DATE_TOKEN = rf"(?<![\w.-]){_DATE}(?![\w-]|\.\d)"
DATE_RULES = (
    (
        rf"\bbetween\s+(?P<a>{_DATE_TOKEN})\s+(?:and|to|-)\s+(?P<b>{_DATE_TOKEN})",
        "range",
    ),
    (
        rf"\bfrom\s+(?P<a>{_DATE_TOKEN})\s+(?:to|until|till|-)\s+(?P<b>{_DATE_TOKEN})",
        "range",
    ),
    (rf"(?P<a>{_DATE_TOKEN})\s+(?:to|until|till|-|–)\s+(?P<b>{_DATE_TOKEN})", "range"),
    (rf"\b(?:since|from|starting)\s+(?P<a>{_DATE_TOKEN})", "since"),
    (rf"\bafter\s+(?P<a>{_DATE_TOKEN})", "after"),
    (rf"\bbefore\s+(?P<a>{_DATE_TOKEN})", "before"),
    (rf"\b(?:until|till|up\s+to)\s+(?P<a>{_DATE_TOKEN})", "until"),
    (rf"(?P<a>{_DATE_TOKEN})", "on"),
)
_NUMBER = "|".join(
    [r"\d+", *sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)]
)
RELATIVE_PERIOD = re.compile(
    r"\b(?:(?:in|over|during|within|from)\s+)?(?:the\s+)?"
    r"(?:last|past|previous|recent)\s+"
    rf"(?:(?P<count>{_NUMBER})\s*)?(?P<unit>hour|hr|h|day|week|month|year)s?\b"
)
CURRENT_PERIOD = re.compile(
    r"\b(?:(?:in|during)\s+)?this\s+(?P<unit>week|month|year)\b"
)
NEGATION = re.compile(
    r"\b(?:but\s+not|not|no|except(?:\s+for)?|excluding|exclude|without"
    r"|other\s+than|apart\s+from)\b(?:\s+(?:at|in|during|from|on)\b)?"
)
NEGATION_END = re.compile(
    r"[.;!?]|\b(?:only|but|from|during|since|between|of|last|past|this|before"
    r"|after|until|on|in|at|taken|show|with|where|when)\b"
)
_TEMPERATURE_UNIT = r"°\s*c?\b|°|degrees?(?:\s+celsius)?\b|celsius\b|c\b"
TEMPERATURE = re.compile(
    r"(?P<subject>\btemperatures?\s+(?:is\s+|was\s+)?)?"
    r"(?P<op>\b(?:below|under|less\s+than|lower\s+than|colder\s+than|above|over"
    r"|more\s+than|higher\s+than|warmer\s+than)\b|[<>])\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_TEMPERATURE_UNIT})?"
)
FREEZING = re.compile(r"\b(?:below\s+(?:zero|freezing)|freezing|frost(?:y)?)\b")
NOT_REVIEWED = re.compile(
    r"\b(?:unreviewed|unlabell?ed|unchecked"
    r"|not\s+(?:yet\s+)?(?:been\s+)?(?:reviewed|labell?ed|checked)"
    r"|without\s+(?:a\s+)?manual\s+labels?)\b"
)
MISCLASSIFIED = re.compile(
    r"\b(?:misclassified|misclassifications?|corrected"
    r"|(?:wrong|incorrect)(?:ly)?\s+(?:predicted|classified|predictions?"
    r"|classifications?|labels?))\b"
)
REVIEWED = re.compile(
    r"\b(?:reviewed|manually\s+labell?ed|labell?ed|verified|checked"
    r"|with\s+(?:a\s+)?manual\s+labels?)\b"
)
NIGHT = re.compile(
    r"\b(?:at\s+night|night(?:s|time)?|nocturnal|in\s+the\s+dark|dark)\b"
)
DAY = re.compile(
    r"\b(?:during\s+the\s+day|by\s+day|in\s+daylight|daylight|daytime|bright)\b"
)
NEWEST_LIMIT = re.compile(r"\b(?:last|latest|newest|most\s+recent|recent)\s+(\d+)\b")
OLDEST_LIMIT = re.compile(r"\b(?:first|oldest|earliest)\s+(\d+)\b")
OLDEST_FIRST = re.compile(
    r"\b(?:oldest|earliest)\s+first\b|\bchronological(?:ly)?\b"
    r"|\bsorted\s+by\s+oldest\b"
)
NEWEST_FIRST = re.compile(r"\b(?:newest|latest|most\s+recent)\s+first\b")


@dataclass
class GalleryFilter:
    labels: set[str] = field(default_factory=set)
    excluded_labels: set[str] = field(default_factory=set)
    locations: set[str] = field(default_factory=set)
    excluded_locations: set[str] = field(default_factory=set)
    start: date | None = None
    end: date | None = None
    months: set[int] = field(default_factory=set)
    lighting: str | None = None
    min_temperature: float | None = None
    max_temperature: float | None = None
    reviewed: bool | None = None
    misclassified: bool = False
    oldest_first: bool = False
    limit: int | None = None
    hours: int | None = None

    def is_empty(self):
        return self == GalleryFilter()


def normalize_text(text):
    return unicodedata.normalize("NFC", text).casefold().replace("−", "-")


def ascii_variants(text):
    folded = normalize_text(text)
    expanded = folded.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    stripped = "".join(
        character
        for character in unicodedata.normalize("NFKD", folded)
        if not unicodedata.combining(character)
    )
    return {folded, expanded, stripped}


def row_label(row):
    return (row.get("manual_label") or "").strip() or (row.get("class") or "").strip()


def row_datetime(row):
    for key in ("timestamp", "timestamp_exif"):
        value = (row.get(key) or "").strip()
        if value:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


def recent_rows(rows, now=None):
    return filter_rows(
        rows, GalleryFilter(labels=RECENT_SPECIES | {"", "unknown"}, hours=24), now=now
    )


def shift_months(day, months):
    year, month_index = divmod(day.year * 12 + day.month - 1 + months, 12)
    month = month_index + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def date_span(text):
    """Return the first and last day covered by a date expression."""
    text = text.strip().rstrip(".,")
    try:
        if match := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
            day = date(*map(int, match.groups()))
            return day, day
        if match := re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text):
            day_number, month, year = map(int, match.groups())
            day = date(year, month, day_number)
            return day, day
        if match := re.fullmatch(
            rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\.?,?\s+(\d{{4}})",
            text,
        ):
            day = date(int(match[3]), MONTHS[match[2]], int(match[1]))
            return day, day
        if match := re.fullmatch(
            rf"({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", text
        ):
            day = date(int(match[3]), MONTHS[match[1]], int(match[2]))
            return day, day
        if match := re.fullmatch(rf"({_MONTH})\.?\s+(\d{{4}})", text):
            year, month = int(match[2]), MONTHS[match[1]]
            return date(year, month, 1), date(
                year, month, calendar.monthrange(year, month)[1]
            )
        if re.fullmatch(_YEAR, text):
            return date(int(text), 1, 1), date(int(text), 12, 31)
    except ValueError:
        return None
    return None


def label_terms(labels):
    canonical = {normalize_text(label) for label in labels if label.strip()}
    canonical |= {normalize_text(label) for label in SHORTCUTS.values()}
    canonical |= {normalize_text(label) for label in LABEL_EMOJIS}
    terms = {}
    for label in canonical:
        variants = {label, f"{label}s", f"{label}es"}
        if label.endswith("y"):
            variants.add(f"{label[:-1]}ies")
        for variant in variants:
            terms.setdefault(variant, set()).add(label)
    for alias, targets in LABEL_ALIASES.items():
        terms.setdefault(alias, set()).update(targets)
    return terms


def location_terms(locations):
    terms = {}
    for location in locations:
        name = nfc(location).strip()
        if not name:
            continue
        # "Sauloch" selects both "Sauloch vorne" and "Sauloch hinten".
        for text in {name, name.split()[0]}:
            for variant in ascii_variants(text):
                terms.setdefault(variant, set()).add(name)
    return terms


class _Prompt:
    def __init__(self, prompt):
        self.text = normalize_text(prompt)

    def take(self, pattern):
        matches = list(re.finditer(pattern, self.text))
        characters = list(self.text)
        for match in matches:
            start, end = match.span()
            characters[start:end] = " " * (end - start)
        self.text = "".join(characters)
        return matches


def _negation_spans(text):
    spans = []
    for match in NEGATION.finditer(text):
        end = NEGATION_END.search(text, match.end())
        spans.append((match.start(), end.start() if end else len(text)))
    return spans


def _is_negated(position, spans):
    return any(start <= position < end for start, end in spans)


def _narrow_dates(result, start, end):
    if start is not None and (result.start is None or start > result.start):
        result.start = start
    if end is not None and (result.end is None or end < result.end):
        result.end = end


def _extract_dates(prompt, result, today):
    for pattern, kind in DATE_RULES:
        for match in prompt.take(pattern):
            first = date_span(match["a"])
            last = date_span(match["b"]) if kind == "range" else first
            if first is None or last is None:
                continue
            if kind == "range":
                _narrow_dates(result, first[0], last[1])
            elif kind == "since":
                _narrow_dates(result, first[0], None)
            elif kind == "after":
                _narrow_dates(result, first[1] + timedelta(days=1), None)
            elif kind == "before":
                _narrow_dates(result, None, first[0] - timedelta(days=1))
            elif kind == "until":
                _narrow_dates(result, None, first[1])
            else:
                _narrow_dates(result, first[0], first[1])

    for match in prompt.take(RELATIVE_PERIOD):
        count = match["count"] or "1"
        amount = int(count) if count.isdigit() else NUMBER_WORDS[count]
        unit = match["unit"]
        if unit in {"hour", "hr", "h"}:
            result.hours = (
                min(result.hours, amount) if result.hours is not None else amount
            )
            continue
        if unit == "day":
            start = today - timedelta(days=amount)
        elif unit == "week":
            start = today - timedelta(weeks=amount)
        else:
            start = shift_months(today, -amount * (12 if unit == "year" else 1))
        _narrow_dates(result, start, today)
    for match in prompt.take(CURRENT_PERIOD):
        unit = match["unit"]
        if unit == "week":
            start = today - timedelta(days=today.weekday())
        elif unit == "month":
            start = today.replace(day=1)
        else:
            start = today.replace(month=1, day=1)
        _narrow_dates(result, start, today)
    if prompt.take(r"\byesterday\b"):
        yesterday = today - timedelta(days=1)
        _narrow_dates(result, yesterday, yesterday)
    if prompt.take(r"\btoday\b"):
        _narrow_dates(result, today, today)

    month_names = "|".join(name for name in MONTHS if name != "may")
    for pattern in (
        rf"\b(?:in|during|of)\s+(?P<month>{_MONTH})\b",
        rf"\b(?P<month>{month_names})\b",
    ):
        for match in prompt.take(pattern):
            result.months.add(MONTHS[match["month"]])
    for match in prompt.take(rf"\b(?P<season>{'|'.join(SEASONS)})\b"):
        result.months.update(SEASONS[match["season"]])


def _extract_order(prompt, result):
    if prompt.take(OLDEST_FIRST):
        result.oldest_first = True
    prompt.take(NEWEST_FIRST)
    for match in prompt.take(OLDEST_LIMIT):
        result.oldest_first = True
        result.limit = int(match[1])
    for match in prompt.take(NEWEST_LIMIT):
        result.limit = int(match[1])


def _extract_temperature(prompt, result):
    if prompt.take(FREEZING):
        result.max_temperature = 0.0
    for match in prompt.take(TEMPERATURE):
        op = re.sub(r"\s+", " ", match["op"])
        if not (
            match["unit"] or match["subject"] or op in {"colder than", "warmer than"}
        ):
            continue
        value = float(match["value"])
        if op in {"below", "under", "less than", "lower than", "colder than", "<"}:
            if result.max_temperature is None or value < result.max_temperature:
                result.max_temperature = value
        elif result.min_temperature is None or value > result.min_temperature:
            result.min_temperature = value


def _extract_terms(prompt, terms, included, excluded, negations):
    if not terms:
        return
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        + r")(?!\w)"
    )
    for match in prompt.take(pattern):
        target = excluded if _is_negated(match.start(), negations) else included
        target.update(terms[match.group()])


def parse_filter_prompt(prompt, labels=(), locations=(), today=None):
    """Translate a natural-language prompt into a structured, validated filter."""
    today = today or date.today()
    result = GalleryFilter()
    prompt = _Prompt(prompt)

    if prompt.take(NOT_REVIEWED):
        result.reviewed = False
    if prompt.take(MISCLASSIFIED):
        result.misclassified = True
    if prompt.take(REVIEWED):
        result.reviewed = True
    negations = _negation_spans(prompt.text)

    _extract_temperature(prompt, result)
    _extract_dates(prompt, result, today)
    _extract_order(prompt, result)
    for pattern, lighting in ((NIGHT, "dark"), (DAY, "bright")):
        for match in prompt.take(pattern):
            if _is_negated(match.start(), negations):
                lighting = "bright" if lighting == "dark" else "dark"
            result.lighting = lighting
    _extract_terms(
        prompt,
        location_terms(locations),
        result.locations,
        result.excluded_locations,
        negations,
    )
    _extract_terms(
        prompt, label_terms(labels), result.labels, result.excluded_labels, negations
    )
    result.labels -= result.excluded_labels
    result.locations -= result.excluded_locations
    return result


def filter_rows(rows, gallery_filter, now=None):
    """Return matching rows, newest first unless the filter asks otherwise."""
    f = gallery_filter
    has_date_filter = bool(f.start or f.end or f.months or f.hours is not None)
    if f.hours is not None:
        now = (now or datetime.now().astimezone()).astimezone()
        cutoff = now - timedelta(hours=f.hours)
    locations = {normalize_text(location) for location in f.locations}
    excluded_locations = {normalize_text(location) for location in f.excluded_locations}
    dated, undated = [], []
    for row in rows:
        label = normalize_text(row_label(row))
        if f.labels and label not in f.labels or label in f.excluded_labels:
            continue
        location = normalize_text(nfc(row.get("location_id") or ""))
        if locations and location not in locations or location in excluded_locations:
            continue
        manual_label = (row.get("manual_label") or "").strip()
        if f.reviewed is not None and bool(manual_label) != f.reviewed:
            continue
        if f.misclassified and (
            not manual_label
            or normalize_text(manual_label)
            == normalize_text((row.get("class") or "").strip())
        ):
            continue
        if (
            f.lighting
            and normalize_text((row.get("lighting") or "").strip()) != f.lighting
        ):
            continue
        if f.min_temperature is not None or f.max_temperature is not None:
            try:
                temperature = float(row.get("temperature_celsius") or "")
            except ValueError:
                continue
            if f.min_temperature is not None and temperature <= f.min_temperature:
                continue
            if f.max_temperature is not None and temperature >= f.max_temperature:
                continue
        taken = row_datetime(row)
        if has_date_filter:
            if taken is None:
                continue
            if f.hours is not None:
                taken = taken.astimezone(now.tzinfo)
                if not cutoff <= taken <= now:
                    continue
            day = taken.date()
            if f.start and day < f.start or f.end and day > f.end:
                continue
            if f.months and day.month not in f.months:
                continue
        (dated if taken else undated).append((taken, row))
    dated.sort(key=lambda item: item[0], reverse=not f.oldest_first)
    matches = [row for _, row in dated + undated]
    return matches[: f.limit] if f.limit else matches


def describe_filter(gallery_filter):
    f = gallery_filter
    parts = []
    if f.labels:
        parts.append("only " + ", ".join(map(display_label, sorted(f.labels))))
    if f.excluded_labels:
        parts.append(
            "excluding " + ", ".join(map(display_label, sorted(f.excluded_labels)))
        )
    if f.locations:
        parts.append("at " + ", ".join(sorted(f.locations)))
    if f.excluded_locations:
        parts.append("not at " + ", ".join(sorted(f.excluded_locations)))
    if f.start or f.end:
        parts.append(f"📅 {f.start or '…'} – {f.end or '…'}")
    if f.hours is not None:
        parts.append(f"last {f.hours} hours")
    if f.months:
        parts.append(
            "in " + ", ".join(calendar.month_name[month] for month in sorted(f.months))
        )
    if f.lighting:
        parts.append("🌙 night" if f.lighting == "dark" else "☀️ day")
    if f.min_temperature is not None:
        parts.append(f"🌡️ > {f.min_temperature:g} °C")
    if f.max_temperature is not None:
        parts.append(f"🌡️ < {f.max_temperature:g} °C")
    if f.reviewed is not None:
        parts.append("manually labelled" if f.reviewed else "not yet labelled")
    if f.misclassified:
        parts.append("prediction corrected")
    if f.limit:
        parts.append(f"{'oldest' if f.oldest_first else 'newest'} {f.limit}")
    elif f.oldest_first:
        parts.append("oldest first")
    return " · ".join(parts) or "all images"


def image_source(row, local_images, images_dir, base_url):
    """Prefer the local photo; fall back to the cloud copy under images/<year>/."""
    local_path = local_images.get((nfc(row["location_id"]), row["image_file"]))
    if local_path is not None:
        relative_path = local_path.relative_to(images_dir).as_posix()
        return f"{LOCAL_IMAGE_ROUTE}/{quote(relative_path, safe='/')}"
    taken = row_datetime(row)
    if taken is None:
        return None
    return cloud_image_url(
        images_dir / str(taken.year) / row["location_id"] / row["image_file"],
        images_dir,
        base_url,
    )


def load_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"location_id", "image_file", "class"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV needs location_id, image_file, and class columns")
        return list(reader)


def rows_with_originals(rows, local_images):
    merged = [dict(row) for row in rows]
    known = {(nfc(row["location_id"]), row["image_file"]) for row in rows}
    for location, image_file in local_images:
        if (location, image_file) not in known:
            merged.append(
                {"location_id": location, "image_file": image_file, "class": ""}
            )
    for row in merged:
        if row_datetime(row) is not None:
            continue
        path = local_images.get((nfc(row["location_id"]), row["image_file"]))
        if path is None:
            continue
        try:
            with Image.open(path) as image:
                taken = capture_time_from_exif(image.getexif())
        except (ValueError, SyntaxError):
            taken = None
        except OSError:
            continue
        taken = taken or parse_camera_filename_time(path)
        if taken is not None:
            row["timestamp_exif"] = taken.isoformat()
    return merged


def annotation_labels(row):
    labels = []
    for title, column in (("Manual", "manual_label"), ("Automatic", "class")):
        label = (row.get(column) or "").strip()
        if label and label.casefold() != "unknown":
            labels.append(f"{title}: {display_label(label)}")
    return " · ".join(labels) or "Unlabelled"


def boxed_image(row, source, classes, display_size=(320, 180), fit_viewport=False):
    image = ui.interactive_image(source).classes(classes)

    def on_loaded(event):
        width, height = event.args["width"], event.args["height"]
        if fit_viewport:
            image.style(f"width: min(85vw, {60 * width / height:g}vh)")
        image.set_content(
            box_overlay_svg(
                row,
                (width, height),
                display_size=display_size,
                show_label=annotation_labels(row) != "Unlabelled",
            )
        )

    image.on("loaded", on_loaded)
    return image


def gallery_card(row, source, on_click, show_boxes=False):
    taken = row_datetime(row)
    when = taken.strftime("%Y-%m-%d %H:%M") if taken else "unknown date"
    with ui.card().tight().classes("w-full"):
        image = (
            boxed_image(row, source, "w-full cursor-pointer")
            if show_boxes
            else ui.image(source).props("fit=contain").classes("w-full cursor-pointer")
        )
        image.on("click", on_click)
        with ui.card_section().classes("p-2"):
            ui.label(
                annotation_labels(row) if show_boxes else display_label(row_label(row))
            ).classes("font-medium")
            ui.label(f"{row['location_id']} · {when}").classes("text-xs text-gray-600")


def build_image_viewer(items, show_boxes=False):
    visible_items = []
    current_index = None
    marker = "recent-image" if show_boxes else "gallery-image"

    def close_image_viewer():
        nonlocal current_index
        current_index = None
        dialog.close()

    def move_image(offset):
        if current_index is not None:
            show_image(current_index + offset)

    with ui.dialog() as dialog, ui.card().classes("max-w-[95vw] max-h-[95vh]"):
        viewer = ui.column().classes("w-full")
        caption = ui.label()
        with ui.row().classes("w-full items-center justify-between gap-2"):
            previous_button = (
                ui.button(
                    "" if show_boxes else "Previous image",
                    icon="chevron_left",
                    on_click=lambda: move_image(-1),
                )
                .props("flat round" if show_boxes else "outline no-caps")
                .props('aria-label="Previous image"')
                .mark(f"{marker}-previous")
            )
            previous_button.tooltip("Previous image")
            next_button = (
                ui.button(
                    "" if show_boxes else "Next image",
                    icon="chevron_right",
                    on_click=lambda: move_image(1),
                )
                .props("flat round" if show_boxes else "outline no-caps")
                .props('aria-label="Next image"')
                .mark(f"{marker}-next")
            )
            next_button.tooltip("Next image")
            ui.button(
                "" if show_boxes else "Close",
                icon="close",
                on_click=close_image_viewer,
            ).props("flat round" if show_boxes else "flat no-caps").props(
                'aria-label="Close image"' if show_boxes else 'aria-label="Close"'
            ).mark(f"{marker}-close").tooltip("Close image" if show_boxes else "Close")

    def show_image(index):
        nonlocal current_index
        if index < 0 or index >= len(visible_items):
            return
        current_index = index
        row, source = visible_items[index]
        viewer.clear()
        with viewer:
            if show_boxes:
                boxed_image(
                    row, source, "max-w-full", (960, 540), fit_viewport=True
                ).mark("recent-viewer-image")
            else:
                ui.image(source).props("fit=contain").classes(
                    "w-[85vw] h-[75vh] max-w-full"
                )
        taken = row_datetime(row)
        when = taken.strftime("%Y-%m-%d %H:%M") if taken else "unknown date"
        labels = annotation_labels(row) if show_boxes else display_label(row_label(row))
        caption.set_text(
            f"{index + 1} / {len(visible_items)} · {labels} · {row['location_id']} · {when}"
        )
        previous_button.set_enabled(index > 0)
        next_button.set_enabled(index < len(visible_items) - 1)
        dialog.open()

    def open_image(index):
        visible_items[:] = items
        show_image(index)

    def on_dialog_hide(_):
        nonlocal current_index
        if not dialog.value:
            current_index = None

    def on_key(event):
        if current_index is None or not event.action.keydown:
            return
        if (
            event.action.repeat
            or event.modifiers.alt
            or event.modifiers.ctrl
            or event.modifiers.meta
        ):
            return
        if event.key.arrow_left:
            move_image(-1)
        elif event.key.arrow_right:
            move_image(1)
        elif event.key.escape:
            close_image_viewer()

    dialog.on("hide", on_dialog_hide)
    ui.keyboard(on_key=on_key, repeating=False, ignore=["input", "select", "textarea"])
    return open_image, close_image_viewer


def build_recent_gallery(csv_path, images_dir=None, image_base_url=None):
    images_dir = Path(images_dir) if images_dir is not None else IMAGES_DIR
    all_items = []
    selected_species = set()
    missing_count = 0
    species_names = {
        species: display_label(species) for species in sorted(RECENT_SPECIES)
    }
    species_names["unlabelled"] = "Unlabelled"
    species_buttons = {}
    with ui.column().classes("w-full gap-3"):
        ui.label("Last 24 hours").classes("text-xl font-semibold")
        with ui.row().classes("w-full items-center gap-2"):
            for species, title in {"all": "All", **species_names}.items():
                species_buttons[species] = (
                    ui.button(
                        title, on_click=lambda species=species: toggle_species(species)
                    )
                    .props("no-caps dense")
                    .mark(f"recent-species-{species.replace(' ', '-')}")
                )
        species_summary = (
            ui.label().classes("text-sm text-gray-700").mark("recent-species-summary")
        )
        status = ui.label().classes("text-sm text-gray-600")
        grid = ui.grid(
            columns="repeat(auto-fit, minmax(min(100%, 220px), 1fr))"
        ).classes("w-full gap-3")
        pagination = ui.pagination(1, 1, on_change=lambda: show_page()).props(
            "direction-links max-pages=5"
        )
    items = []
    show_image, close_image_viewer = build_image_viewer(items, show_boxes=True)

    def species_key(row):
        label = normalize_text(row_label(row))
        return "unlabelled" if label in {"", "unknown"} else label

    def update_counts():
        photos = {species: set() for species in species_names}
        for row, _ in all_items:
            photos[species_key(row)].add((nfc(row["location_id"]), row["image_file"]))
        counts = {species: len(keys) for species, keys in photos.items()}
        counts["all"] = len(set().union(*photos.values()))
        for species, button in species_buttons.items():
            title = "All" if species == "all" else species_names[species]
            button.set_text(f"{title} ({counts[species]})")
            active = (
                not selected_species
                if species == "all"
                else species in selected_species
            )
            button.props(
                "unelevated" if active else "outline",
                remove="outline" if active else "unelevated",
            ).props(f'aria-pressed="{str(active).lower()}"')
        species_summary.set_text(
            " · ".join(
                f"{species_names[species]}: {counts[species]}"
                for species in species_names
                if counts[species]
            )
            or "No sightings in the last 24 hours"
        )

    def toggle_species(species):
        if species == "all":
            selected_species.clear()
        elif species in selected_species:
            selected_species.remove(species)
        else:
            selected_species.add(species)
        close_image_viewer()
        update_counts()
        apply_species(reset_page=True)

    def apply_species(reset_page=False):
        items[:] = [
            (row, source)
            for row, source in all_items
            if not selected_species or species_key(row) in selected_species
        ]
        status.set_text(
            f"{len(items)} images"
            if items
            else (
                "No sightings for selected species"
                if selected_species
                else "No sightings in the last 24 hours"
            )
        )
        if missing_count:
            status.set_text(
                f"{len(items)} images · {missing_count} not available locally or in the cloud"
            )
        pagination.max = max(1, math.ceil(len(items) / PAGE_SIZE))
        pagination.set_visibility(pagination.max > 1)
        pagination.set_value(1 if reset_page else min(pagination.value, pagination.max))
        show_page()

    def show_page():
        grid.clear()
        first = (pagination.value - 1) * PAGE_SIZE
        with grid:
            for index, (row, source) in enumerate(
                items[first : first + PAGE_SIZE], start=first
            ):
                gallery_card(
                    row,
                    source,
                    lambda index=index: show_image(index),
                    show_boxes=True,
                )

    def refresh():
        nonlocal missing_count
        try:
            local_images = image_paths(images_dir)
            try:
                rows = load_rows(csv_path)
            except FileNotFoundError:
                if not local_images:
                    raise
                rows = []
            rows = recent_rows(rows_with_originals(rows, local_images))
            all_items[:] = [
                (row, source)
                for row in rows
                if (
                    source := image_source(
                        row, local_images, Path(images_dir), image_base_url
                    )
                )
            ]
        except (OSError, ValueError, csv.Error):
            status.set_text("Recent images unavailable")
            all_items.clear()
            items.clear()
            update_counts()
            pagination.set_visibility(False)
            grid.clear()
            return
        missing_count = len(rows) - len(all_items)
        update_counts()
        apply_species()

    refresh()
    ui.timer(60, refresh, immediate=False)


def build_gallery_page(
    csv_path, images_dir=IMAGES_DIR, image_base_url=None, today=None
):
    ui.label("Gallery").classes("text-xl font-semibold")
    try:
        rows = load_rows(csv_path)
    except (OSError, ValueError, csv.Error) as error:
        ui.label("Gallery data unavailable").classes("text-lg font-semibold")
        ui.label(str(error))
        return

    images_dir = Path(images_dir)
    local_images = image_paths(images_dir)
    labels = {
        str(value).strip()
        for row in rows
        for value in (row.get("class"), row.get("manual_label"))
        if str(value or "").strip()
    }
    locations = {nfc(row["location_id"]) for row in rows}
    species_options = {
        f"only {species}": display_label(species)
        for species in sorted(labels, key=normalize_text)
    }
    location_options = {
        f"at {location}": location for location in sorted(locations, key=normalize_text)
    }
    date_options = dict(FILTER_EXAMPLES["Dates"])
    years = {taken.year for row in rows if (taken := row_datetime(row)) is not None}
    date_options.update({f"in {year}": str(year) for year in sorted(years)})
    date_options.update(
        {
            f"in {calendar.month_name[month]}": f"{calendar.month_name[month]} "
            "(any year)"
            for month in range(1, 13)
        }
    )
    items = []
    show_image, close_image_viewer = build_image_viewer(items)

    with ui.row().classes("w-full items-center gap-2"):
        prompt = (
            ui.input("Smart filter", placeholder=f'e.g. "{EXAMPLE_PROMPT}"')
            .props("clearable")
            .classes("flex-grow")
        )
        ui.button("Filter", icon="search", on_click=lambda: apply()).props("no-caps")
    with ui.expansion("Filter guide", icon="info").classes("w-full"):
        ui.markdown(FILTER_GUIDE).classes("w-full")
    with ui.card().classes("w-full gap-2 border border-green-200 bg-green-50"):
        ui.label("Example filters").classes("font-medium")
        ui.label(
            "Selections combine across categories. Species and Locations allow "
            "multiple choices; clear a category to remove it. Quick examples "
            "remain below."
        ).classes("text-sm text-gray-600")
        selections = {}
        selectors = {}
        selection_order = (
            "Species",
            "Locations",
            "Dates",
            "Lighting",
            "Temperature",
            "Review status",
            "Order",
        )

        def update_selection(category, value):
            if value:
                selections[category] = value if isinstance(value, list) else [value]
            else:
                selections.pop(category, None)
            prompt.value = " ".join(
                fragment
                for selected_category in selection_order
                for fragment in selections.get(selected_category, [])
            )
            prompt.update()
            apply()

        with ui.grid(columns="repeat(auto-fit, minmax(190px, 1fr))").classes(
            "w-full gap-2"
        ):
            dropdowns = (
                ("Species", species_options, True),
                ("Locations", location_options, True),
                ("Dates", date_options, False),
                *(
                    (category, examples, False)
                    for category, examples in FILTER_EXAMPLES.items()
                    if category != "Dates"
                ),
            )
            for category, options, multiple in dropdowns:
                selectors[category] = ui.select(
                    options,
                    label=category,
                    multiple=multiple,
                    clearable=True,
                    on_change=lambda event, category=category: update_selection(
                        category, event.value
                    ),
                ).classes("w-full")
        with ui.row().classes("w-full flex-wrap gap-2"):
            for example in (
                "Show only wild boar of the last two weeks",
                "Only foxes",
                "Foxes at night in winter",
                "Badgers at Sauloch in March 2025",
                "Unlabelled images below 0 °C",
                "Latest 20 misclassified",
            ):
                ui.button(
                    example,
                    on_click=lambda example=example: select_example(example),
                ).props("flat no-caps dense")
    summary = ui.label().classes("text-sm text-gray-600")
    pagination = ui.pagination(
        1, 1, direction_links=True, on_change=lambda: show_page()
    )
    grid = ui.grid(columns="repeat(auto-fill, minmax(220px, 1fr))").classes(
        "w-full gap-3"
    )

    def show_page():
        grid.clear()
        first = (pagination.value - 1) * PAGE_SIZE
        with grid:
            for index, (row, source) in enumerate(
                items[first : first + PAGE_SIZE], start=first
            ):
                gallery_card(row, source, lambda index=index: show_image(index))

    def apply():
        close_image_viewer()
        text = prompt.value or ""
        gallery_filter = parse_filter_prompt(text, labels, locations, today)
        if text.strip() and gallery_filter.is_empty():
            ui.notify("Filter not understood; showing all images", type="warning")
        matches = filter_rows(rows, gallery_filter)
        items[:] = [
            (row, source)
            for row in matches
            if (source := image_source(row, local_images, images_dir, image_base_url))
        ]
        missing = len(matches) - len(items)
        summary.set_text(
            f"{len(items)} images · {describe_filter(gallery_filter)}"
            + (f" · {missing} not available locally or in the cloud" if missing else "")
        )
        pages = max(1, math.ceil(len(items) / PAGE_SIZE))
        pagination.max = pages
        pagination.set_visibility(pages > 1)
        if pagination.value != 1:
            pagination.set_value(1)
        else:
            show_page()

    def select_example(example):
        selections.clear()
        for category, selector in selectors.items():
            selector.value = [] if category in {"Species", "Locations"} else None
            selector.update()
        prompt.value = example
        prompt.update()
        apply()

    prompt.on("keydown.enter", apply)
    apply()
