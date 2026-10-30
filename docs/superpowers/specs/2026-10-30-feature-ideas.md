# Feature Ideas

Status: proposals only. Nothing here is designed or approved.

## Data Basis

- About 19.7k labelled images, 9 sites, Aug 2024 to Oct 2026.
- Per row: `location_id`, `timestamp`, `image_file`, `class`, `box`, `lighting`,
  `confidence`, `classification_confidence`, `temperature_celsius`, `manual_label`.
- Main species: wild boar, roe deer, pigeon, fox, badger, crow, raccoon, marten, hare.
  Humans and vehicles are also detected.
- Existing: email download, organiser, MegaDetector + DeepFaune labelling,
  fine-tuning, review app, gallery, activity/baiting/moon/temperature charts,
  activity forecast worktree with Open-Meteo weather.

## Foundations

These make every later result credible. Do them first.

1. **Independent visits instead of image counts.** Merge bursts into visits per
   site and species (common rule: 30 min independence). Image counts overweight
   animals that linger at bait sites. Visits also give dwell time.
2. **Camera uptime from the data.** Infer active periods from image gaps, empty
   triggers, and OCR failures. Alert after N silent days. This produces the
   coverage CSV the forecast needs and the effort that abundance indices need.
3. **All detections per image.** The CSV stores one `box` per row. If only the
   top detection is kept, store all boxes. Group size depends on this.

## For Hunters

| Idea | Based on | Effort |
|---|---|---|
| Live push alert, e.g. "wild boar at Wendeplatte now", via ntfy or Telegram. Mail polling already runs every minute. | existing pipeline | S |
| Best hide-seat window per site: likely activity hours overlaid with Baden-Württemberg open seasons, legal shooting hours, and sunset. | activity data, astral, rules table | M |
| Wind-aware site choice: recommend sites where forecast wind blows from the expected animals toward the seat. Needs one bearing per site. | forecast weather module | S-M |
| Sounder size and piglets: count individuals and estimate piglets by box size and season. Flag sows with piglets, which are protected while they lead young. | all boxes, fixed camera geometry | M |
| Bait-site management: time since last human visit against falling boar activity, "likely empty" hint, suggested refill interval. | extends baiting charts | S |
| Roe deer buck / doe / fawn classes and antler cycle timing (casting, velvet). | `finetune_classifier.py` + new labels | L |
| Crop-damage early warning: boar trend at field-edge sites in maize and grain season. | site trends | S |
| Harvest log: record kills and compare with activity and the roe deer cull plan. | new input | M |

## For Conservationists

| Idea | Based on | Effort |
|---|---|---|
| Camtrap DP export (TDWG standard) for GBIF, Wildlife Insights, or Agouti. `DATA-LICENCE.md` already exists. | CSV, coordinates, uptime | M |
| Relative abundance index: independent visits per 100 camera-days by site, season, and year. Trends without density claims. | visits + uptime | S |
| Occupancy with detection probability: separate "absent" from "not detected" across the 9 sites. | detection histories | M |
| Temporal overlap between species (kernel overlap coefficient Δ): fox vs hare, raccoon vs others, boar vs roe deer. Also time to next species at the same site. | timestamps | S-M |
| Human disturbance: do animals shift to night after people, dogs, or vehicles? Compare by site and weekday. | human/vehicle classes | S-M |
| Invasive raccoon: trend, spread across sites, alert on first sighting at a new site. | existing class | S |
| Rare-species alert: high-confidence lynx, wolf, or wildcat goes to review first, then an exportable report for FVA wildlife monitoring. | DeepFaune classes | S |
| Movement between sites: same species at site A then site B within minutes suggests a corridor. Show on a map. | site coordinates | M |
| Species richness and Shannon diversity per site and season. | classes | S |

## Usability

- Map page with site markers, activity heat, and a time slider.
- Weekly digest: per-site summary, unusual events, best photos.
- Privacy: blur people before publishing or exporting (GDPR). Check whether
  `publish_images.py` already does this.
- Camera health: detect moved cameras (scene change), false-trigger rate, and
  clock drift against sunrise and the `lighting` column.

## Data Limits

- Images are not individuals. Without visits and uptime, counts are not abundance.
- 9 cameras in one area describe these sites, not the population.
- Bait sites bias activity. Keep them separate in conservation analyses.
- OCR temperature is the camera sensor and reads high in sun. Prefer Open-Meteo
  for weather analyses.

## Suggested Order

1. Independent visits.
2. Camera uptime.
3. Live alerts.
4. Hide-seat window with wind.
5. Camtrap DP export.
