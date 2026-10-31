# Web UI Follow-up Design

## Purpose

Complete the requested usability updates across the existing wildl-id web
application, building on the partial work already present in the worktree.

## Scope and behavior

- Keep the home page's recent-sightings content, but remove its side navigation.
  Center the brand logo and use this brief summary: "Explore wildlife
  sightings, review detections, browse camera images, and view activity across
  your cameras."
- Keep descriptive hover summaries on side-navigation links and use **Review**
  as the navigation label for `/review`.
- Apply one bounding-box color rule consistently across all project renderers:
  yellow for predicted and unchecked, green for checked and correct, and red
  for checked and incorrect.
- Toggling boxes in the gallery carousel must not resize its image frame or add
  blank space.
- Keep only the first two review modes and provide the existing empty-image
  removal script as an action in the review UI.
- Limit the number of visible gallery page links for large result sets while
  retaining access to all pages.
- Generate visualisations each time the visualisations page is visited; do not
  show a button for triggering the standard visualisation run. Keep unrelated
  filtered-plot actions unchanged.

## Implementation approach

Preserve the current NiceGUI routes and data flow. Reuse the shared bounding-box
color helper wherever possible, and fix carousel sizing at its rendering
boundary so overlay updates do not alter the image frame. Keep the existing
script and visualisation run-store APIs. Update only directly related tests and
user-facing documentation. Preserve unrelated worktree changes.

## Acceptance and verification

- Home has no side drawer, displays the centered logo and summary, and retains
  recent sightings.
- Side-navigation hover summaries remain available, with **Review** as the
  review link's label.
- Tests cover all three bounding-box states in every renderer, and confirm
  toggling an overlay leaves the carousel image frame unchanged.
- Review exposes exactly the top two modes and the empty-image removal action.
- Gallery pagination remains usable with many pages without displaying every
  page number.
- Each visualisations-page visit starts one standard run and exposes no
  standard-generation button; filtered plot controls remain available.
- Run the focused tests and relevant project test suite before committing the
  implementation.
