# Video Frame Interpolation

Interpolate converts CFR video to a higher frame rate while protecting temporal boundaries that should not be synthesized.

## Language

**Content preset**:
A named processing strategy selected according to the source material.
_Avoid_: Encoder preset, style

**Movie preset**:
The compatibility content preset that interpolates each adjacent source-frame pair without cadence analysis.

**Anime preset**:
A content preset that smooths confidently detected short held-drawing runs while preserving long or ambiguous holds.

**Modified content preset**:
A content preset whose user-overridable settings differ from its resolved defaults.

**Held drawing**:
A visually repeated animation frame intentionally occupying two or more source-frame intervals.
_Avoid_: Dropped frame, skipped frame

**Confident duplicate**:
A held drawing classified only when global, changed-pixel, and localized tile differences are all below conservative limits.

**Cadence protection**:
The policy that identifies held drawings and limits which runs may be smoothed.
_Avoid_: Deduplication, because output duration and timing are preserved

**Scene protection**:
The policy that prevents synthesis across a detected shot boundary.

## Relationships

- A **Content preset** is either the **Movie preset** or the **Anime preset**.
- The **Anime preset** always applies **Scene protection**.
- The **Anime preset** applies **Cadence protection** to confidently detected held-drawing runs of two or three frames.
- **Cadence protection** preserves long or ambiguous holds unchanged.
- **Scene protection** takes precedence over **Cadence protection**.
- The **Anime preset** enables half-scale UHD flow by default for 4K sources, but the user may override it.
- Overriding a preset setting produces a **Modified content preset**.
- Selecting a **Content preset** resets its settings to deterministic defaults and clears prior overrides.

## Example dialogue

> **Dev:** “Should the **Anime preset** smooth a six-frame static hold before the next drawing?”
> **Domain expert:** “No. **Cadence protection** only smooths confident two- or three-frame held-drawing runs; long or ambiguous holds remain unchanged.”

## Flagged ambiguities

- “Skipped frame” described both missing output and intentional repeated anime drawings — resolved: use **Held drawing** for intentional repeated source frames.
- “Preset” could mean encoder settings or content-aware interpolation behavior — resolved: use **Content preset** for Movie and Anime strategies.
