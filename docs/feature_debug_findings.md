# Feature Debug Findings

This file records runtime values observed from `agent_ppo/debug` platform runs.
Use these values as the current feature-engineering anchors for rebuilding
`agent_diy` observation processing. They are empirical, not official protocol
constants.

## 1. Actor Type Mapping

The current hok1v1 runtime mapping observed in debug logs is:

| Entity | actor_type | sub_type |
| --- | ---: | ---: |
| Hero | 0 | 0 |
| Monster | 1 | 0 |
| Soldier | 1 | 11 |
| Tower | 2 | 21 |
| Crystal | 2 | 23 |
| Spring / Spawn | 2 | 24 |

`DumpCollector` should classify entities by `(actor_type, sub_type)` first.
Do not infer tower/monster from `actor_type` alone.

## 2. Coordinate Remapping

The map is not well represented as an axis-aligned square. Runtime positions
show a long strip near the diagonal. The recommended remapping is therefore:

```text
lane_projected  = (x + z) / sqrt(2)
width_projected = (x - z) / sqrt(2)

lane_norm  = lane_projected  / lane_scale
width_norm = width_projected / width_scale
```

Coordinate policy is now frozen for the first feature rebuild:

- Long axis: `x + z`, with `sqrt(2)` projection in feature code.
- Width axis: `x - z`, with `sqrt(2)` projection in feature code.
- Global map length is treated as approximately `45000 * 2`.
- Most lane/tower/soldier calculations use width half-range around `14000`.
- The central health-pack zone is wider and should use a separate center-region
  width half-range around `20000`.
- Raw `x`/`z` values can contain sentinel or outlier coordinates. Clip or mask
  out samples with `abs(x) > 60000` or `abs(z) > 60000` before projection.

Recommended first-pass constants:

| Constant | Initial value | Reason |
| --- | ---: | --- |
| `LANE_HALF_RANGE` | `45000` | Approximate half-length from center to base along projected lane axis. |
| `NORMAL_WIDTH_HALF_RANGE` | `14000` | Main lane / tower / soldier area. |
| `CENTER_LANE_HALF_RANGE` | `15000` | Central core between the two health packs. |
| `CENTER_WIDTH_HALF_RANGE` | `20000` | Wider center zone around the two health packs. |
| `COORD_CLIP_ABS` | `60000` | Filters sentinel/outlier actor coordinates before projection. |

If code temporarily uses raw `x + z` and `x - z` without the `sqrt(2)`
projection, multiply the scale constants by `sqrt(2)` consistently. Feature
implementation should clip normalized values to `[-1, 1]` after projection.
Keep debug counters for clipped samples, but do not block feature rebuild on
additional coordinate observation.

Suggested normalization:

```python
lane = (x + z) / 1.41421356237
width = (x - z) / 1.41421356237
width_scale = CENTER_WIDTH_HALF_RANGE if abs(lane) <= CENTER_LANE_HALF_RANGE else NORMAL_WIDTH_HALF_RANGE
lane_norm = clip(lane / LANE_HALF_RANGE, -1.0, 1.0)
width_norm = clip(width / width_scale, -1.0, 1.0)
```

## 3. Confirmed Map Anchors

Approximate anchor positions observed so far:

| Anchor | Approximate coordinate | Notes |
| --- | --- | --- |
| Spring / spawn | around `(+40000, +40000)` and `(-40000, -40000)` | Use as base-side anchor. Some raw logs may contain larger sentinel values; use valid filtered coordinates. |
| Outer tower | around `(+13000, +13000)` and `(-13000, -13000)` | `actor_type=2, sub_type=21`; two camps are observed. |
| Cake / health pack | `[15340, 15100]` and `[-15220, -15120]` | Observed from `cakes[].collider.location`; first appearance around frame `1778` in debug logs. |

These anchors support using `x + z` as progress along the lane:

- Friendly-to-enemy progress should be computed after red/blue viewpoint
  mirroring.
- Tower distance and forward reward should use projected lane distance, not
  raw Euclidean `x/z` alone.
- Width features should use `x - z` to represent lateral deviation from lane.

## 4. Confirmed Tower Fields

Debug logs now show tower entities:

```text
actor=2|sub=21 -> tower
tower_per_camp_count=2
behave_tower includes 1 and 5
attack_range observed stable around 8800
```

For feature engineering, collect at least:

| Field | Use |
| --- | --- |
| `camp` | Identify self/enemy tower after viewpoint mapping. |
| `config_id` / `runtime_id` | Stable entity identity and attack target lookup. |
| `location` | Tower anchor and tower-distance feature. |
| `hp` / `max_hp` | Win-condition progress and tower reward. |
| `attack_range` | Tower danger zone. |
| `sight_area` | Visibility/danger context if stable. |
| `attack_target` | Detect tower aggro. |
| `behav_mode` | Tower state diagnostics. |
| `camp_visible` | Mask hidden/unobservable tower state if needed. |

Tower coordinates should be taken from `tower_per_camp` or
`coordinate_analysis.valid_by_kind.tower`, not from mixed global map bounds.

For the first feature rebuild, tower attack range can be treated as confirmed
at `8800` in runtime units. Keep reading `attack_range` dynamically from the
observation, but normalize tower-danger features against `8800` unless a later
complete `summary.json` contradicts this.

## 5. Confirmed Monster Fields

Observed monster behavior IDs so far:

```text
behave_monster = [0, 23, 26]
```

The semantic names of these behavior IDs are not required for first-pass feature
engineering. Encode behavior as an ID/bucket or compact one-hot, and keep
`attack_range`, `sight_area`, `hp/max_hp`, `camp`, `config_id`, and projected
position using the frozen coordinate projection.

## 6. Remaining Debug Gaps

Coordinate values are considered sufficiently determined for the first feature
rebuild. Tower and health-pack anchors are fixed above. Monster, bullet,
soldier, hero, and other entity coordinates should all go through the same
projection method and clipping fallback.

The largest remaining feature-debug gap is hero buff/mark interpretation:

- Buffs and marks are currently visible as IDs, but need grouping by
  `hero_config_id`, `camp`, `origin_actorId`, first/last frame, and max layer.
- This is required to distinguish Luban No.7 passive state, Di Renjie passive
  stacks, summoner effects, and generic combat buffs.
- The official protocol path is `buff_state.buff_marks[]`, with fields
  `configId`, `layer`, and `origin_actorId`.
- `DumpCollector` now also logs raw mark samples and field-key patterns through
  `[FREEZE] buff_mark_raw_samples=...`, so platform logs can prove whether
  marks are absent or only using a field-name variant.

Other still-useful additions:

- `hero_by_config_camp` summary, because aggregating only by `config_id` mixes
  blue/red same-hero cases.
- Per-frame target candidate mapping for the 9 target slots:
  `None, enemy hero, self, soldier1-4, tower, monster`.
- Final exact projection constants from raw `summary.json`, not screenshots.

## 7. No-Summary Fallback Policy

Platform access may only expose visible logs, not the full `summary.json`.
In that case, freeze feature engineering from the compact `[FREEZE]` log lines
and use robust encodings instead of depending on a perfectly exhaustive ID list.

Use this final policy for buff/mark features:

- Keep explicit known mark layers:
  - `11200: 5`
  - `13300: 5`
  - `13310: 1`
- Keep the common/112/133 buff candidate families from `agent_diy.conf.conf`.
- Add an unknown/hash fallback for any unseen buff or mark ID. Do not drop an
  unseen ID to all-zero; encode its type, holder relation, raw-ID bucket/hash,
  layer bucket, and origin relation.
- Use `buff_detail_by_holder` log lines only to refine the explicit whitelist.
  The whitelist is an optimization for common IDs, not a hard protocol boundary.

Skill cooldowns are bucketed. Exact public cooldown values are only used as
scale references. Runtime fields `cooldown`, `cooldown_max`, and `usable` remain
authoritative when present.

## 8. Hero Mechanism Notes

The final feature engineering must model 112 and 133 as different heroes, but
it should not hand-code a tactical rule for each skill. Encode the mechanics
through observed state fields: hero ID, skill slot state, cooldown, usable,
buff/mark presence, mark layer, attack target, projectile source, hit/take-hurt
events, and tower danger.

Luban No.7 (`112`) important mechanics:

- Passive is the main DPS driver. After repeated normal attacks and after using
  skills, Luban enters an enhanced/sweeping attack state. This should be learned
  from passive mark layer, buff presence, skill-used recency, attack target, and
  projectile features.
- Skill 1 is an area projectile/control/damage skill that also helps trigger
  passive pressure.
- Skill 2 is a long-range projectile with defensive/knockback value nearby and
  map pressure value at range.
- Skill 3 creates a persistent area pressure zone. Treat it mostly through
  cooldown, projectile/bullet source, hit events, and enemy distance to zone
  sources if observable.

Di Renjie (`133`) important mechanics:

- Passive stacks from normal attacks and affects sustained combat tempo. Encode
  mark layer, attack speed, movement speed, and recent attack/hit state.
- Skill 1 is token/projectile damage and control pressure.
- Skill 2 is defensive cleanse/self-protection. Its availability is strategically
  important, so `usable` and cooldown buckets matter.
- Skill 3 is a high-impact control/debuff projectile. Encode cooldown/usable,
  hit event, enemy hurt/debuff buff IDs, and target legality.

Summoner skills should be encoded by selected skill ID plus cooldown bucket.
Known local IDs:

| ID | Name |
| ---: | --- |
| 80102 | Heal |
| 80109 | Sprint |
| 80104 | Smite |
| 80108 | Execute |
| 80110 | Frenzy |
| 80105 | Interference |
| 80103 | Stun |
| 80107 | Cleanse |
| 80121 | Weaken |
| 80115 | Flash |
