# 🏷️ Labelling guide — 61 plans ready

## What you got
- `floorplancad/` — 30× 1000×1000 CAD blocks. Clean symbols, no dims. Train eyes for Wall/Door/Window.
- `houses/` — 31× whole-house plans, up to 1080×1538. These carry room names + dims. **Label these first.**
- ⚠️ Some `houses/` are tiny (~207×295). Skip those for dim-text; use for room boxes only.

## Label order (fastest win first)
1. **10 best houses** → room boxes + dim strings linked (`Bedroom 12 x 14`). This is your golden demo set.
2. **Next 20 houses** → same. Now you have 30.
3. **FloorPlanCAD 30** → Wall/Door/Window boxes only (dims don't exist there). Pre-train material.

## The 4 labels (nothing else)
`Room` (box + text incl. dims) / `Wall` / `Door` / `Window` (+ openings count per room for headers).

Export Label Studio JSON → `python data_convert.py --labelstudio custom.json --imgdir ./blueprints/houses --out ./data/custom.jsonl`
