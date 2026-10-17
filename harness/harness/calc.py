"""Deterministic lumber takeoff: blueprint + intake + pricing -> quote dict. Pure, no I/O."""

from __future__ import annotations

import re

BF_DIVISOR = 144.0
SPEC_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)\s*$")


def _num(value, default: float | None = None) -> float | None:
    """Coerce int/float/numeric-string to float; bools and junk -> `default`."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _qty(value, default: int = 1) -> int:
    """Piece count; missing, non-numeric, zero or negative -> `default`."""
    n = _num(value)
    return default if n is None or n <= 0 else int(n)


def _spec(raw) -> str:
    """Canonical "TxW" from a nominal spec (" 2 X 10 " -> "2x10"); "" when unparseable."""
    m = SPEC_RE.match(str(raw if raw is not None else ""))
    if not m:
        return ""

    def side(text: str) -> str:
        v = float(text)
        return str(int(v)) if v.is_integer() else str(v)

    return f"{side(m[1])}x{side(m[2])}"


def _key(raw) -> str:
    """Case-insensitive lookup key for species/grade names."""
    return str(raw if raw is not None else "").strip().lower()


def _table(raw) -> dict:
    """Normalize a {name: number} pricing table; drop unparseable rows."""
    out: dict[str, float] = {}
    for k, v in (raw or {}).items():
        n = _num(v)
        if n is not None:
            out[_key(k)] = n
    return out


def _mult(table: dict, name: str, warnings: list[str], label: str, owner: str) -> float:
    """Multiplier for `name`, defaulting to 1.0 (silently when unset, loudly when unknown)."""
    if not name:
        return 1.0
    n = table.get(name)
    if n is None:
        warnings.append(f"{owner}: unknown {label} '{name}' - multiplier 1.0")
        return 1.0
    return n


def _match_override(overrides: list[dict], spec: str, member: dict) -> dict | None:
    """Last override matching the member's spec, and its grade/species when the override sets them."""
    hit = None
    for ov in overrides:
        if _spec(ov.get("spec")) != spec:
            continue
        for field in ("grade", "species"):
            want = ov.get(field)
            if want is not None and _key(want) != _key(member.get(field)):
                break
        else:
            hit = ov
    return hit


def quote(context: dict, pricing: dict) -> dict:
    """Price a blueprint against a pricing table, applying intake overrides. Deterministic."""
    context = context or {}
    pricing = pricing or {}
    intake = context.get("intake") or {}
    blueprint = context.get("blueprint") or {}
    warnings: list[str] = []

    catalog: dict[str, float] = {}
    for k, v in (pricing.get("catalog") or {}).items():
        spec, price = _spec(k), _num(v)
        if spec and price is not None:
            catalog[spec] = price
    species_t = _table(pricing.get("species"))
    grades_t = _table(pricing.get("grades"))
    overrides = [o for o in (intake.get("overrides") or []) if isinstance(o, dict)]

    currency = str(pricing.get("currency") or "USD")
    waste_pct = _num(intake.get("waste_pct"), _num(pricing.get("waste_pct"), 0.0))
    margin_pct = _num(intake.get("margin_pct"), _num(pricing.get("margin_pct"), 0.0))
    delivery = _num(intake.get("delivery"), _num(pricing.get("delivery"), 0.0))
    tax_rate = _num(intake.get("tax_rate"), _num(pricing.get("tax_rate"), 0.0))
    min_charge = _num(pricing.get("min_charge"), 0.0)

    members = blueprint.get("members")
    if not isinstance(members, list) or not members:
        warnings.append("no members in blueprint - quote has no line items")
        members = []

    items: list[dict] = []
    total_bf = 0.0
    for raw in members:
        if not isinstance(raw, dict):
            warnings.append(f"member {raw!r}: not an object - member skipped")
            continue
        name = str(raw.get("name") or "member")
        spec = _spec(raw.get("spec"))
        length = _num(raw.get("length_in"))
        if length is None:
            warnings.append(f"{name}: non-numeric length {raw.get('length_in')!r} - member skipped")
            continue
        if not spec:
            warnings.append(f"{name}: unparseable spec {raw.get('spec')!r} - member skipped")
            continue

        ov = _match_override(overrides, spec, raw)
        qty_src = raw.get("qty")
        if ov is not None and ov.get("qty") is not None:
            qty_src = ov["qty"]
        qty = _qty(qty_src)

        over_bf = _num(ov.get("price_per_bf")) if ov is not None else None
        if over_bf is not None:
            unit = round(over_bf, 4)
        else:
            base = catalog.get(spec)
            if base is None:
                warnings.append(f"{name}: unknown spec '{spec}' - member excluded")
                continue
            thick, wide = (float(p) for p in spec.split("x"))
            unit = round(base * _mult(species_t, _key(raw.get("species")), warnings, "species", name)
                         * _mult(grades_t, _key(raw.get("grade")), warnings, "grade", name), 4)

        thick, wide = (float(p) for p in spec.split("x"))
        bf = thick * wide * length / BF_DIVISOR * qty
        items.append({"name": name, "spec": spec, "length_in": length, "qty": qty,
                      "grade": raw.get("grade"), "species": raw.get("species"),
                      "board_feet": round(bf, 3), "unit_price": unit,
                      "ext": round(bf * unit, 2)})
        total_bf += bf

    material = round(sum(i["ext"] for i in items), 2)
    waste_cost = round(material * waste_pct / 100.0, 2)
    labor = round((material + waste_cost) * margin_pct / 100.0, 2)
    subtotal = round(material + waste_cost + labor + delivery, 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    min_applied = total < min_charge
    if min_applied:
        gap = round(min_charge - total, 2)
        labor = round(labor + gap, 2)
        total = round(min_charge, 2)
        warnings.append(f"min charge {currency} {min_charge:.2f} applied - labor raised by {gap:.2f}")

    return {"line_items": items, "board_feet": round(total_bf, 3), "material": material,
            "waste_pct": waste_pct, "waste_cost": waste_cost, "margin_pct": margin_pct,
            "labor": labor, "delivery": delivery, "tax_rate": tax_rate, "tax": tax,
            "subtotal": subtotal, "total": total, "min_charge_applied": min_applied,
            "currency": currency, "warnings": warnings}


def _default_mult(raw) -> float:
    """Baseline multiplier: the table entry equal to 1.0, else 1.0."""
    for v in (raw or {}).values():
        n = _num(v)
        if n is not None and round(n, 6) == 1.0:
            return n
    return 1.0


def price_catalog(pricing: dict) -> dict:
    """Effective $/BF per spec (catalog x baseline species x baseline grade) for `harness price`."""
    pricing = pricing or {}
    smult, gmult = _default_mult(pricing.get("species")), _default_mult(pricing.get("grades"))
    out: dict[str, float] = {}
    for k, v in (pricing.get("catalog") or {}).items():
        spec, price = _spec(k), _num(v)
        if spec and price is not None:
            out[spec] = round(price * smult * gmult, 4)
    return out
