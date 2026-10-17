"""calc.quote: money-rule enforcement, floor reconciliation, member handling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.calc import BF_DIVISOR, quote  # noqa: E402

PRICING = {
    "currency": "USD", "waste_pct": 10, "margin_pct": 20, "delivery": 85.0,
    "tax_rate": 0.0825, "min_charge": 250.0,
    "catalog": {"2x4": 0.62, "2x6": 0.95, "2x10": 1.55, "1x6": 0.78, "4x4": 1.40},
    "species": {"SPF": 1.0, "DF": 1.25, "SYP": 1.35, "PT": 1.90},
    "grades": {"#2": 1.0, "#1": 1.18, "select": 1.35},
}
JOIST = {"name": "Rim joist", "spec": "2x6", "length_in": 192, "qty": 4,
         "species": "SPF", "grade": "#2"}
MONEY = ("material", "waste_cost", "labor", "subtotal", "tax", "total")


def priced(members, pricing=None, intake=None):
    return quote({"blueprint": {"members": members}, "intake": intake or {}},
                 dict(PRICING if pricing is None else pricing))


class Reconcile(unittest.TestCase):
    """Every printed field must add up, floored or not."""

    def check(self, q):
        self.assertEqual(round(q["subtotal"] + q["tax"], 2), q["total"],
                         f"subtotal+tax != total: {q}")
        self.assertEqual(round(q["material"] + q["waste_cost"] + q["labor"]
                               + q["delivery"], 2), q["subtotal"],
                         f"components != subtotal: {q}")
        self.assertEqual(round(sum(i["ext"] for i in q["line_items"]), 2),
                         q["material"], f"line items != material: {q}")

    def test_table(self):
        cases = [
            ([JOIST], PRICING),                                    # floored
            ([dict(JOIST, qty=40)], PRICING),                      # well past the floor
            ([JOIST], {**PRICING, "min_charge": 0}),               # no floor
            ([JOIST], {**PRICING, "tax_rate": 0.0}),               # zero tax
            ([], PRICING),                                         # empty
            ([], {**PRICING, "min_charge": 500.0, "tax_rate": 0.0}),  # floor, no tax
            ([JOIST], {**PRICING, "waste_pct": 0, "margin_pct": 0,
                       "delivery": 0, "tax_rate": 0, "min_charge": 0}),
            ([JOIST], {**PRICING, "tax_rate": 0.2, "min_charge": 1000.0}),
        ]
        for members, pricing in cases:
            self.check(quote({"blueprint": {"members": members}}, pricing))

    def test_floor_is_post_tax(self):
        q = priced([JOIST])
        self.assertTrue(q["min_charge_applied"])
        self.assertEqual(q["total"], 250.0)
        # grossed up through tax, not the raw gap
        self.assertGreater(q["labor"], 13.38)
        self.assertLess(q["labor"], 13.38 + 71.11)
        self.check(q)

    def test_no_floor_when_over(self):
        q = priced([dict(JOIST, qty=40)])
        self.assertFalse(q["min_charge_applied"])
        self.assertGreater(q["total"], 250.0)
        self.check(q)


class MoneyRule(unittest.TestCase):
    """intake is LLM output: it negotiates specs, it never sets business constants."""

    def test_intake_cannot_move_money(self):
        base = priced([JOIST])
        hijack = priced([JOIST], intake={"tax_rate": 0.9, "margin_pct": 500,
                                         "waste_pct": 99, "delivery": 9999,
                                         "min_charge": 99999, "currency": "EUR",
                                         "catalog": {"2x6": 99.0},
                                         "species": {"SPF": 99.0}, "grades": {"#2": 99.0}})
        for key in MONEY:
            self.assertEqual(base[key], hijack[key], f"intake.{key} moved the money")
        self.assertEqual(base["currency"], hijack["currency"])

    def test_ignored_keys_are_reported(self):
        q = priced([JOIST], intake={"tax_rate": 0.9, "margin_pct": 500,
                                    "waste_pct": 99, "delivery": 9999})
        for key, val in (("tax_rate", 0.9), ("margin_pct", 500),
                         ("waste_pct", 99), ("delivery", 9999)):
            self.assertTrue(any(f"intake.{key}={val!r} ignored" in w for w in q["warnings"]),
                            f"no warning for intake.{key}: {q['warnings']}")

    def test_clean_intake_is_silent(self):
        q = priced([JOIST], intake={"customer": "Ada", "overrides": []})
        self.assertEqual([w for w in q["warnings"] if "ignored" in w], [])

    def test_intake_can_override_qty(self):
        base = priced([JOIST])
        over = priced([JOIST], intake={"overrides": [{"spec": "2x6", "qty": 8}]})
        self.assertEqual(over["line_items"][0]["qty"], 8)
        self.assertEqual(round(over["board_feet"], 1), round(base["board_feet"] * 2, 1))

    def test_intake_can_override_price_per_bf(self):
        q = priced([JOIST], intake={"overrides": [{"spec": "2x6", "price_per_bf": 1.25}]})
        self.assertEqual(q["line_items"][0]["unit_price"], 1.25)
        self.assertEqual(q["line_items"][0]["ext"], round(64.0 * 1.25, 2))

    def test_override_scoped_to_matching_spec(self):
        q = priced([JOIST, {"name": "Stud", "spec": "2x4", "length_in": 96,
                            "qty": 10, "species": "SPF", "grade": "#2"}],
                   intake={"overrides": [{"spec": "2x4", "price_per_bf": 2.0}]})
        by = {i["name"]: i for i in q["line_items"]}
        self.assertEqual(by["Stud"]["unit_price"], 2.0)
        self.assertEqual(by["Rim joist"]["unit_price"], 0.95)


class Golden(unittest.TestCase):
    def test_handoff_case(self):
        q = priced([JOIST])
        self.assertEqual(q["board_feet"], 64.0)
        self.assertEqual(q["material"], 60.80)
        self.assertEqual(q["waste_cost"], 6.08)
        self.assertEqual(q["delivery"], 85.0)
        self.assertEqual(q["waste_pct"], 10)
        self.assertEqual(q["margin_pct"], 20)
        self.assertEqual(q["currency"], "USD")
        self.assertEqual(len(q["line_items"]), 1)
        self.assertEqual(q["line_items"][0]["board_feet"], 64.0)

    def test_board_feet_formula(self):
        q = priced([{"name": "m", "spec": "2x10", "length_in": 96, "qty": 7,
                     "species": "SPF", "grade": "#2"}])
        self.assertAlmostEqual(q["board_feet"], 2 * 10 * 96 / 144 * 7, places=3)
        self.assertEqual(q["line_items"][0]["unit_price"], 1.55)

    def test_species_and_grade_multipliers(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": 192, "qty": 1,
                     "species": "DF", "grade": "#1"}])
        self.assertEqual(q["line_items"][0]["unit_price"], round(0.95 * 1.25 * 1.18, 4))

    def test_missing_species_defaults_silently(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": 192, "qty": 1}])
        self.assertEqual(q["line_items"][0]["unit_price"], 0.95)
        self.assertEqual([w for w in q["warnings"] if "multiplier" in w], [])


class Members(unittest.TestCase):
    def test_unknown_spec_excluded(self):
        q = priced([{"name": "m", "spec": "9x99", "length_in": 96, "qty": 1}])
        self.assertEqual(q["line_items"], [])
        self.assertEqual(q["material"], 0.0)
        self.assertTrue(any("unknown spec '9x99'" in w for w in q["warnings"]))

    def test_unknown_species_warns_multiplier_one(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": 192, "qty": 1,
                     "species": "Mysterywood"}])
        self.assertEqual(q["line_items"][0]["unit_price"], 0.95)
        self.assertTrue(any("unknown species 'Mysterywood'" in w for w in q["warnings"]))

    def test_unknown_grade_warns_multiplier_one(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": 192, "qty": 1,
                     "grade": "blue"}])
        self.assertTrue(any("unknown grade 'blue'" in w for w in q["warnings"]))

    def test_non_numeric_length_skipped(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": "twelve", "qty": 1}])
        self.assertEqual(q["line_items"], [])
        self.assertTrue(any("non-numeric length" in w for w in q["warnings"]))

    def test_unparseable_spec_skipped(self):
        q = priced([{"name": "m", "spec": "two-by-six", "length_in": 96, "qty": 1}])
        self.assertEqual(q["line_items"], [])
        self.assertTrue(any("unparseable spec" in w for w in q["warnings"]))

    def test_non_dict_member_skipped(self):
        q = priced([JOIST, "junk", None])
        self.assertEqual(len(q["line_items"]), 1)
        self.assertTrue(any("not an object" in w for w in q["warnings"]))

    def test_no_members_warns(self):
        for ctx in ({}, {"blueprint": {}}, {"blueprint": {"members": []}}):
            q = quote(ctx, PRICING)
            self.assertEqual(q["line_items"], [])
            self.assertEqual(q["total"], 250.0)     # floor on an empty quote
            self.assertTrue(any("no members" in w for w in q["warnings"]))


class Coercion(unittest.TestCase):
    def test_numeric_strings(self):
        q = priced([{"name": "m", "spec": " 2 X 10 ", "length_in": "192", "qty": "3",
                     "species": " spf ", "grade": " #2 "}])
        self.assertEqual(q["line_items"][0]["spec"], "2x10")
        self.assertEqual(q["line_items"][0]["qty"], 3)
        self.assertAlmostEqual(q["board_feet"], 2 * 10 * 192 / 144 * 3, places=6)

    def test_bool_is_not_a_number(self):
        q = priced([{"name": "m", "spec": "2x6", "length_in": True, "qty": 1}])
        self.assertEqual(q["line_items"], [])

    def test_qty_non_positive_defaults_to_one(self):
        for raw in (0, -5, None, "many"):
            q = priced([{"name": "m", "spec": "2x6", "length_in": 192, "qty": raw}])
            self.assertEqual(q["line_items"][0]["qty"], 1, f"qty={raw!r}")

    def test_spec_canonicalisation(self):
        q = priced([{"name": "m", "spec": " 2.0 x 6.0 ", "length_in": 192, "qty": 1}])
        self.assertEqual(q["line_items"][0]["spec"], "2x6")

    def test_bf_divisor(self):
        self.assertEqual(BF_DIVISOR, 144.0)


class Producer(unittest.TestCase):
    def test_lumber_pieces_preferred(self):
        ctx = {"lumber": {"pieces": [{"name": "Stud", "spec": "2x4", "length_in": 96,
                                      "qty": 5, "species": "SPF", "grade": "#2"}]},
               "blueprint": {"members": [JOIST]}}
        q = quote(ctx, PRICING)
        self.assertEqual([i["name"] for i in q["line_items"]], ["Stud"])

    def test_blueprint_members_fallback(self):
        q = quote({"blueprint": {"members": [JOIST]}}, PRICING)
        self.assertEqual([i["name"] for i in q["line_items"]], ["Rim joist"])

    def test_lumber_present_but_empty_is_authoritative(self):
        q = quote({"lumber": {"pieces": []}, "blueprint": {"members": [JOIST]}}, PRICING)
        self.assertEqual(q["line_items"], [])

    def test_non_dict_intake_tolerated(self):
        q = quote({"blueprint": {"members": [JOIST]}, "intake": "not a dict"}, PRICING)
        self.assertEqual(len(q["line_items"]), 1)


class Dead(unittest.TestCase):
    def test_price_catalog_gone(self):
        from harness import calc
        self.assertFalse(hasattr(calc, "price_catalog"))
        self.assertFalse(hasattr(calc, "_default_mult"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
