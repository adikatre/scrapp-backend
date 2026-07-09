"""Tests for waste route mappings."""

import sys
from pathlib import Path

import pytest

FINAL_DIR = Path(__file__).resolve().parents[1] / "final"
sys.path.insert(0, str(FINAL_DIR))

from categories import (  # noqa: E402
    NON_WASTE_ROUTES,
    ROUTE_E_WASTE,
    ROUTE_LIVING_THINGS,
    ROUTE_RECYCLE,
    VALID_ROUTES,
    build_coco_to_bin,
    default_route,
    e_waste,
    is_non_waste_route,
    normalize_route,
    recycle,
)


def test_clock_maps_to_e_waste_not_recycle():
    mapping = build_coco_to_bin({0: "clock"})
    assert mapping["clock"] == ROUTE_E_WASTE
    assert "clock" not in recycle
    assert "clock" in e_waste


def test_no_class_appears_in_multiple_category_sets():
    category_sets = [
        recycle,
        e_waste,
    ]
    seen = {}
    for category_set in category_sets:
        for item in category_set:
            assert item not in seen, f"{item} appears in multiple category sets"
            seen[item] = True


def test_normalize_route_accepts_valid_routes():
    for route in VALID_ROUTES:
        assert normalize_route(route) == route


def test_normalize_route_falls_back_for_unknown_route():
    assert normalize_route("Not A Real Route") == default_route


def test_is_non_waste_route_identifies_living_things_and_city_infra():
    assert is_non_waste_route(ROUTE_LIVING_THINGS) is True
    assert is_non_waste_route(ROUTE_RECYCLE) is False
    assert ROUTE_LIVING_THINGS in NON_WASTE_ROUTES
