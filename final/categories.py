# categories.py

# Disposal route constants
ROUTE_RECYCLE = "Recycle"
ROUTE_COMPOST = "Compost"
ROUTE_E_WASTE = "E-Waste"
ROUTE_BULKY_ITEMS = "Bulky Items (Donate)"
ROUTE_HAZARDOUS = "Hazardous Waste"
ROUTE_SINGLE_USE = "Single-Use Items"
ROUTE_CITY_INFRA = "City Infrastructure"
ROUTE_LIVING_THINGS = "Living Things"
ROUTE_GENERAL_TRASH = "General Trash"

VALID_ROUTES = frozenset({
    ROUTE_RECYCLE,
    ROUTE_COMPOST,
    ROUTE_E_WASTE,
    ROUTE_BULKY_ITEMS,
    ROUTE_HAZARDOUS,
    ROUTE_SINGLE_USE,
    ROUTE_CITY_INFRA,
    ROUTE_LIVING_THINGS,
    ROUTE_GENERAL_TRASH,
})

# Routes that are not household waste and should be ignored for classification.
NON_WASTE_ROUTES = frozenset({
    ROUTE_LIVING_THINGS,
    ROUTE_CITY_INFRA,
})

# Fallback route for anything uncategorized
default_route = "Landfill / Donate / Check rules"

# City of San Diego household bin destinations. Trash bins are gray as of
# July 1, 2026 — the City no longer collects from the old black bins.
BIN_BLUE = "Blue Bin (Recycling)"
BIN_GREEN = "Green Bin (Organics)"
BIN_GRAY = "Gray Bin (Trash)"
BIN_SPECIAL = "Special Drop-off"
BIN_NONE = "Not Applicable"

VALID_BINS = frozenset({
    BIN_BLUE,
    BIN_GREEN,
    BIN_GRAY,
    BIN_SPECIAL,
    BIN_NONE,
})

# Default San Diego curbside bin for each disposal route, used when the
# classifier omits or invents a bin and for YOLO-only fallback items.
ROUTE_TO_DEFAULT_BIN = {
    ROUTE_RECYCLE: BIN_BLUE,
    ROUTE_COMPOST: BIN_GREEN,
    ROUTE_GENERAL_TRASH: BIN_GRAY,
    ROUTE_SINGLE_USE: BIN_GRAY,
    ROUTE_E_WASTE: BIN_SPECIAL,
    ROUTE_HAZARDOUS: BIN_SPECIAL,
    ROUTE_BULKY_ITEMS: BIN_SPECIAL,
    ROUTE_CITY_INFRA: BIN_NONE,
    ROUTE_LIVING_THINGS: BIN_NONE,
    default_route: BIN_SPECIAL,
}

# Waste categories with COCO class names

recycle = {
    "bottle", "wine glass", "cup", "bowl",
    "book", "vase"
}

compost = {
    "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza"
}

e_waste = {
    "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "clock"
}

bulky_items = {
    "chair", "couch", "potted plant", "bed", "dining table"
}

hazardous = {
    "microwave", "oven", "toaster", "sink",
    "refrigerator", "hair drier", "toothbrush"
}

single_use = {
    "backpack", "umbrella", "handbag", "suitcase",
    "sports ball", "fork", "knife", "spoon"
}

city_infra = {
    "traffic light", "fire hydrant", "stop sign"
}

living_things = {
    "person", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
}

general_trash = {
    "parking meter", "bench", "tie", "frisbee",
    "skis", "snowboard", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "donut", "cake", "scissors", "teddy bear"
}


def normalize_route(route: str) -> str:
    """Validate and normalize a disposal route string."""
    if route in VALID_ROUTES:
        return route
    return default_route


def normalize_bin(bin_str: str, route: str) -> str:
    """Validate a bin string, falling back to the route's default San Diego bin."""
    if bin_str in VALID_BINS:
        return bin_str
    return ROUTE_TO_DEFAULT_BIN.get(route, BIN_SPECIAL)


def is_non_waste_route(route: str) -> bool:
    """Return True when a route represents people, animals, or city infrastructure."""
    return route in NON_WASTE_ROUTES


def build_coco_to_bin(model_names):
    """Map COCO class names to disposal bins."""
    coco_to_bin = {}
    for _, name in model_names.items():
        if name in recycle:
            coco_to_bin[name] = ROUTE_RECYCLE
        elif name in compost:
            coco_to_bin[name] = ROUTE_COMPOST
        elif name in e_waste:
            coco_to_bin[name] = ROUTE_E_WASTE
        elif name in bulky_items:
            coco_to_bin[name] = ROUTE_BULKY_ITEMS
        elif name in hazardous:
            coco_to_bin[name] = ROUTE_HAZARDOUS
        elif name in single_use:
            coco_to_bin[name] = ROUTE_SINGLE_USE
        elif name in city_infra:
            coco_to_bin[name] = ROUTE_CITY_INFRA
        elif name in living_things:
            coco_to_bin[name] = ROUTE_LIVING_THINGS
        elif name in general_trash:
            coco_to_bin[name] = ROUTE_GENERAL_TRASH
        else:
            coco_to_bin[name] = default_route
    return coco_to_bin
