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

# Fallback route for anything uncategorized
default_route = "Landfill / Donate / Check rules"

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
