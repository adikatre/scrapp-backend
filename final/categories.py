# categories.py

# ♻️ Waste categories with COCO class names

recycle = {
    "bottle", "wine glass", "cup", "bowl",
    "book", "clock", "vase"
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

# Fallback route for anything uncategorized
default_route = "Landfill / Donate / Check rules"


def build_coco_to_bin(model_names):
    """Map COCO class names to disposal bins."""
    COCO_TO_BIN = {}
    for _, name in model_names.items():
        if name in recycle:
            COCO_TO_BIN[name] = "Recycle"
        elif name in compost:
            COCO_TO_BIN[name] = "Compost"
        elif name in e_waste:
            COCO_TO_BIN[name] = "E-Waste"
        elif name in bulky_items:
            COCO_TO_BIN[name] = "Bulky Items"
        elif name in hazardous:
            COCO_TO_BIN[name] = "Hazardous Waste"
        elif name in single_use:
            COCO_TO_BIN[name] = "Single-Use Items"
        elif name in city_infra:
            COCO_TO_BIN[name] = "City Infrastructure"
        elif name in living_things:
            COCO_TO_BIN[name] = "Living Things"
        elif name in general_trash:
            COCO_TO_BIN[name] = "General Trash"
        else:
            COCO_TO_BIN[name] = default_route
    return COCO_TO_BIN
