"""
AI2-THOR skill set definition for REI-Bench.

Defines the robot's available actions in the simulator environment,
matching the skill set used in the SayCan framework.
"""

# ──────────────────────────────────────────────
# Core robot skills (actions the robot can perform)
# ──────────────────────────────────────────────
ROBOT_SKILLS = [
    "find",
    "pick_up",
    "put_down",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "slice",
    "clean",
    "heat",
    "cool",
    "examine",
    "done",
]

# ──────────────────────────────────────────────
# Objects available in AI2-THOR scenes
# ──────────────────────────────────────────────
SCENE_OBJECTS = [
    "AlarmClock", "Apple", "BaseballBat", "BasketBall", "Bowl",
    "GarbageCan", "HousePlant", "Laptop", "Mug", "RemoteControl",
    "SprayBottle", "Television", "Vase", "ArmChair", "Bed",
    "Book", "Bottle", "Box", "ButterKnife", "Candle", "CD",
    "CellPhone", "Chair", "CoffeeTable", "Cup", "DeskLamp",
    "Desk", "DiningTable", "Drawer", "Dresser", "FloorLamp",
    "Fork", "Newspaper", "Painting", "Pencil", "Pen",
    "PepperShaker", "Pillow", "Plate", "Pot", "SaltShaker",
    "Shelf", "SideTable", "Sofa", "Statue", "TeddyBear",
    "TennisRacket", "TVStand", "Watch", "Bread", "Lettuce",
    "Tomato", "Potato", "Egg", "Knife", "SoapBottle",
    "CreditCard", "KeyChain", "Microwave", "Fridge", "Sink",
    "StoveBurner", "Toaster", "CounterTop", "Cabinet",
    "LightSwitch", "Pan", "SoapBar", "Cloth", "Toilet",
    "HandTowel", "Bathtub", "ShowerHead", "TowelHolder",
]

# ──────────────────────────────────────────────
# Receptacles (locations where objects can be placed)
# ──────────────────────────────────────────────
RECEPTACLES = [
    "CounterTop", "DiningTable", "CoffeeTable", "SideTable",
    "Desk", "Shelf", "Drawer", "Cabinet", "Fridge",
    "Microwave", "Sink", "GarbageCan", "Bowl", "Plate",
    "Mug", "Cup", "Pot", "Pan", "Box", "Dresser",
    "TVStand", "ArmChair", "Sofa", "Bed", "Toaster",
    "StoveBurner", "Bathtub", "Toilet", "TowelHolder",
]


def get_skill_descriptions() -> dict:
    """
    Get human-readable descriptions of each skill.

    Returns:
        Dictionary mapping skill names to their descriptions.
    """
    return {
        "find": "Navigate to and locate an object in the environment",
        "pick_up": "Pick up an object (must be near it first)",
        "put_down": "Put down the held object at a location",
        "open": "Open a container or appliance (fridge, microwave, etc.)",
        "close": "Close a container or appliance",
        "toggle_on": "Turn on an appliance (lamp, microwave, etc.)",
        "toggle_off": "Turn off an appliance",
        "slice": "Slice an object (requires holding a knife)",
        "clean": "Clean an object using water (sink)",
        "heat": "Heat an object (microwave or stove)",
        "cool": "Cool an object (fridge)",
        "examine": "Examine an object closely (may need light)",
        "done": "Signal that the task is complete",
    }


def format_skill_set_for_prompt(scene_objects: list = None) -> str:
    """
    Format the skill set as a string for inclusion in prompts.

    Args:
        scene_objects: Optional list of objects in the current scene.
                      If None, uses the full default list.

    Returns:
        Formatted string listing available actions.
    """
    objects = scene_objects or SCENE_OBJECTS
    obj_str = ", ".join(objects[:30])  # Limit to avoid prompt overflow

    skills_text = """Available robot actions:
1. find [object] - Navigate to an object
2. pick_up [object] - Pick up an object
3. put_down [object] [location] - Place object at location
4. open [object] - Open a container/appliance
5. close [object] - Close a container/appliance
6. toggle_on [object] - Turn on an appliance
7. toggle_off [object] - Turn off an appliance
8. slice [object] - Slice an object
9. clean [object] - Clean an object using sink
10. heat [object] - Heat object using microwave/stove
11. cool [object] - Cool object using fridge
12. examine [object] - Examine an object
13. done - Task is complete

Objects in the scene: {objects}""".format(objects=obj_str)

    return skills_text


def parse_action(action_str: str) -> dict:
    """
    Parse an action string into structured format.

    Args:
        action_str: Raw action string like "find apple" or "put_down apple countertop"

    Returns:
        Dictionary with 'skill', 'object', and optionally 'location'.
    """
    parts = action_str.strip().lower().split()
    if not parts:
        return {"skill": "unknown", "object": None, "location": None}

    skill = parts[0]
    obj = parts[1] if len(parts) > 1 else None
    location = parts[2] if len(parts) > 2 else None

    return {"skill": skill, "object": obj, "location": location}


# ──────────────────────────────────────────────
# Ground-truth action templates for each task type
# ──────────────────────────────────────────────
TASK_ACTION_TEMPLATES = {
    "pick_and_place": [
        "find {object}",
        "pick_up {object}",
        "find {receptacle}",
        "put_down {object} {receptacle}",
        "done",
    ],
    "stack_and_place": [
        "find {object1}",
        "pick_up {object1}",
        "find {object2}",
        "put_down {object1} {object2}",
        "find {receptacle}",
        "pick_up {object2}",
        "put_down {object2} {receptacle}",
        "done",
    ],
    "clean_and_place": [
        "find {object}",
        "pick_up {object}",
        "find sink",
        "clean {object}",
        "find {receptacle}",
        "put_down {object} {receptacle}",
        "done",
    ],
    "heat_and_place": [
        "find {object}",
        "pick_up {object}",
        "find microwave",
        "open microwave",
        "heat {object}",
        "close microwave",
        "find {receptacle}",
        "put_down {object} {receptacle}",
        "done",
    ],
    "cool_and_place": [
        "find {object}",
        "pick_up {object}",
        "find fridge",
        "open fridge",
        "cool {object}",
        "close fridge",
        "find {receptacle}",
        "put_down {object} {receptacle}",
        "done",
    ],
    "examine_in_light": [
        "find {object}",
        "pick_up {object}",
        "find {lamp}",
        "toggle_on {lamp}",
        "examine {object}",
        "done",
    ],
}
