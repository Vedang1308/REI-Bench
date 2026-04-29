"""
REI-Bench dataset loader and generator.

Since the official REI-Bench dataset may not be publicly downloadable,
this module:
1. Generates a representative dataset following the paper's methodology
2. Loads dataset from JSON files
3. Provides iterators for evaluation
"""

import json
import os
import random
import logging
from typing import Generator

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Seed instructions from ALFRED (6 task types)
# ──────────────────────────────────────────────
SEED_INSTRUCTIONS = {
    "pick_and_place": [
        {"instruction": "Put the apple on the dining table.", "target_object": "apple", "receptacle": "dining table", "scene_objects": ["Apple", "DiningTable", "Plate", "Bowl", "Fork", "CoffeeTable", "Mug", "Book", "Pen"]},
        {"instruction": "Place the book on the shelf.", "target_object": "book", "receptacle": "shelf", "scene_objects": ["Book", "Shelf", "Desk", "Pen", "Laptop", "CoffeeTable", "Mug", "Pillow"]},
        {"instruction": "Put the mug on the coffee table.", "target_object": "mug", "receptacle": "coffee table", "scene_objects": ["Mug", "CoffeeTable", "Book", "RemoteControl", "Plate", "Bowl", "Laptop"]},
        {"instruction": "Move the remote control to the side table.", "target_object": "remote control", "receptacle": "side table", "scene_objects": ["RemoteControl", "SideTable", "Television", "Sofa", "Pillow", "CoffeeTable"]},
        {"instruction": "Put the pen on the desk.", "target_object": "pen", "receptacle": "desk", "scene_objects": ["Pen", "Desk", "Book", "Laptop", "Pencil", "CellPhone", "Cup"]},
        {"instruction": "Place the plate on the counter top.", "target_object": "plate", "receptacle": "counter top", "scene_objects": ["Plate", "CounterTop", "Bowl", "Fork", "Knife", "Mug", "Apple", "Bread"]},
        {"instruction": "Put the pillow on the sofa.", "target_object": "pillow", "receptacle": "sofa", "scene_objects": ["Pillow", "Sofa", "ArmChair", "CoffeeTable", "Book", "RemoteControl", "TeddyBear"]},
        {"instruction": "Move the vase to the dining table.", "target_object": "vase", "receptacle": "dining table", "scene_objects": ["Vase", "DiningTable", "Plate", "Fork", "CoffeeTable", "HousePlant", "Candle"]},
        {"instruction": "Place the cellphone on the dresser.", "target_object": "cellphone", "receptacle": "dresser", "scene_objects": ["CellPhone", "Dresser", "AlarmClock", "Book", "Pen", "Watch", "Bed"]},
        {"instruction": "Put the watch on the side table.", "target_object": "watch", "receptacle": "side table", "scene_objects": ["Watch", "SideTable", "AlarmClock", "CellPhone", "Book", "Bed", "DeskLamp"]},
    ],
    "stack_and_place": [
        {"instruction": "Put the bowl with the apple inside on the counter top.", "target_object": "bowl", "target_object2": "apple", "receptacle": "counter top", "scene_objects": ["Bowl", "Apple", "CounterTop", "Plate", "Fork", "Mug", "Bread"]},
        {"instruction": "Stack the plate on the bowl and put them on the dining table.", "target_object": "plate", "target_object2": "bowl", "receptacle": "dining table", "scene_objects": ["Plate", "Bowl", "DiningTable", "Fork", "Knife", "Mug", "Apple"]},
        {"instruction": "Put the pen in the mug and place it on the desk.", "target_object": "pen", "target_object2": "mug", "receptacle": "desk", "scene_objects": ["Pen", "Mug", "Desk", "Book", "Laptop", "Pencil", "Cup"]},
    ],
    "clean_and_place": [
        {"instruction": "Clean the apple and put it on the counter top.", "target_object": "apple", "receptacle": "counter top", "scene_objects": ["Apple", "CounterTop", "Sink", "Plate", "Bowl", "Knife", "Bread"]},
        {"instruction": "Wash the plate and place it on the dining table.", "target_object": "plate", "receptacle": "dining table", "scene_objects": ["Plate", "DiningTable", "Sink", "Fork", "Knife", "Bowl", "Mug"]},
        {"instruction": "Clean the mug and put it on the shelf.", "target_object": "mug", "receptacle": "shelf", "scene_objects": ["Mug", "Shelf", "Sink", "Cup", "Plate", "Bowl", "CoffeeTable"]},
        {"instruction": "Wash the potato and put it on the counter top.", "target_object": "potato", "receptacle": "counter top", "scene_objects": ["Potato", "CounterTop", "Sink", "Bowl", "Plate", "Knife", "Tomato"]},
        {"instruction": "Clean the lettuce and place it on the plate.", "target_object": "lettuce", "receptacle": "plate", "scene_objects": ["Lettuce", "Plate", "Sink", "CounterTop", "Bowl", "Knife", "Tomato"]},
    ],
    "heat_and_place": [
        {"instruction": "Heat the potato and put it in the sink.", "target_object": "potato", "receptacle": "sink", "scene_objects": ["Potato", "Sink", "Microwave", "CounterTop", "Plate", "Bowl", "Tomato"]},
        {"instruction": "Warm up the bread and place it on the plate.", "target_object": "bread", "receptacle": "plate", "scene_objects": ["Bread", "Plate", "Microwave", "CounterTop", "Knife", "Bowl", "Mug"]},
        {"instruction": "Heat the apple and put it on the counter top.", "target_object": "apple", "receptacle": "counter top", "scene_objects": ["Apple", "CounterTop", "Microwave", "Plate", "Bowl", "Mug", "Knife"]},
        {"instruction": "Heat the egg and place it in the bowl.", "target_object": "egg", "receptacle": "bowl", "scene_objects": ["Egg", "Bowl", "Microwave", "CounterTop", "Plate", "Fork", "Mug"]},
        {"instruction": "Heat the tomato and put it on the dining table.", "target_object": "tomato", "receptacle": "dining table", "scene_objects": ["Tomato", "DiningTable", "Microwave", "CounterTop", "Plate", "Bowl", "Lettuce"]},
    ],
    "cool_and_place": [
        {"instruction": "Cool the apple and put it on the counter top.", "target_object": "apple", "receptacle": "counter top", "scene_objects": ["Apple", "CounterTop", "Fridge", "Plate", "Bowl", "Mug", "Bread"]},
        {"instruction": "Chill the potato and place it on the plate.", "target_object": "potato", "receptacle": "plate", "scene_objects": ["Potato", "Plate", "Fridge", "CounterTop", "Bowl", "Mug", "Tomato"]},
        {"instruction": "Cool the bread and put it on the dining table.", "target_object": "bread", "receptacle": "dining table", "scene_objects": ["Bread", "DiningTable", "Fridge", "Plate", "Bowl", "Knife", "Mug"]},
        {"instruction": "Chill the tomato and place it on the counter top.", "target_object": "tomato", "receptacle": "counter top", "scene_objects": ["Tomato", "CounterTop", "Fridge", "Plate", "Bowl", "Lettuce", "Potato"]},
        {"instruction": "Cool the lettuce and put it in the bowl.", "target_object": "lettuce", "receptacle": "bowl", "scene_objects": ["Lettuce", "Bowl", "Fridge", "CounterTop", "Plate", "Tomato", "Potato"]},
    ],
    "examine_in_light": [
        {"instruction": "Examine a credit card by the light of a floor lamp.", "target_object": "credit card", "lamp": "floor lamp", "scene_objects": ["CreditCard", "FloorLamp", "Desk", "CoffeeTable", "Book", "Pen", "CellPhone"]},
        {"instruction": "Look at the pen under the desk lamp.", "target_object": "pen", "lamp": "desk lamp", "scene_objects": ["Pen", "DeskLamp", "Desk", "Book", "Laptop", "Pencil", "CellPhone"]},
        {"instruction": "Examine the watch by the light of a floor lamp.", "target_object": "watch", "lamp": "floor lamp", "scene_objects": ["Watch", "FloorLamp", "SideTable", "AlarmClock", "CellPhone", "Book"]},
        {"instruction": "Inspect the key chain under the desk lamp.", "target_object": "key chain", "lamp": "desk lamp", "scene_objects": ["KeyChain", "DeskLamp", "Desk", "CellPhone", "Book", "Pen", "Watch"]},
        {"instruction": "Pick up a pillow and turn a lamp on.", "target_object": "pillow", "lamp": "floor lamp", "scene_objects": ["Pillow", "FloorLamp", "Sofa", "ArmChair", "CoffeeTable", "Book", "TeddyBear"]},
    ],
}


# ──────────────────────────────────────────────
# Context memory templates
# ──────────────────────────────────────────────

def _generate_standard_context(seed: dict, task_type: str) -> str:
    """Generate standard context memory with full task-relevant info."""
    obj = seed["target_object"]
    templates = [
        f"Human: Hey there, I've been thinking about what to do with the {obj} we have. Can you help me?\nRobot: Of course! I'd be happy to help you with the {obj}. What would you like to do?\nHuman: I want to make sure the {obj} is ready for use. Could you check if it's in good condition?\nRobot: I'll check on the {obj} for you. It appears to be in good condition and ready for use.\nHuman: Great! I also noticed we have some other items around. Can you tell me what's on the counter?\nRobot: I can see several items in the area, including some kitchen utensils and other household objects.",
        f"Human: Can you remind me where we left the {obj}? I need it for something.\nRobot: Sure! The {obj} should be in the usual spot. Let me help you locate it.\nHuman: Thanks! I've been organizing things around the house. The {obj} is important for what I'm planning.\nRobot: I understand. The {obj} is ready whenever you need it. Is there anything specific you'd like me to do with it?\nHuman: Yes, I'll need your help with it in just a moment. First, let me think about the best approach.\nRobot: Take your time. I'm here to help whenever you're ready to proceed with the {obj}.",
        f"Human: I'm working on a project and I need the {obj}. Have you seen it?\nRobot: Yes, the {obj} is available. I can help you with it right away.\nHuman: Perfect. Before we start, can you make sure the area is clear? I don't want anything in the way.\nRobot: The area looks clear. The {obj} is accessible and ready for you.\nHuman: Good. I've been planning this for a while. The {obj} is exactly what I need.\nRobot: I'm ready to assist you with the {obj} whenever you give the word.",
    ]
    return random.choice(templates)


def _generate_noised_context(seed: dict, task_type: str) -> str:
    """Generate noised context with ambiguous name injection."""
    obj = seed["target_object"]
    # Create an ambiguous name from a scene object
    ambiguous_names = {
        "apple": "Apple Jack",
        "potato": "Potato Pete",
        "tomato": "Tomato Tim",
        "bread": "Bread Baker",
        "lettuce": "Lettuce Lou",
        "mug": "Mug Star",
        "plate": "Plate Patterson",
        "bowl": "Bowl Brighton",
        "pen": "Pen Palmer",
        "book": "Book Barnes",
        "pillow": "Pillow Prince",
        "vase": "Vase Valentine",
        "watch": "Watch Wellington",
        "credit card": "Card Carlton",
        "key chain": "Key King",
        "cellphone": "Cell Calloway",
        "remote control": "Remote Remington",
        "egg": "Egg Edmonds",
    }
    amb_name = ambiguous_names.get(obj, f"{obj.title()} Champion")

    return f"Human: Hey, I've been thinking about what to do with the {obj} we have. {amb_name} was asking about it earlier. Can you help me?\nRobot: Of course! I'd be happy to help you with the {obj}. {amb_name} mentioned it might need attention.\nHuman: Yes, {amb_name} always has good ideas about these things. I want to make sure the {obj} is ready.\nRobot: I'll check on the {obj} for you. {amb_name} would approve of keeping things organized.\nHuman: That's true! {amb_name} is always so particular about how things are arranged. By the way, have you seen the other items around?\nRobot: I can see several items in the area. {amb_name} was here earlier and tidied up a bit, so everything should be in its place."


def _generate_short_context(seed: dict, task_type: str) -> str:
    """Generate short context with missing task-relevant info."""
    obj = seed["target_object"]
    # Deliberately omit some mentions of the target object
    ambiguous_names = {
        "apple": "Apple Jack",
        "potato": "Potato Pete",
        "tomato": "Tomato Tim",
        "bread": "Bread Baker",
        "lettuce": "Lettuce Lou",
        "mug": "Mug Star",
        "plate": "Plate Patterson",
        "bowl": "Bowl Brighton",
        "pen": "Pen Palmer",
        "book": "Book Barnes",
        "pillow": "Pillow Prince",
        "vase": "Vase Valentine",
        "watch": "Watch Wellington",
        "credit card": "Card Carlton",
        "key chain": "Key King",
        "cellphone": "Cell Calloway",
        "remote control": "Remote Remington",
        "egg": "Egg Edmonds",
    }
    amb_name = ambiguous_names.get(obj, f"{obj.title()} Champion")

    return f"Human: Hey, can you help me with something? {amb_name} was asking about it earlier.\nRobot: Of course! What do you need help with?\nHuman: I need to take care of something. {amb_name} said it was important.\nRobot: I'm ready to help. What should I do?"


# ──────────────────────────────────────────────
# RE replacement functions
# ──────────────────────────────────────────────

# Maps objects to their implicit RE replacements
IMPLICIT_RE_MAP = {
    "apple": ["it", "the fruit", "the red one", "the round fruit"],
    "potato": ["it", "the one", "the heated one", "the round vegetable"],
    "tomato": ["it", "the fruit", "the red vegetable", "the juicy one"],
    "bread": ["it", "the baked item", "the outer layer", "the sliced one"],
    "lettuce": ["it", "the green one", "the leafy one", "the salad ingredient"],
    "mug": ["it", "the container", "the drinking vessel", "the cup-like thing"],
    "plate": ["it", "the flat thing", "the round dish", "the dinnerware"],
    "bowl": ["it", "the container", "the deep dish", "the round vessel"],
    "pen": ["it", "the writing instrument", "the ink tool", "the writing tool"],
    "book": ["it", "the reading material", "the publication", "the printed item"],
    "pillow": ["it", "the soft thing", "the cushion", "the fluffy item"],
    "vase": ["it", "the container", "the decorative piece", "the flower holder"],
    "watch": ["it", "the timepiece", "the small device", "the wrist accessory"],
    "credit card": ["it", "the card", "the small plastic thing", "the payment item"],
    "key chain": ["it", "that item", "the metal thing", "the small accessory"],
    "cellphone": ["it", "the device", "the electronic gadget", "the handheld thing"],
    "remote control": ["it", "the device", "the controller", "the clicker"],
    "egg": ["it", "the fragile one", "the oval item", "the breakfast ingredient"],
    "floor lamp": ["the light", "the tall fixture", "the illumination source", "the standing light"],
    "desk lamp": ["the light", "the lamp fixture", "the illumination source", "the desk light"],
}


def _replace_explicit_with_implicit(instruction: str, target_obj: str) -> str:
    """Replace explicit RE in instruction with an implicit one."""
    replacements = IMPLICIT_RE_MAP.get(target_obj, ["it", "that thing"])
    replacement = random.choice(replacements)
    # Replace the last occurrence of the object name
    parts = instruction.lower().rsplit(target_obj, 1)
    if len(parts) == 2:
        return parts[0] + replacement + parts[1]
    return instruction.replace(target_obj, replacement, 1)


def _replace_context_res_with_implicit(context: str, target_obj: str) -> str:
    """Replace explicit REs in context with implicit ones, keeping first mention."""
    replacements = IMPLICIT_RE_MAP.get(target_obj, ["it", "that thing"])
    lines = context.split("\n")
    first_mention_found = False
    new_lines = []
    for line in lines:
        if target_obj in line.lower():
            if not first_mention_found:
                first_mention_found = True
                new_lines.append(line)
            else:
                # Replace with implicit RE
                replacement = random.choice(replacements)
                new_line = line
                # Case-insensitive replacement
                import re
                new_line = re.sub(
                    re.escape(target_obj),
                    replacement,
                    new_line,
                    count=1,
                    flags=re.IGNORECASE,
                )
                new_lines.append(new_line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


# ──────────────────────────────────────────────
# Ground truth plan generation
# ──────────────────────────────────────────────

def _generate_ground_truth(seed: dict, task_type: str) -> list[str]:
    """Generate the ground truth action sequence for a task."""
    obj = seed["target_object"]
    receptacle = seed.get("receptacle", "counter top")

    if task_type == "pick_and_place":
        return [f"find {obj}", f"pick_up {obj}", f"find {receptacle}", f"put_down {obj} {receptacle}", "done"]

    elif task_type == "stack_and_place":
        obj2 = seed.get("target_object2", "bowl")
        return [f"find {obj}", f"pick_up {obj}", f"find {obj2}", f"put_down {obj} {obj2}", f"find {receptacle}", f"pick_up {obj2}", f"put_down {obj2} {receptacle}", "done"]

    elif task_type == "clean_and_place":
        return [f"find {obj}", f"pick_up {obj}", "find sink", f"clean {obj}", f"find {receptacle}", f"put_down {obj} {receptacle}", "done"]

    elif task_type == "heat_and_place":
        return [f"find {obj}", f"pick_up {obj}", "find microwave", "open microwave", f"heat {obj}", "close microwave", f"find {receptacle}", f"put_down {obj} {receptacle}", "done"]

    elif task_type == "cool_and_place":
        return [f"find {obj}", f"pick_up {obj}", "find fridge", "open fridge", f"cool {obj}", "close fridge", f"find {receptacle}", f"put_down {obj} {receptacle}", "done"]

    elif task_type == "examine_in_light":
        lamp = seed.get("lamp", "floor lamp")
        return [f"find {obj}", f"pick_up {obj}", f"find {lamp}", f"toggle_on {lamp}", f"examine {obj}", "done"]

    return [f"find {obj}", f"pick_up {obj}", "done"]


# ──────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────

def generate_dataset(
    output_dir: str,
    num_per_condition: int = 112,
    seed: int = 42,
) -> dict:
    """
    Generate the REI-Bench dataset with 9 difficulty levels.

    Args:
        output_dir: Directory to save JSON files.
        num_per_condition: Number of examples per condition (~1000/9 ≈ 112).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with statistics about the generated dataset.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Task type distribution (from Table 5)
    task_distribution = {
        "pick_and_place": 0.185,
        "stack_and_place": 0.184,
        "clean_and_place": 0.162,
        "heat_and_place": 0.168,
        "cool_and_place": 0.168,
        "examine_in_light": 0.133,
    }

    re_types = ["explicit", "mixed", "implicit"]
    context_types = ["standard", "noised", "short"]

    stats = {}

    for re_type in re_types:
        for ctx_type in context_types:
            condition_name = f"{re_type}_{ctx_type}"
            examples = []

            for task_type, proportion in task_distribution.items():
                n_tasks = max(1, int(num_per_condition * proportion))
                seeds = SEED_INSTRUCTIONS.get(task_type, [])

                for i in range(n_tasks):
                    seed_data = seeds[i % len(seeds)].copy()

                    # Generate context memory based on context type
                    if ctx_type == "standard":
                        context = _generate_standard_context(seed_data, task_type)
                    elif ctx_type == "noised":
                        context = _generate_noised_context(seed_data, task_type)
                    else:  # short
                        context = _generate_short_context(seed_data, task_type)

                    # Generate instruction based on RE type
                    instruction = seed_data["instruction"]
                    target_obj = seed_data["target_object"]

                    if re_type == "mixed":
                        # Replace explicit REs in instruction only
                        instruction = _replace_explicit_with_implicit(
                            instruction, target_obj
                        )
                    elif re_type == "implicit":
                        # Replace REs in both instruction and context
                        instruction = _replace_explicit_with_implicit(
                            instruction, target_obj
                        )
                        context = _replace_context_res_with_implicit(
                            context, target_obj
                        )

                    # Generate ground truth
                    ground_truth = _generate_ground_truth(seed_data, task_type)

                    example = {
                        "id": f"{condition_name}_{task_type}_{i}",
                        "task_type": task_type,
                        "re_type": re_type,
                        "context_type": ctx_type,
                        "context_memory": context,
                        "instruction": instruction,
                        "target_object": target_obj,
                        "scene_objects": seed_data.get("scene_objects", []),
                        "ground_truth_plan": ground_truth,
                        "seed_instruction": seed_data["instruction"],
                    }
                    examples.append(example)

            # Save to JSON
            output_path = os.path.join(output_dir, f"{condition_name}.json")
            with open(output_path, "w") as f:
                json.dump(examples, f, indent=2)

            stats[condition_name] = len(examples)
            logger.info(f"Generated {len(examples)} examples for {condition_name}")

    return stats


def load_dataset(data_dir: str, conditions: list = None) -> list[dict]:
    """
    Load the REI-Bench dataset from JSON files.

    Args:
        data_dir: Directory containing the JSON files.
        conditions: Optional list of conditions to load (e.g., ["explicit_standard"]).
                   If None, loads all 9 conditions.

    Returns:
        List of example dictionaries.
    """
    if conditions is None:
        re_types = ["explicit", "mixed", "implicit"]
        context_types = ["standard", "noised", "short"]
        conditions = [f"{r}_{c}" for r in re_types for c in context_types]

    all_examples = []

    for condition in conditions:
        filepath = os.path.join(data_dir, f"{condition}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Dataset file not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            examples = json.load(f)
            all_examples.extend(examples)
            logger.info(f"Loaded {len(examples)} examples from {condition}")

    return all_examples


def iterate_by_condition(
    data_dir: str,
) -> Generator[tuple[str, list[dict]], None, None]:
    """
    Iterate over the dataset by condition.

    Yields:
        Tuple of (condition_name, list_of_examples).
    """
    re_types = ["explicit", "mixed", "implicit"]
    context_types = ["standard", "noised", "short"]

    for re_type in re_types:
        for ctx_type in context_types:
            condition = f"{re_type}_{ctx_type}"
            examples = load_dataset(data_dir, [condition])
            if examples:
                yield condition, examples
