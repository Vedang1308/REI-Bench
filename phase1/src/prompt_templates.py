"""
Prompt templates for SayCan-style task planning on REI-Bench.

Implements the prompting strategy from the paper:
- System prompt establishing the robot's role
- Context memory (multi-turn dialogue history)
- Human pending instruction
- Skill set constraint
- Step-by-step planning
"""


SYSTEM_PROMPT = """You are a helpful home robot assistant. You can interact with objects in the environment using a predefined set of skills. Your task is to help the human by generating a step-by-step plan to accomplish their request.

Important rules:
1. You must ONLY use actions from the available skill set.
2. You must specify the exact object name for each action.
3. Generate one action per line.
4. End your plan with "done" when the task is complete.
5. Be precise about which objects to interact with based on the conversation context."""


def build_saycan_prompt(
    context_memory: str,
    instruction: str,
    skill_set_str: str,
    scene_objects: list = None,
) -> str:
    """
    Build a complete SayCan-style prompt for task planning.

    Args:
        context_memory: Multi-turn dialogue history between human and robot.
        instruction: The human's pending instruction to execute.
        skill_set_str: Formatted string of available robot actions.
        scene_objects: Optional list of objects in the current scene.

    Returns:
        Complete prompt string ready for model input.
    """
    prompt = f"""{SYSTEM_PROMPT}

{skill_set_str}

--- Previous Conversation ---
{context_memory}

--- Human Pending Instruction ---
{instruction}

--- Your Plan ---
Based on the conversation above and the human's instruction, generate a step-by-step plan using ONLY the available actions. Output one action per line.

Plan:
"""
    return prompt


def build_saycan_chat_messages(
    context_memory: str,
    instruction: str,
    skill_set_str: str,
) -> list:
    """
    Build chat-style messages for instruction-tuned models.

    Args:
        context_memory: Multi-turn dialogue history.
        instruction: Human's pending instruction.
        skill_set_str: Available skills.

    Returns:
        List of message dicts for chat template.
    """
    system_msg = f"""{SYSTEM_PROMPT}

{skill_set_str}"""

    user_msg = f"""Here is the previous conversation between the human and the robot:

{context_memory}

Now the human gives the following instruction:
"{instruction}"

Generate a step-by-step plan to accomplish this instruction. Use ONLY the available actions listed above. Output one action per line, and end with "done".

Plan:"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_step_prompt(
    context_memory: str,
    instruction: str,
    skill_set_str: str,
    previous_actions: list,
) -> list:
    """
    Build a prompt for per-step planning (SayCan iterative style).

    At each step, the model sees the context, instruction, and previous
    actions, then generates the next single action.

    Args:
        context_memory: Multi-turn dialogue history.
        instruction: Human's pending instruction.
        skill_set_str: Available skills.
        previous_actions: List of actions already taken.

    Returns:
        List of message dicts for chat template.
    """
    prev_actions_str = "\n".join(
        [f"Step {i+1}: {a}" for i, a in enumerate(previous_actions)]
    )

    system_msg = f"""{SYSTEM_PROMPT}

{skill_set_str}"""

    user_msg = f"""Previous conversation:
{context_memory}

Human instruction: "{instruction}"

Actions taken so far:
{prev_actions_str if prev_actions_str else "(none yet)"}

What is the next action? Output ONLY the next single action (e.g., "find apple" or "pick_up potato" or "done").

Next action:"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ──────────────────────────────────────────────
# Prompting method variants from the paper
# ──────────────────────────────────────────────

def build_tocc_prompt(context_memory: str, instruction: str) -> list:
    """
    Build TOCC (Task-Oriented Context Cognition) prompt.

    TOCC first rewrites the vague instruction into a clear one,
    then uses the clear instruction for planning.

    Args:
        context_memory: Multi-turn dialogue history.
        instruction: Human's (potentially vague) instruction.

    Returns:
        Chat messages for the TOCC rewriting step.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a language understanding assistant. Your task is to "
                "rewrite vague human instructions into clear, unambiguous ones "
                "by resolving all referring expressions using the conversation context."
            ),
        },
        {
            "role": "user",
            "content": f"""Human pending instruction may contain vague referring expressions, such as "electronic devices", "beverages", "fruits", and "containers", which are not specific items. Use the previous context below to resolve the referring expressions:

Previous conversation:
{context_memory}

Instruction to rewrite: "{instruction}"

Do not add extra commentary or conversation. Only output the clear, rewritten instruction with all vague references resolved to specific object names.

Clear instruction:""",
        },
    ]


def build_aware_prompt_addition() -> str:
    """
    Get the Aware Prompt (AP) addition from the paper.

    Returns:
        AP text to prepend to the system prompt.
    """
    return (
        'I will check whether the "Human Pending Instruction" contains '
        "implicit or ambiguous references.\n"
        'I understand that "Human Pending Instruction" may include vague '
        "referring expressions, and I can infer their meaning based on "
        "context and antecedents in the preceding dialogue."
    )


def build_cot_prompt_addition() -> str:
    """
    Get the Chain-of-Thought (CoT) prompt addition from the paper.

    Returns:
        CoT text to add to the user prompt.
    """
    return (
        'The "Human Pending Instruction" may contain vague referring '
        "expressions. Before planning, I will first identify any referring "
        "expressions and reason about their intended objects based on the "
        "context below, and then restate the instruction with the resolved "
        "entities.\n"
        "Step: Identify referring expressions → infer their referents → "
        "rewrite the instruction with explicit object names."
    )
