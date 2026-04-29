"""
SayCan-style task planner for REI-Bench.

Implements step-by-step planning where the LLM generates one action
at a time, constrained to the available skill set.
"""

import logging
import re
from typing import Optional

from .model_wrapper import ModelWrapper
from .skill_set import (
    ROBOT_SKILLS,
    SCENE_OBJECTS,
    format_skill_set_for_prompt,
    parse_action,
)
from .prompt_templates import (
    build_saycan_chat_messages,
    build_step_prompt,
    build_tocc_prompt,
)

logger = logging.getLogger(__name__)


class SayCanPlanner:
    """
    SayCan-style task planner using constrained LLM generation.

    Supports two planning modes:
    1. Full-plan mode: Generate the entire plan at once
    2. Step-by-step mode: Generate one action at a time (paper's approach)
    """

    def __init__(
        self,
        model: ModelWrapper,
        max_steps: int = 15,
        max_new_tokens: int = 64,
    ):
        """
        Initialize the planner.

        Args:
            model: Loaded ModelWrapper instance.
            max_steps: Maximum planning steps before forced termination.
            max_new_tokens: Max tokens per generation call.
        """
        self.model = model
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

    def plan_full(
        self,
        context_memory: str,
        instruction: str,
        scene_objects: list = None,
    ) -> list[str]:
        """
        Generate a full plan at once.

        Args:
            context_memory: Multi-turn dialogue context.
            instruction: Human's pending instruction.
            scene_objects: Objects in the current scene.

        Returns:
            List of action strings.
        """
        skill_set_str = format_skill_set_for_prompt(scene_objects)
        messages = build_saycan_chat_messages(
            context_memory, instruction, skill_set_str
        )

        # Generate with more tokens for full plan
        response = self.model.generate_with_chat_template(
            messages,
            max_new_tokens=256,
            temperature=0.0,
        )

        # Parse response into individual actions
        actions = self._parse_plan_response(response)
        return actions

    def plan_stepwise(
        self,
        context_memory: str,
        instruction: str,
        scene_objects: list = None,
    ) -> list[str]:
        """
        Generate a plan step-by-step (paper's SayCan approach).

        At each step, the model sees all previous actions and generates
        the next single action.

        Args:
            context_memory: Multi-turn dialogue context.
            instruction: Human's pending instruction.
            scene_objects: Objects in the current scene.

        Returns:
            List of action strings.
        """
        skill_set_str = format_skill_set_for_prompt(scene_objects)
        actions = []

        for step in range(self.max_steps):
            messages = build_step_prompt(
                context_memory, instruction, skill_set_str, actions
            )

            response = self.model.generate_with_chat_template(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
            )

            # Extract the next action from response
            action = self._extract_single_action(response)

            if action is None:
                logger.warning(
                    f"Could not parse action at step {step+1}: '{response}'"
                )
                actions.append(f"unparseable: {response[:50]}")
                break

            actions.append(action)

            if action.strip().lower() == "done":
                break

        return actions

    def plan_with_tocc(
        self,
        context_memory: str,
        instruction: str,
        scene_objects: list = None,
    ) -> tuple[str, list[str]]:
        """
        Plan using TOCC: first rewrite instruction, then plan.

        Args:
            context_memory: Multi-turn dialogue context.
            instruction: Human's (potentially vague) instruction.
            scene_objects: Objects in the current scene.

        Returns:
            Tuple of (rewritten_instruction, action_list).
        """
        # Step 1: Rewrite the instruction
        tocc_messages = build_tocc_prompt(context_memory, instruction)
        clear_instruction = self.model.generate_with_chat_template(
            tocc_messages,
            max_new_tokens=128,
            temperature=0.0,
        )

        logger.info(f"TOCC rewrite: '{instruction}' → '{clear_instruction}'")

        # Step 2: Plan with the clear instruction
        actions = self.plan_full(context_memory, clear_instruction, scene_objects)

        return clear_instruction, actions

    def _parse_plan_response(self, response: str) -> list[str]:
        """
        Parse a multi-line plan response into individual actions.

        Handles various output formats:
        - "1. find apple"
        - "Step 1: find apple"
        - "find apple"
        - "- find apple"
        """
        actions = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes
            line = re.sub(r"^(step\s*\d+[:\.]?\s*)", "", line, flags=re.IGNORECASE)
            line = re.sub(r"^(\d+[:\.\)]\s*)", "", line)
            line = re.sub(r"^[-•]\s*", "", line)
            line = line.strip()

            if not line:
                continue

            # Check if it looks like a valid action
            first_word = line.split()[0].lower() if line.split() else ""
            if first_word in ROBOT_SKILLS or first_word in [
                s.replace("_", "") for s in ROBOT_SKILLS
            ]:
                # Normalize skill names
                line = line.replace("pickup", "pick_up")
                line = line.replace("putdown", "put_down")
                line = line.replace("toggleon", "toggle_on")
                line = line.replace("toggleoff", "toggle_off")
                actions.append(line)

            if first_word == "done":
                actions.append("done")
                break

        return actions if actions else [f"unparseable: {response[:100]}"]

    def _extract_single_action(self, response: str) -> Optional[str]:
        """
        Extract a single action from a model response.

        The model might output extra text; we extract just the action.
        """
        # Take the first line that looks like an action
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Remove numbering/bullets
            line = re.sub(r"^(step\s*\d+[:\.]?\s*)", "", line, flags=re.IGNORECASE)
            line = re.sub(r"^(\d+[:\.\)]\s*)", "", line)
            line = re.sub(r"^[-•]\s*", "", line)
            line = line.strip()

            if not line:
                continue

            first_word = line.split()[0].lower()
            if first_word in ROBOT_SKILLS:
                return line
            # Handle unseparated skill names
            for skill in ROBOT_SKILLS:
                if line.lower().startswith(skill):
                    return line

        # If nothing matched, return the first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                return line

        return None
