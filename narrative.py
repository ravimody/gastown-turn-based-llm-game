"""
Narrative engine for The Age of AI: 2026.

Coordinates prompt templates, character profiles, and LLM generation
for dynamic story elements.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from characters import (
    NPC_PROFILES,
    get_npc_profile,
    list_npcs,
    DialogueManager,
    RelationshipTracker,
)
from prompts import PromptEngine, render_template, get_prompt_engine
from llm_provider import LLMProvider, MockLLMProvider, create_llm_provider


class NarrativeEngine:
    """
    Central narrative coordinator.

    Bridges game state → prompts → LLM → narrative output.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or create_llm_provider('mock')
        self.prompts = get_prompt_engine()
        self.dialogue = DialogueManager(llm_provider=self.llm)
        self.relationships = RelationshipTracker()

    def generate_opening_scenario(self, game_state: Dict[str, Any]) -> str:
        """Generate the game's opening narrative."""
        context = {
            'bank_account': game_state.get('bank_account', 5000),
            'year': 2026,
            'goal': 20000000,
            **game_state
        }

        # Try template first
        prompt = render_template('scenarios/opening.j2', context)

        # Use LLM to enhance
        return self.llm.generate_scenario(context)

    def generate_victory_scenario(
        self,
        game_state: Dict[str, Any],
        turns: int
    ) -> str:
        """Generate victory ending narrative."""
        context = {
            'net_worth': game_state.get('net_worth', 0),
            'turns': turns,
            'years': round(turns / 52, 1),
            **game_state
        }

        # Render template for structure
        template_result = render_template('scenarios/victory.j2', context)

        # Use template as prompt for LLM
        return template_result

    def generate_game_over_scenario(
        self,
        game_state: Dict[str, Any],
        reason: str,
        turns: int
    ) -> str:
        """Generate game over narrative."""
        templates = {
            'bankruptcy': 'scenarios/bankruptcy.j2',
            'burnout': 'scenarios/bankruptcy.j2',  # Reuse with different context
            'time_out': 'scenarios/bankruptcy.j2',
        }

        context = {
            'bank_account': game_state.get('bank_account', 0),
            'net_worth': game_state.get('net_worth', 0),
            'turns': turns,
            'context': self._get_game_over_context(reason),
            **game_state
        }

        template_name = templates.get(reason, 'scenarios/bankruptcy.j2')
        return render_template(template_name, context)

    def _get_game_over_context(self, reason: str) -> str:
        """Get flavor text for game over reason."""
        contexts = {
            'bankruptcy': (
                "The debts piled up faster than the income. The creditors stopped being polite. "
                "Silicon Valley has no safety net for the broke."
            ),
            'burnout': (
                "Your body finally rebelled. The stress, the sleepless nights, the constant hustle—"
                "it all caught up. The doctor's orders were clear: stop, or risk permanent damage."
            ),
            'time_out': (
                "Ten years. A decade of grinding, and still short of the goal. "
                "The window closes. The AI boom moves on to a new generation."
            ),
        }
        return contexts.get(reason, "The journey ends here.")

    def generate_action_outcome(
        self,
        action: str,
        numerical_result: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        """Generate narrative description of action outcome."""
        # Try action-specific template
        template_name = f'outcomes/{action}.j2'
        context = {
            'action': action,
            'numerical_result': numerical_result,
            **game_state,
            **numerical_result
        }

        # Render template for prompt
        prompt = render_template(template_name, context)

        # Fall back to generic if specific template missing
        if prompt.startswith('[Template'):
            prompt = render_template('outcomes/generic.j2', context)

        # Use LLM for final narrative
        return self.llm.generate_outcome_narrative(
            action=action,
            numerical_result=numerical_result,
            context=game_state
        )

    def generate_npc_introduction(self, npc_id: str) -> str:
        """Generate introduction text for an NPC."""
        profile = get_npc_profile(npc_id)
        if not profile:
            return f"[Unknown character: {npc_id}]"

        context = profile.to_dict()
        return render_template('characters/intro.j2', context)

    def start_npc_dialogue(self, npc_id: str) -> Dict[str, Any]:
        """Begin a dialogue interaction with an NPC."""
        profile = get_npc_profile(npc_id)
        if not profile:
            return {'error': f'Unknown NPC: {npc_id}'}

        # Get relationship state
        rel = self.relationships.get_relationship(npc_id)

        # Start dialogue tree
        node = self.dialogue.start_dialogue(npc_id)

        # Generate dynamic greeting using LLM
        game_context = {
            'trust_level': rel.get('trust', 0),
            'market_sentiment': 1.0,  # Would come from actual game state
            'ai_hype_cycle': 50,
            'turn_number': 1,
        }

        greeting = self.llm.generate_npc_dialogue(
            npc_id=npc_id,
            npc_profile=profile.to_dict(),
            player_history=self.dialogue.get_history(),
            current_context=game_context
        )

        return {
            'npc': profile.to_dict(),
            'greeting': greeting,
            'trust_level': rel.get('trust', 0),
            'choices': [
                {'text': c.text, 'index': i}
                for i, c in enumerate(node.choices)
            ] if node else []
        }

    def continue_dialogue(
        self,
        npc_id: str,
        choice_index: int
    ) -> Dict[str, Any]:
        """Continue dialogue after player choice."""
        node = self.dialogue.make_choice(choice_index)

        if not node or node.node_type.name == 'FAREWELL':
            self.dialogue.end_dialogue()
            return {'ended': True}

        profile = get_npc_profile(npc_id)
        rel = self.relationships.get_relationship(npc_id)

        # Generate response
        game_context = {
            'trust_level': rel.get('trust', 0),
            'market_sentiment': 1.0,
            'ai_hype_cycle': 50,
            'turn_number': 1,
        }

        response = self.llm.generate_npc_dialogue(
            npc_id=npc_id,
            npc_profile=profile.to_dict(),
            player_history=self.dialogue.get_history(),
            current_context=game_context
        )

        # Apply trust changes from choice
        prev_history = self.dialogue.get_history()
        if prev_history:
            last = prev_history[-1]
            if last.get('trust_change'):
                self.relationships.modify_trust(npc_id, last['trust_change'])

        return {
            'response': response,
            'trust_level': rel.get('trust', 0),
            'choices': [
                {'text': c.text, 'index': i}
                for i, c in enumerate(node.choices)
            ]
        }

    def generate_market_news(self, market_state: Dict[str, Any]) -> str:
        """Generate news/flavor for market phase."""
        return self.llm.generate_market_news(market_state)

    def get_narrative_context(self) -> Dict[str, Any]:
        """Get full narrative state for save/load."""
        return {
            'relationships': self.relationships.to_dict(),
            'dialogue_history': self.dialogue.get_history(),
        }

    def load_narrative_context(self, data: Dict[str, Any]):
        """Restore narrative state from save."""
        if 'relationships' in data:
            self.relationships = RelationshipTracker.from_dict(data['relationships'])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_npcs(
    min_reputation: int = 0,
    max_reputation: int = 100
) -> List[Dict[str, Any]]:
    """Get list of available NPCs with basic info."""
    return [
        {
            'npc_id': npc_id,
            'name': profile.name,
            'role': profile.role,
            'company': profile.company,
        }
        for npc_id, profile in NPC_PROFILES.items()
    ]


def get_npc_details(npc_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed info about a specific NPC."""
    profile = get_npc_profile(npc_id)
    if profile:
        return profile.to_dict()
    return None


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing NarrativeEngine with MockLLMProvider...")

    engine = NarrativeEngine()

    # Test opening scenario
    print("\n--- Opening Scenario ---")
    opening = engine.generate_opening_scenario({
        'bank_account': 5000,
        'coding_skill': 20,
    })
    print(opening[:500] + "...")

    # Test NPC introduction
    print("\n--- NPC Introduction: Sarah Chen ---")
    intro = engine.generate_npc_introduction('sarah_chen')
    print(intro)

    # Test action outcome
    print("\n--- Action Outcome: code_sprint ---")
    outcome = engine.generate_action_outcome(
        'code_sprint',
        {'coding_skill_gain': 5, 'revenue': 0, 'stress_delta': 10},
        {'energy': 70, 'stress': 20, 'bank_account': 5000}
    )
    print(outcome)

    # Test dialogue
    print("\n--- Dialogue with Marcus Webb ---")
    dialogue = engine.start_npc_dialogue('marcus_webb')
    print(f"Greeting: {dialogue['greeting']}")
    print(f"Trust: {dialogue['trust_level']}")
    print(f"Choices: {[c['text'] for c in dialogue['choices']]}")

    print("\nNarrative system ready.")
