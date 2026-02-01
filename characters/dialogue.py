"""
Dialogue tree system for NPC interactions.

Manages conversation flow, choices, and consequences.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto


class DialogueType(Enum):
    """Types of dialogue nodes."""
    GREETING = auto()      # Initial greeting
    INFO = auto()          # Information sharing
    QUEST = auto()         # Task/quest offering
    TRADE = auto()         # Transaction
    CONFRONT = auto()      # Difficult conversation
    FAREWELL = auto()      # Exit dialogue


@dataclass
class DialogueChoice:
    """A player choice in dialogue."""
    text: str                              # What player says
    next_node: Optional[str] = None        # ID of next node
    trust_change: int = 0                  # Trust impact
    condition: Optional[Callable] = None   # Prerequisite check
    action: Optional[str] = None           # Game action triggered


@dataclass
class DialogueNode:
    """A single node in a dialogue tree."""
    node_id: str
    npc_id: str
    text: str                              # NPC response text
    node_type: DialogueType = DialogueType.INFO
    choices: List[DialogueChoice] = field(default_factory=list)
    condition: Optional[Callable] = None   # Visibility condition
    one_time: bool = False                 # Can only be seen once
    seen: bool = False                     # Has been viewed
    trust_change: int = 0                  # Trust impact for visiting this node


class DialogueManager:
    """
    Manages dialogue trees for all NPCs.

    Coordinates with llm_provider.py for dynamic responses
    while maintaining structured conversation flow.
    """

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self._trees: Dict[str, Dict[str, DialogueNode]] = {}
        self._current_node: Optional[str] = None
        self._current_npc: Optional[str] = None
        self._conversation_history: List[Dict] = []

        # Initialize default trees
        self._init_default_trees()

    def _init_default_trees(self):
        """Set up default dialogue trees for each NPC."""
        from .profiles import NPC_PROFILES

        for npc_id in NPC_PROFILES:
            self._trees[npc_id] = self._create_tree_for_npc(npc_id)

    def _create_tree_for_npc(self, npc_id: str) -> Dict[str, DialogueNode]:
        """Create a basic dialogue tree for an NPC."""
        from .profiles import get_npc_profile

        profile = get_npc_profile(npc_id)
        if not profile:
            return {}

        # Create standard tree structure
        tree = {
            'greeting': DialogueNode(
                node_id='greeting',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} greets the player]",
                node_type=DialogueType.GREETING,
                choices=[
                    DialogueChoice("Ask for advice", 'advice'),
                    DialogueChoice("Discuss industry news", 'news'),
                    DialogueChoice("Request help with a project", 'help'),
                    DialogueChoice("Say goodbye", 'farewell'),
                ]
            ),
            'advice': DialogueNode(
                node_id='advice',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} offers career advice]",
                node_type=DialogueType.INFO,
                choices=[
                    DialogueChoice("Thanks for the advice", 'greeting'),
                    DialogueChoice("I should go", 'farewell'),
                ]
            ),
            'news': DialogueNode(
                node_id='news',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} discusses recent AI developments]",
                node_type=DialogueType.INFO,
                choices=[
                    DialogueChoice("Tell me more", 'news_deep'),
                    DialogueChoice("Back to other topics", 'greeting'),
                ]
            ),
            'news_deep': DialogueNode(
                node_id='news_deep',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} shares insider perspective]",
                node_type=DialogueType.INFO,
                trust_change=2,  # Sharing builds trust
                choices=[
                    DialogueChoice("Interesting, thanks", 'greeting'),
                ]
            ),
            'help': DialogueNode(
                node_id='help',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} considers helping]",
                node_type=DialogueType.QUEST,
                choices=[
                    DialogueChoice("Yes, I'd appreciate that", 'help_yes', trust_change=5),
                    DialogueChoice("Maybe another time", 'greeting'),
                ]
            ),
            'help_yes': DialogueNode(
                node_id='help_yes',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} agrees to help]",
                node_type=DialogueType.QUEST,
                one_time=True,
                choices=[
                    DialogueChoice("Thank you!", 'greeting'),
                ]
            ),
            'farewell': DialogueNode(
                node_id='farewell',
                npc_id=npc_id,
                text=f"[Dynamic: {profile.name} says goodbye]",
                node_type=DialogueType.FAREWELL,
                choices=[]
            ),
        }

        return tree

    def start_dialogue(self, npc_id: str) -> Optional[DialogueNode]:
        """Begin a conversation with an NPC."""
        if npc_id not in self._trees:
            return None

        self._current_npc = npc_id
        self._current_node = 'greeting'
        self._conversation_history = []

        return self.get_current_node()

    def get_current_node(self) -> Optional[DialogueNode]:
        """Get the current dialogue node."""
        if not self._current_npc or not self._current_node:
            return None

        tree = self._trees.get(self._current_npc, {})
        node = tree.get(self._current_node)

        if node:
            node.seen = True

        return node

    def make_choice(self, choice_index: int) -> Optional[DialogueNode]:
        """Select a dialogue choice."""
        node = self.get_current_node()
        if not node or choice_index >= len(node.choices):
            return None

        choice = node.choices[choice_index]

        # Record interaction
        self._conversation_history.append({
            'node': self._current_node,
            'player_choice': choice.text,
            'trust_change': choice.trust_change,
        })

        # Move to next node
        self._current_node = choice.next_node

        return self.get_current_node()

    def get_available_choices(self) -> List[DialogueChoice]:
        """Get current available choices."""
        node = self.get_current_node()
        if not node:
            return []

        # Filter by conditions
        return [
            choice for choice in node.choices
            if choice.condition is None or choice.condition()
        ]

    def is_dialogue_active(self) -> bool:
        """Check if dialogue is in progress."""
        return (
            self._current_npc is not None and
            self._current_node is not None and
            self._current_node != 'farewell'
        )

    def end_dialogue(self):
        """Force end current dialogue."""
        self._current_npc = None
        self._current_node = None

    def get_history(self) -> List[Dict]:
        """Get conversation history for LLM context."""
        return self._conversation_history.copy()

    def generate_dynamic_response(
        self,
        npc_id: str,
        player_input: str,
        game_context: Dict[str, Any]
    ) -> str:
        """
        Generate dynamic NPC response using LLM.

        Falls back to static tree if LLM unavailable.
        """
        from .profiles import get_npc_profile

        if not self.llm_provider:
            return "[They consider your words carefully.]"

        profile = get_npc_profile(npc_id)
        if not profile:
            return "[No response]"

        return self.llm_provider.generate_npc_dialogue(
            npc_id=npc_id,
            npc_profile=profile.to_dict(),
            player_history=self._conversation_history,
            current_context=game_context,
            player_message=player_input
        )


# =============================================================================
# NPC-SPECIFIC DIALOGUE TREES
# =============================================================================

def create_marcus_pitch_tree() -> Dict[str, DialogueNode]:
    """Special pitch dialogue tree for Marcus Webb (investor)."""
    return {
        'pitch': DialogueNode(
            node_id='pitch',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus evaluates your pitch]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice(
                    "I'm building the next generation of AI infrastructure",
                    'pitch_tech',
                    trust_change=5
                ),
                DialogueChoice(
                    "This is a $10B market opportunity",
                    'pitch_market',
                    trust_change=3
                ),
                DialogueChoice(
                    "I'm still exploring ideas",
                    'pitch_weak',
                    trust_change=-5
                ),
            ]
        ),
        'pitch_tech': DialogueNode(
            node_id='pitch_tech',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus is intrigued by the technical angle]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Here's my technical deep-dive", 'pitch_success'),
                DialogueChoice("Let me tell you about the team", 'pitch_team'),
            ]
        ),
        'pitch_market': DialogueNode(
            node_id='pitch_market',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus considers the market size]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Competition is weak right now", 'pitch_success'),
                DialogueChoice("We have first-mover advantage", 'pitch_success'),
            ]
        ),
        'pitch_weak': DialogueNode(
            node_id='pitch_weak',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus looks disappointed]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Actually, I have a specific plan...", 'pitch_recovery'),
            ]
        ),
        'pitch_recovery': DialogueNode(
            node_id='pitch_recovery',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus gives you another chance]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Here's what I'm really building...", 'pitch_tech'),
            ]
        ),
        'pitch_success': DialogueNode(
            node_id='pitch_success',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus offers investment terms]",
            node_type=DialogueType.TRADE,
            one_time=True,
            choices=[
                DialogueChoice("Accept the terms", 'pitch_accept', trust_change=10),
                DialogueChoice("Negotiate better terms", 'pitch_negotiate'),
            ]
        ),
        'pitch_accept': DialogueNode(
            node_id='pitch_accept',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus seals the deal]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Let's build something great", 'farewell'),
            ]
        ),
        'pitch_negotiate': DialogueNode(
            node_id='pitch_negotiate',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus counters]",
            node_type=DialogueType.TRADE,
            choices=[
                DialogueChoice("Deal", 'pitch_accept'),
                DialogueChoice("I need to think about it", 'farewell'),
            ]
        ),
        'farewell': DialogueNode(
            node_id='farewell',
            npc_id='marcus_webb',
            text="[Dynamic: Marcus ends the meeting]",
            node_type=DialogueType.FAREWELL,
            choices=[]
        ),
    }


def create_sarah_mentorship_tree() -> Dict[str, DialogueNode]:
    """Special mentorship tree for Sarah Chen."""
    return {
        'mentorship': DialogueNode(
            node_id='mentorship',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah considers mentoring you]",
            node_type=DialogueType.QUEST,
            choices=[
                DialogueChoice(
                    "I want to learn about AI safety",
                    'safety_path',
                    trust_change=10
                ),
                DialogueChoice(
                    "I want to build something profitable",
                    'profit_path',
                    trust_change=-5
                ),
                DialogueChoice(
                    "I want to understand the fundamentals",
                    'fundamentals_path',
                    trust_change=5
                ),
            ]
        ),
        'safety_path': DialogueNode(
            node_id='safety_path',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah is pleased]",
            node_type=DialogueType.INFO,
            choices=[
                DialogueChoice("What's the biggest risk?", 'safety_risks'),
                DialogueChoice("How do I get started?", 'safety_start'),
            ]
        ),
        'profit_path': DialogueNode(
            node_id='profit_path',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah is skeptical but listens]",
            node_type=DialogueType.INFO,
            choices=[
                DialogueChoice(
                    "Profit can fund safety research",
                    'safety_path',
                    trust_change=5
                ),
            ]
        ),
        'fundamentals_path': DialogueNode(
            node_id='fundamentals_path',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah approves of your approach]",
            node_type=DialogueType.INFO,
            choices=[
                DialogueChoice("Recommend papers to read", 'resources'),
                DialogueChoice("Who should I talk to?", 'networking'),
            ]
        ),
        'safety_risks': DialogueNode(
            node_id='safety_risks',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah explains existential risks]",
            node_type=DialogueType.INFO,
            one_time=True,
            choices=[
                DialogueChoice("That's terrifying", 'safety_commit'),
                DialogueChoice("That seems far-fetched", 'safety_skeptic'),
            ]
        ),
        'safety_commit': DialogueNode(
            node_id='safety_commit',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah offers guidance]",
            node_type=DialogueType.QUEST,
            choices=[
                DialogueChoice("I'm in. What do I do?", 'mentorship_start'),
            ]
        ),
        'safety_skeptic': DialogueNode(
            node_id='safety_skeptic',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah is patient but firm]",
            node_type=DialogueType.INFO,
            choices=[
                DialogueChoice("Maybe I'm wrong", 'safety_commit'),
                DialogueChoice("Let's agree to disagree", 'farewell'),
            ]
        ),
        'mentorship_start': DialogueNode(
            node_id='mentorship_start',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah becomes your mentor]",
            node_type=DialogueType.QUEST,
            one_time=True,
            choices=[
                DialogueChoice("Thank you for believing in me", 'farewell'),
            ]
        ),
        'resources': DialogueNode(
            node_id='resources',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah recommends resources]",
            node_type=DialogueType.INFO,
            one_time=True,
            choices=[
                DialogueChoice("I'll read these", 'greeting'),
            ]
        ),
        'networking': DialogueNode(
            node_id='networking',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah offers introductions]",
            node_type=DialogueType.INFO,
            one_time=True,
            choices=[
                DialogueChoice("That would be amazing", 'greeting'),
            ]
        ),
        'farewell': DialogueNode(
            node_id='farewell',
            npc_id='sarah_chen',
            text="[Dynamic: Sarah says goodbye]",
            node_type=DialogueType.FAREWELL,
            choices=[]
        ),
    }
