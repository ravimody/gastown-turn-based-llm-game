"""
Character system for The Age of AI: 2026.

NPC profiles, dialogue trees, and relationship tracking.
"""

from .profiles import (
    NPC_PROFILES,
    get_npc_profile,
    list_npcs,
    get_npc_by_role,
    RelationshipTracker,
)

from .dialogue import DialogueManager, DialogueNode

__all__ = [
    'NPC_PROFILES',
    'get_npc_profile',
    'list_npcs',
    'get_npc_by_role',
    'RelationshipTracker',
    'DialogueManager',
    'DialogueNode',
]
