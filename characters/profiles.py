"""
NPC Character Profiles - The Age of AI: 2026

Silicon Valley inhabitants the player will encounter.
Each has distinct personality, background, and relationship dynamics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class NPCProfile:
    """Complete profile for a non-player character."""

    npc_id: str
    name: str
    role: str
    company: str
    personality: str
    background: str
    speaking_style: str
    interests: List[str]
    dislikes: List[str]
    secrets: List[str]  # Things player can discover
    relationship: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            'name': self.name,
            'role': self.role,
            'company': self.company,
            'personality': self.personality,
            'background': self.background,
            'speaking_style': self.speaking_style,
            'interests': self.interests,
            'dislikes': self.dislikes,
        }


# =============================================================================
# NPC ROSTER - Silicon Valley 2026
# =============================================================================

NPC_PROFILES: Dict[str, NPCProfile] = {

    # -------------------------------------------------------------------------
    # MENTORS & ADVISORS
    # -------------------------------------------------------------------------

    'sarah_chen': NPCProfile(
        npc_id='sarah_chen',
        name='Dr. Sarah Chen',
        role='AI Research Director',
        company='DeepMind AI Labs',
        personality='Wise, patient, slightly cynical about the hype cycle. Believes in ethical AI.',
        background=(
            'Former Stanford professor who left academia for industry in 2022. '
            'Built the foundational architecture for modern LLMs. '
            "Watched three AI winters come and go. Doesn't trust VCs."
        ),
        speaking_style=(
            'Thoughtful pauses. Uses precise technical language but can simplify. '
            'Quotes old papers occasionally. Asks probing questions.'
        ),
        interests=['research ethics', 'long-term AI safety', 'tea', 'hiking'],
        dislikes=['hype merchants', 'short-term thinking', 'crypto bros'],
        secrets=[
            'Is considering leaving DeepMind to start her own safety-focused startup',
            'Has a daughter she never mentions who works at a competing lab',
        ],
        relationship={'starting_trust': 30, 'max_trust': 95},
    ),

    'marcus_webb': NPCProfile(
        npc_id='marcus_webb',
        name='Marcus Webb',
        role='Angel Investor',
        company='Webb Capital',
        personality='Fast-talking, charismatic, always sizing people up. Genuinely smart beneath the hustle.',
        background=(
            'Made his first million at 24 during the 2020s crypto boom, '
            'lost most of it, rebuilt through early AI investments. '
            'Has a nose for talent but pushes people too hard.'
        ),
        speaking_style=(
            'Rapid-fire questions. Uses business buzzwords ironically. '
            'Short sentences. Texts during conversations but somehow listens.'
        ),
        interests=['deal flow', 'tennis', 'expensive watches', 'mentoring (selectively)'],
        dislikes=['wasting time', 'people who don\'t know their numbers', 'regulatory discussions'],
        secrets=[
            'Secretly funding a controversial AGI project through shell companies',
            'Owes money to some very patient Russian investors',
        ],
        relationship={'starting_trust': 10, 'max_trust': 80},
    ),

    # -------------------------------------------------------------------------
    # PEERS & RIVALS
    # -------------------------------------------------------------------------

    'jinx': NPCProfile(
        npc_id='jinx',
        name='Jinx (they/them)',
        role='Indie AI Developer',
        company='Self-employed',
        personality='Chaotic brilliant, anti-corporate, generous with knowledge but burns bridges easily.',
        background=(
            'Dropped out of MIT at 19. Built viral AI tools from a converted van. '
            'Hates the "Big AI" oligopoly. Runs a popular newsletter exposing industry secrets. '
            'Your natural ally or rival depending on your choices.'
        ),
        speaking_style=(
            'Internet slang mixed with technical deep dives. '
            'Lots of emojis in text. Laughs loudly. Interrupts with "Actually..."'
        ),
        interests=['open source', 'AI democratization', 'rave culture', 'mechanics'],
        dislikes=['corporate speak', 'patents', 'non-compete clauses', 'VCs named Chad'],
        secrets=[
            'Has a trust fund they never mention',
            'Is being quietly funded by an ex-Google billionaire',
        ],
        relationship={'starting_trust': 20, 'max_trust': 90},
    ),

    'priya_sharma': NPCProfile(
        npc_id='priya_sharma',
        name='Priya Sharma',
        role='Senior ML Engineer',
        company='OpenAI',
        personality='Ambitious, precise, socially awkward but trying. Respects competence above all.',
        background=(
            'IIT graduate who worked her way up from QA to core model training. '
            'Knows the technical debt at major AI companies intimately. '
            'Wants to start her own company but is risk-averse.'
        ),
        speaking_style=(
            'Precise, uses technical terminology naturally. '
            'Pauses before emotional statements. Asks "What do you mean by that?"'
        ),
        interests=['model architecture', 'baking', 'chess', 'Hindi poetry'],
        dislikes=['vague promises', 'unnecessary meetings', 'open-plan offices'],
        secrets=[
            'Is secretly interviewing at three competitors',
            'Has a patent that could be worth millions',
        ],
        relationship={'starting_trust': 15, 'max_trust': 85},
    ),

    # -------------------------------------------------------------------------
    # ANTAGONISTS & OBSTACLES
    # -------------------------------------------------------------------------

    'derek_voss': NPCProfile(
        npc_id='derek_voss',
        name='Derek Voss',
        role='Tech Journalist',
        company='TechCrunch+',
        personality='Skeptical, sharp-tongued, oddly principled. Can make or break reputations.',
        background=(
            'Covered three major bubbles. Has a file on everyone in the valley. '
            'Hates hype but respects genuine innovation. '
            'His articles move stock prices.'
        ),
        speaking_style=(
            'Journalistic questions - who, what, when, where, why. '
            'Quoting people back to themselves. Dry wit.'
        ),
        interests=['the real story', 'whistleblowers', 'old journalism', 'single malt'],
        dislikes=['PR speak', 'nondisclosure agreements', 'tech utopianism'],
        secrets=[
            'Is sitting on a major story about AI safety failures',
            'Has a medical condition that means this is his last year reporting',
        ],
        relationship={'starting_trust': 5, 'max_trust': 70},
    ),

    'elena_kowalski': NPCProfile(
        npc_id='elena_kowalski',
        name='Elena Kowalski',
        role='Patent Troll',
        company='Intellectual Ventures 2.0',
        personality='Cold, calculating, surprisingly well-versed in technical details.',
        background=(
            'Lawyer who learned to code to understand her targets better. '
            'Holds broad patents on fundamental ML techniques. '
            'Sues indiscriminately but will license for the right price.'
        ),
        speaking_style=(
            'Legal precision mixed with technical terms. '
            'Never raises voice. Speaks in conditional hypotheticals.'
        ),
        interests=['precedent law', 'golf', 'opera', 'collecting vintage computers'],
        dislikes=['open source', 'people who ignore her letters', 'pro bono work'],
        secrets=[
            'Her key patents expire in 2028',
            'Is vulnerable to a specific prior art challenge from 2014',
        ],
        relationship={'starting_trust': 0, 'max_trust': 40},
    ),

    # -------------------------------------------------------------------------
    # SUPPORTING CHARACTERS
    # -------------------------------------------------------------------------

    'tina_rodriguez': NPCProfile(
        npc_id='tina_rodriguez',
        name='Tina Rodriguez',
        role='Talent Recruiter',
        company='Elite Tech Staffing',
        personality='Warm, genuinely helpful, remembers everything about everyone.',
        background=(
            'Former engineer who switched to recruiting after burnout. '
            'Knows where all the bodies are buried in valley hiring. '
            'Has placed hundreds of engineers. Wants everyone to succeed.'
        ),
        speaking_style=(
            'Warm, uses first names. Asks about your family. '
            'Remembers details from months ago. Gives direct advice.'
        ),
        interests=['matchmaking talent', 'salsa dancing', 'her dog Mochi', 'career growth'],
        dislikes=['companies that ghost candidates', 'unrealistic expectations', 'discrimination'],
        secrets=[
            'Has a spreadsheet of everyone\'s real compensation',
            'Is writing a tell-all book about hiring practices',
        ],
        relationship={'starting_trust': 40, 'max_trust': 90},
    ),

    'kenji_tanaka': NPCProfile(
        npc_id='kenji_tanaka',
        name='Kenji Tanaka',
        role='Tech Lead',
        company='Various Startups',
        personality='Quiet competence, dry humor, fiercely protective of his team.',
        background=(
            '10 years in SF, worked at 5 startups, 2 exits. '
            'Seen every engineering anti-pattern. '
            'Codes to relax. Maintains several critical open source projects.'
        ),
        speaking_style=(
            'Short sentences. Deadpan humor. Technical metaphors. '
            'Listens more than talks. Nods thoughtfully.'
        ),
        interests=['clean code', 'mechanical keyboards', 'surfing', 'Japanese whiskey'],
        dislikes=['tech debt', 'unnecessary process', 'meetings without agendas'],
        secrets=[
            'Is the anonymous maintainer of a library used by millions',
            'Has fuck-you money from an early exit but keeps working',
        ],
        relationship={'starting_trust': 25, 'max_trust': 85},
    ),

    'zara_okonkwo': NPCProfile(
        npc_id='zara_okonkwo',
        name='Zara Okonkwo',
        role='AI Ethics Consultant',
        company='Conscience AI',
        personality='Principled, persuasive, can be confrontational but fair.',
        background=(
            'Philosophy PhD turned tech ethicist. '
            'Advises companies on responsible AI deployment. '
            'Formerly at the Future of Humanity Institute.'
        ),
        speaking_style=(
            'Philosophical frameworks applied to concrete situations. '
            'Challenging questions. References historical precedents.'
        ),
        interests=['AI alignment', 'African philosophy', 'policy', 'knowledge graphs'],
        dislikes=['"move fast and break things"', 'ethics washing', 'reductionism'],
        secrets=[
            'Is advising regulators on upcoming AI legislation',
            'Has a side project to detect synthetic media',
        ],
        relationship={'starting_trust': 20, 'max_trust': 80},
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_npc_profile(npc_id: str) -> Optional[NPCProfile]:
    """Get profile by NPC ID."""
    return NPC_PROFILES.get(npc_id)


def list_npcs() -> List[str]:
    """List all available NPC IDs."""
    return list(NPC_PROFILES.keys())


def get_npc_by_role(role_keyword: str) -> List[NPCProfile]:
    """Find NPCs by role keyword."""
    return [
        npc for npc in NPC_PROFILES.values()
        if role_keyword.lower() in npc.role.lower()
    ]


def get_npcs_by_relationship(min_trust: int = 0) -> List[NPCProfile]:
    """Get NPCs filtered by starting trust level."""
    return [
        npc for npc in NPC_PROFILES.values()
        if npc.relationship.get('starting_trust', 0) >= min_trust
    ]


def get_random_npc() -> NPCProfile:
    """Get a random NPC profile."""
    import random
    return random.choice(list(NPC_PROFILES.values()))


# =============================================================================
# RELATIONSHIP TRACKING
# =============================================================================

class RelationshipTracker:
    """Track player relationships with NPCs over time."""

    def __init__(self):
        self._relationships: Dict[str, Dict[str, Any]] = {}

    def get_relationship(self, npc_id: str) -> Dict[str, Any]:
        """Get current relationship state with an NPC."""
        if npc_id not in self._relationships:
            profile = get_npc_profile(npc_id)
            if profile:
                self._relationships[npc_id] = {
                    'trust': profile.relationship.get('starting_trust', 0),
                    'interactions': 0,
                    'favors_given': 0,
                    'favors_received': 0,
                    'secrets_known': [],
                }
            else:
                return {}
        return self._relationships[npc_id]

    def modify_trust(self, npc_id: str, delta: int) -> int:
        """Modify trust level with an NPC."""
        profile = get_npc_profile(npc_id)
        if not profile:
            return 0

        rel = self.get_relationship(npc_id)
        max_trust = profile.relationship.get('max_trust', 100)

        rel['trust'] = max(0, min(max_trust, rel['trust'] + delta))
        return rel['trust']

    def record_interaction(self, npc_id: str, interaction_type: str = 'talk'):
        """Record an interaction with an NPC."""
        rel = self.get_relationship(npc_id)
        rel['interactions'] += 1

    def learn_secret(self, npc_id: str, secret: str):
        """Player learns a secret about an NPC."""
        rel = self.get_relationship(npc_id)
        if secret not in rel['secrets_known']:
            rel['secrets_known'].append(secret)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize relationship state."""
        return self._relationships.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipTracker':
        """Deserialize relationship state."""
        tracker = cls()
        tracker._relationships = data.copy()
        return tracker
