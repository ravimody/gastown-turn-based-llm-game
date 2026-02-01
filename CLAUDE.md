# The Age of AI: 2026 - Architecture Guide

> **Turn-based strategy game where an ML engineer races to $20M net worth.**
>
> **Core Principle:** Game logic is HARD-CODED in Python. The LLM provides narrative flavor, not rule authority.

---

## Folder Structure

```
project-billionaire/
├── CLAUDE.md              # This document - architectural source of truth
├── engine.py              # Core game state and turn logic (UNHACKABLE)
├── llm_provider.py        # Ollama client abstraction
├── narrative.py           # Prompt templates and scenario generation
├── characters/
│   ├── __init__.py
│   ├── profiles.py        # NPC character definitions
│   └── dialogue.py        # Dialogue tree handlers
├── prompts/
│   ├── __init__.py
│   ├── scenarios/         # Dynamic start/end scenario templates
│   ├── npc/               # NPC-specific prompt templates
│   └── outcomes/          # Narrative outcome templates
├── data/
│   ├── savegames/         # JSON save files
│   └── events/            # Predefined event pools
├── tests/
│   └── test_engine.py     # State machine tests (CRITICAL)
└── main.py                # Entry point
```

---

## GameState State Machine

### Core Resources (Hard-Coded Math Only)

```python
@dataclass
class GameState:
    # Primary metrics - LLM CANNOT MODIFY DIRECTLY
    bank_account: Decimal      # Player's liquid cash
    net_worth: Decimal         # Total assets - liabilities
    energy: int                # 0-100, actions consume energy
    reputation: int            # 0-100, affects deal quality
    stress: int                # 0-100, high stress = penalties

    # Progress tracking
    turn_number: int           # Current turn (1 turn = 1 week)
    coding_skill: int          # 0-100, affects project quality
    ml_expertise: int          # 0-100, unlocks advanced projects

    # Market state (calculated each Market Phase)
    market_sentiment: float    # 0.5-2.0 multiplier on deals
    ai_hype_cycle: int         # 0-100, affects AI sector returns
    recession_risk: float      # 0.0-1.0, random event trigger
```

### Mathematical Constraints (Immutable)

```python
# engine.py - These are the laws of the game universe

ENERGY_RECOVERY_PER_TURN = 20          # Base recovery
MAX_ENERGY = 100
MIN_ENERGY = 0

BURNOUT_THRESHOLD = 85                 # Stress level for burnout check
BURNOUT_PENALTY = 0.5                  # 50% efficiency loss

MARKET_VOLATILITY = 0.15               # 15% standard deviation
HYPE_DECAY = 5                         # Hype drops 5pts/turn without news
HYPE_BOOST_NEWS = 15                   # News events boost hype

# Action costs (fixed, non-negotiable)
ACTION_COSTS = {
    'code_sprint': {'energy': 30, 'stress': 10},
    'side_project': {'energy': 20, 'stress': 5},
    'networking': {'energy': 15, 'stress': 5},
    'rest': {'energy': -40, 'stress': -20},  # Negative = recovery
    'apply_job': {'energy': 10, 'stress': 8},
}
```

### Turn Loop Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TURN START                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: MARKET PHASE (Deterministic Math)                      │
│  ─────────────────────────────────────────                       │
│  1. Calculate market_sentiment = base * random.normal(1, 0.15)   │
│  2. Decay ai_hype_cycle by HYPE_DECAY                            │
│  3. Check recession_risk against random() for event trigger      │
│  4. Apply passive income/expenses to bank_account                │
│  5. Regenerate energy: min(MAX_ENERGY, energy + ENERGY_RECOVERY) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: ACTION PHASE (Player Choice + Hard Math)               │
│  ────────────────────────────────────────────────                │
│  1. Present valid actions (energy >= action_cost)                │
│  2. Player selects action                                        │
│  3. Deduct energy and add stress per ACTION_COSTS                │
│  4. Calculate outcome using:                                     │
│     - Player stats (coding_skill, ml_expertise)                  │
│     - Market state (market_sentiment, ai_hype_cycle)             │
│     - Random factor (bounded, seeded)                            │
│  5. Update bank_account, stats, progress toward $20M             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: NARRATIVE OUTCOME (LLM Flavor Layer)                   │
│  ─────────────────────────────────────────────                   │
│  1. Gather context: action taken, numerical outcome, new state   │
│  2. Call llm_provider.generate_narrative(context)                │
│  3. LLM returns descriptive text (NO NUMBERS - cosmetic only)    │
│  4. Display narrative to player                                  │
│  5. Check win condition (net_worth >= 20_000_000)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TURN END                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interface Contracts

### Engine → LLM Provider

```python
# llm_provider.py interface - engine.py calls these:

class LLMProvider(ABC):
    @abstractmethod
    def generate_scenario(self, context: GameState) -> str:
        """Generate opening scenario. Returns narrative text only."""
        pass

    @abstractmethod
    def generate_npc_dialogue(
        self,
        npc_id: str,
        player_history: list,
        current_context: GameState
    ) -> str:
        """Generate NPC response. Returns dialogue text only."""
        pass

    @abstractmethod
    def generate_outcome_narrative(
        self,
        action: str,
        numerical_result: dict,  # {'revenue': 5000, 'stress_delta': 5}
        context: GameState
    ) -> str:
        """Generate narrative description of action outcome.

        CRITICAL: numerical_result contains the ACTUAL math results.
        LLM only describes them, cannot modify.
        """
        pass
```

### Engine → Narrative

```python
# narrative.py - Prompt template management

class NarrativeEngine:
    def load_prompt(self, template_name: str, context: dict) -> str:
        """Load and fill a Jinja2 prompt template."""
        pass

    def get_npc_profile(self, npc_id: str) -> dict:
        """Return character traits for consistent personality."""
        pass
```

---

## Ollama Integration

### Model Configuration

```python
# llm_provider.py

OLLAMA_CONFIG = {
    'model': 'gpt-oss-20b',
    'base_url': 'http://localhost:11434',
    'timeout': 30,
    'max_retries': 3,
    'temperature': 0.7,  # Creative but consistent
    'system_prompt': '''You are a narrative engine for "The Age of AI: 2026".

    RULES:
    1. You describe events but NEVER invent numbers or outcomes
    2. All financial/stats changes are provided in context
    3. Stay in 2026 near-future AI industry setting
    4. Keep responses under 150 words
    5. Character voice should match their profile'''
}
```

### Error Handling Strategy

```python
class OllamaClient(LLMProvider):
    def __init__(self):
        self.fallback_narratives = {
            'scenario': 'Welcome to Silicon Valley, 2026. The AI boom is in full swing...',
            'dialogue': '[Character nods thoughtfully]',
            'outcome': 'The week passes. You check your accounts to see the results.'
        }

    def _with_fallback(self, operation: Callable, fallback_key: str):
        """Execute LLM call with fallback on failure."""
        for attempt in range(MAX_RETRIES):
            try:
                return operation()
            except (ConnectionError, Timeout, ModelNotFound) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.warning(f"LLM failed, using fallback: {e}")
                    return self.fallback_narratives[fallback_key]
                time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Agent Work Boundaries

### agent-engine (engine.py)
- **OWNS:** GameState, turn loop, all mathematical calculations
- **MUST NOT:** Call LLM for state modifications
- **MUST:** Provide read-only state snapshots to narrative layer
- **EXPORTS:** GameState dataclass, turn() function, action handlers

### agent-llm (llm_provider.py)
- **OWNS:** Ollama communication, error handling, prompt formatting
- **MUST NOT:** Modify game state or invent numbers
- **MUST:** Return strings only, accept context dicts
- **EXPORTS:** OllamaClient class implementing LLMProvider interface

### agent-narrative (narrative.py, characters/, prompts/)
- **OWNS:** Character profiles, prompt templates, scenario generators
- **MUST NOT:** Hard-code any game logic or math
- **MUST:** Use Jinja2 templates, load from files
- **EXPORTS:** NarrativeEngine, character profile loaders

---

## Win/Lose Conditions

```python
# engine.py - Hard-coded victory conditions

VICTORY_NET_WORTH = 20_000_000  # $20M

GAME_OVER_CONDITIONS = {
    'bankruptcy': lambda s: s.bank_account < -50000,
    'burnout': lambda s: s.stress >= 100 and s.energy <= 0,
    'time_out': lambda s: s.turn_number > 520,  # 10 years
}
```

---

## Testing Requirements

```python
# tests/test_engine.py - CRITICAL: Test math is unhackable

def test_energy_never_exceeds_max():
    """Energy must be clamped at MAX_ENERGY."""

def test_negative_bank_triggers_bankruptcy():
    """Bank account < -50000 triggers game over."""

def test_llm_cannot_modify_state():
    """Mock LLM returning evil JSON - state must remain unchanged."""

def test_market_math_is_deterministic_with_seed():
    """Same seed produces same market conditions."""
```

---

## State Persistence

```python
# engine.py - Save/Load

def save_game(state: GameState, slot: int) -> None:
    """Serialize GameState to JSON."""

def load_game(slot: int) -> GameState:
    """Deserialize GameState from JSON."""
    # Validates all numeric bounds on load
```

---

**Remember:** The LLM is the storyteller. The Python code is the game master. Never confuse the two.
