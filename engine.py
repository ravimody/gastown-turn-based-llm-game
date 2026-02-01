"""
The Age of AI: 2026 - Game Engine

Core game state and turn logic. ALL MATH IS HARD-CODED.
The LLM provides narrative flavor but CANNOT modify game state.

This module is the single source of truth for:
- GameState dataclass and all game metrics
- Turn loop (Market → Action → Narrative phases)
- Mathematical calculations (deterministic, seeded)
- Win/lose conditions
- Save/load persistence
"""

from dataclasses import dataclass, asdict, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Callable, Dict, Any, List, Tuple
from enum import Enum, auto
import json
import random
import os
from pathlib import Path


# =============================================================================
# MATHEMATICAL CONSTANTS - THE LAWS OF THE GAME UNIVERSE
# These are immutable. No LLM can override these.
# =============================================================================

ENERGY_RECOVERY_PER_TURN = 20
MAX_ENERGY = 100
MIN_ENERGY = 0
MIN_STRESS = 0
MAX_STRESS = 100
MIN_REPUTATION = 0
MAX_REPUTATION = 100
MIN_SKILL = 0
MAX_SKILL = 100
MIN_HYPE = 0
MAX_HYPE = 100

BURNOUT_THRESHOLD = 85
BURNOUT_PENALTY = 0.5

MARKET_VOLATILITY = 0.15
MARKET_BASE_SENTIMENT = 1.0
MARKET_MIN_SENTIMENT = 0.5
MARKET_MAX_SENTIMENT = 2.0

HYPE_DECAY = 5
HYPE_BOOST_NEWS = 15

VICTORY_NET_WORTH = Decimal('20000000')  # $20M
BANKRUPTCY_THRESHOLD = Decimal('-50000')  # -$50k
MAX_TURNS = 520  # 10 years

# Living expenses per turn (1 week)
BASE_LIVING_EXPENSES = Decimal('2000')

# Passive income multipliers (applied to bank_account for investment returns)
PASSIVE_RETURN_RATE = Decimal('0.0005')  # 0.05% per week ~= 2.6% annually

# Action costs: energy and stress deltas (positive = cost, negative = recovery)
ACTION_COSTS = {
    'code_sprint': {'energy': 30, 'stress': 10},
    'side_project': {'energy': 20, 'stress': 5},
    'networking': {'energy': 15, 'stress': 5},
    'rest': {'energy': -40, 'stress': -20},
    'apply_job': {'energy': 10, 'stress': 8},
    'freelance': {'energy': 25, 'stress': 8},
    'study_ml': {'energy': 20, 'stress': 5},
    'interview': {'energy': 20, 'stress': 15},
}

# Revenue calculations by action type (base values modified by stats/market)
ACTION_REVENUE_BASE = {
    'code_sprint': Decimal('0'),      # No direct revenue, improves skills
    'side_project': Decimal('500'),   # Small potential payout
    'networking': Decimal('0'),       # No direct revenue, improves reputation
    'rest': Decimal('0'),             # Recovery only
    'apply_job': Decimal('0'),        # No direct revenue
    'freelance': Decimal('2000'),     # Immediate payment
    'study_ml': Decimal('0'),         # Skill improvement
    'interview': Decimal('0'),        # Potential job offer
}


# =============================================================================
# GAME STATE DATACLASS
# All fields have validation bounds. LLM cannot bypass these.
# =============================================================================

@dataclass
class GameState:
    """
    Complete game state. All numeric fields are bounded on init and update.

    CRITICAL: This class is READ-ONLY to the narrative layer.
    Only engine.py modifies these values through hard-coded math.
    """

    # Primary metrics - LLM CANNOT MODIFY DIRECTLY
    bank_account: Decimal = field(default_factory=lambda: Decimal('5000'))
    net_worth: Decimal = field(default_factory=lambda: Decimal('5000'))
    energy: int = MAX_ENERGY
    reputation: int = 50
    stress: int = 10

    # Progress tracking
    turn_number: int = 1
    coding_skill: int = 20
    ml_expertise: int = 10

    # Market state (calculated each Market Phase)
    market_sentiment: float = MARKET_BASE_SENTIMENT
    ai_hype_cycle: int = 50
    recession_risk: float = 0.05

    # Job/income state
    current_salary: Decimal = field(default_factory=lambda: Decimal('0'))
    job_title: Optional[str] = None
    side_project_progress: int = 0  # 0-100 completion

    # Turn history for narrative context
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    # Random seed for reproducibility
    _rng_seed: int = field(default_factory=lambda: random.randint(0, 2**32))

    def __post_init__(self):
        """Validate and clamp all values to valid ranges."""
        self._clamp_all_values()
        self._rng = random.Random(self._rng_seed)

    def _clamp_all_values(self):
        """Ensure all numeric fields are within valid bounds."""
        # Energy: 0-100
        self.energy = max(MIN_ENERGY, min(MAX_ENERGY, int(self.energy)))

        # Stress: 0-100
        self.stress = max(MIN_STRESS, min(MAX_STRESS, int(self.stress)))

        # Reputation: 0-100
        self.reputation = max(MIN_REPUTATION, min(MAX_REPUTATION, int(self.reputation)))

        # Skills: 0-100
        self.coding_skill = max(MIN_SKILL, min(MAX_SKILL, int(self.coding_skill)))
        self.ml_expertise = max(MIN_SKILL, min(MAX_SKILL, int(self.ml_expertise)))

        # Hype: 0-100
        self.ai_hype_cycle = max(MIN_HYPE, min(MAX_HYPE, int(self.ai_hype_cycle)))

        # Market sentiment: 0.5-2.0
        self.market_sentiment = max(MARKET_MIN_SENTIMENT,
                                    min(MARKET_MAX_SENTIMENT, float(self.market_sentiment)))

        # Recession risk: 0.0-1.0
        self.recession_risk = max(0.0, min(1.0, float(self.recession_risk)))

        # Side project progress: 0-100
        self.side_project_progress = max(0, min(100, int(self.side_project_progress)))

        # Ensure Decimal precision
        self.bank_account = Decimal(self.bank_account).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        self.net_worth = Decimal(self.net_worth).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        self.current_salary = Decimal(self.current_salary).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get_valid_actions(self) -> List[str]:
        """Return list of actions the player can afford given current energy."""
        return [
            action for action, costs in ACTION_COSTS.items()
            if self.energy >= costs['energy']
        ]

    def is_burned_out(self) -> bool:
        """Check if player is in burnout state (high stress + low energy)."""
        return self.stress >= BURNOUT_THRESHOLD and self.energy <= 30

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary (for JSON save)."""
        data = asdict(self)
        # Convert Decimal to string for JSON serialization
        data['bank_account'] = str(self.bank_account)
        data['net_worth'] = str(self.net_worth)
        data['current_salary'] = str(self.current_salary)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """Deserialize state from dictionary."""
        # Convert string back to Decimal
        data['bank_account'] = Decimal(data['bank_account'])
        data['net_worth'] = Decimal(data['net_worth'])
        data['current_salary'] = Decimal(data.get('current_salary', '0'))

        # Remove _rng if present (will be recreated)
        data.pop('_rng', None)

        return cls(**data)


# =============================================================================
# GAME ENGINE CLASS
# Handles turn logic, calculations, and state transitions.
# =============================================================================

class GameEngine:
    """
    Main game engine. Coordinates turn phases and enforces game rules.

    All mathematical operations are deterministic given a seed.
    The LLM layer receives read-only state snapshots.
    """

    def __init__(self, state: Optional[GameState] = None, llm_provider=None):
        self.state = state or GameState()
        self.llm_provider = llm_provider  # Optional narrative layer
        self._game_over = False
        self._victory = False
        self._game_over_reason: Optional[str] = None

    # -------------------------------------------------------------------------
    # TURN LOOP
    # -------------------------------------------------------------------------

    def take_turn(self, action: str) -> Dict[str, Any]:
        """
        Execute one complete turn.

        Args:
            action: The player's chosen action (must be in ACTION_COSTS)

        Returns:
            Dict with turn results including state changes and narrative
        """
        if self._game_over:
            raise GameOverError(f"Game is over: {self._game_over_reason}")

        if action not in ACTION_COSTS:
            raise InvalidActionError(f"Unknown action: {action}")

        if action not in self.state.get_valid_actions():
            raise InsufficientEnergyError(f"Not enough energy for {action}")

        # Store turn start state for diff calculation
        start_state = self.state.to_dict()

        # Phase 1: Market Phase
        self._market_phase()

        # Phase 2: Action Phase
        numerical_result = self._action_phase(action)

        # Phase 3: Check game over conditions
        self._check_game_over()

        # Phase 4: Narrative Outcome (if LLM provider available)
        narrative = None
        if self.llm_provider and not self._game_over:
            narrative = self._narrative_phase(action, numerical_result)

        # Record action in history
        turn_record = {
            'turn': self.state.turn_number - 1,  # Already incremented in market phase
            'action': action,
            'numerical_result': numerical_result,
            'narrative': narrative,
        }
        self.state.action_history.append(turn_record)

        return {
            'action': action,
            'state_changes': self._calculate_changes(start_state, self.state.to_dict()),
            'numerical_result': numerical_result,
            'narrative': narrative,
            'game_over': self._game_over,
            'victory': self._victory,
            'game_over_reason': self._game_over_reason,
        }

    def _market_phase(self):
        """
        Phase 1: Market Phase - Deterministic calculations only.

        1. Update market sentiment
        2. Decay AI hype
        3. Check for recession events
        4. Apply passive income/expenses
        5. Regenerate energy
        6. Advance turn counter
        """
        rng = self.state._rng

        # 1. Calculate market sentiment (random walk with bounds)
        sentiment_change = rng.gauss(0, MARKET_VOLATILITY)
        self.state.market_sentiment += sentiment_change
        self.state.market_sentiment = max(MARKET_MIN_SENTIMENT,
                                          min(MARKET_MAX_SENTIMENT, self.state.market_sentiment))

        # 2. Decay AI hype cycle
        self.state.ai_hype_cycle -= HYPE_DECAY
        if self.state.ai_hype_cycle < MIN_HYPE:
            self.state.ai_hype_cycle = MIN_HYPE

        # 3. Check for recession event
        if rng.random() < self.state.recession_risk:
            # Recession hits: sentiment drops, hype falls
            self.state.market_sentiment *= 0.7
            self.state.ai_hype_cycle = max(MIN_HYPE, self.state.ai_hype_cycle - 30)
            self.state.recession_risk = 0.02  # Reset risk after event
        else:
            # Gradually increase recession risk over time
            self.state.recession_risk = min(1.0, self.state.recession_risk + 0.01)

        # 4. Apply passive income/expenses
        # Living expenses (always deducted)
        self.state.bank_account -= BASE_LIVING_EXPENSES

        # Salary income (if employed)
        if self.state.current_salary > 0:
            weekly_salary = self.state.current_salary / 52
            self.state.bank_account += weekly_salary

        # Investment returns on positive balance
        if self.state.bank_account > 0:
            returns = self.state.bank_account * PASSIVE_RETURN_RATE
            self.state.bank_account += returns

        # 5. Regenerate energy
        recovery = ENERGY_RECOVERY_PER_TURN
        if self.state.stress > 50:
            # High stress reduces recovery
            recovery = int(recovery * 0.7)
        self.state.energy = min(MAX_ENERGY, self.state.energy + recovery)

        # 6. Advance turn
        self.state.turn_number += 1

        # Recalculate net worth
        self._update_net_worth()

        # Validate all bounds
        self.state._clamp_all_values()

    def _action_phase(self, action: str) -> Dict[str, Any]:
        """
        Phase 2: Action Phase - Execute player action with hard-coded math.

        Returns:
            Dict with numerical results of the action
        """
        costs = ACTION_COSTS[action]
        rng = self.state._rng

        # Deduct energy and add stress
        self.state.energy -= costs['energy']
        self.state.stress += costs['stress']

        # Apply burnout penalty if applicable
        efficiency = 1.0
        if self.state.is_burned_out():
            efficiency = BURNOUT_PENALTY

        # Calculate action outcome based on type
        result = {
            'revenue': Decimal('0'),
            'coding_skill_gain': 0,
            'ml_expertise_gain': 0,
            'reputation_gain': 0,
            'stress_delta': costs['stress'],
            'energy_cost': costs['energy'],
            'burnout_applied': self.state.is_burned_out(),
        }

        if action == 'code_sprint':
            # Improves coding skill, small chance of freelance income
            skill_gain = int(5 * efficiency * (1 + self.state.market_sentiment * 0.1))
            self.state.coding_skill += skill_gain
            result['coding_skill_gain'] = skill_gain

            # 20% chance of finding a quick freelance gig
            if rng.random() < 0.2:
                revenue = Decimal('500') * Decimal(str(self.state.market_sentiment))
                revenue = revenue.quantize(Decimal('0.01'))
                self.state.bank_account += revenue
                result['revenue'] = revenue
                result['freelance_found'] = True

        elif action == 'side_project':
            # Builds side project progress
            progress_gain = int(10 * efficiency * (1 + self.state.coding_skill / 100))
            self.state.side_project_progress += progress_gain
            result['side_project_progress_gain'] = progress_gain

            # If project completes, payout based on market conditions
            if self.state.side_project_progress >= 100:
                base_payout = Decimal('5000')
                hype_multiplier = 1 + (self.state.ai_hype_cycle / 100)
                market_multiplier = Decimal(str(self.state.market_sentiment))
                payout = base_payout * Decimal(str(hype_multiplier)) * market_multiplier
                payout = payout.quantize(Decimal('0.01'))

                self.state.bank_account += payout
                self.state.side_project_progress = 0
                result['side_project_completed'] = True
                result['side_project_payout'] = payout
                result['revenue'] = payout

                # Reputation boost from successful project
                self.state.reputation += 5
                result['reputation_gain'] = 5

        elif action == 'networking':
            # Improves reputation, chance of job lead
            rep_gain = int(3 * efficiency * (1 + self.state.market_sentiment * 0.2))
            self.state.reputation += rep_gain
            result['reputation_gain'] = rep_gain

            # Chance of AI hype news
            if rng.random() < 0.3:
                self.state.ai_hype_cycle += HYPE_BOOST_NEWS
                result['hype_boost'] = True

            # 15% chance of job offer if not employed or seeking better
            if rng.random() < 0.15 and self.state.reputation > 40:
                result['job_lead'] = True

        elif action == 'rest':
            # Major stress reduction (handled by negative stress cost)
            # Bonus energy recovery beyond base
            bonus_recovery = int(10 * efficiency)
            self.state.energy = min(MAX_ENERGY, self.state.energy + bonus_recovery)
            result['bonus_recovery'] = bonus_recovery

        elif action == 'apply_job':
            # Apply for jobs based on skills and reputation
            success_chance = 0.3 + (self.state.coding_skill / 200) + (self.state.reputation / 200)
            success_chance *= efficiency

            if rng.random() < success_chance:
                # Got a job offer
                base_salary = Decimal('80000')
                skill_bonus = Decimal(str(self.state.coding_skill * 500))
                rep_bonus = Decimal(str(self.state.reputation * 200))
                salary = base_salary + skill_bonus + rep_bonus
                salary = salary.quantize(Decimal('0.01'))

                self.state.current_salary = salary
                self.state.job_title = self._generate_job_title()
                result['job_offer'] = True
                result['salary'] = salary
                result['job_title'] = self.state.job_title

        elif action == 'freelance':
            # Immediate income based on skills and market
            base_rate = Decimal('100')
            skill_multiplier = 1 + (self.state.coding_skill / 50)
            market_mult = Decimal(str(self.state.market_sentiment))
            hype_mult = 1 + (self.state.ai_hype_cycle / 200)

            revenue = base_rate * Decimal(str(skill_multiplier)) * market_mult * Decimal(str(hype_mult))
            revenue = revenue * Decimal('20')  # Weekly freelance income
            revenue = revenue.quantize(Decimal('0.01'))

            self.state.bank_account += revenue
            result['revenue'] = revenue

            # Small skill improvement
            skill_gain = int(1 * efficiency)
            self.state.coding_skill += skill_gain
            result['coding_skill_gain'] = skill_gain

        elif action == 'study_ml':
            # Improves ML expertise
            ml_gain = int(4 * efficiency * (1 + self.state.coding_skill / 150))
            self.state.ml_expertise += ml_gain
            result['ml_expertise_gain'] = ml_gain

            # High ML expertise can generate consulting income
            if self.state.ml_expertise > 50:
                consulting_revenue = Decimal(str(self.state.ml_expertise * 10))
                consulting_revenue = consulting_revenue.quantize(Decimal('0.01'))
                self.state.bank_account += consulting_revenue
                result['revenue'] = consulting_revenue
                result['consulting_income'] = True

        elif action == 'interview':
            # Higher-stakes job application for better positions
            if self.state.ml_expertise < 30:
                result['interview_failed'] = True
                result['failure_reason'] = 'insufficient_ml_expertise'
                self.state.stress += 10  # Extra stress from failure
            else:
                success_chance = 0.2 + (self.state.ml_expertise / 200) + (self.state.reputation / 200)
                success_chance *= efficiency

                if rng.random() < success_chance:
                    # High-level ML position
                    base_salary = Decimal('150000')
                    ml_bonus = Decimal(str(self.state.ml_expertise * 1000))
                    salary = base_salary + ml_bonus
                    salary = salary.quantize(Decimal('0.01'))

                    self.state.current_salary = salary
                    self.state.job_title = self._generate_ml_job_title()
                    result['job_offer'] = True
                    result['salary'] = salary
                    result['job_title'] = self.state.job_title
                    result['ml_position'] = True
                else:
                    result['interview_failed'] = True
                    result['failure_reason'] = 'random_chance'

        # Apply bounds after all modifications
        self.state._clamp_all_values()
        self._update_net_worth()

        return result

    def _narrative_phase(self, action: str, numerical_result: Dict[str, Any]) -> Optional[str]:
        """
        Phase 3: Narrative Outcome - Call LLM for flavor text.

        The LLM receives context but CANNOT modify state.
        """
        if not self.llm_provider:
            return None

        try:
            return self.llm_provider.generate_outcome_narrative(
                action=action,
                numerical_result=numerical_result,
                context=self.state
            )
        except Exception as e:
            # LLM failure doesn't break the game - return fallback
            return f"The week passes. You check your accounts to see the results. ({action} completed)"

    def _check_game_over(self):
        """Check win/lose conditions."""
        # Victory condition
        if self.state.net_worth >= VICTORY_NET_WORTH:
            self._game_over = True
            self._victory = True
            self._game_over_reason = "victory"
            return

        # Bankruptcy
        if self.state.bank_account < BANKRUPTCY_THRESHOLD:
            self._game_over = True
            self._game_over_reason = "bankruptcy"
            return

        # Burnout
        if self.state.stress >= MAX_STRESS and self.state.energy <= MIN_ENERGY:
            self._game_over = True
            self._game_over_reason = "burnout"
            return

        # Time out (10 years)
        if self.state.turn_number > MAX_TURNS:
            self._game_over = True
            self._game_over_reason = "time_out"
            return

    def _update_net_worth(self):
        """Recalculate net worth from all assets."""
        # For now, net worth = bank account + side project value
        side_project_value = Decimal('0')
        if self.state.side_project_progress > 0:
            # Partial projects have some value
            side_project_value = Decimal(str(self.state.side_project_progress * 50))

        self.state.net_worth = self.state.bank_account + side_project_value
        self.state.net_worth = self.state.net_worth.quantize(Decimal('0.01'))

    def _generate_job_title(self) -> str:
        """Generate a job title based on skills."""
        if self.state.coding_skill < 30:
            return "Junior Developer"
        elif self.state.coding_skill < 60:
            return "Software Engineer"
        elif self.state.coding_skill < 80:
            return "Senior Engineer"
        else:
            return "Staff Engineer"

    def _generate_ml_job_title(self) -> str:
        """Generate an ML-specific job title."""
        if self.state.ml_expertise < 40:
            return "ML Engineer"
        elif self.state.ml_expertise < 70:
            return "Senior ML Engineer"
        elif self.state.ml_expertise < 90:
            return "Staff ML Engineer"
        else:
            return "Principal AI Researcher"

    def _calculate_changes(self, start: Dict, end: Dict) -> Dict[str, Any]:
        """Calculate delta between two states."""
        changes = {}
        for key in start:
            if key in end and key != 'action_history':
                if start[key] != end[key]:
                    changes[key] = {
                        'from': start[key],
                        'to': end[key],
                        'delta': self._compute_delta(start[key], end[key])
                    }
        return changes

    def _compute_delta(self, old, new):
        """Compute numeric delta between two values."""
        try:
            return new - old
        except TypeError:
            return None

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def get_state(self) -> GameState:
        """Return read-only state snapshot."""
        # Return a copy to prevent external modification
        return GameState.from_dict(self.state.to_dict())

    def is_game_over(self) -> bool:
        return self._game_over

    def is_victory(self) -> bool:
        return self._victory

    def get_game_over_reason(self) -> Optional[str]:
        return self._game_over_reason

    def get_valid_actions(self) -> List[str]:
        """Get list of actions player can currently take."""
        return self.state.get_valid_actions()

    def get_turn_summary(self) -> Dict[str, Any]:
        """Get summary of current turn state."""
        return {
            'turn': self.state.turn_number,
            'bank_account': self.state.bank_account,
            'net_worth': self.state.net_worth,
            'energy': self.state.energy,
            'stress': self.state.stress,
            'reputation': self.state.reputation,
            'coding_skill': self.state.coding_skill,
            'ml_expertise': self.state.ml_expertise,
            'job_title': self.state.job_title,
            'salary': self.state.current_salary,
            'market_sentiment': round(self.state.market_sentiment, 2),
            'ai_hype_cycle': self.state.ai_hype_cycle,
        }


# =============================================================================
# SAVE / LOAD
# =============================================================================

SAVE_DIR = Path("data/savegames")


def ensure_save_dir():
    """Ensure save directory exists."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_game(state: GameState, slot: int = 1) -> Path:
    """Serialize GameState to JSON file."""
    ensure_save_dir()
    save_path = SAVE_DIR / f"save_{slot}.json"

    data = state.to_dict()
    data['_save_version'] = '1.0'
    data['_save_timestamp'] = str(random.randint(0, 2**32))  # Placeholder for timestamp

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    return save_path


def load_game(slot: int = 1) -> Optional[GameState]:
    """Deserialize GameState from JSON file."""
    save_path = SAVE_DIR / f"save_{slot}.json"

    if not save_path.exists():
        return None

    with open(save_path, 'r') as f:
        data = json.load(f)

    state = GameState.from_dict(data)
    state._clamp_all_values()  # Validate on load
    return state


def list_saves() -> List[int]:
    """List available save slots."""
    if not SAVE_DIR.exists():
        return []

    slots = []
    for f in SAVE_DIR.glob("save_*.json"):
        try:
            slot = int(f.stem.split('_')[1])
            slots.append(slot)
        except (IndexError, ValueError):
            continue

    return sorted(slots)


def delete_save(slot: int) -> bool:
    """Delete a save file."""
    save_path = SAVE_DIR / f"save_{slot}.json"
    if save_path.exists():
        save_path.unlink()
        return True
    return False


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GameOverError(Exception):
    """Raised when attempting to play after game over."""
    pass


class InvalidActionError(Exception):
    """Raised when invalid action is specified."""
    pass


class InsufficientEnergyError(Exception):
    """Raised when player lacks energy for action."""
    pass


# =============================================================================
# NEW GAME FACTORY
# =============================================================================

def new_game(seed: Optional[int] = None) -> GameEngine:
    """Create a new game with default starting state."""
    state = GameState(
        bank_account=Decimal('5000'),
        net_worth=Decimal('5000'),
        energy=MAX_ENERGY,
        reputation=50,
        stress=10,
        turn_number=1,
        coding_skill=20,
        ml_expertise=10,
        _rng_seed=seed if seed is not None else random.randint(0, 2**32)
    )
    return GameEngine(state=state)


# =============================================================================
# MAIN ENTRY (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test
    engine = new_game(seed=42)
    print("Initial state:", engine.get_turn_summary())

    # Take a few turns
    for turn in range(5):
        valid = engine.get_valid_actions()
        action = valid[0] if valid else 'rest'
        result = engine.take_turn(action)
        print(f"\nTurn {turn + 1}: {action}")
        print(f"  State changes: {result['state_changes']}")
        print(f"  Numerical result: {result['numerical_result']}")
