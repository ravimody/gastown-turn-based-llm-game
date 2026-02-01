"""
Tests for game engine - CRITICAL: Verify math is unhackable
"""

import pytest
from decimal import Decimal
from engine import (
    GameState, GameEngine, new_game, save_game, load_game,
    MAX_ENERGY, MIN_ENERGY, MAX_STRESS, VICTORY_NET_WORTH,
    BANKRUPTCY_THRESHOLD, MAX_TURNS, ACTION_COSTS,
    GameOverError, InvalidActionError, InsufficientEnergyError
)


class TestGameState:
    """Test GameState validation and bounds."""

    def test_energy_never_exceeds_max(self):
        """Energy must be clamped at MAX_ENERGY."""
        state = GameState(energy=150)
        assert state.energy == MAX_ENERGY

    def test_energy_never_below_min(self):
        """Energy must be clamped at MIN_ENERGY."""
        state = GameState(energy=-50)
        assert state.energy == MIN_ENERGY

    def test_stress_bounds(self):
        """Stress must be clamped to 0-100."""
        state = GameState(stress=150)
        assert state.stress == MAX_STRESS

        state = GameState(stress=-10)
        assert state.stress == 0

    def test_reputation_bounds(self):
        """Reputation must be clamped to 0-100."""
        state = GameState(reputation=150)
        assert state.reputation == 100

        state = GameState(reputation=-10)
        assert state.reputation == 0

    def test_skill_bounds(self):
        """Skills must be clamped to 0-100."""
        state = GameState(coding_skill=150, ml_expertise=-10)
        assert state.coding_skill == 100
        assert state.ml_expertise == 0

    def test_market_sentiment_bounds(self):
        """Market sentiment must be clamped to 0.5-2.0."""
        state = GameState(market_sentiment=5.0)
        assert state.market_sentiment == 2.0

        state = GameState(market_sentiment=0.1)
        assert state.market_sentiment == 0.5

    def test_bank_account_decimal_precision(self):
        """Bank account uses Decimal with 2 decimal places."""
        state = GameState(bank_account=Decimal('100.999'))
        assert state.bank_account == Decimal('101.00')

    def test_serialization_roundtrip(self):
        """State can be serialized and deserialized."""
        original = GameState(
            bank_account=Decimal('12345.67'),
            energy=75,
            coding_skill=50,
            _rng_seed=42
        )
        data = original.to_dict()
        restored = GameState.from_dict(data)

        assert restored.bank_account == original.bank_account
        assert restored.energy == original.energy
        assert restored.coding_skill == original.coding_skill


class TestGameEngine:
    """Test game engine turn logic."""

    def test_new_game_initial_state(self):
        """New game has correct initial values."""
        engine = new_game(seed=42)
        state = engine.get_state()

        assert state.bank_account == Decimal('5000')
        assert state.energy == MAX_ENERGY
        assert state.turn_number == 1

    def test_market_phase_advances_turn(self):
        """Market phase advances turn counter."""
        engine = new_game(seed=42)
        initial_turn = engine.get_state().turn_number

        engine.take_turn('rest')

        assert engine.get_state().turn_number == initial_turn + 1

    def test_action_consumes_energy(self):
        """Actions consume energy according to ACTION_COSTS."""
        engine = new_game(seed=42)
        initial_energy = engine.get_state().energy

        engine.take_turn('code_sprint')

        expected_energy = initial_energy - ACTION_COSTS['code_sprint']['energy']
        assert engine.get_state().energy == expected_energy

    def test_insufficient_energy_raises_error(self):
        """Cannot take action without sufficient energy."""
        engine = new_game(seed=42)
        # Exhaust energy
        engine.state.energy = 5
        engine.state._clamp_all_values()

        with pytest.raises(InsufficientEnergyError):
            engine.take_turn('code_sprint')  # Requires 30 energy

    def test_invalid_action_raises_error(self):
        """Invalid action raises InvalidActionError."""
        engine = new_game(seed=42)

        with pytest.raises(InvalidActionError):
            engine.take_turn('invalid_action')

    def test_rest_recoveries_energy_and_reduces_stress(self):
        """Rest action recovers energy and reduces stress."""
        engine = new_game(seed=42)
        engine.state.energy = 50
        engine.state.stress = 50
        engine.state._clamp_all_values()

        engine.take_turn('rest')

        assert engine.get_state().energy > 50
        assert engine.get_state().stress < 50

    def test_game_over_prevents_further_actions(self):
        """Cannot take actions after game over."""
        engine = new_game(seed=42)
        # Force bankruptcy
        engine.state.bank_account = BANKRUPTCY_THRESHOLD - Decimal('1')
        engine.state._clamp_all_values()

        # Take any action to trigger game over check
        with pytest.raises(GameOverError):
            engine.take_turn('rest')

    def test_victory_condition(self):
        """Reach $20M net worth triggers victory."""
        engine = new_game(seed=42)
        engine.state.bank_account = VICTORY_NET_WORTH
        engine.state._update_net_worth()
        engine.state._clamp_all_values()

        result = engine.take_turn('rest')

        assert result['game_over'] is True
        assert result['victory'] is True
        assert result['game_over_reason'] == 'victory'

    def test_bankruptcy_condition(self):
        """Bank account below threshold triggers bankruptcy."""
        engine = new_game(seed=42)
        engine.state.bank_account = BANKRUPTCY_THRESHOLD - Decimal('1')
        engine.state._clamp_all_values()

        result = engine.take_turn('rest')

        assert result['game_over'] is True
        assert result['victory'] is False
        assert result['game_over_reason'] == 'bankruptcy'

    def test_burnout_condition(self):
        """Max stress + zero energy triggers burnout."""
        engine = new_game(seed=42)
        engine.state.stress = MAX_STRESS
        engine.state.energy = MIN_ENERGY
        engine.state._clamp_all_values()

        result = engine.take_turn('rest')

        assert result['game_over'] is True
        assert result['game_over_reason'] == 'burnout'

    def test_time_out_condition(self):
        """Exceeding MAX_TURNS triggers time out."""
        engine = new_game(seed=42)
        engine.state.turn_number = MAX_TURNS + 1
        engine.state._clamp_all_values()

        result = engine.take_turn('rest')

        assert result['game_over'] is True
        assert result['game_over_reason'] == 'time_out'

    def test_burnout_reduces_efficiency(self):
        """Burnout state reduces action effectiveness."""
        engine = new_game(seed=42)
        engine.state.stress = 90
        engine.state.energy = 20
        engine.state._clamp_all_values()

        result = engine.take_turn('code_sprint')

        assert result['numerical_result']['burnout_applied'] is True


class TestSaveLoad:
    """Test save/load functionality."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Game can be saved and loaded."""
        import engine
        original_save_dir = engine.SAVE_DIR
        engine.SAVE_DIR = tmp_path

        try:
            original = new_game(seed=42)
            original.take_turn('code_sprint')

            save_path = save_game(original.get_state(), slot=1)
            loaded = load_game(slot=1)

            assert loaded is not None
            assert loaded.bank_account == original.get_state().bank_account
            assert loaded.energy == original.get_state().energy
        finally:
            engine.SAVE_DIR = original_save_dir

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading non-existent save returns None."""
        import engine
        original_save_dir = engine.SAVE_DIR
        engine.SAVE_DIR = tmp_path

        try:
            result = load_game(slot=999)
            assert result is None
        finally:
            engine.SAVE_DIR = original_save_dir


class TestDeterminism:
    """Test that game logic is deterministic with same seed."""

    def test_same_seed_produces_same_market(self):
        """Same seed produces identical market conditions."""
        engine1 = new_game(seed=42)
        engine2 = new_game(seed=42)

        # Take same actions
        for _ in range(5):
            engine1.take_turn('rest')
            engine2.take_turn('rest')

        # Market conditions should match
        assert engine1.get_state().market_sentiment == engine2.get_state().market_sentiment
        assert engine1.get_state().ai_hype_cycle == engine2.get_state().ai_hype_cycle

    def test_different_seeds_produce_different_markets(self):
        """Different seeds produce different market conditions."""
        engine1 = new_game(seed=42)
        engine2 = new_game(seed=999)

        # Take same actions
        for _ in range(10):
            engine1.take_turn('rest')
            engine2.take_turn('rest')

        # Market conditions should differ (with high probability)
        assert engine1.get_state().market_sentiment != engine2.get_state().market_sentiment


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
