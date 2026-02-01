"""
LLM Provider Module - The Age of AI: 2026

Abstracts Ollama (gpt-oss-20b) integration with robust error handling.
Engine.py calls these methods without knowing implementation details.

CRITICAL: This module returns narrative text ONLY. It CANNOT modify game state.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
from decimal import Decimal
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ollama, provide graceful fallback if not installed
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama library not installed. OllamaClient will use fallbacks.")


# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_CONFIG = {
    'model': 'gpt-oss-20b',
    'base_url': 'http://localhost:11434',
    'timeout': 30,
    'max_retries': 3,
    'temperature': 0.7,
    'system_prompt': '''You are a narrative engine for "The Age of AI: 2026".

RULES:
1. You describe events but NEVER invent numbers or outcomes
2. All financial/stats changes are provided in context
3. Stay in 2026 near-future AI industry setting
4. Keep responses under 150 words
5. Character voice should match their profile
6. Never break character or mention you are an AI'''
}

FALLBACK_NARRATIVES = {
    'scenario': (
        "Welcome to Silicon Valley, 2026. The AI boom is in full swing. "
        "You're an ML engineer with big dreams and a modest bank account. "
        "The path to $20M starts with a single line of code."
    ),
    'dialogue': "[They nod thoughtfully, considering your words.]",
    'outcome': "The week passes. You check your accounts to see the results.",
    'market_news': "Market conditions shift as the AI sector continues its volatile growth.",
}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM provider errors."""
    pass


class ConnectionError(LLMError):
    """Failed to connect to Ollama server."""
    pass


class ModelNotFoundError(LLMError):
    """Requested model not available."""
    pass


class TimeoutError(LLMError):
    """Request timed out."""
    pass


class GenerationError(LLMError):
    """Error during text generation."""
    pass


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    Engine.py uses this interface - it doesn't know about Ollama.
    All methods return narrative text ONLY. No state modification.
    """

    @abstractmethod
    def generate_scenario(self, context: Dict[str, Any]) -> str:
        """
        Generate opening scenario text.

        Args:
            context: GameState as dict with player stats

        Returns:
            Narrative text describing the starting scenario
        """
        pass

    @abstractmethod
    def generate_npc_dialogue(
        self,
        npc_id: str,
        npc_profile: Dict[str, Any],
        player_history: List[Dict],
        current_context: Dict[str, Any],
        player_message: Optional[str] = None
    ) -> str:
        """
        Generate NPC response dialogue.

        Args:
            npc_id: Unique identifier for the NPC
            npc_profile: Character traits and background
            player_history: Previous interactions with this NPC
            current_context: Current game state
            player_message: What the player said (if any)

        Returns:
            Dialogue text for the NPC
        """
        pass

    @abstractmethod
    def generate_outcome_narrative(
        self,
        action: str,
        numerical_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate narrative description of action outcome.

        CRITICAL: numerical_result contains the ACTUAL math results.
        LLM only describes them, cannot modify.

        Args:
            action: The action taken (e.g., 'code_sprint', 'freelance')
            numerical_result: Dict with actual numbers from engine
            context: Current game state

        Returns:
            Narrative description of what happened
        """
        pass

    @abstractmethod
    def generate_market_news(self, market_state: Dict[str, Any]) -> str:
        """
        Generate news/flavor text for market phase.

        Args:
            market_state: Dict with market_sentiment, ai_hype_cycle, etc.

        Returns:
            News headline or market flavor text
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if LLM service is available."""
        pass


# =============================================================================
# OLLAMA CLIENT IMPLEMENTATION
# =============================================================================

class OllamaClient(LLMProvider):
    """
    Production LLM provider using Ollama (gpt-oss-20b).

    Features:
    - Connection pooling and reuse
    - Exponential backoff retry logic
    - Graceful fallbacks on failure
    - Timeout handling
    - Model availability checking
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: int = None,
        max_retries: int = None,
        temperature: float = None
    ):
        self.model = model or OLLAMA_CONFIG['model']
        self.base_url = base_url or OLLAMA_CONFIG['base_url']
        self.timeout = timeout or OLLAMA_CONFIG['timeout']
        self.max_retries = max_retries or OLLAMA_CONFIG['max_retries']
        self.temperature = temperature or OLLAMA_CONFIG['temperature']
        self.system_prompt = OLLAMA_CONFIG['system_prompt']

        self._client = None
        self._model_available = None  # Cache model check

        if OLLAMA_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """Initialize Ollama client with configured base URL."""
        try:
            self._client = ollama.Client(host=self.base_url)
            logger.info(f"Ollama client initialized: {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self._client = None

    def _with_retry(
        self,
        operation: Callable,
        fallback_key: str,
        operation_name: str = "LLM operation"
    ) -> str:
        """
        Execute operation with exponential backoff retry.

        Args:
            operation: Callable that performs the LLM call
            fallback_key: Key for fallback narrative if all retries fail
            operation_name: Description for logging

        Returns:
            Generated text or fallback narrative
        """
        for attempt in range(self.max_retries):
            try:
                return operation()
            except (ConnectionError, TimeoutError, ModelNotFoundError) as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"{operation_name} unexpected error: {e}")
                break

        # All retries exhausted or unexpected error
        logger.warning(f"{operation_name} using fallback after {self.max_retries} retries")
        return FALLBACK_NARRATIVES.get(fallback_key, FALLBACK_NARRATIVES['outcome'])

    def _call_ollama(self, prompt: str) -> str:
        """
        Make actual call to Ollama API.

        Args:
            prompt: Formatted prompt text

        Returns:
            Generated text

        Raises:
            ConnectionError: If cannot connect to server
            ModelNotFoundError: If model not available
            TimeoutError: If request times out
            GenerationError: If generation fails
        """
        if not OLLAMA_AVAILABLE:
            raise ConnectionError("ollama library not installed")

        if self._client is None:
            self._init_client()
            if self._client is None:
                raise ConnectionError("Ollama client not initialized")

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 200,  # Limit response length
                }
            )

            generated_text = response.get('response', '').strip()

            if not generated_text:
                raise GenerationError("Empty response from model")

            return generated_text

        except ollama.ResponseError as e:
            if 'not found' in str(e).lower() or e.status_code == 404:
                raise ModelNotFoundError(f"Model '{self.model}' not found: {e}")
            raise GenerationError(f"Ollama response error: {e}")

        except Exception as e:
            error_str = str(e).lower()
            if 'connection' in error_str or 'refused' in error_str:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")
            if 'timeout' in error_str:
                raise TimeoutError(f"Request timed out after {self.timeout}s: {e}")
            raise GenerationError(f"Unexpected error: {e}")

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format game state context for prompts."""
        return json.dumps(context, indent=2, default=str)

    def health_check(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        if not OLLAMA_AVAILABLE:
            return False

        if self._client is None:
            self._init_client()

        try:
            # Try a simple generation to verify connectivity
            self._client.generate(
                model=self.model,
                prompt="test",
                options={'num_predict': 1}
            )
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # INTERFACE IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    def generate_scenario(self, context: Dict[str, Any]) -> str:
        """Generate opening scenario."""
        def _generate():
            prompt = f"""Generate an engaging opening scenario for "The Age of AI: 2026".

Player starting context:
{self._format_context(context)}

Write a vivid 2-3 paragraph introduction that:
- Sets the scene in Silicon Valley, 2026
- Describes the AI boom atmosphere
- Introduces the player as an ambitious ML engineer
- Mentions their modest starting funds (${context.get('bank_account', 'unknown')})
- Hints at the path to $20M wealth

Keep it under 150 words. Be atmospheric and immersive."""

            return self._call_ollama(prompt)

        return self._with_retry(_generate, 'scenario', "Generate scenario")

    def generate_npc_dialogue(
        self,
        npc_id: str,
        npc_profile: Dict[str, Any],
        player_history: List[Dict],
        current_context: Dict[str, Any],
        player_message: Optional[str] = None
    ) -> str:
        """Generate NPC dialogue response."""
        def _generate():
            history_str = ""
            if player_history:
                recent = player_history[-3:]  # Last 3 exchanges
                history_str = "\nRecent conversation:\n" + "\n".join([
                    f"Player: {h.get('player', '')}\n{h.get('npc', '')}: {h.get('response', '')}"
                    for h in recent
                ])

            player_input = player_message or "[The player approaches you]"

            prompt = f"""You are {npc_profile.get('name', npc_id)}, {npc_profile.get('role', 'a character')}.

Your personality: {npc_profile.get('personality', 'professional and neutral')}
Your background: {npc_profile.get('background', 'Works in tech')}
Speaking style: {npc_profile.get('speaking_style', 'direct and concise')}

Current situation (AI industry, 2026):
{self._format_context(current_context)}{history_str}

Respond to: "{player_input}"

Provide ONLY the dialogue response, no stage directions or explanations.
Stay in character. 1-2 sentences maximum."""

            return self._call_ollama(prompt)

        return self._with_retry(_generate, 'dialogue', f"Generate dialogue for {npc_id}")

    def generate_outcome_narrative(
        self,
        action: str,
        numerical_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate narrative description of action outcome."""
        def _generate():
            # Build result description
            results = []
            if 'revenue' in numerical_result and numerical_result['revenue']:
                results.append(f"Earned: ${numerical_result['revenue']}")
            if 'coding_skill_gain' in numerical_result and numerical_result['coding_skill_gain']:
                results.append(f"Coding skill +{numerical_result['coding_skill_gain']}")
            if 'ml_expertise_gain' in numerical_result and numerical_result['ml_expertise_gain']:
                results.append(f"ML expertise +{numerical_result['ml_expertise_gain']}")
            if 'reputation_gain' in numerical_result and numerical_result['reputation_gain']:
                results.append(f"Reputation +{numerical_result['reputation_gain']}")
            if 'job_offer' in numerical_result and numerical_result['job_offer']:
                results.append(f"Job offer: {numerical_result.get('job_title', 'position')}")
            if 'side_project_completed' in numerical_result and numerical_result['side_project_completed']:
                results.append("Side project completed!")

            results_str = "\n".join(results) if results else "No significant changes"

            prompt = f"""Describe the outcome of a player's action in "The Age of AI: 2026".

Action taken: {action}
Player's current state:
{self._format_context(context)}

Actual results (YOU MUST REFERENCE THESE ACCURATELY):
{results_str}

Write 1-2 sentences describing what happened in narrative form.
- Describe the action and its result
- Mention financial/skill changes naturally
- Keep the 2026 Silicon Valley atmosphere
- NEVER invent numbers not listed above
- Be concise but immersive"""

            return self._call_ollama(prompt)

        return self._with_retry(_generate, 'outcome', f"Generate outcome for {action}")

    def generate_market_news(self, market_state: Dict[str, Any]) -> str:
        """Generate market phase news/flavor."""
        def _generate():
            sentiment = market_state.get('market_sentiment', 1.0)
            hype = market_state.get('ai_hype_cycle', 50)

            sentiment_desc = "bullish" if sentiment > 1.2 else "bearish" if sentiment < 0.8 else "stable"
            hype_desc = "peak hype" if hype > 80 else "cooling" if hype < 30 else "moderate interest"

            prompt = f"""Generate a news headline or market update for the AI sector in 2026.

Market conditions: {sentiment_desc} (sentiment: {sentiment:.2f})
AI hype level: {hype_desc} (hype: {hype})

Write a brief news snippet (under 50 words) about:
- A fictional AI company or product
- Market movements
- Industry trends

Keep it realistic for a near-future setting."""

            return self._call_ollama(prompt)

        return self._with_retry(_generate, 'market_news', "Generate market news")


# =============================================================================
# MOCK PROVIDER (for testing without Ollama)
# =============================================================================

class MockLLMProvider(LLMProvider):
    """
    Mock provider that returns templated responses.

    Useful for:
    - Unit testing without Ollama running
    - Development/debugging
    - CI/CD pipelines
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.call_history: List[Dict] = []

    def _record_call(self, method: str, **kwargs):
        """Record method call for testing verification."""
        self.call_count += 1
        self.call_history.append({'method': method, 'args': kwargs})

    def generate_scenario(self, context: Dict[str, Any]) -> str:
        """Return mock scenario."""
        self._record_call('generate_scenario', context=context)
        return self.responses.get('scenario', FALLBACK_NARRATIVES['scenario'])

    def generate_npc_dialogue(
        self,
        npc_id: str,
        npc_profile: Dict[str, Any],
        player_history: List[Dict],
        current_context: Dict[str, Any],
        player_message: Optional[str] = None
    ) -> str:
        """Return mock dialogue."""
        self._record_call(
            'generate_npc_dialogue',
            npc_id=npc_id,
            npc_profile=npc_profile,
            player_message=player_message
        )
        return self.responses.get(
            'dialogue',
            f"[{npc_id}]: {FALLBACK_NARRATIVES['dialogue']}"
        )

    def generate_outcome_narrative(
        self,
        action: str,
        numerical_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Return mock outcome."""
        self._record_call(
            'generate_outcome_narrative',
            action=action,
            numerical_result=numerical_result
        )

        # Build dynamic response based on results
        parts = [f"You spent the week {action.replace('_', ' ')}."]

        if 'revenue' in numerical_result and numerical_result['revenue']:
            parts.append(f"You earned ${numerical_result['revenue']}.")
        if 'job_offer' in numerical_result and numerical_result['job_offer']:
            parts.append(f"You received a job offer as {numerical_result.get('job_title', 'a developer')}!")
        if 'side_project_completed' in numerical_result and numerical_result['side_project_completed']:
            parts.append("Your side project is complete!")

        return " ".join(parts) if len(parts) > 1 else FALLBACK_NARRATIVES['outcome']

    def generate_market_news(self, market_state: Dict[str, Any]) -> str:
        """Return mock market news."""
        self._record_call('generate_market_news', market_state=market_state)
        return self.responses.get('market_news', FALLBACK_NARRATIVES['market_news'])

    def health_check(self) -> bool:
        """Always healthy."""
        return True


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_llm_provider(
    provider_type: str = 'ollama',
    **kwargs
) -> LLMProvider:
    """
    Factory function to create appropriate LLM provider.

    Args:
        provider_type: 'ollama' or 'mock'
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLMProvider instance
    """
    if provider_type == 'ollama':
        return OllamaClient(**kwargs)
    elif provider_type == 'mock':
        return MockLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    # Test with mock provider
    print("Testing MockLLMProvider:")
    mock = MockLLMProvider()

    test_context = {
        'bank_account': 5000,
        'net_worth': 5000,
        'energy': 100,
        'stress': 10,
        'turn_number': 1,
        'coding_skill': 20,
        'ml_expertise': 10,
    }

    print("\nScenario:", mock.generate_scenario(test_context))
    print("\nDialogue:", mock.generate_npc_dialogue(
        'mentor',
        {'name': 'Sarah', 'personality': 'wise and encouraging'},
        [],
        test_context,
        'Should I quit my job?'
    ))
    print("\nOutcome:", mock.generate_outcome_narrative(
        'code_sprint',
        {'revenue': 0, 'coding_skill_gain': 5, 'stress_delta': 10},
        test_context
    ))
    print("\nMarket news:", mock.generate_market_news({
        'market_sentiment': 1.3,
        'ai_hype_cycle': 75
    }))
    print(f"\nTotal calls: {mock.call_count}")

    # Test Ollama if available
    if OLLAMA_AVAILABLE:
        print("\n\nTesting OllamaClient (health check):")
        client = OllamaClient()
        healthy = client.health_check()
        print(f"Ollama healthy: {healthy}")

        if healthy:
            print("\nGenerating scenario with Ollama:")
            scenario = client.generate_scenario(test_context)
            print(scenario)
