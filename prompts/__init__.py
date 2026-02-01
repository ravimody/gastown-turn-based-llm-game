"""
Prompt template system for The Age of AI: 2026.

Jinja2-based templates for dynamic narrative generation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, BaseLoader, DictLoader

# Template directories
PROMPT_DIR = Path(__file__).parent
SCENARIOS_DIR = PROMPT_DIR / 'scenarios'
NPC_DIR = PROMPT_DIR / 'npc'
OUTCOMES_DIR = PROMPT_DIR / 'outcomes'


class PromptEngine:
    """
    Jinja2-based prompt template engine.

    Loads and renders templates for LLM generation.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or PROMPT_DIR

        # Set up Jinja2 environment
        if self.template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            # Fallback to dict loader with inline templates
            self.env = Environment(loader=DictLoader(DEFAULT_TEMPLATES))

        # Register custom filters
        self.env.filters['currency'] = self._format_currency
        self.env.filters['percent'] = self._format_percent

    def _format_currency(self, value) -> str:
        """Format number as currency."""
        try:
            return f"${int(float(value)):,}"
        except (ValueError, TypeError):
            return f"${value}"

    def _format_percent(self, value) -> str:
        """Format number as percentage."""
        try:
            return f"{float(value):.1f}%"
        except (ValueError, TypeError):
            return f"{value}%"

    def load_template(self, template_name: str) -> Optional[Any]:
        """Load a Jinja2 template by name."""
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            # Try with .j2 extension
            try:
                return self.env.get_template(f"{template_name}.j2")
            except Exception:
                return None

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with context."""
        template = self.load_template(template_name)
        if template:
            return template.render(**context)

        # Fallback to inline template
        if template_name in DEFAULT_TEMPLATES:
            return self.env.from_string(DEFAULT_TEMPLATES[template_name]).render(**context)

        return f"[Template '{template_name}' not found]"


# =============================================================================
# INLINE FALLBACK TEMPLATES
# Used when template files don't exist
# =============================================================================

DEFAULT_TEMPLATES = {
    # Scenarios
    'scenarios/opening.txt': '''
# Opening Scenario: The Age of AI: 2026

Welcome to {{ location | default('Silicon Valley') }}, {{ year | default('2026') }}.

You are {{ player_name | default('an ambitious ML engineer') }} with ${{ bank_account | default('5000') | currency }} to your name.

The AI boom is in full swing. Every week brings new billion-dollar valuations, new breakthrough models, new opportunities—and new ways to fail.

Your goal: reach ${{ goal | default('20000000') | currency }} net worth.

{{ hype_description | default('The hype cycle is at a fever pitch. Investors are throwing money at anything with "AI" in the pitch deck.') }}

What will you do first?
''',

    # NPC Dialogue
    'npc/greeting.txt': '''
{{ npc_name }} ({{ npc_role }} at {{ npc_company }}) regards you.

{% if trust_level > 50 %}
They seem genuinely pleased to see you.
{% elif trust_level > 20 %}
They nod politely, recognising you.
{% else %}
They maintain a professional distance.
{% endif %}

Personality: {{ personality }}
Speaking style: {{ speaking_style }}

Respond as {{ npc_name }} would.
''',

    # Action Outcomes
    'outcomes/code_sprint.txt': '''
Describe a coding sprint session.

Player context:
- Energy level: {{ energy }}%
- Stress level: {{ stress }}%
- Coding skill: {{ coding_skill }}
- Current project: {{ project | default('a new AI tool') }}

Results:
{% if coding_skill_gain > 0 %}
- Coding skill improved by {{ coding_skill_gain }} points
{% endif %}
{% if freelance_found %}
- Found a quick freelance gig earning ${{ revenue }}
{% endif %}
{% if burnout_applied %}
- Note: Player is burned out, efficiency was reduced
{% endif %}

Write 1-2 sentences describing the session—late nights, breakthroughs, frustrations, the glow of monitors.
''',

    'outcomes/side_project.txt': '''
Describe progress on a side project.

Player has been working on their own AI project in spare time.

Progress: {{ side_project_progress }}% complete
{% if side_project_completed %}
MILESTONE: Project completed! Payout: ${{ side_project_payout | currency }}
Reputation increased by {{ reputation_gain }}.
{% endif %}

Capture the indie hacker spirit—building in public, shipping features, the uncertainty of entrepreneurship.
''',

    'outcomes/networking.txt': '''
Describe a networking event in 2026 Silicon Valley.

Player attended {{ event_type | default('a tech meetup') }}.

{% if reputation_gain > 0 %}
Made valuable connections. Reputation +{{ reputation_gain }}.
{% endif %}
{% if hype_boost %}
Heard breaking AI news—hype cycle accelerating.
{% endif %}
{% if job_lead %}
Discovered a promising job opportunity.
{% endif %}

The scene: craft cocktails, startup T-shirts, the constant pitch, the eternal hustle.
''',

    'outcomes/rest.txt': '''
Describe a week of rest and recovery.

Player stepped back to recharge.

Energy recovered to {{ energy }}%.
Stress reduced to {{ stress }}%.
{% if burnout_applied %}
Despite burnout, some recovery achieved.
{% endif %}

What did they do? Sleep? Exercise? Video games? Time with family?
Capture the rare moments of peace in startup life.
''',

    'outcomes/freelance.txt': '''
Describe freelance work completed this week.

Revenue: ${{ revenue | currency }}
{% if coding_skill_gain > 0 %}
Skill improvement: +{{ coding_skill_gain }} coding
{% endif %}

The gig economy of 2026—quick contracts, demanding clients, immediate pay.
The freedom and precarity of freelance life.
''',

    'outcomes/apply_job.txt': '''
Describe a job application process.

{% if job_offer %}
SUCCESS: Offer received!
Position: {{ job_title }}
Salary: ${{ salary | currency }} / year
{% else %}
Application submitted. Waiting to hear back.
The anxiety of the job search—resume polishing, algorithm interviews, the silence between communications.
{% endif %}
''',

    'outcomes/interview.txt': '''
Describe a job interview experience.

{% if job_offer %}
SUCCESS: High-level position offered!
Role: {{ job_title }}
Compensation: ${{ salary | currency }} / year
ML position: Senior technical track.
{% elif interview_failed %}
Rejection. {{ failure_reason | default('Not the right fit') }}.
The sting of rejection, the questioning of worth, the resolve to try again.
{% endif %}

The whiteboard coding, the system design questions, the culture fit interviews.
''',

    'outcomes/study_ml.txt': '''
Describe a week of intensive ML study.

ML expertise gained: +{{ ml_expertise_gain }}
{% if consulting_income %}
Applied knowledge generated consulting income: ${{ revenue | currency }}
{% endif %}

Papers read, notebooks filled, models trained.
The deepening expertise that separates engineers from researchers.
''',

    # Market News
    'market/bullish.txt': '''
Generate bullish AI market news (2026).

Sentiment: {{ sentiment | percent }}
Hype level: {{ hype | percent }}

Possible angles:
- New model release from major lab
- Billion-dollar funding round
- Regulatory approval for AI product
- Breakthrough research paper

Keep it realistic for near-future tech.
''',

    'market/bearish.txt': '''
Generate bearish AI market news (2026).

Sentiment: {{ sentiment | percent }}
Hype level: {{ hype | percent }}

Possible angles:
- Regulatory crackdown announced
- Major AI safety incident
- Market correction
- Layoffs at prominent startup

Maintain 2026 realism—challenges but not dystopia.
''',

    # Character Introductions
    'characters/intro.txt': '''
Introduce {{ npc_name }} to the player.

Role: {{ role }}
Company: {{ company }}
Personality: {{ personality }}
Background: {{ background }}

First impression—how do they come across?
What vibe do they give off?
2-3 sentences maximum.
''',
}


# Global prompt engine instance
_engine: Optional[PromptEngine] = None


def get_prompt_engine() -> PromptEngine:
    """Get or create the global prompt engine."""
    global _engine
    if _engine is None:
        _engine = PromptEngine()
    return _engine


def render_template(template_name: str, context: Dict[str, Any]) -> str:
    """Convenience function to render a template."""
    return get_prompt_engine().render(template_name, context)
