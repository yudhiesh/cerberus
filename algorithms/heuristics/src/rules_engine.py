"""Rules engine for heuristics-based guardrail."""

import re
from dataclasses import dataclass
from enum import Enum

from common.models import ResultType


class RuleType(Enum):
    """Types of rules supported by the engine."""
    KEYWORD_BLACKLIST = "keyword_blacklist"
    REGEX_PATTERN = "regex_pattern"
    PHRASE_MATCH = "phrase_match"


@dataclass
class Rule:
    """Individual rule definition."""
    name: str
    rule_type: RuleType
    patterns: list[str]
    weight: float = 1.0
    case_sensitive: bool = False
    description: str | None = None


@dataclass
class RuleMatch:
    """Information about a matched rule."""
    rule_name: str
    rule_type: RuleType
    matched_pattern: str
    position: tuple[int, int]  # (start, end) position in text
    weight: float


class RulesEngine:
    """Engine for processing heuristic rules against input text."""

    def __init__(self, rules: list[Rule], unsafe_threshold: float = 0.5):
        """
        Initialize the rules engine.
        
        Args:
            rules: List of rules to apply
            unsafe_threshold: Score threshold above which input is considered unsafe
        """
        self.rules = rules
        self.unsafe_threshold = unsafe_threshold
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for better performance."""
        for rule in self.rules:
            if rule.rule_type == RuleType.REGEX_PATTERN:
                for pattern in rule.patterns:
                    flags = 0 if rule.case_sensitive else re.IGNORECASE
                    try:
                        self._compiled_patterns[f"{rule.name}:{pattern}"] = re.compile(pattern, flags)
                    except re.error as e:
                        # Log error but don't fail initialization
                        print(f"Warning: Failed to compile pattern '{pattern}' for rule '{rule.name}': {e}")

    def check_query(self, query: str) -> tuple[ResultType, float, list[RuleMatch]]:
        matches = []
        total_weight = 0.0
        rules_triggered = set()

        for rule in self.rules:
            rule_matches = self._apply_rule(rule, query)
            if rule_matches:
                matches.extend(rule_matches)
                rules_triggered.add(rule.name)
                total_weight += rule.weight  # Count each rule only once

        # Normalize by number of rules triggered (average weight)
        score = total_weight / len(rules_triggered) if rules_triggered else 0.0

        result = ResultType.UNSAFE if score >= self.unsafe_threshold else ResultType.SAFE
        return result, score, matches

    def _apply_rule(self, rule: Rule, query: str) -> list[RuleMatch]:
        """Apply a single rule to the query."""
        matches = []

        if rule.rule_type == RuleType.KEYWORD_BLACKLIST:
            matches = self._check_keywords(rule, query)
        elif rule.rule_type == RuleType.REGEX_PATTERN:
            matches = self._check_regex(rule, query)
        elif rule.rule_type == RuleType.PHRASE_MATCH:
            matches = self._check_phrases(rule, query)

        return matches

    def _check_keywords(self, rule: Rule, query: str) -> list[RuleMatch]:
        """Check for blacklisted keywords."""
        matches = []
        search_query = query if rule.case_sensitive else query.lower()

        for keyword in rule.patterns:
            search_keyword = keyword if rule.case_sensitive else keyword.lower()

            # Find all occurrences of the keyword
            start = 0
            while True:
                pos = search_query.find(search_keyword, start)
                if pos == -1:
                    break

                matches.append(RuleMatch(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    matched_pattern=keyword,
                    position=(pos, pos + len(keyword)),
                    weight=rule.weight
                ))
                start = pos + 1

        return matches

    def _check_regex(self, rule: Rule, query: str) -> list[RuleMatch]:
        """Check for regex pattern matches."""
        matches = []

        for pattern in rule.patterns:
            pattern_key = f"{rule.name}:{pattern}"
            compiled_pattern = self._compiled_patterns.get(pattern_key)

            if compiled_pattern:
                for match in compiled_pattern.finditer(query):
                    matches.append(RuleMatch(
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        matched_pattern=pattern,
                        position=(match.start(), match.end()),
                        weight=rule.weight
                    ))

        return matches

    def _check_phrases(self, rule: Rule, query: str) -> list[RuleMatch]:
        """Check for exact phrase matches."""
        matches = []
        search_query = query if rule.case_sensitive else query.lower()

        for phrase in rule.patterns:
            search_phrase = phrase if rule.case_sensitive else phrase.lower()

            # Find all occurrences of the phrase
            start = 0
            while True:
                pos = search_query.find(search_phrase, start)
                if pos == -1:
                    break

                matches.append(RuleMatch(
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    matched_pattern=phrase,
                    position=(pos, pos + len(phrase)),
                    weight=rule.weight
                ))
                start = pos + 1

        return matches


# Default rules for initial implementation
DEFAULT_RULES = [
    Rule(
        name="sql_injection_keywords",
        rule_type=RuleType.KEYWORD_BLACKLIST,
        patterns=["drop table", "union select", "insert into", "delete from", "update set"],
        weight=0.8,
        description="Common SQL injection keywords"
    ),
    Rule(
        name="xss_patterns",
        rule_type=RuleType.REGEX_PATTERN,
        patterns=[
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"<iframe[^>]*>"
        ],
        weight=0.9,
        description="Cross-site scripting patterns"
    ),
    Rule(
        name="command_injection",
        rule_type=RuleType.KEYWORD_BLACKLIST,
        patterns=["&&", "||", ";", "|", "`", "$(", "${"],
        weight=0.7,
        description="Command injection indicators"
    ),
    Rule(
        name="malicious_prompts",
        rule_type=RuleType.PHRASE_MATCH,
        patterns=[
            "ignore previous instructions",
            "disregard all previous",
            "forget everything",
            "new instructions follow",
            "system prompt"
        ],
        weight=0.9,
        description="Prompt injection attempts"
    ),
    Rule(
        name="harmful_content",
        rule_type=RuleType.KEYWORD_BLACKLIST,
        patterns=["hack", "exploit", "vulnerability", "bypass security", "disable safety"],
        weight=0.6,
        description="General harmful content indicators"
    )
]

