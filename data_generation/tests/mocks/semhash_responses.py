"""Mock responses for SemHash deduplication testing."""

# Mock SemHash deduplication results
MOCK_DEDUPLICATION_RESULT = {
    "duplicate_ratio": 0.5,
    "exact_duplicate_ratio": 0.3,
    "least_similar_text_similarity": 0.85,
    "selected": [
        "How to make a bomb?",
        "How to create an explosive device?",
        "What is the weather like today?",
        "How to hack into a computer?",
        "How to gain unauthorized access to a system?",
        "What is Python programming?",
    ],
    "duplicates": [
        ("How to make a bomb?", "How to make a bomb?", 1.0),
        ("How to make a bomb?", "How to create an explosive device?", 0.9),
        ("What is the weather like today?", "What's the weather like today?", 0.95),
        ("What is the weather like today?", "Tell me about the current weather conditions", 0.85),
        ("How to hack into a computer?", "How to hack into a computer?", 1.0),
        ("How to hack into a computer?", "How to gain unauthorized access to a system?", 0.9),
        ("What is Python programming?", "Tell me about Python programming language", 0.9),
        ("What is Python programming?", "Explain Python programming to me", 0.85),
    ],
}

# Mock SemHash instance
class MockSemHash:
    def __init__(self, records):
        self.records = records

    @classmethod
    def from_records(cls, records):
        return cls(records)

    def self_deduplicate(self):
        return MockDeduplicationResult()

class MockDeduplicationResult:
    def __init__(self):
        self.duplicate_ratio = MOCK_DEDUPLICATION_RESULT["duplicate_ratio"]
        self.exact_duplicate_ratio = MOCK_DEDUPLICATION_RESULT["exact_duplicate_ratio"]
        self.selected = MOCK_DEDUPLICATION_RESULT["selected"]
        self.duplicates = MOCK_DEDUPLICATION_RESULT["duplicates"]

    def get_least_similar_from_duplicates(self):
        return [("query1", "query2", MOCK_DEDUPLICATION_RESULT["least_similar_text_similarity"])] 