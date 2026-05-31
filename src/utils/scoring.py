"""
Repetition and Intent scoring modules.
These provide quantitative measures for evaluating the degree
of repetition and harmful intent in conversations.
"""
import re
from collections import Counter, defaultdict


class RepetitionScorer:
    """
    Measures repetition of harmful behavior.
    Analyzes message frequency and pattern repetition per user/target pair.
    
    Repetition Score = (repeat_messages / total_messages) * frequency_weight
    """

    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.user_history = defaultdict(list)

    def add_message(self, user_id, message, target_id=None):
        """Record a message from a user."""
        self.user_history[user_id].append({
            'text': message.lower().strip(),
            'target': target_id
        })

    def compute_score(self, user_id):
        """
        Compute repetition score for a user.
        Returns a float between 0.0 and 1.0.
        """
        messages = self.user_history.get(user_id, [])
        if len(messages) <= 1:
            return 0.0

        texts = [m['text'] for m in messages]
        total = len(texts)

        # Count similar message pairs
        repeat_count = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self._jaccard_similarity(texts[i], texts[j])
                if similarity >= self.similarity_threshold:
                    repeat_count += 1

        max_pairs = total * (total - 1) / 2
        if max_pairs == 0:
            return 0.0

        # Repetition ratio with frequency weight
        repetition_ratio = repeat_count / max_pairs
        frequency_weight = min(total / 10.0, 1.0)  # Cap at 10 messages

        score = repetition_ratio * frequency_weight
        return min(score, 1.0)

    def _jaccard_similarity(self, text1, text2):
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get_targeted_score(self, user_id, target_id):
        """Get repetition score for a specific user→target pair."""
        messages = self.user_history.get(user_id, [])
        targeted = [m for m in messages if m['target'] == target_id]
        if len(targeted) <= 1:
            return 0.0

        texts = [m['text'] for m in targeted]
        total = len(texts)
        frequency_weight = min(total / 5.0, 1.0)
        return frequency_weight


class IntentScorer:
    """
    Estimates intent to harm based on linguistic cues.
    Uses keyword matching, severity weighting, and pattern detection.

    Intent Score = weighted_keyword_score + pattern_bonus
    """

    # Severity-weighted keyword categories
    INTENT_KEYWORDS = {
        'death_threat': {
            'weight': 1.0,
            'keywords': ['kill', 'die', 'murder', 'dead', 'mar dunga', 'mar dalunga',
                         'khatam kar dunga', 'jaan se maar']
        },
        'harm_threat': {
            'weight': 0.8,
            'keywords': ['hurt', 'beat', 'punch', 'slap', 'marunga', 'peetunga',
                         'tujhe dekh lunga', 'tera kaam tamam']
        },
        'intimidation': {
            'weight': 0.6,
            'keywords': ['watch out', 'beware', 'warning', 'careful', 'khabardar',
                         'dekh lena', 'bach ke rehna', 'anjaam bura hoga']
        },
        'degradation': {
            'weight': 0.5,
            'keywords': ['worthless', 'loser', 'pathetic', 'useless', 'nikamma',
                         'bekar', 'wahiyat', 'ghatiya', 'tameez nahi']
        },
        'isolation': {
            'weight': 0.4,
            'keywords': ['nobody likes', 'no friends', 'alone', 'koi pasand nahi',
                         'akela', 'tujhse koi baat nahi karega']
        }
    }

    # Patterns indicating deliberate intent
    INTENT_PATTERNS = [
        r'\bi will\b.*\b(kill|hurt|destroy|ruin)\b',
        r'\byou (will|are going to)\b.*\b(pay|suffer|regret)\b',
        r'\b(wait and see|just wait)\b',
        r'\bnext time\b',
    ]

    def compute_score(self, text):
        """
        Compute intent-to-harm score for a text.
        Returns a float between 0.0 and 1.0.
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Keyword matching with severity weights
        max_severity = 0.0
        keyword_hits = 0

        for category, data in self.INTENT_KEYWORDS.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    keyword_hits += 1
                    max_severity = max(max_severity, data['weight'])

        # Pattern matching bonus
        pattern_bonus = 0.0
        for pattern in self.INTENT_PATTERNS:
            if re.search(pattern, text_lower):
                pattern_bonus = 0.2
                break

        # Compute final score
        keyword_score = min(keyword_hits * 0.15, 0.8)
        score = max(max_severity * 0.6 + keyword_score, max_severity * 0.5) + pattern_bonus

        return min(score, 1.0)

    def get_detailed_analysis(self, text):
        """Return detailed breakdown of intent analysis."""
        text_lower = text.lower()

        analysis = {
            'score': self.compute_score(text),
            'detected_categories': [],
            'matched_keywords': [],
            'pattern_match': False
        }

        for category, data in self.INTENT_KEYWORDS.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    if category not in analysis['detected_categories']:
                        analysis['detected_categories'].append(category)
                    analysis['matched_keywords'].append(keyword)

        for pattern in self.INTENT_PATTERNS:
            if re.search(pattern, text_lower):
                analysis['pattern_match'] = True
                break

        return analysis


if __name__ == '__main__':
    # Demo
    print("=== Repetition Scorer Demo ===")
    rep_scorer = RepetitionScorer()
    rep_scorer.add_message('user1', 'you are stupid', 'target1')
    rep_scorer.add_message('user1', 'you are so stupid', 'target1')
    rep_scorer.add_message('user1', 'you are really stupid', 'target1')
    rep_scorer.add_message('user1', 'hey how are you', 'target2')
    print(f"Repetition score for user1: {rep_scorer.compute_score('user1'):.3f}")
    print(f"Targeted score (user1→target1): {rep_scorer.get_targeted_score('user1', 'target1'):.3f}")

    print("\n=== Intent Scorer Demo ===")
    intent_scorer = IntentScorer()
    examples = [
        "I will kill you",
        "You are worthless and nobody likes you",
        "Tujhe dekh lunga, khabardar!",
        "Let's play cricket tomorrow",
    ]
    for text in examples:
        analysis = intent_scorer.get_detailed_analysis(text)
        print(f"\nText: '{text}'")
        print(f"  Score: {analysis['score']:.3f}")
        print(f"  Categories: {analysis['detected_categories']}")
        print(f"  Keywords: {analysis['matched_keywords']}")
