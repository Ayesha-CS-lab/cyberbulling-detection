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

    # Severity-weighted keyword categories (English + Roman Urdu)
    INTENT_KEYWORDS = {
        'death_threat': {
            'weight': 1.0,
            'keywords': ['kill', 'murder', 'slay', 'behead', 'jaan se maar', 'maar dunga',
                         'maar dalunga', 'mar dunga', 'khatam kar dunga', 'zinda nahi chodunga',
                         'zinda nahi chhodunga', 'jaan le lunga']
        },
        'harm_threat': {
            'weight': 0.85,
            'keywords': ['hit', 'hurt', 'harm', 'beat', 'punch', 'kick', 'slap', 'smash',
                         'bash', 'thrash', 'choke', 'strangle', 'stab', 'shoot', 'burn',
                         'destroy', 'ruin', 'break your', 'break ur', 'marunga', 'maarunga',
                         'peetunga', 'tujhe dekh lunga', 'dekh lunga', 'tera kaam tamam',
                         'mun tor', 'muh tor', 'mooh tor', 'haddi tor', 'haddi tod',
                         'haddiyan tor', 'thok dunga', 'pel dunga', 'thappar', 'chamaat',
                         'maar dunga', 'toor dunga', 'tod dunga', 'phod dunga']
        },
        'intimidation': {
            'weight': 0.6,
            'keywords': ['watch out', 'beware', 'warning', 'be careful', 'you will pay',
                         'you will regret', 'you will suffer', 'khabardar', 'dekh lena',
                         'bach ke rehna', 'anjaam bura hoga', 'maza chakhaunga',
                         'pata chal jayega', 'dekh lunga tujhe']
        },
        'degradation': {
            'weight': 0.5,
            'keywords': ['worthless', 'pathetic', 'useless', 'nikamma', 'bekar', 'wahiyat',
                         'ghatiya', 'tameez nahi']
        },
        'isolation': {
            'weight': 0.4,
            'keywords': ['nobody likes you', 'no one likes you', 'no friends', 'koi pasand nahi',
                         'tujhse koi baat nahi karega']
        }
    }

    # Order-independent threat-construction vocabulary
    VIOLENCE_VERBS = {
        'hit', 'hits', 'hitting', 'hurt', 'hurts', 'harm', 'beat', 'beats', 'beating',
        'punch', 'punching', 'kick', 'kicking', 'slap', 'slapping', 'smash', 'smashing',
        'break', 'breaking', 'bash', 'thrash', 'choke', 'strangle', 'stab', 'stabbing',
        'shoot', 'shooting', 'kill', 'killing', 'murder', 'destroy', 'ruin', 'burn',
        'burning', 'maar', 'maarun', 'maarunga', 'peet', 'peetunga', 'thok', 'thappar',
        'todunga', 'phodunga', 'pel',
    }
    VIOLENCE_PHRASES = [
        'jaan se maar', 'maar dunga', 'maar dalunga', 'mar dunga', 'khatam kar',
        'dekh lunga', 'dekh lena', 'haddi tor', 'haddi tod', 'muh tor', 'mun tor',
        'zinda nahi chod', 'maza chakha', 'tod dunga', 'phod dunga', 'thok dunga',
        'kaam tamam',
    ]
    TARGET_WORDS = {
        'you', 'your', 'yours', 'u', 'ur', 'tujhe', 'tujhy', 'tujh', 'tumhe', 'tumhein',
        'tera', 'teri', 'tere', 'tumhara', 'tumhari', 'tumhare', 'apko', 'tum', 'tm',
    }
    FUTURE_WORDS = {'will', 'gonna', 'going', 'gon', 'shall', 'wanna'}
    FUTURE_SUFFIXES = ('unga', 'ungi', 'lunga', 'lungi', 'dunga', 'dungi', 'oge', 'ega', 'egi', 'enge')
    BODYPARTS = {
        'face', 'head', 'teeth', 'tooth', 'nose', 'jaw', 'mouth', 'neck', 'throat',
        'legs', 'leg', 'arm', 'arms', 'bones', 'bone', 'body', 'muh', 'mun', 'mooh',
        'sar', 'gardan', 'haddi', 'haddiyan', 'chehra', 'tang',
    }

    # Deliberate-intent regex patterns (flexible: verb need not immediately follow)
    INTENT_PATTERNS = [
        r'\b(i|we|main|mein)\b[\w\s]{0,20}\b(kill|hurt|harm|beat|hit|punch|kick|slap|smash|break|destroy|ruin|stab|shoot|burn|maar)\b',
        r'\byou(\s+will|\s+are going to|\s+gonna|.?ll)\b[\w\s]{0,15}\b(pay|suffer|regret|die|be sorry)\b',
        r'\b(wait and see|just wait|you.?ll see|dekh lena|pata chal)\b',
        r'\bnext time\b',
    ]

    @staticmethod
    def _tokens(text_lower):
        return set(re.findall(r"[a-z']+", text_lower))

    def _construction_score(self, text_lower, tokens):
        """Order-independent detection of a threat construction."""
        has_verb = bool(tokens & self.VIOLENCE_VERBS) or any(p in text_lower for p in self.VIOLENCE_PHRASES)
        has_target = bool(tokens & self.TARGET_WORDS)
        has_future = bool(tokens & self.FUTURE_WORDS) or any(t.endswith(self.FUTURE_SUFFIXES) for t in tokens)
        has_body = bool(tokens & self.BODYPARTS)
        if has_verb and (has_target or has_body):
            return 0.9 if has_future else 0.75
        if has_verb and has_future:
            return 0.6
        return 0.0

    def compute_score(self, text):
        """
        Compute intent-to-harm score for a text (0.0 - 1.0).
        Combines severity-weighted keyword matching with an order-independent
        threat-construction detector, so threats phrased in novel word orders or
        with common misspellings (e.g. "i hit will on your face") are still caught.
        """
        text_lower = text.lower()
        tokens = self._tokens(text_lower)

        max_severity = 0.0
        for data in self.INTENT_KEYWORDS.values():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    max_severity = max(max_severity, data['weight'])

        construction = self._construction_score(text_lower, tokens)
        pattern_match = any(re.search(p, text_lower) for p in self.INTENT_PATTERNS)

        score = max(max_severity, construction)
        if score < 0.5 and pattern_match:
            score = 0.5
        elif pattern_match:
            score = min(score + 0.1, 1.0)

        return round(min(score, 1.0), 4)

    def get_detailed_analysis(self, text):
        """Return detailed breakdown of intent analysis."""
        text_lower = text.lower()
        tokens = self._tokens(text_lower)

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

        # surface the threat-construction cue too
        if self._construction_score(text_lower, tokens) > 0:
            if 'threat_construction' not in analysis['detected_categories']:
                analysis['detected_categories'].append('threat_construction')
            for w in sorted((tokens & self.VIOLENCE_VERBS) | (tokens & self.BODYPARTS)):
                if w not in analysis['matched_keywords']:
                    analysis['matched_keywords'].append(w)

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
