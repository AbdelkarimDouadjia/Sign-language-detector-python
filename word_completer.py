"""
Word autocomplete engine for the ASL Sign Language Detector.
Provides smart word suggestions as you spell letter by letter.
"""

# Top 500 most common English words for autocomplete
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "great", "between", "need", "large", "often",
    "hand", "high", "help", "line", "turn", "cause", "much", "mean", "before",
    "move", "right", "boy", "old", "too", "same", "tell", "does", "set",
    "three", "air", "here", "play", "small", "end", "put", "home", "read",
    "head", "start", "might", "story", "far", "sea", "draw", "left", "late",
    "run", "while", "press", "close", "night", "real", "life", "few", "north",
    "open", "seem", "together", "next", "white", "children", "begin", "got",
    "walk", "example", "ease", "paper", "group", "always", "music", "those",
    "both", "mark", "book", "letter", "until", "mile", "river", "car", "feet",
    "care", "second", "enough", "plain", "girl", "usual", "young", "ready",
    "above", "ever", "red", "list", "though", "feel", "talk", "bird", "soon",
    "body", "dog", "family", "direct", "pose", "leave", "song", "measure",
    "door", "product", "black", "short", "number", "class", "wind", "question",
    "happen", "complete", "ship", "area", "half", "rock", "order", "fire",
    "south", "problem", "piece", "told", "knew", "pass", "since", "top",
    "whole", "king", "space", "heard", "best", "hour", "better", "true",
    "during", "hundred", "five", "remember", "step", "early", "hold", "west",
    "ground", "interest", "reach", "fast", "verb", "sing", "listen", "six",
    "table", "travel", "less", "morning", "ten", "simple", "several", "vowel",
    "toward", "war", "lay", "against", "pattern", "slow", "center", "love",
    "person", "money", "serve", "appear", "road", "map", "rain", "rule",
    "govern", "pull", "cold", "notice", "voice", "unit", "power", "town",
    "fine", "certain", "fly", "fall", "lead", "cry", "dark", "machine",
    "note", "wait", "plan", "figure", "star", "box", "noun", "field", "rest",
    "correct", "able", "pound", "done", "beauty", "drive", "stood", "contain",
    "front", "teach", "week", "final", "gave", "green", "oh", "quick",
    "develop", "ocean", "warm", "free", "minute", "strong", "special", "mind",
    "behind", "clear", "tail", "produce", "fact", "street", "inch", "multiply",
    "nothing", "course", "stay", "wheel", "full", "force", "blue", "object",
    "decide", "surface", "deep", "moon", "island", "foot", "system", "busy",
    "test", "record", "boat", "common", "gold", "possible", "plane", "stead",
    "dry", "wonder", "laugh", "thousand", "ago", "ran", "check", "game",
    "shape", "equate", "hot", "miss", "brought", "heat", "snow", "tire",
    "bring", "yes", "distant", "fill", "east", "paint", "language", "among",
    "hello", "world", "please", "thank", "thanks", "sorry", "excuse",
    "goodbye", "welcome", "friend", "happy", "sad", "angry", "hungry",
    "thirsty", "tired", "sick", "pain", "water", "food", "eat", "drink",
    "sleep", "wake", "stop", "again", "more", "less", "where", "what",
    "why", "name", "sign", "language", "learn", "practice", "understand",
    "repeat", "slow", "fast", "deaf", "hearing", "speak", "quiet", "loud",
]


class WordCompleter:
    """Lightweight prefix-based word completer."""

    def __init__(self, word_list=None):
        self.words = sorted(set(w.upper() for w in (word_list or COMMON_WORDS)))

    def suggest(self, prefix: str, max_results: int = 3) -> list[str]:
        """Return up to *max_results* words starting with *prefix*."""
        if not prefix or len(prefix) < 1:
            return []
        prefix = prefix.upper().strip()
        # Get the last word being typed
        parts = prefix.split(' ')
        current_word = parts[-1] if parts else prefix
        if not current_word:
            return []
        matches = [w for w in self.words if w.startswith(current_word) and w != current_word]
        return matches[:max_results]

    def complete(self, sentence: str, chosen_word: str) -> str:
        """Replace the last partial word in *sentence* with *chosen_word*."""
        parts = sentence.rsplit(' ', 1)
        if len(parts) > 1:
            return parts[0] + ' ' + chosen_word + ' '
        return chosen_word + ' '
