"""Selective Safeguard Compression (SSC).

Sentence-level compression of retrieved chunks that (i) classifies every sentence into
one of five regulatory roles and (ii) protects Exception / Condition / Numeric sentences
that fall in the middle of the importance-score distribution.

Internal/legacy name of this method in earlier result files: "ProLingua" / "ProtectiveLingua".
"""

import dataclasses
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .data import ContentChunk, chunk_text_as_str

ROLES = ("exception", "condition", "numeric", "definition", "background")


# --------------------------------------------------------------------------- roles
class SentenceRoleClassifier:
    """Rule-based role classifier (regex / keyword patterns for Korean billing text).

    Roles are checked in the order exception -> condition -> numeric -> definition;
    a sentence that matches none of them is 'background'.
    """

    ROLE_PATTERNS = {
        "exception": [
            r"다만", r"단,\s*", r"제외", r"제한", r"불가",
            r"하지\s*(?:않|못)", r"이외는", r"이를\s*제외", r"예외",
        ],
        "condition": [
            r"\s*AND\s*", r"그리고", r"또한", r"~인\s*경우", r"~할\s*때", r"만약", r"~만",
            r"~일\s*경우", r"다음을\s*만족", r"반드시", r"필수", r"동시에", r"함께",
            r"더불어", r"조건", r"경우",
        ],
        "numeric": [
            r"\d+\s*(?:만원|원|세|개월|일|시간|회|배)",
            r"(?:만원|원|세|개월|일|시간|회|배)",
            r"\d+\s*[%％]",
            r"\d+~\d+",
            r"최대|최소",
            r"이상|이하|초과|미만",
        ],
        "definition": [
            r"은\s*(?:어떤|뭔가)?", r"는\s*(?:어떤|뭔가)?", r"이다", r"뜻", r"의미",
            r"즉", r"다시\s*말해", r"말하자면", r"다른\s*말로", r"정의",
        ],
    }

    def __init__(self):
        self.compiled = {
            role: [re.compile(p, re.IGNORECASE) for p in patterns]
            for role, patterns in self.ROLE_PATTERNS.items()
        }

    def classify(self, sentence: str) -> str:
        s = sentence.lower()
        for role in ("exception", "condition", "numeric", "definition"):
            for pattern in self.compiled[role]:
                if pattern.search(s):
                    return role
        return "background"


def split_sentences(text: str) -> List[str]:
    """Sentence split shared by SSC and the LongLLMLingua re-implementation."""
    return [s.strip() for s in re.split(r"[.!?。！？\n]+", text) if s.strip()]


# --------------------------------------------------------------------- compression
class SSCCompressor:
    """Selective Safeguard Compression.

    Per chunk:
      1. split into sentences and classify each sentence's role;
      2. score every sentence (heuristic: query word overlap + length + negation markers);
      3. compute mean/std of the chunk's scores and apply the decision zones below;
      4. fill up to the target number of sentences from the 'uncertain' bands.

    Decision zones (mu, sigma = mean / std of the chunk's sentence scores):
      score >  mu + high_threshold*sigma                      -> keep
      score <  mu + low_threshold*sigma   (low_threshold<0)   -> delete
      |score - mu| <= protection_zone_width*sigma             -> keep if role in {exception, condition, numeric}
                                                                 else delete
      anything else (the two remaining bands)                 -> 'uncertain', resolved by score rank
    """

    PROTECTED_ROLES = ("exception", "condition", "numeric")
    NEGATION_MARKERS = ("아니", "없", "불", "제외", "제한", "불가")

    def __init__(
        self,
        compression_ratio: float = 0.4,
        high_threshold: float = 1.0,
        low_threshold: float = -1.0,
        protection_zone_width: float = 0.5,
        verbose: bool = False,
    ):
        self.compression_ratio = compression_ratio
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.protection_zone_width = protection_zone_width
        self.verbose = verbose
        self.role_classifier = SentenceRoleClassifier()

    # ------------------------------------------------------------- scoring
    def _score_sentences(self, sentences: List[str], question: str) -> List[float]:
        question_words = set(question.split())
        scores = []
        for sent in sentences:
            score = 0.0
            words = sent.split()
            if len(words) < 3:
                score -= 2.0
            elif len(words) > 3:
                score += 0.5
            overlap = len(question_words & set(words))
            if overlap > 0:
                score += overlap * 2.0
            for marker in self.NEGATION_MARKERS:
                if marker in sent:
                    score += 1.5
            scores.append(max(0.0, 1.0 + score))
        return scores

    def _thresholds(self, scores: List[float]) -> Dict[str, float]:
        arr = np.array(scores)
        mean, std = float(np.mean(arr)), float(np.std(arr))
        if std == 0:
            std = 1.0
        return {
            "mean": mean,
            "std": std,
            "high": mean + self.high_threshold * std,
            "high_protection": mean + self.protection_zone_width * std,
            "low_protection": mean - self.protection_zone_width * std,
            "low": mean + self.low_threshold * std,
        }

    # ------------------------------------------------------------- one chunk
    def compress_text(self, text: str, question: str) -> Tuple[str, Dict, Dict]:
        """Returns (compressed_text, role_counts_before, role_counts_kept)."""
        sentences = split_sentences(text)
        if not sentences:
            return text, {}, {}

        roles = [self.role_classifier.classify(s) for s in sentences]
        scores = self._score_sentences(sentences, question)
        th = self._thresholds(scores)

        keep, delete, uncertain = [], [], []
        for i, (role, score) in enumerate(zip(roles, scores)):
            if score > th["high"]:
                keep.append(i)
            elif score < th["low"]:
                delete.append(i)
            elif th["low_protection"] <= score <= th["high_protection"]:
                (keep if role in self.PROTECTED_ROLES else delete).append(i)
            else:
                uncertain.append(i)

        keep_count = max(1, int(len(sentences) * (1 - self.compression_ratio)))
        selected = set(keep)
        remaining = keep_count - len(selected)
        if remaining > 0 and uncertain:
            ranked = sorted(uncertain, key=lambda i: scores[i], reverse=True)
            selected.update(ranked[:remaining])

        kept_sentences = [sentences[i] for i in sorted(selected)]
        compressed = ". ".join(kept_sentences)
        if kept_sentences and not compressed.endswith("."):
            compressed += "."

        before = dict(Counter(roles))
        kept = dict(Counter(roles[i] for i in selected))
        return compressed, before, kept

    # ------------------------------------------------------------- retrieved set
    def compress(
        self, documents: List[Tuple[ContentChunk, float]], question: str
    ) -> List[Tuple[ContentChunk, float]]:
        """Compress every retrieved chunk; retrieval order is preserved."""
        out = []
        for chunk, score in documents:
            text = chunk_text_as_str(chunk.content)
            compressed, before, kept = self.compress_text(text, question)
            out.append(
                (dataclasses.replace(chunk, content=compressed, role_dist=before, role_dist_kept=kept), score)
            )
            if self.verbose:
                red = (1 - len(compressed) / len(text)) * 100 if text else 0.0
                print(f"    [SSC] {len(text)} -> {len(compressed)} chars ({red:.1f}% reduction)")
        return out
