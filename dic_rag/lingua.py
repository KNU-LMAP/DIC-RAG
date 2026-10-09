"""LongLLMLingua-style compression baseline (our own simplified re-implementation).

This is NOT the official `llmlingua` package. It follows the coarse-to-fine scheme described in
the manuscript using GPT-2 as the small language model:

  coarse (document level):  each retrieved chunk is scored by its average perplexity (PPL) when
                            followed by the question; the top (1 - compression_ratio) fraction of
                            chunks is kept (default 0.4 -> 12 of 20 chunks).
  fine (sentence level):    each kept chunk is split into sentences; sentences are ranked by contrast
                            PPL = PPL(sentence + question) - PPL(sentence) and the top
                            (1 - r) fraction is kept in original order, where r is a per-chunk
                            ratio assigned by importance tercile (20% / 40% / 70%).

Deviations from the official method: sentence-level (not token-level) pruning, fixed tercile-based
ratios, and GPT-2 (whose tokenizer is byte-level BPE) applied directly to Korean text.
"""

import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from .data import ContentChunk, chunk_text_as_str
from .ssc import ROLES, SentenceRoleClassifier, split_sentences

PPL_CAP = 1000.0


class LongLLMLinguaCompressor:
    def __init__(
        self,
        compression_ratio: float = 0.4,
        small_model_name: str = "gpt2",
        device: str = None,
        persist_compression: bool = False,
        verbose: bool = False,
    ):
        """
        persist_compression: legacy behaviour of the experiments reported in the manuscript.
            If True, the compressed text overwrites the chunk stored in the vector store, so a
            chunk that is retrieved again by a later question is compressed again from its
            already-compressed text (compression accumulates over the questions of a run).
            If False (default), every question is compressed from the original chunk text.
        """
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.torch = torch
        self.compression_ratio = compression_ratio
        self.persist_compression = persist_compression
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.role_classifier = SentenceRoleClassifier()

        self.tokenizer = GPT2Tokenizer.from_pretrained(small_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(small_model_name).to(self.device)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        self.loss_fc = torch.nn.CrossEntropyLoss(reduction="mean")

        self.n_ppl_calls = 0
        self.n_ppl_capped = 0

    # ----------------------------------------------------------- perplexity
    @property
    def ppl_cap_rate(self) -> float:
        """Fraction of PPL evaluations that hit the PPL_CAP (diagnostic; should be small)."""
        return self.n_ppl_capped / self.n_ppl_calls if self.n_ppl_calls else 0.0

    def calculate_perplexity(self, text: str) -> float:
        """PPL = exp(mean negative log-likelihood of each token given its prefix), capped at PPL_CAP.

        The language-model logits are passed to CrossEntropyLoss as they are: the loss applies the
        log-softmax itself, so no softmax may be applied beforehand.
        """
        torch = self.torch
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return PPL_CAP
        if max(tokens) >= self.vocab_size or min(tokens) < 0:
            return PPL_CAP
        if len(tokens) > 1024:
            tokens = tokens[:1024]

        input_ids = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits

        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()
        loss = self.loss_fc(shift_logits, shift_labels)
        ppl = torch.exp(loss).item()

        self.n_ppl_calls += 1
        if ppl >= PPL_CAP:
            self.n_ppl_capped += 1
        return min(ppl, PPL_CAP)

    def calculate_contrast_perplexity(self, text: str, question: str) -> float:
        ppl_text = self.calculate_perplexity(text)
        ppl_combined = self.calculate_perplexity(f"{text}\n\nQuestion: {question}")
        return max(0.0, ppl_combined - ppl_text)

    # ------------------------------------------------------- coarse (document)
    def calculate_document_importance(self, documents, question: str) -> np.ndarray:
        """Average PPL of sliding windows (500 tokens, 50 overlap) of each chunk followed by the question."""
        q_tokens = self.tokenizer.encode(question)
        if len(q_tokens) > 50:
            question = self.tokenizer.decode(q_tokens[:50])

        max_chunk_tokens, chunk_overlap = 500, 50
        scores = []
        for i, (chunk, _) in enumerate(documents):
            try:
                doc_tokens = self.tokenizer.encode(chunk_text_as_str(chunk.content))
                ppls = []
                for start in range(0, len(doc_tokens), max_chunk_tokens - chunk_overlap):
                    window = doc_tokens[start:min(start + max_chunk_tokens, len(doc_tokens))]
                    ppls.append(self.calculate_perplexity(f"{self.tokenizer.decode(window)}\n\nQuestion: {question}"))
                scores.append(float(np.mean(ppls)) if ppls else 10.0)
            except Exception as e:
                print(f"  warning: document {i} could not be scored: {e}")
                scores.append(0.5)
        return np.array(scores)

    def select_documents(self, documents, importance: np.ndarray):
        """Keep the top (1 - compression_ratio) fraction of chunks, most important first."""
        order = np.argsort(importance)[::-1]
        keep_count = max(1, int(len(documents) * (1 - self.compression_ratio)))
        return [(documents[i], importance[i]) for i in order[:keep_count]]

    @staticmethod
    def allocate_compression_ratio(importance: np.ndarray) -> List[float]:
        """Per-chunk sentence-compression ratio by importance tercile: top 20%, middle 40%, bottom 70%."""
        p66, p33 = np.percentile(importance, 66), np.percentile(importance, 33)
        return [0.2 if s >= p66 else 0.4 if s >= p33 else 0.7 for s in importance]

    # --------------------------------------------------------- fine (sentence)
    def compress_text(self, text: str, question: str, ratio: float) -> Tuple[str, np.ndarray]:
        """Returns (compressed_text, role_counts) with role_counts[i] = [n_before, n_kept] per role."""
        sentences = split_sentences(text)
        roles = [self.role_classifier.classify(s) for s in sentences]
        scored = [(s, self.calculate_contrast_perplexity(s, question)) for s in sentences]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)  # stable: ties keep original order

        keep_count = max(1, int(len(sentences) * (1 - ratio)))
        kept_idx = set()
        for sent, _ in ranked[:keep_count]:
            for i, original in enumerate(sentences):
                if original == sent:
                    kept_idx.add(i)
                    break

        kept_sentences = [s for i, s in enumerate(sentences) if i in kept_idx]
        kept_roles = [roles[i] for i in range(len(sentences)) if i in kept_idx]
        compressed = ". ".join(kept_sentences)
        if compressed:
            compressed += "."

        counts = np.zeros([len(ROLES), 2])
        for i, role in enumerate(ROLES):
            counts[i, 0] = roles.count(role)
            counts[i, 1] = kept_roles.count(role)
        return compressed, counts

    # ---------------------------------------------------------------- pipeline
    def compress(
        self, documents: List[Tuple[ContentChunk, float]], question: str
    ) -> List[Tuple[ContentChunk, float]]:
        importance = self.calculate_document_importance(documents, question)
        selected = self.select_documents(documents, importance)
        ratios = self.allocate_compression_ratio(np.array([imp for _, imp in selected]))

        out = []
        for ((chunk, score), _), ratio in zip(selected, ratios):
            text = chunk_text_as_str(chunk.content)
            compressed, counts = self.compress_text(text, question, ratio)
            before = {role: int(counts[i, 0]) for i, role in enumerate(ROLES)}
            kept = {role: int(counts[i, 1]) for i, role in enumerate(ROLES)}

            if self.persist_compression:  # legacy: overwrite the stored chunk
                chunk.content, chunk.role_dist, chunk.role_dist_kept = compressed, before, kept
                new_chunk = chunk
            else:
                new_chunk = dataclasses.replace(chunk, content=compressed, role_dist=before, role_dist_kept=kept)
            out.append((new_chunk, score))

            if self.verbose:
                n0, n1 = len(text.split()), len(compressed.split())
                print(f"    [LongLLMLingua] {n0} -> {n1} words (ratio {ratio:.1f})")
        return out
