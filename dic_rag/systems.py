"""The four evaluated systems: llm (no retrieval), vrag, longllmlingua and ssc (all RAG-based)."""

import time
from typing import Callable, Dict, List, Optional

import pandas as pd

from .data import ContentChunk
from .ssc import ROLES

# The prompts were written in Korean for the experiments and are kept verbatim.
SYSTEM_MESSAGE = "당신은 치과 의료 전문가입니다. 객관식 문제의 정답을 선택하고 이유를 설명하세요."
MAX_TRIALS = 5
RETRY_SLEEP_SEC = 3.0


def _call_llm(client, model: str, messages: List[Dict], is_valid: Callable[[Optional[str]], bool]):
    """Call the chat API; retry (up to MAX_TRIALS) while the reply is not valid.

    Returns (response, answer_text, n_trials). Token usage is taken from the LAST call only.
    """
    for trial in range(MAX_TRIALS):
        response = client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content
        if is_valid(answer):
            break
        time.sleep(RETRY_SLEEP_SEC)
    return response, answer, trial + 1


def _usage(response) -> Dict[str, int]:
    return {
        "tokens_used": response.usage.total_tokens,
        "tokens_prompt": response.usage.prompt_tokens,
        "tokens_completion": response.usage.completion_tokens,
    }


class LLMSystem:
    """Standard LLM baseline: the prompt is sent as it is, without retrieval."""

    name = "llm"

    def __init__(self, client, model: str = "gpt-5.5", answer_parser: Callable = None):
        self.client = client
        self.model = model
        # the LLM baseline retries while the reply contains no parsable answer number
        self.answer_parser = answer_parser

    def query(self, prompt: str) -> Dict:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]
        valid = (lambda a: a is not None and self.answer_parser(a) is not None) if self.answer_parser else (lambda a: a is not None)
        response, answer, n_trials = _call_llm(self.client, self.model, messages, valid)
        return {"answer": answer, "ntrial": n_trials, "n_chunks": 0, "role_counts": None, **_usage(response)}


class RAGSystem:
    """Retrieval-augmented system; `compressor` is None (vrag), LongLLMLinguaCompressor or SSCCompressor."""

    def __init__(self, client, vector_store, model: str = "gpt-5.5", k: int = 20, compressor=None, name: str = "vrag"):
        self.client = client
        self.vector_store = vector_store
        self.model = model
        self.k = k
        self.compressor = compressor
        self.name = name

    @staticmethod
    def _format_chunk(chunk: ContentChunk, rank: int) -> str:
        if chunk.content_type == "table" and isinstance(chunk.content, pd.DataFrame):
            return f"[Table {rank}] (Page {chunk.page_number})\n{chunk.content.to_markdown(index=False)}\n"
        if chunk.content_type == "table":  # table whose text has been compressed
            return f"[Table {rank}] (Page {chunk.page_number})\n{chunk.content}\n"
        return f"[Document {rank}] (Page {chunk.page_number})\n{chunk.content}\n"

    @staticmethod
    def _role_counts(chunks: List[ContentChunk]) -> Dict[str, List[int]]:
        counts = {role: [0, 0] for role in ROLES}
        for chunk in chunks:
            for role in ROLES:
                counts[role][0] += (chunk.role_dist or {}).get(role, 0)
                counts[role][1] += (chunk.role_dist_kept or {}).get(role, 0)
        return counts

    def query(self, prompt: str) -> Dict:
        # the whole prompt (instruction + question + options) is used as the retrieval query
        documents = self.vector_store.search(prompt, k=self.k)
        if self.compressor is not None:
            documents = self.compressor.compress(documents, prompt)

        chunks = [chunk for chunk, _ in documents]
        context = "\n".join(self._format_chunk(c, r) for r, c in enumerate(chunks, start=1))
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"},
        ]
        response, answer, n_trials = _call_llm(self.client, self.model, messages, lambda a: a is not None)
        return {
            "answer": answer,
            "ntrial": n_trials,
            "n_chunks": len(chunks),
            "role_counts": self._role_counts(chunks) if self.compressor is not None else None,
            **_usage(response),
        }
