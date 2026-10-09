"""Offline tests: no network, no model downloads (stubs replace GPT-2, the encoder and OpenAI)."""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dic_rag.data import ContentChunk, DocumentProcessor
from dic_rag.evaluation import (
    aggregate_ipr, build_prompt, extract_answer_number, grade_letter, parse_correct_answers, parse_questions,
)
from dic_rag.ssc import SSCCompressor, SentenceRoleClassifier
from dic_rag.systems import RAGSystem

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
TEXT = (
    "치아 우식 치료는 보험 급여 대상이다. 다만 6세 미만 소아는 예외로 한다. "
    "본인부담금은 30% 이다. 해당 항목은 연 1회만 산정한다. 참고로 이는 일반 설명이다. "
    "검사 결과를 반드시 기록해야 한다."
)


def chunk(i, text):
    return ContentChunk(id=f"c{i}", content_type="text", content=text, page_number=1, metadata={})


def test_parsing_and_prompt():
    qs = parse_questions(os.path.join(EX, "sample_questions.txt"))
    ans = parse_correct_answers(os.path.join(EX, "sample_answers.txt"))
    assert len(qs) == 2 and all(len(q["options"]) == 5 for q in qs)
    assert ans == {1: 2, 2: 3}
    assert "선택지:\n1: " in build_prompt(qs[0])


@pytest.mark.parametrize("text,expected", [("정답: 3\n이유: x", 3), ("정답은 4번입니다", 4), ("답: 2", 2), ("none", None)])
def test_extract_answer(text, expected):
    assert extract_answer_number(text) == expected


def test_grade():
    assert [grade_letter(x) for x in (95, 85, 75, 65, 10)] == list("ABCDF")


def test_roles():
    c = SentenceRoleClassifier()
    assert c.classify("다만 예외로 한다") == "exception"
    assert c.classify("참고 사항입니다") == "background"


def test_ssc_is_stateless_and_reduces():
    ssc = SSCCompressor(compression_ratio=0.4)
    original = chunk(0, TEXT)
    out = ssc.compress([(original, 0.9)], "소아 우식 치료 본인부담금")
    assert original.content == TEXT  # no in-place mutation
    new = out[0][0]
    assert len(new.content) < len(TEXT)
    assert sum(new.role_dist_kept.values()) <= sum(new.role_dist.values())


def _stub_lingua(persist):
    """LongLLMLinguaCompressor without loading GPT-2: deterministic fake perplexity."""
    from dic_rag.lingua import LongLLMLinguaCompressor

    obj = LongLLMLinguaCompressor.__new__(LongLLMLinguaCompressor)
    obj.compression_ratio, obj.persist_compression, obj.verbose = 0.4, persist, False
    obj.role_classifier = SentenceRoleClassifier()
    obj.n_ppl_calls = obj.n_ppl_capped = 0
    obj.tokenizer = types.SimpleNamespace(
        encode=lambda t: list(range(len(t.split()))), decode=lambda toks: " ".join("w" * 1 for _ in toks)
    )
    obj.calculate_perplexity = lambda text: 1.0 + (len(text) % 17)
    return obj


@pytest.mark.parametrize("persist", [False, True])
def test_lingua_state(persist):
    lingua = _stub_lingua(persist)
    docs = [(chunk(i, TEXT + f" 문장 번호 {i} 입니다."), 0.5) for i in range(5)]
    before = [d[0].content for d in docs]
    out = lingua.compress(docs, "질문")
    assert len(out) == int(5 * 0.6)  # 40% of chunks dropped
    after = [d[0].content for d in docs]
    assert (after != before) == persist  # only the legacy flag mutates stored chunks


def test_pipeline_with_fake_client_and_encoder():
    from dic_rag.retrieval import VectorStore

    class Enc:
        def encode(self, texts, **kw):
            return np.array([[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts])

    store = VectorStore(encoder=Enc())
    store.build_index([chunk(i, TEXT[: 20 + i * 10]) for i in range(6)])

    calls = []

    def create(model, messages):
        calls.append(messages)
        msg = types.SimpleNamespace(content="정답: 2\n이유: 테스트")
        usage = types.SimpleNamespace(total_tokens=10, prompt_tokens=8, completion_tokens=2)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    system = RAGSystem(client, store, k=4, compressor=SSCCompressor(), name="ssc")
    out = system.query("소아 우식 치료")
    assert out["answer"].startswith("정답: 2") and out["n_chunks"] == 4
    assert "Context:" in calls[0][1]["content"]
    ipr = aggregate_ipr([out])
    assert ipr is not None and "overall" in ipr
