# DIC-RAG

Code for the paper *Domain-aware prompt compression improves the efficiency of retrieval-augmented large language models for dental insurance claims* (Scientific Reports).

It evaluates four systems on multiple-choice questions about dental insurance claims:

| `--method` | Description |
|---|---|
| `llm` | LLM only, no retrieval |
| `vrag` | Vanilla RAG (top-k chunks, no compression) |
| `longllmlingua` | RAG + our simplified re-implementation of LongLLMLingua (GPT-2, chunk- and sentence-level pruning). Not the official `llmlingua` package. |
| `ssc` | RAG + Selective Safeguard Compression (role-aware sentence compression) |

## Install

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

## Run

```bash
python main.py --method ssc --questions examples/sample_questions.txt \
    --answers examples/sample_answers.txt --pdf guide.pdf --output-dir results
```

Defaults match the paper: k=20, chunk size 1000, overlap 100, compression ratio 0.4 (both methods),
`paraphrase-multilingual-mpnet-base-v2`, `gpt-5.5` (no max-token or temperature setting).

Outputs in `--output-dir`: `results_<method>.json` (per-question records and summary: accuracy, token usage, IPR) and `answers_<method>.txt`.
For LongLLMLingua the summary also reports `ppl_cap_rate`, the fraction of GPT-2 perplexity evaluations that hit the cap of 1000 (a diagnostic; it should be small).

### Input formats

- Questions: one per line, `<no>. <question>  1: <opt>  2: <opt>  3: <opt>  4: <opt>  5: <opt>` (two or more spaces before `1:`).
- Answers: one `<no>. <1-5>` per line. See `examples/` (synthetic items, not from the study set).
- The knowledge base is the (Korean) insurance guideline PDF(s); only text and tables are indexed. The PDFs and the study question sets are not distributed (see "Data availability"); please contact the authors if you need them.

### Information preservation rate (IPR)

For methods with sentence-level compression, IPR per role = kept / original sentences, pooled over all questions.
Overall = 0.40·Exception + 0.35·Condition + 0.25·Numeric. For LongLLMLingua, only chunks that survive the document-level stage are counted.

### LongLLMLingua state (reproducibility note)

In the original experiment scripts the compressed text overwrote the chunk stored in the index, so a chunk retrieved by several questions was compressed repeatedly. This release compresses each question from the original chunk text by default.
Pass `--legacy-lingua-state` to reproduce the original behaviour.

## Notes

- Prompts are in Korean and kept verbatim; the retrieval query is the full question prompt.
- The role classifier and sentence scorer are rule-based and tuned for Korean billing text; the splitter breaks on `.`, `!`, `?` and newlines.
- If the API reply contains no parsable answer, the `llm` method retries up to 5 times; RAG methods retry only if the reply is empty. Token counts are from the last call.

## Tests

```bash
pytest tests    # offline, uses stubs (no network or model downloads)
```

## Data availability

The question sets and the source PDF documents used to build the vector index cannot be released because of licensing restrictions on the source material. Researchers who need access to them should contact the corresponding author(s) of the paper. The files in `examples/` are synthetic and only illustrate the input format.

## License and citation

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`): commercial use is not permitted, and academic or other non-commercial use requires attribution, i.e. citing the paper:

> *Domain-aware prompt compression improves the efficiency of retrieval-augmented large language models for dental insurance claims.* Scientific Reports (citation details to be added upon publication).

For commercial use, please contact the authors.
