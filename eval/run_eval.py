"""Evaluate the RAG pipeline on eval/dataset.jsonl across configurations.

Runs against a freshly seeded database with the real models, so Postgres and Ollama must
be up. Use a separate database so your own uploads don't change the results:

    docker compose exec db psql -U cardocs -d postgres -c "CREATE DATABASE cardocs_eval"
    docker compose run --rm \
        -e DATABASE_URL=postgresql+psycopg://cardocs:cardocs@db:5432/cardocs_eval \
        api sh -c "alembic upgrade head && python scripts/seed.py"
    export DATABASE_URL=postgresql+psycopg://cardocs:cardocs@localhost:5432/cardocs_eval

Then, from the repo root:

    backend/.venv/Scripts/python eval/run_eval.py                 # all configs
    backend/.venv/Scripts/python eval/run_eval.py --configs full  # just one
    backend/.venv/Scripts/python eval/run_eval.py --retrieval-only  # scores, no LLM

Writes eval/results/<config>.json (per-question detail) and eval/results/summary.md.
"""

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.generation import (  # noqa: E402
    PARSE_FAILURE_ANSWER,
    PipelineConfig,
    answer_question,
    looks_filipino,
    translate_to_english,
)
from app.services.providers import get_embedder  # noqa: E402
from app.services.providers.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.services.providers.ollama import OllamaLLM  # noqa: E402
from app.services.retrieval import RetrievalConfig, resolve_vehicle, retrieve  # noqa: E402
from app.services.verification import verify_answer  # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS = ROOT / "eval" / "results"


@dataclass(frozen=True)
class EvalConfig:
    name: str
    label: str
    pipeline: PipelineConfig
    llm_model: str | None = None  # None = settings.llm_model


def _pipeline(
    mode="hybrid", rerank=True, verify=True, auto_scope=True, translate=True
) -> PipelineConfig:
    s = get_settings()
    return PipelineConfig(
        retrieval=RetrievalConfig(mode=mode, top_k=s.retrieval_top_k, final_k=s.final_k,
                                  rerank=rerank, rerank_candidates=s.rerank_candidates),
        abstain_rerank_threshold=s.abstain_rerank_threshold,
        abstain_vector_threshold=s.abstain_vector_threshold,
        verify=verify,
        auto_scope=auto_scope,
        translate_taglish=translate,
    )  # fmt: skip


CONFIGS = {
    c.name: c
    for c in [
        EvalConfig("vector", "Vector only", _pipeline("vector", rerank=False, verify=False,
                                                      auto_scope=False, translate=False)),
        EvalConfig("hybrid", "Hybrid (vector + keyword, RRF)",
                   _pipeline(rerank=False, verify=False, auto_scope=False, translate=False)),
        EvalConfig("hybrid_rerank", "Hybrid + rerank + abstain",
                   _pipeline(verify=False, auto_scope=False, translate=False)),
        EvalConfig("verify", "+ verification",
                   _pipeline(auto_scope=False, translate=False)),
        EvalConfig("full", "+ vehicle scoping + Taglish translation (full system)",
                   _pipeline()),
        # Smaller model, with and without verification: shows what verification catches.
        EvalConfig("noverify_llama3b", "llama3.2:3b, everything except verification",
                   _pipeline(verify=False), llm_model="llama3.2:3b"),
        EvalConfig("full_llama3b", "llama3.2:3b, full system", _pipeline(),
                   llm_model="llama3.2:3b"),
    ]
}  # fmt: skip


def norm(text: str) -> str:
    """Lowercase and drop spaces, hyphens and thousands separators for keyword matching."""
    return re.sub(r"[\s\-,]", "", text.lower())


def load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keywords_match(answer: str, keywords: list[str]) -> bool:
    """Every keyword must appear; 'a|b' accepts either spelling (e.g. English or Filipino)."""
    text = norm(answer)
    return all(any(norm(alt) in text for alt in k.split("|")) for k in keywords)


def hit_at_k(item: dict, candidates) -> bool:
    expected = {(doc, page) for doc, page in item["expected_sources"]}
    return any((c.filename, c.page) in expected for c in candidates)


def is_shown_as_fact(answered: bool, verified: bool, answer: str, verify_enabled: bool) -> bool:
    """Presented to staff as a trustworthy answer: not abstained, not flagged unverified,
    and not the parse-failure notice (which makes no factual claim)."""
    if not answered or answer == PARSE_FAILURE_ANSWER:
        return False
    return verified or not verify_enabled


def score_answer(item: dict, ans, verify_enabled: bool) -> dict:
    """Classify one answer. 'shown_as_fact': answered and not flagged as unverified
    (without runtime verification, every answer is shown as fact)."""
    answered = not ans.abstained
    correct = answered and keywords_match(ans.answer, item["answer_keywords"])
    cand = ans.retrieval.candidates if ans.retrieval else []
    cited_texts = [cand[c.n - 1].text for c in ans.citations if 0 < c.n <= len(cand)]
    # Measured the same way for every config, even ones that don't verify at runtime.
    supported = answered and verify_answer(ans.answer, ans.supporting_quote, cited_texts).ok
    shown_as_fact = is_shown_as_fact(answered, ans.verified, ans.answer, verify_enabled)
    return {
        "answered": answered,
        "correct": bool(correct) if item["answerable"] else False,
        "supported": supported,
        "shown_as_fact": shown_as_fact,
    }


def run_config(cfg: EvalConfig, items: list[dict], embedder, reranker) -> dict:
    s = get_settings()
    llm = OllamaLLM(s.ollama_url, cfg.llm_model or s.llm_model)
    rows = []
    with SessionLocal() as db:
        # Warm-up so model loading doesn't count toward latency.
        answer_question(db, "warm up", embedder=embedder, llm=llm, reranker=reranker,
                        config=replace(cfg.pipeline, verify=False))  # fmt: skip
        for item in items:
            t0 = time.perf_counter()
            ans = answer_question(
                db, item["question"], embedder=embedder, llm=llm,
                reranker=reranker if cfg.pipeline.retrieval.rerank else None,
                config=cfg.pipeline,
            )  # fmt: skip
            latency = time.perf_counter() - t0
            cands = ans.retrieval.candidates if ans.retrieval else []
            row = {
                "id": item["id"],
                "lang": item["lang"],
                "ocr": item.get("ocr", False),
                "answerable": item["answerable"],
                "question": item["question"],
                "expected": item["expected_answer"],
                "answer": ans.answer,
                "abstained": ans.abstained,
                "abstain_reason": ans.abstain_reason,
                "verified": ans.verified,
                "attempts": ans.attempts,
                "problems": ans.problems,
                "confidence": round(ans.retrieval.confidence, 4) if ans.retrieval else None,
                "hit_at_5": hit_at_k(item, cands) if item["answerable"] else None,
                "top5": [f"{c.filename}#p{c.page}" for c in cands],
                "citations": [f"{c.filename}#p{c.page}" for c in ans.citations],
                "latency_s": round(latency, 3),
                **score_answer(item, ans, cfg.pipeline.verify),
            }
            rows.append(row)
            mark = "abstain" if row["abstained"] else ("ok" if row["correct"] else "--")
            print(f"  [{item['id']:>2}] {mark:7} {latency:5.1f}s  {item['question'][:60]}")
    return {
        "config": cfg.name,
        "label": cfg.label,
        "summary": summarize(rows),
        "rows": rows,
    }


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    lat = sorted(r["latency_s"] for r in rows)
    # Confidently wrong: shown as fact, but wrong (answerable) or invented (unanswerable).
    conf_wrong = [
        r for r in rows
        if r["shown_as_fact"] and ((r["answerable"] and not r["correct"]) or not r["answerable"])
    ]  # fmt: skip
    taglish = [r for r in ans if r["lang"] == "taglish"]
    photo_scan = [r for r in ans if r.get("ocr")]
    return {
        "n": len(rows),
        "hit_at_5": _pct(sum(r["hit_at_5"] for r in ans), len(ans)),
        "answer_accuracy": _pct(sum(r["correct"] for r in ans), len(ans)),
        "taglish_accuracy": _pct(sum(r["correct"] for r in taglish), len(taglish)),
        "photo_scan_accuracy": _pct(sum(r["correct"] for r in photo_scan), len(photo_scan)),
        "false_abstain": _pct(sum(r["abstained"] for r in ans), len(ans)),
        "abstention_accuracy": _pct(sum(r["abstained"] for r in unans), len(unans)),
        "faithfulness": _pct(sum(r["supported"] for r in answered), len(answered)),
        "confidently_wrong": _pct(len(conf_wrong), len(rows)),
        "confidently_wrong_ids": [r["id"] for r in conf_wrong],
        "flagged_unverified": sum(1 for r in rows if r["answered"] and not r["shown_as_fact"]),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else None,
    }


def write_summary(results: list[dict]) -> Path:
    head = (
        "| Configuration | Hit@5 | Answer accuracy | Taglish accuracy | Photo/scan accuracy "
        "| Abstention accuracy | False abstain | Faithfulness | Confidently wrong "
        "| p50 / p95 latency |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r['label']} | {s['hit_at_5']}% | {s['answer_accuracy']}% | "
            f"{s['taglish_accuracy']}% | {s.get('photo_scan_accuracy')}% | "
            f"{s['abstention_accuracy']}% | {s['false_abstain']}% | "
            f"{s['faithfulness']}% | {s['confidently_wrong']}% | "
            f"{s['latency_p50_s']}s / {s['latency_p95_s']}s |"
        )
    out = RESULTS / "summary.md"
    existing = {}
    if out.exists():  # keep rows for configs not re-run this time
        for line in out.read_text(encoding="utf-8").splitlines()[2:]:
            if line.startswith("| "):
                existing[line.split("|")[1].strip()] = line
    for line, r in zip(lines, results, strict=True):
        existing[r["label"]] = line
    # Rows in config order; labels no longer in CONFIGS are dropped.
    body = "\n".join(existing[c.label] for c in CONFIGS.values() if c.label in existing)
    out.write_text(head + body + "\n", encoding="utf-8")
    return out


def rescore(names: list[str]) -> None:
    """Re-grade saved results after editing answer keywords, without re-running models."""
    items = {i["id"]: i for i in load_dataset()}
    results = []
    for name in names:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for row in result["rows"]:
            item = items[row["id"]]
            row["correct"] = (
                item["answerable"]
                and row["answered"]
                and keywords_match(row["answer"], item["answer_keywords"])
            )
            row["shown_as_fact"] = is_shown_as_fact(
                row["answered"], row["verified"], row["answer"], CONFIGS[name].pipeline.verify
            )
        result["summary"] = summarize(result["rows"])
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(name, json.dumps(result["summary"]))
        results.append(result)
    print(f"Wrote {write_summary(results)}")


def retrieval_only(items: list[dict], embedder, reranker) -> None:
    """Print the best rerank score per question, to choose the abstain threshold."""
    cfg = RetrievalConfig()
    s = get_settings()
    llm = OllamaLLM(s.ollama_url, s.llm_model)
    with SessionLocal() as db:
        for item in items:
            q = item["question"]
            vid = resolve_vehicle(db, q)
            alt = [translate_to_english(llm, q)] if looks_filipino(q) else None
            r = retrieve(db, q, embedder, reranker, cfg, vehicle_id=vid, alt_queries=alt)
            hit = hit_at_k(item, r.candidates) if item["answerable"] else None
            tag = "ANS " if item["answerable"] else "NONE"
            print(f"{tag} conf={r.confidence:.3f} hit={hit!s:5} [{item['id']:>2}] "
                  f"{item['question'][:60]}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="re-grade saved results only")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.configs)
        return

    items = load_dataset()
    s = get_settings()
    embedder = get_embedder()
    reranker = CrossEncoderReranker(s.reranker_model, s.reranker_device)

    if args.retrieval_only:
        retrieval_only(items, embedder, reranker)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n== {cfg.label}")
        result = run_config(cfg, items, embedder, reranker)
        (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("  ", json.dumps(result["summary"]))
        results.append(result)
    print(f"\nWrote {write_summary(results)}")


if __name__ == "__main__":
    main()
