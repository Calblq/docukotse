"use client";

import { useEffect, useRef, useState } from "react";

import Stamp from "@/components/Stamp";
import { api, ApiError, openDocumentPage } from "@/lib/api";
import {
  type ChatResponse,
  type Citation,
  DOC_TYPE_LABELS,
  type DocType,
  OCR_LOW_CONFIDENCE,
  type Vehicle,
  vehicleLabel,
} from "@/lib/types";

interface Selection {
  turn: number;
  citation: Citation;
  quote: string;
}

interface Turn {
  question: string;
  result?: ChatResponse;
  error?: string;
}

const EXAMPLES = [
  "Who bought the Vios with plate NBC-1234, and for how much?",
  "Magkano ang monthly amortization ng Fortuner?",
  "What is the deductible on the Honda City's insurance?",
  "Kailan mag-e-expire ang registration ng CAR 7890?",
];

export default function AskPage() {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [vehicleId, setVehicleId] = useState("");
  const [docType, setDocType] = useState("");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api<Vehicle[]>("/vehicles")
      .then(setVehicles)
      .catch(() => setVehicles([]));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ask(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [...t, { question: q }]);
    try {
      const result = await api<ChatResponse>("/chat", {
        method: "POST",
        body: JSON.stringify({
          question: q,
          vehicle_id: vehicleId ? Number(vehicleId) : null,
          doc_type: docType || null,
          conversation_id: conversationId,
        }),
      });
      setConversationId(result.conversation_id);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, result } : turn)));
      const first = result.citations[0];
      setSelected(
        first ? { turn: turns.length, citation: first, quote: result.supporting_quote } : null,
      );
    } catch (err) {
      const error = err instanceof ApiError ? err.message : "Something went wrong. Try again.";
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error } : turn)));
    } finally {
      setBusy(false);
    }
  }

  async function openSource(c: Citation) {
    setOpenError(null);
    try {
      await openDocumentPage(c.document_id, c.page);
    } catch (err) {
      setOpenError(err instanceof ApiError ? err.message : "Couldn't open the file.");
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <section aria-label="Conversation" className="flex min-h-[70vh] flex-col">
        <div className="flex flex-wrap gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-ink-soft">Vehicle</span>
            <select
              value={vehicleId}
              onChange={(e) => setVehicleId(e.target.value)}
              className="rounded border border-rule bg-sheet px-2 py-1.5"
            >
              <option value="">Any vehicle</option>
              {vehicles.map((v) => (
                <option key={v.id} value={v.id}>
                  {vehicleLabel(v)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="text-ink-soft">Document</span>
            <select
              value={docType}
              onChange={(e) => setDocType(e.target.value)}
              className="rounded border border-rule bg-sheet px-2 py-1.5"
            >
              <option value="">Any type</option>
              {Object.entries(DOC_TYPE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mt-6 flex-1 space-y-8">
          {turns.length === 0 && (
            <div className="max-w-prose">
              <h1 className="text-xl font-bold tracking-tight">Ask about any vehicle&apos;s papers</h1>
              <p className="mt-2 text-ink-soft">
                Answers come only from uploaded documents, in English, Filipino or Taglish. If the
                papers don&apos;t say, you&apos;ll be told so instead of getting a guess.
              </p>
              <ul className="mt-5 space-y-2">
                {EXAMPLES.map((ex) => (
                  <li key={ex}>
                    <button
                      onClick={() => ask(ex)}
                      className="text-left text-form underline decoration-rule underline-offset-4 hover:decoration-form"
                    >
                      {ex}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turns.map((turn, i) => (
            <article key={i} className="max-w-prose">
              <p className="font-semibold">{turn.question}</p>
              {!turn.result && !turn.error && (
                <p className="mt-3 text-ink-soft" aria-live="polite">
                  Searching the documents…
                </p>
              )}
              {turn.error && (
                <p role="alert" className="mt-3 text-danger">
                  {turn.error}
                </p>
              )}
              {turn.result && (
                <Answer
                  result={turn.result}
                  selectedId={selected?.turn === i ? selected.citation.chunk_id : null}
                  onSelect={(c) =>
                    setSelected({ turn: i, citation: c, quote: turn.result!.supporting_quote })
                  }
                />
              )}
              {selected?.turn === i && (
                // On small screens the source opens right under its answer.
                <div className="mt-4 lg:hidden">
                  <SourcePanel selection={selected} onOpen={openSource} error={openError} />
                </div>
              )}
            </article>
          ))}
          <div ref={endRef} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
          className="sticky bottom-0 mt-6 flex gap-2 border-t border-rule bg-paper pt-4 pb-2"
        >
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <textarea
            id="question"
            rows={2}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(question);
              }
            }}
            placeholder="e.g. Sino ang buyer ng DAE 9012?"
            className="flex-1 resize-none rounded border border-rule bg-sheet px-3 py-2"
          />
          <button
            type="submit"
            disabled={busy || !question.trim()}
            className="self-end rounded bg-form px-5 py-2 font-semibold text-white hover:bg-form-dark disabled:opacity-50"
          >
            Ask
          </button>
        </form>
      </section>

      <aside aria-label="Source" className="hidden lg:sticky lg:top-6 lg:block lg:self-start">
        <SourcePanel selection={selected} onOpen={openSource} error={openError} />
      </aside>
    </div>
  );
}

function Answer({
  result,
  selectedId,
  onSelect,
}: {
  result: ChatResponse;
  selectedId: number | null;
  onSelect: (c: Citation) => void;
}) {
  return (
    <div className="mt-3">
      <Stamp result={result} />
      <p className="mt-3 text-lg leading-relaxed">{result.answer}</p>

      {result.abstained && (
        <p className="mt-2 text-sm text-ink-soft">
          Try naming the plate number or vehicle, or upload the document that has this
          information.
        </p>
      )}
      {!result.abstained && !result.verified && (
        <p className="mt-2 text-sm text-stamp-warn">
          This answer couldn&apos;t be matched word-for-word to the source text. Check the
          excerpts below before relying on it.
        </p>
      )}
      {result.verified && result.supporting_quote && (
        <blockquote className="mt-3 border-l-2 border-stamp-ok pl-3 text-sm text-ink-soft">
          “{result.supporting_quote}”
        </blockquote>
      )}

      {result.citations.length > 0 && (
        <ol className="mt-4 flex flex-wrap gap-2" aria-label="Sources">
          {result.citations.map((c) => {
            const active = selectedId === c.chunk_id;
            return (
              <li key={c.chunk_id}>
                <button
                  onClick={() => onSelect(c)}
                  aria-pressed={active}
                  className={`rounded border px-2.5 py-1 text-sm ${
                    active
                      ? "border-form bg-form-tint text-form-dark"
                      : "border-rule bg-sheet hover:border-form"
                  }`}
                >
                  <span className="font-semibold">[{c.n}]</span> {c.filename}, p. {c.page}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function SourcePanel({
  selection,
  onOpen,
  error,
}: {
  selection: Selection | null;
  onOpen: (c: Citation) => void;
  error: string | null;
}) {
  if (!selection) {
    return (
      <div className="rounded-md border border-dashed border-rule p-4 text-sm text-ink-soft">
        Sources for an answer appear here. Select one to read the excerpt and open the page.
      </div>
    );
  }
  const { citation } = selection;
  return (
    <div className="rounded-md border border-rule bg-sheet">
      <div className="border-b border-rule px-4 py-3">
        <p className="font-semibold">
          [{citation.n}] {citation.filename}
        </p>
        <p className="text-sm text-ink-soft">
          {DOC_TYPE_LABELS[citation.doc_type as DocType] ?? citation.doc_type}, page{" "}
          {citation.page}
        </p>
        {citation.ocr_confidence != null && <OcrNote confidence={citation.ocr_confidence} />}
      </div>
      <p className="whitespace-pre-line px-4 py-3 text-sm leading-relaxed">
        <Highlighted text={citation.snippet} quote={selection.quote} />
      </p>
      <div className="border-t border-rule px-4 py-3">
        <button
          onClick={() => onOpen(citation)}
          className="rounded bg-form px-3 py-1.5 text-sm font-semibold text-white hover:bg-form-dark"
        >
          Open page {citation.page}
        </button>
        {error && (
          <p role="alert" className="mt-2 text-sm text-danger">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

/** Marks the supporting quote inside an excerpt (whitespace-insensitive, case-insensitive). */
function Highlighted({ text, quote }: { text: string; quote: string }) {
  const words = quote.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return <>{text}</>;
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const match = new RegExp(escaped.join("\\s+"), "i").exec(text);
  if (!match) return <>{text}</>;
  return (
    <>
      {text.slice(0, match.index)}
      <mark className="rounded-sm bg-stamp-ok/15 px-0.5 text-ink">{match[0]}</mark>
      {text.slice(match.index + match[0].length)}
    </>
  );
}

/** Pages read from a photo can contain misread characters (e.g. 2021 read as 2024). */
function OcrNote({ confidence }: { confidence: number }) {
  const low = confidence < OCR_LOW_CONFIDENCE;
  return (
    <p className={`mt-2 text-sm ${low ? "font-medium text-stamp-warn" : "text-ink-soft"}`}>
      {low
        ? `Low-quality photo/scan (${Math.round(confidence)}% read confidence). Open the page and check it before relying on this.`
        : `Read from a photo/scan (${Math.round(confidence)}% read confidence). Check numbers against the original.`}
    </p>
  );
}
