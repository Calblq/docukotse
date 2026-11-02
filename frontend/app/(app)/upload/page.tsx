"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useUser } from "@/components/AppShell";
import { api, ApiError, openDocumentPage } from "@/lib/api";
import {
  DOC_TYPE_LABELS,
  type DocStatus,
  isUploadable,
  OCR_LOW_CONFIDENCE,
  UPLOAD_ACCEPT,
  type DocType,
  type DocumentRow,
  type Vehicle,
  vehicleLabel,
} from "@/lib/types";

const STATUS_TEXT: Record<DocStatus, string> = {
  pending: "Waiting",
  processing: "Reading",
  ready: "Searchable",
  failed: "Failed",
};

const STATUS_CLASS: Record<DocStatus, string> = {
  pending: "text-ink-soft",
  processing: "text-form",
  ready: "text-stamp-ok",
  failed: "text-danger",
};

export default function DocumentsPage() {
  const user = useUser();
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [vehicleId, setVehicleId] = useState("");
  const [docType, setDocType] = useState<DocType | "">("");
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  const loadDocs = useCallback(() => {
    api<DocumentRow[]>("/documents")
      .then(setDocs)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    api<Vehicle[]>("/vehicles")
      .then(setVehicles)
      .catch(() => undefined);
    loadDocs();
  }, [loadDocs]);

  // Poll while anything is still being read, so status updates live.
  const inFlight = docs.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!inFlight) return;
    const id = setInterval(loadDocs, 2000);
    return () => clearInterval(id);
  }, [inFlight, loadDocs]);

  function addFiles(list: FileList | null) {
    if (!list) return;
    const accepted = Array.from(list).filter(isUploadable);
    if (accepted.length < list.length) {
      setMessage({
        kind: "error",
        text: "Only PDFs and photos (JPG, PNG, WEBP, HEIC) can be uploaded. Other files were skipped.",
      });
    } else {
      setMessage(null);
    }
    setFiles((prev) => [...prev, ...accepted]);
  }

  async function upload() {
    if (!docType || files.length === 0) return;
    setUploading(true);
    setMessage(null);
    const failed: string[] = [];
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      form.append("doc_type", docType);
      if (vehicleId) form.append("vehicle_id", vehicleId);
      try {
        await api("/documents", { method: "POST", body: form });
      } catch (err) {
        failed.push(`${file.name}: ${err instanceof ApiError ? err.message : "upload failed"}`);
      }
    }
    setUploading(false);
    setFiles([]);
    loadDocs();
    setMessage(
      failed.length
        ? { kind: "error", text: failed.join(" ") }
        : { kind: "ok", text: "Uploaded. Documents become searchable once they show Searchable." },
    );
  }

  async function reread(doc: DocumentRow) {
    try {
      await api(`/documents/${doc.id}/reprocess`, { method: "POST" });
      loadDocs();
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof ApiError ? err.message : "Re-read failed." });
    }
  }

  async function remove(doc: DocumentRow) {
    if (!confirm(`Delete ${doc.filename}? This removes the file and its search index.`)) return;
    try {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      loadDocs();
    } catch (err) {
      setMessage({ kind: "error", text: err instanceof ApiError ? err.message : "Delete failed." });
    }
  }

  const plateOf = new Map(vehicles.map((v) => [v.id, v.plate ?? `${v.make} ${v.model}`]));

  return (
    <div className="space-y-10">
      <section aria-labelledby="upload-heading" className="max-w-2xl">
        <h1 id="upload-heading" className="text-xl font-bold tracking-tight">
          Upload documents
        </h1>
        <p className="mt-1 text-ink-soft">
          PDFs or photos of OR/CRs, deeds of sale, insurance, inspection and loan papers. Photos
          are read automatically. For best results, lay the paper flat in good light and fill the
          frame.
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="font-medium">Vehicle</span>
            <select
              value={vehicleId}
              onChange={(e) => setVehicleId(e.target.value)}
              className="mt-1 w-full rounded border border-rule bg-sheet px-2 py-2"
            >
              <option value="">Not linked to a vehicle</option>
              {vehicles.map((v) => (
                <option key={v.id} value={v.id}>
                  {vehicleLabel(v)}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="font-medium">Document type</span>
            <select
              required
              value={docType}
              onChange={(e) => setDocType(e.target.value as DocType)}
              className="mt-1 w-full rounded border border-rule bg-sheet px-2 py-2"
            >
              <option value="" disabled>
                Choose a type
              </option>
              {Object.entries(DOC_TYPE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`mt-4 rounded-md border-2 border-dashed p-6 text-center ${
            dragging ? "border-form bg-form-tint" : "border-rule bg-sheet"
          }`}
        >
          <p className="text-ink-soft">Drop PDFs or photos here, or</p>
          <div className="mt-2 flex flex-wrap justify-center gap-2">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="rounded border border-form px-3 py-1.5 text-sm font-semibold text-form hover:bg-form-tint"
            >
              Choose files
            </button>
            <button
              type="button"
              onClick={() => cameraRef.current?.click()}
              className="rounded border border-form px-3 py-1.5 text-sm font-semibold text-form hover:bg-form-tint"
            >
              Take photo
            </button>
          </div>
          <input
            ref={cameraRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="sr-only"
            aria-label="Take a photo of a document"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={inputRef}
            type="file"
            accept={UPLOAD_ACCEPT}
            multiple
            className="sr-only"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          {files.length > 0 && (
            <ul className="mt-4 space-y-1 text-left text-sm">
              {files.map((f, i) => (
                <li key={`${f.name}-${i}`} className="flex justify-between gap-2">
                  <span className="truncate">{f.name}</span>
                  <button
                    onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                    className="text-ink-soft hover:text-danger"
                    aria-label={`Remove ${f.name}`}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="mt-4 flex items-center gap-4">
          <button
            onClick={upload}
            disabled={uploading || !docType || files.length === 0}
            className="rounded bg-form px-5 py-2 font-semibold text-white hover:bg-form-dark disabled:opacity-50"
          >
            {uploading
              ? "Uploading…"
              : files.length === 0
                ? "Upload"
                : `Upload ${files.length} ${files.length === 1 ? "file" : "files"}`}
          </button>
          {!docType && files.length > 0 && (
            <span className="text-sm text-ink-soft">Choose a document type first.</span>
          )}
        </div>
        {message && (
          <p
            role={message.kind === "error" ? "alert" : "status"}
            className={`mt-3 text-sm ${message.kind === "error" ? "text-danger" : "text-stamp-ok"}`}
          >
            {message.text}
          </p>
        )}
      </section>

      <section aria-labelledby="docs-heading">
        <h2 id="docs-heading" className="text-lg font-bold tracking-tight">
          All documents
        </h2>
        {docs.length === 0 ? (
          <p className="mt-2 text-ink-soft">No documents yet. Upload the first one above.</p>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-md border border-rule bg-sheet">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-rule text-ink-soft">
                <tr>
                  <th className="px-3 py-2 font-medium">File</th>
                  <th className="px-3 py-2 font-medium">Vehicle</th>
                  <th className="px-3 py-2 font-medium">Type</th>
                  <th className="px-3 py-2 font-medium">Pages</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="border-b border-rule last:border-0 align-top">
                    <td className="px-3 py-2">
                      <button
                        onClick={() => openDocumentPage(d.id, 1).catch(() => undefined)}
                        className="text-left text-form hover:underline"
                      >
                        {d.filename}
                      </button>
                    </td>
                    <td className="px-3 py-2 font-mono whitespace-nowrap">
                      {d.vehicle_id ? plateOf.get(d.vehicle_id) : "–"}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap">{DOC_TYPE_LABELS[d.doc_type]}</td>
                    <td className="px-3 py-2">{d.page_count ?? "–"}</td>
                    <td className={`px-3 py-2 whitespace-nowrap font-medium ${STATUS_CLASS[d.status]}`}>
                      {STATUS_TEXT[d.status]}
                      {d.error && <p className="max-w-xs font-normal text-ink-soft">{d.error}</p>}
                      {d.ocr_confidence != null && (
                        <p
                          className={`font-normal ${
                            d.ocr_confidence < OCR_LOW_CONFIDENCE ? "text-stamp-warn" : "text-ink-soft"
                          }`}
                        >
                          Read from a photo/scan ({Math.round(d.ocr_confidence)}% confidence)
                        </p>
                      )}
                    </td>
                    <td className="space-x-3 px-3 py-2 text-right whitespace-nowrap">
                      {(d.status === "ready" || d.status === "failed") && (
                        <button
                          onClick={() => reread(d)}
                          className="text-ink-soft hover:text-form"
                          title="Read the file again, e.g. after OCR improvements"
                        >
                          Re-read
                        </button>
                      )}
                      {user?.role === "admin" && (
                        <button onClick={() => remove(d)} className="text-ink-soft hover:text-danger">
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
