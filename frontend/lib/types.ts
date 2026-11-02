export type Role = "admin" | "staff";

export interface User {
  id: number;
  email: string;
  role: Role;
}

export interface Vehicle {
  id: number;
  plate: string | null;
  vin: string | null;
  make: string;
  model: string;
  year: number | null;
  status: "acquired" | "listed" | "sold";
}

export type DocType =
  | "or_cr"
  | "deed_of_sale"
  | "insurance"
  | "inspection"
  | "financing"
  | "id"
  | "other";

export type DocStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentRow {
  id: number;
  vehicle_id: number | null;
  doc_type: DocType;
  filename: string;
  status: DocStatus;
  error: string | null;
  page_count: number | null;
  ocr_confidence?: number | null;
  created_at: string;
}

export interface Citation {
  n: number;
  chunk_id: number;
  document_id: number;
  filename: string;
  doc_type: DocType;
  page: number;
  snippet: string;
  ocr_confidence?: number | null;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  abstained: boolean;
  abstain_reason?: "low_confidence" | "llm_not_answerable" | null;
  verified: boolean;
  supporting_quote: string;
  conversation_id: number;
  message_id: number;
}

export const DOC_TYPE_LABELS: Record<DocType, string> = {
  or_cr: "OR/CR",
  deed_of_sale: "Deed of sale",
  insurance: "Insurance",
  inspection: "Inspection report",
  financing: "Financing",
  id: "ID",
  other: "Other",
};

/** Same threshold as the API's OCR_LOW_CONFIDENCE: below it, a page may be misread. */
export const OCR_LOW_CONFIDENCE = 70;

/** File types the upload accepts (photos are converted to PDF and read with OCR). */
export const UPLOAD_ACCEPT =
  "application/pdf,.pdf,image/jpeg,.jpg,.jpeg,image/png,.png,image/webp,.webp,image/heic,image/heif,.heic,.heif";
const UPLOAD_EXTENSIONS = [".pdf", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"];

export function isUploadable(file: File): boolean {
  const name = file.name.toLowerCase();
  return UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext)) || file.type.startsWith("image/");
}

export function vehicleLabel(v: Vehicle): string {
  return [v.plate, v.make, v.model, v.year].filter(Boolean).join(" ");
}
