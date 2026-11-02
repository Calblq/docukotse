import type { ChatResponse } from "@/lib/types";

/** Answer status as a rubber stamp: the one place the UI raises its voice. */
export default function Stamp({ result }: { result: ChatResponse }) {
  if (result.abstained) {
    return <span className="stamp stamp-in text-stamp-none">Not in the documents</span>;
  }
  if (!result.verified) {
    return <span className="stamp stamp-in text-stamp-warn">Unverified: check the sources</span>;
  }
  return <span className="stamp stamp-in text-stamp-ok">Verified against source</span>;
}
