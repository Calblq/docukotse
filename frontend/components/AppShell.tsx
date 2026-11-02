"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";

import { api, getToken, setToken } from "@/lib/api";
import type { User } from "@/lib/types";

const UserContext = createContext<User | null>(null);

export function useUser(): User | null {
  return useContext(UserContext);
}

const NAV = [
  { href: "/", label: "Ask" },
  { href: "/upload", label: "Documents" },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    api<User>("/me")
      .then(setUser)
      .catch(() => router.replace("/login"));
  }, [router]);

  function signOut() {
    setToken(null);
    router.replace("/login");
  }

  if (!user) {
    return <div className="flex-1" aria-busy="true" />;
  }

  return (
    <UserContext.Provider value={user}>
      <header className="border-b border-rule bg-sheet">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="font-bold tracking-tight">DocuKotse</span>
          <nav className="flex gap-1" aria-label="Main">
            {NAV.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={`rounded px-3 py-1.5 text-sm ${
                    active ? "bg-form-tint font-semibold text-form-dark" : "text-ink-soft hover:text-ink"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm text-ink-soft">
            <span className="hidden sm:inline">
              {user.email} ({user.role})
            </span>
            <button onClick={signOut} className="rounded px-2 py-1 hover:text-ink">
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
    </UserContext.Provider>
  );
}
