# DocuKotse frontend

Next.js (App Router) + Tailwind. Pages: login, Ask (chat with citations), Documents
(upload with live status). Talks to the FastAPI backend at `NEXT_PUBLIC_API_URL`
(default `http://localhost:8000`).

```sh
npm install
npm run dev          # http://localhost:3000, needs the API running
npm run lint
npm run build && npx playwright install chromium && npm run test:e2e   # API is mocked
```
