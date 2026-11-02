import { expect, type Page, test } from "@playwright/test";

const API = "http://localhost:8000";

const CITATION = {
  n: 1,
  chunk_id: 11,
  document_id: 3,
  filename: "nbc1234_deed_of_sale.pdf",
  doc_type: "deed_of_sale",
  page: 1,
  snippet: "Seller: JUAN P. DELA CRUZ\nBuyer: MARIA L. SANTOS\nPrice: PHP 485,000.00",
};

const ANSWERS = {
  verified: {
    answer: "Maria L. Santos bought it for PHP 485,000.00.",
    citations: [CITATION],
    abstained: false,
    verified: true,
    supporting_quote: "Price: PHP 485,000.00",
  },
  notFound: {
    answer: "I couldn't find this in the uploaded documents.",
    citations: [],
    abstained: true,
    verified: true,
    supporting_quote: "",
  },
  unverified: {
    answer: "It was sold for PHP 500,000.",
    citations: [CITATION, { ...CITATION, n: 2, chunk_id: 12, filename: "nbc1234_orcr.pdf" }],
    abstained: false,
    verified: false,
    supporting_quote: "",
  },
};

/** Mock the backend: a signed-in staff user and a scripted /chat reply. */
async function mockApi(page: Page, chat: object) {
  await page.route(`${API}/**`, async (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (body: unknown) => route.fulfill({ json: body });
    if (path === "/auth/login") return json({ access_token: "test-token", token_type: "bearer" });
    if (path === "/me") return json({ id: 1, email: "staff@cardocs.local", role: "staff" });
    if (path === "/vehicles")
      return json([
        { id: 1, plate: "NBC 1234", vin: null, make: "Toyota", model: "Vios", year: 2019, status: "sold" },
      ]);
    if (path === "/chat") return json({ ...chat, conversation_id: 1, message_id: 2 });
    if (path === "/documents/3/file")
      return json({ url: "http://storage.test/doc.pdf?sig=x", expires_in: 600 });
    if (path === "/documents/7/reprocess") return json({ id: 7, status: "pending" });
    if (path === "/documents")
      return json([
        {
          id: 7, vehicle_id: 1, doc_type: "or_cr", filename: "orcr_photo.jpg", mime_type: "application/pdf",
          status: "ready", error: null, page_count: 1, ocr_confidence: 64.2, uploaded_by: 1,
          created_at: "2026-10-21T00:00:00Z",
        },
      ]);
    return route.fulfill({ status: 404, json: { detail: "not mocked" } });
  });
}

async function signIn(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill("staff@cardocs.local");
  await page.getByLabel("Password").fill("staff123");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: /Ask about any vehicle/ })).toBeVisible();
}

async function askQuestion(page: Page, question: string) {
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
}

test("signed-out visitors are sent to the login page", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/login$/);
});

test("verified answer shows the stamp, quote and a citation that opens the source", async ({
  page,
}) => {
  await mockApi(page, ANSWERS.verified);
  await signIn(page);
  await askQuestion(page, "Who bought the Vios?");

  await expect(page.getByText("Verified against source")).toBeVisible();
  await expect(page.getByText(ANSWERS.verified.answer)).toBeVisible();
  await page.getByRole("button", { name: /\[1\] nbc1234_deed_of_sale\.pdf, p\. 1/ }).click();
  const source = page.locator("mark:visible").filter({ hasText: "Price: PHP 485,000.00" });
  await expect(source).toBeVisible();
});

test("not-found answer says so and shows no sources", async ({ page }) => {
  await mockApi(page, ANSWERS.notFound);
  await signIn(page);
  await askQuestion(page, "What is the Mirage's insurance policy number?");

  await expect(page.getByText("Not in the documents")).toBeVisible();
  await expect(page.getByText(/upload the document that has this information/)).toBeVisible();
  await expect(page.getByRole("list", { name: "Sources" })).toHaveCount(0);
});

test("unverified answer warns and lists every retrieved excerpt", async ({ page }) => {
  await mockApi(page, ANSWERS.unverified);
  await signIn(page);
  await askQuestion(page, "How much was the Vios sold for?");

  await expect(page.getByText("Unverified: check the sources")).toBeVisible();
  await expect(page.getByText(/couldn't be matched word-for-word/)).toBeVisible();
  await expect(page.getByRole("list", { name: "Sources" }).getByRole("button")).toHaveCount(2);
});

test("vehicle filter is sent with the question", async ({ page }) => {
  await mockApi(page, ANSWERS.verified);
  await signIn(page);
  await page.getByLabel("Vehicle").selectOption({ label: "NBC 1234 Toyota Vios 2019" });

  const request = page.waitForRequest(`${API}/chat`);
  await askQuestion(page, "Who bought it?");
  expect((await request).postDataJSON()).toMatchObject({ question: "Who bought it?", vehicle_id: 1 });
});

test("a citation read from a poor photo warns staff to check the original", async ({ page }) => {
  await mockApi(page, {
    ...ANSWERS.verified,
    citations: [{ ...CITATION, filename: "orcr_photo.jpg", ocr_confidence: 58.4 }],
  });
  await signIn(page);
  await askQuestion(page, "Who bought the Vios?");

  await page.getByRole("button", { name: /\[1\] orcr_photo\.jpg/ }).click();
  await expect(page.getByText(/Low-quality photo\/scan \(58% read confidence\)/).locator("visible=true")).toBeVisible();
});

test("documents page accepts photos and shows how well they were read", async ({ page }) => {
  await mockApi(page, ANSWERS.verified);
  await signIn(page);
  await page.getByRole("link", { name: "Documents" }).click();

  await expect(page.getByRole("button", { name: "Take photo" })).toBeVisible();
  await expect(page.getByText("Read from a photo/scan (64% confidence)")).toBeVisible();

  await page.locator('input[type=file][multiple]').setInputFiles({
    name: "orcr.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from([0xff, 0xd8, 0xff, 0xe0]),
  });
  await expect(page.getByText("orcr.jpg")).toBeVisible();
  await expect(page.getByText(/can be uploaded/)).toHaveCount(0);

  const reread = page.waitForRequest((r) => r.url().endsWith("/documents/7/reprocess"));
  await page.getByRole("button", { name: "Re-read" }).click();
  expect((await reread).method()).toBe("POST");
});
