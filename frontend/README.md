This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Aletheia + RAG API

The UI POSTs to **`NEXT_PUBLIC_API_URL`** (default **`http://localhost:8000`**) at **`/query`** with JSON **`{ "question": "<text>", "pipeline_version": "v1" }`** (same as `QueryRequest` in FastAPI).

1. Start the FastAPI app on port **8000** (from repo root):

   `uv run uvicorn src.api.main:app --reload`

   Or use Docker Compose (`api` service maps **host `8000` → container `8000`**).

2. If the API runs elsewhere, set e.g. `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` in **`frontend/.env.local`** and restart `next dev`.

The API enables CORS for **`http(s)://localhost` / `127.0.0.1` with any port** in dev. For locked-down deployments, set **`CORS_ALLOW_ORIGINS`** on the API (comma-separated list) instead of relying on the default regex.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
