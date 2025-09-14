// netlify/functions/recommend.mjs
import fs from "fs/promises";
import path from "path";
import { pipeline } from "@xenova/transformers";

let DATA = null;        // { model, items: [{ food_name, ingredients, procedures, embedding: number[] }, ...] }
let READY = false;
let EMBEDDER = null;    // lazy-initialized at first semantic request

const norm = (v) => {
  const s = Math.sqrt(v.reduce((a, b) => a + b*b, 0)) || 1;
  return v.map(x => x / s);
};
const cosine = (a, b) => a.reduce((s, x, i) => s + x*b[i], 0); // if both are normalized

async function loadData() {
  if (DATA) return DATA;
  const p = path.join(process.cwd(), "public", "data", "recipes_with_embeddings.json");
  const raw = await fs.readFile(p, "utf8");
  DATA = JSON.parse(raw);
  READY = true;
  return DATA;
}

async function ensureEmbedder() {
  if (!EMBEDDER) {
    EMBEDDER = await pipeline("feature-extraction", "sentence-transformers/paraphrase-MiniLM-L3-v2");
  }
  return EMBEDDER;
}

function json(statusCode, body) {
  return {
    statusCode,
    headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
    body: JSON.stringify(body),
  };
}

function text(statusCode, body) {
  return {
    statusCode,
    headers: { "content-type": "text/plain", "access-control-allow-origin": "*" },
    body,
  };
}

function matchByIngredients(q, items, threshold = 0.7) {
  const wanted = q.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
  const out = [];
  for (const it of items) {
    const recipeIngs = it.ingredients.toLowerCase().split(/,|\n|\r/).map(s => s.trim()).filter(Boolean);
    const matches = wanted.filter(w => recipeIngs.includes(w));
    if (!wanted.length) continue;
    const ratio = matches.length / wanted.length;
    if (ratio >= threshold) out.push(it);
  }
  return out;
}

function matchByName(q, items) {
  const parts = q.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
  return items.filter(it => parts.every(p => it.food_name.toLowerCase().includes(p)));
}

async function matchSemantic(q, items, threshold = 0.6) {
  await ensureEmbedder();
  const out = await EMBEDDER([q.toLowerCase()], { pooling: "mean", normalize: true });
  const emb = Array.from(out.data[0]); // normalized
  const scored = items
    .map((it) => ({ it, score: cosine(emb, it.embedding) }))
    .filter(x => x.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .map(x => x.it);
  return scored;
}

export async function handler(event) {
  try {
    const { httpMethod, path: reqPath, queryStringParameters } = event;
    if (httpMethod === "OPTIONS") {
      return {
        statusCode: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET,OPTIONS",
          "access-control-allow-headers": "Content-Type"
        },
        body: ""
      };
    }

    const data = await loadData();
    const items = data.items;

    // Health
    if (reqPath.endsWith("/health") || reqPath === "/.netlify/functions/recommend") {
      return json(200, { status: READY ? "ok" : "loading", rows: items.length });
    }

    // by-ingredients
    if (reqPath.endsWith("/by-ingredients")) {
      const q = (queryStringParameters?.q || "").trim();
      const t = parseFloat(queryStringParameters?.threshold || "0.7");
      const results = matchByIngredients(q, items, isNaN(t) ? 0.7 : t);
      return json(200, { items: results });
    }

    // by-name
    if (reqPath.endsWith("/by-name")) {
      const q = (queryStringParameters?.q || "").trim();
      const results = matchByName(q, items);
      return json(200, { items: results });
    }

    // semantic (chat / talk to me)
    if (reqPath.endsWith("/semantic")) {
      const q = (queryStringParameters?.q || "").trim();
      const t = parseFloat(queryStringParameters?.threshold || "0.6");
      const results = await matchSemantic(q, items, isNaN(t) ? 0.6 : t);
      return json(200, { items: results });
    }

    // If you’re using the /api/* redirect, map here:
    if (reqPath.startsWith("/api/recommend")) {
      const sub = reqPath.replace("/api/recommend", "");
      event.path = "/.netlify/functions/recommend" + sub;
      return await handler({ ...event, path: event.path });
    }

    return text(404, "Not found");
  } catch (e) {
    console.error(e);
    return json(500, { error: String(e) });
  }
}
