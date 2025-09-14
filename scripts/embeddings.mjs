// scripts/build-embeddings.mjs
import fs from "fs/promises";
import path from "path";
import Papa from "papaparse";
import { pipeline } from "@xenova/transformers";

const ROOT = process.cwd();
const CSV_PATH = path.join(ROOT, "data", "Nigerian Palatable meals.csv");
const OUT_DIR = path.join(ROOT, "public", "data");
const OUT_PATH = path.join(OUT_DIR, "recipes_with_embeddings.json");

async function readCSV(filepath) {
  const buf = await fs.readFile(filepath, "utf8");
  return new Promise((resolve) => {
    Papa.parse(buf, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data)
    });
  });
}

function normalizeRow(row) {
  const food_name = (row.food_name || row.Food_Name || "").toString().trim();
  const ingredients = (row.ingredients || row.Ingredients || "").toString().replace(/\r?\n/g, ", ").trim();
  const procedures = (row.procedures || row.Procedures || "").toString().trim();
  return { food_name, ingredients, procedures };
}

function toCombined(r) {
  return `${r.food_name}, ${r.ingredients}, ${r.procedures}`.toLowerCase();
}

function l2norm(v) {
  const s = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
  return v.map(x => x / s);
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const rows = (await readCSV(CSV_PATH)).map(normalizeRow).filter(r => r.food_name);
  console.log(`Loaded ${rows.length} recipes`);

  // Sentence embedding pipeline
  const embed = await pipeline("feature-extraction", "sentence-transformers/paraphrase-MiniLM-L3-v2");

  // Batch to avoid memory spikes
  const BATCH = 64;
  const combined = rows.map(toCombined);
  const vectors = [];
  for (let i = 0; i < combined.length; i += BATCH) {
    const chunk = combined.slice(i, i + BATCH);
    const out = await embed(chunk, { pooling: "mean", normalize: true });
    // out.data is [batch, dim]
    for (let j = 0; j < out.data.length; j++) {
      vectors.push(Array.from(out.data[j])); // already normalized
    }
    console.log(`Embedded ${Math.min(i + BATCH, combined.length)} / ${combined.length}`);
  }

  const payload = rows.map((r, i) => ({
    food_name: r.food_name,
    ingredients: r.ingredients,
    procedures: r.procedures,
    embedding: vectors[i]
  }));

  await fs.writeFile(OUT_PATH, JSON.stringify({ model: "paraphrase-MiniLM-L3-v2", items: payload }), "utf8");
  console.log(`Wrote ${OUT_PATH}`);
}

main().catch(err => { console.error(err); process.exit(1); });
