/**
 * Generates a small synthetic WeedCOCO dataset (images + dataset.json) for
 * exercising the reviewer app without needing a real Weed-AI download.
 * Run: deno task gen-fixtures
 */
import { encodePng } from "../src/png.ts";
import type { WeedCocoDataset } from "../src/types.ts";

const OUT_DIR = new URL("./", import.meta.url).pathname;
const IMAGES_DIR = `${OUT_DIR}images`;

await Deno.mkdir(IMAGES_DIR, { recursive: true });

const categories = [
  { id: 1, name: "crop: zea mays (whole-plant)" },
  { id: 2, name: "weed: bassia scoparia" },
  { id: 3, name: "weed: amaranthus retroflexus" },
];

const palette: Record<number, [number, number, number]> = {
  1: [60, 160, 60],
  2: [190, 60, 60],
  3: [200, 140, 30],
};

const WIDTH = 320;
const HEIGHT = 240;

const images: WeedCocoDataset["images"] = [];
const annotations: WeedCocoDataset["annotations"] = [];

let annId = 1;
let imgId = 1;

function randInt(max: number) {
  return Math.floor(Math.random() * max);
}

for (const cat of categories) {
  const countForCategory = 8;
  for (let i = 0; i < countForCategory; i++) {
    const fileName = `img_${String(imgId).padStart(3, "0")}.png`;
    const boxCount = 1 + randInt(2);
    const patches = [];
    const bboxes: [number, number, number, number][] = [];

    for (let b = 0; b < boxCount; b++) {
      const w = 30 + randInt(60);
      const h = 30 + randInt(60);
      const x = randInt(WIDTH - w);
      const y = randInt(HEIGHT - h);
      patches.push({ x, y, w, h, color: palette[cat.id] });
      bboxes.push([x, y, w, h]);
    }

    const png = await encodePng(WIDTH, HEIGHT, [230, 230, 220], patches);
    await Deno.writeFile(`${IMAGES_DIR}/${fileName}`, png);

    images.push({ id: imgId, file_name: fileName, width: WIDTH, height: HEIGHT });
    for (const bbox of bboxes) {
      annotations.push({ id: annId++, image_id: imgId, category_id: cat.id, bbox });
    }
    imgId++;
  }
}

const dataset: WeedCocoDataset = {
  info: {
    description: "Synthetic fixture dataset for weedcoco-review",
    generated: new Date().toISOString(),
  },
  categories,
  images,
  annotations,
};

await Deno.writeTextFile(`${OUT_DIR}dataset.json`, JSON.stringify(dataset, null, 2));

console.log(`Wrote ${images.length} images and ${annotations.length} annotations to ${OUT_DIR}`);
