# weedvlm density labeller

A local browser app for working out the weed density rule from hand labels, instead of guessing its
thresholds. You give a sample of images a density category, and the app fits rules to those labels
and shows how well each agrees with you.

| View      | What it's for                                                          |
| --------- | ---------------------------------------------------------------------- |
| **Label** | Step through sampled images and mark each None, Low, Medium or High.   |
| **Fit**   | Rules fitted to the labels so far, and where each disagrees with them. |

Run it from this directory with `deno task serve`, then open the printed `http://localhost:8788`.

| Flag                   | Default                                  | What                                        |
| ---------------------- | ---------------------------------------- | ------------------------------------------- |
| `--dataset <file>`     | every `<repo>/.datasets/*/weedcoco.json` | A WeedCOCO JSON to draw from; repeatable.   |
| `--datasets-dir <dir>` | `<repo>/.datasets`                       | Where to look when no `--dataset` is given. |
| `--labels-out <file>`  | `density-labels.json` in this directory  | Where labels are saved.                     |
| `--config <file>`      | `<repo>/config.example.yaml`             | Generation config holding the current rule. |
| `--port <n>`           | `8788`                                   |                                             |

## Measurements

Every image is reduced to four numbers, all taken from its weed boxes alone (`src/features.ts`):

| Measurement       | Definition                                                               |
| ----------------- | ------------------------------------------------------------------------ |
| Weed count        | Number of weed boxes.                                                    |
| Weed coverage     | Share of the image inside at least one weed box (overlaps counted once). |
| Typical weed size | The median weed box's area as a share of the image.                      |
| Spatial spread    | Share of a 4x4 grid's cells that at least one weed box reaches into.     |

Count and coverage are computed exactly as `src/weedvlm/generation/density.py` does. Typical size
stands in for camera distance, which differs by orders of magnitude between datasets: the median box
is about 17% of the image in CottonWeedDet12 and under 0.1% in CropAndWeed. Spread separates weeds
throughout the frame from the same weeds in one clump.

## Sampling

Most images hold one or two weeds, so a sample in proportion to the data would put nearly every
label where no threshold falls. The queue is instead drawn evenly across the count/coverage plane
(`sample` in `server.ts`): images are grouped into count x coverage cells, and the queue takes one
image from each cell in turn. Within a cell the datasets take turns too, so no dataset supplies a
whole cell. Weed-free images are never sampled: they are "none" under any rule.

## Labelling

Labels are saved to the labels file as each is given, keyed by `<dataset>:<image id>`, and reloaded
on the next run. **Skip** records that an image can't be judged; skipped images stay out of later
samples and out of the fit. Measurements are hidden by default so the label is a judgement of the
picture rather than of its numbers.

Keys: `1`-`4` None/Low/Medium/High, `S` skip, `Space` toggle weed boxes, `I` toggle measurements,
`←`/`→` navigate.

## Fitting

The Fit view refits from the saved labels each time it opens (`src/fit.ts`) and compares three
rules:

- **Current rule**: the pipeline's score rule with the thresholds in the generation config. Count
  and coverage are each placed on a log scale from 0 to 1, averaged, and cut at two scores, with a
  minimum weed count for Medium and for High.
- **Refitted score rule**: the same rule with its thresholds searched for the lowest total category
  error, so being two categories out costs twice as much as one. The search is a random sweep of the
  plausible ranges followed by small nudges to the best rule found. Its output is the `density:`
  keys of the generation config, so it can be pasted in with no code change.
- **Threshold tree**: a small decision tree over all four measurements. Each test is the single
  threshold that leaves its two sides most uniform in category, and no branch may hold fewer than 3%
  of the labelled images. Using it in the pipeline means replacing `classify_density`.

Each rule gets two agreement figures: on the images it was fitted to, and **held out**, where the
labelled images are dealt into five groups and each group is predicted by a rule fitted to the other
four. Held out is the estimate of how the rule does on unlabelled images; a rule far better on its
own images than held out has been fitted too closely, and a shallower tree or more labels will help.

The chart plots the labelled images over any two measurements, coloured by label, and can be
narrowed to the images a chosen rule gets wrong. Clicking a point shows its image with every rule's
answer. **Download labels as CSV** exports each labelled image with its four measurements.
