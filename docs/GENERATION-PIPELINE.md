# Generation pipeline: how the code works

This explains the internals of the question-generation code: the data
model, how each module works, and the design decisions behind it. What each
task *is* (the benchmark spec) lives in the other `docs/*.md` files.

Code: `src/weedvlm/generation/`, `src/weedvlm/types/`, and the orchestrating
script `src/weedvlm/scripts/generate-all-questions.py`.

## The shape of the pipeline

Every task follows the same three-stage pattern, and the orchestrator
script wires the stages together.

1. **Generate.** A task module walks every image and emits *every*
   qualifying question (the full candidate pool). Nothing is capped or
   sampled yet, except where noted below.
2. **Select.** A separate step trims the pool to the exact quota, with a
   seeded RNG.
3. **Render.** Only the questions that survived selection have their
   box-annotated image drawn.

Splitting generation from selection is what makes exact sizing possible:
selection needs to see the whole pool to know which species or classes can
meet their quota. Splitting selection from rendering means the expensive
image drawing is done only for the questions actually kept.

## Data model (`types/`)

**Raw layer, `types/weedcoco.py`.** Pydantic models of the WeedCOCO JSON
(`Image`, `Annotation`, `Category`, ...), with `extra="allow"` because the
schema varies slightly between source datasets. Only the fields the
pipeline reads are declared.

**Reviewed layer, `types/dataset.py`.** The unit every task works from is
`ReviewedImage`: dataset name, image id, resolved file path, width and
height, and a list of `SpeciesAnnotation` (annotation id, category id, a
parsed `Species`, and `bbox` as `(x, y, w, h)` in pixels). The name
"reviewed" is historical, from when images were filtered by the review app.
`load.py` currently passes everything through unfiltered.

**Species, `types/species.py`.** Category names must match the regex
`<role>: <name>[ (<qualifier>)]`, giving `Species(role, name, qualifier)`.
`role` is `crop` or `weed`. `display_name` is the name with its first letter
capitalised, and is the form used in answer choices and `benchmark_class`.
The qualifier is kept as free text because it isn't always a growth stage
(e.g. CropAndWeed uses `(80)`).

**Questions, `types/questions.py`.** `QuestionBase` carries `id`,
`question_type`, `question_text`, `image_path`, `source`
(`QuestionSource`: dataset name, image id, annotation ids), `annotated`,
`benchmark_class`, and a `render` field. Subclasses:

| Class | Extra fields | Used by |
| --- | --- | --- |
| `MultipleChoiceQuestion` | `choices`, `answer_index` | species ID, fine-grained, grounded VQA |
| `OpenEndedQuestion` | `answer_text`, `ablation_of` | species ID ablation |
| `DensityEstimationQuestion` | `density_choices`, true count, true coverage fraction, `density_category` | density |
| `LocalisationQuestion` | `target_species`, `image_width`, `image_height`, `boxes` | localisation |

Two fields do a lot of work:

- `benchmark_class` is the group a question counts against. It is the
  species display name for species ID, grounded VQA and localisation,
  `same_species`/`different_species` for fine-grained, and the density
  category for density. Both selection (`balance.py`) and per-class scoring
  key off it.
- `render` is a `RenderJob` (`types/render.py`): a frozen dataclass of
  source path, output path, line width, labelled boxes, and whether to draw
  numbers. It is excluded from serialisation. It exists so that a question
  can point at an image that hasn't been drawn yet.

Question ids are built deterministically from the dataset, image id, target
and type (e.g. `CropAndWeed:123:Fat hen:species_id_multiple_choice`), with
suffixes such as `:unannotated` or `:open_ended`. The evaluation pipeline
depends on this: resuming a run re-uses ids.

## Loading (`generation/load.py`)

`load_images` resolves the dataset list, defaulting to
`.datasets/*/weedcoco.json`, sorted so ordering is deterministic. For each
dataset, `load_dataset_images`:

1. Validates the JSON into `WeedCocoDataset`.
2. Parses each category name into a `Species` and indexes annotations by
   image id.
3. Resolves each image's file with `resolve_image_path`, which tries
   `<dataset>/images/<file_name>` and then `<dataset>/<file_name>` because
   exports disagree about what `file_name` is relative to.
4. Builds a `ReviewedImage` per image, annotations included.

Missing image files don't get skipped: they are collected and raised as a
single `FileNotFoundError` naming a count and an example. The dataset name
is the folder name that contains `weedcoco.json`.

`scripts/filter-weedcoco-categories.py` is a separate preprocessing tool.
It removes named categories, and the annotations using them, from a
WeedCOCO file, e.g. CropAndWeed's catch-all `none` category. It reports how
many images are left with no annotations.

## Configuration (`generation/config.py`)

`PipelineConfig` is a tree of Pydantic models: one section per task plus
shared settings. Every field has a default equal to the corresponding CLI
default, so a config file only lists overrides. The script merges them with
`merged(cli_value, cfg_value)`: a CLI value wins if it is not `None`.

Two base classes capture the two sizing styles:

- `SpeciesSelection` (`num_species`, `questions_per_species`) is for tasks
  whose `benchmark_class` comes from a large, open set of species.
- `ClassBalance` (`questions_per_class`) is for tasks with a small, fixed
  class set.

Task configs inherit from these. `DensityConfig` also validates its rule
parameters (scales must satisfy `0 < low < high`, and scores must satisfy
`0 <= medium <= high <= 1`). `LocalisationConfig` constrains `max_boxes` and
the scene counts.

`resolve_config_path` falls back to `config.example.yaml` at the repo root
if no path is given. After a run, `generation-config.yaml` is written into
the output directory, and the evaluation pipeline later re-reads it through
the same `PipelineConfig` model.

## Selection (`generation/balance.py`)

`select_questions(questions, num_classes, questions_per_class, rng)` groups
questions by `benchmark_class` and has two modes.

**Fixed class set** (`num_classes is None`). Every class present is kept.
Each class must independently hold at least `questions_per_class` questions,
otherwise `InsufficientQuestionsError` is raised naming the class. Then
`rng.sample` draws exactly that many from each. A benchmark that's missing
one of its own classes would be broken rather than merely unbalanced, so
nothing is silently dropped.

**Species pool** (`num_classes` given). First filter to classes with at
least `questions_per_class` questions (the *eligible* set), and only then
`rng.sample` exactly `num_classes` of them. Filtering before sampling
matters. Sampling first could land on a low-supply species and fail a
request that the data could actually satisfy.

In both modes there is no cross-class backfilling. The output size is
always exactly `classes x quota`. `InsufficientQuestionsError` is a
`RuntimeError`, and the orchestrator catches it and exits with the message,
which says what to loosen.

Passing neither argument returns the questions unchanged.

## Shared helpers

**`geometry.py`**

- `rectangle_union_area` computes the exact union of `(x, y, w, h)` boxes by
  coordinate compression. The boxes' edges define a grid. Each cell is
  either fully inside some box or fully outside all of them, so it tests
  each cell's midpoint against the boxes and sums the covered cell areas.
  This is O(cells x boxes), exact, and counts overlaps once.
- `centre_of_mass` is the area-weighted mean of box centres.
- `centre_offset_fraction` maps that point onto a scale from 0 (image
  centre) to 1 (a corner) by normalising to `[-1, 1]` on each axis and
  dividing the Euclidean distance by `sqrt(2)`. The `max_centre_offset`
  filters use this to keep targets away from corners.

**`colour.py`.** `mean_colours` opens an image once and returns the mean RGB
of each requested box. For JPEGs it asks PIL for a reduced-size decode
(`draft`, 1/8 scale) because it is much cheaper than a full decode. Box
coordinates are scaled down and rounded to nearest-pixel edges, clamped to
at least one pixel. The module comments record the measured trade-off:
faster, but 6% of cross-species pairs flip the colour test at 1/8 scale.
That is acceptable for a coarse pre-filter. `colour_similarity` is
`1 - euclidean_distance / max_possible_distance`.

**`parallel.py`.** `parallel_map` runs a picklable function over a sequence
using a `ProcessPoolExecutor` (processes, not threads, since image decoding
holds the GIL). It falls back to a serial loop for one worker or too few
items. Results always come back in input order. This is what makes a seeded
run independent of the worker count: all RNG use happens serially in the
parent, in a fixed order, and only RNG-free work (colour measurement,
drawing) is parallelised.

**`render.py`.** See [Rendering](#rendering).

## The tasks

### Species ID (`species_id.py`, `open_ended.py`)

Three generation functions plus one selector.

`generate_species_id_mc_questions` (boxed). For each image and each species
of the role (`_group_by_species`), `_is_large_enough` tests the species'
total box area (sum of areas, so overlap is double counted here) against
`min_area_fraction` and its largest box against `min_largest_area_fraction`.
A species that passes becomes one question:

- Distractors come from a per-dataset name table built up front. Only
  species from the same dataset are used, so options are plausible
  neighbours and not taxa from another continent. `min(num_choices-1,
  pool size)` are sampled, the correct name is added, and the list is
  shuffled. The answer index is the correct name's position.
- A `plan_species_boxes` render job draws only this species' boxes. Other
  species in the image are left undrawn, so the picture doesn't leak or
  confuse the answer.
- An image with several qualifying species yields several questions.

`generate_species_id_unannotated_questions`. Only images with exactly one
species of the role (`unique_species_names`) qualify, because with no boxes
drawn the question "what species is present?" must have one answer. The
image path is the original image and there is no render job.

`select_species_id_questions` is the most involved selector, because the
boxed and unannotated sets are sized separately but must be consistent.
Ordering matters:

1. Group both pools by species. `boxed_eligible` is the species with at
   least `quota` boxed questions. `both_eligible` is the subset that also
   has at least `quota` unannotated ones.
2. If `unannotated_num_species` is set, draw those species first from
   `both_eligible`. Then fill the rest of `num_species` from the remaining
   boxed-eligible species. The unannotated set is a subset of the boxed set,
   and the "can supply both" species are not used up by the boxed draw.
3. If it is `None`, draw `num_species` boxed species and give unannotated
   questions to whichever of them can supply them.
4. Take exactly `questions_per_species` from each chosen species in each set.

`_require` raises `InsufficientQuestionsError` with a message describing the
shortfall. Asking for more unannotated species than total species is also
rejected.

`open_ended.generate_open_ended_questions` runs *after* selection on the
final MC list. It samples `round(len x fraction)` of them and creates an
`OpenEndedQuestion` for each, with the same image, render job, source and
class. `answer_text` is the correct choice, and the id is `<mc id>:open_ended`
with `ablation_of` pointing back. The text is the MC question text plus an
"If you don't know, say so" suffix. Sharing the render job means the
ablation and its original resolve to the same image file.

### Fine-grained (`fine_grained.py`)

The unit is a *pair* of annotations in one image. `_candidate_pairs`:

1. Groups annotations by species.
2. **Same-species pairs:** all `itertools.combinations` within each species
   that pass `_passes_selection`.
3. **Different-species pairs:** needs at least two species. It measures mean
   colours for every annotation in the image (one image decode), then for
   each cross-species pair that passes selection, keeps it if area ratio
   `min/max >= area_similarity_ratio` **or** colour similarity
   `>= colour_similarity_ratio`.

The `or` is intentional: each signal alone is a weak, noisy proxy for
looking alike, and requiring both would shrink an already small pool. These
are explicitly placeholders for a later appearance-based check and manual
review.

`_passes_selection` applies to every pair, same or different species, so
neither class is easier to spot by size alone. The combined area must reach
`min_total_area_fraction`, the larger box `min_largest_area_fraction`
(deliberately the larger of the pair, not each box, so "one tiny box" can't
leak the answer), and the pair's centre of mass must be within
`max_centre_offset`.

`generate_fine_grained_questions` computes candidate pairs per image in
parallel (the colour pass is the slow part), then serially shuffles each
image's pairs with the shared RNG and keeps the first `max_pairs_per_image`.
Each kept pair becomes a `MultipleChoiceQuestion` with fixed choices
`["Same species", "Different species"]`, drawn with boxes labelled 1 and 2
in distinct colours. `benchmark_class` is `same_species` or
`different_species`, which is also what the later per-class balancing uses.

### Density (`density.py`)

**Ground truth.** For an image, weeds are the annotations of the role. Count
is their number. Coverage is `rectangle_union_area(boxes) / image_area`.

**Category rule.** `density_score` log-scales count and coverage each to
`[0, 1]` between a (low, high) anchor pair, then averages them. Log because
both are heavily right-skewed: going from 1 to 2 weeds matters as much as
10 to 20. `classify_density` maps the score to a category: zero weeds is
`none`, score >= `high_score` with at least `high_min_count` weeds is `high`,
score >= `medium_score` with at least `medium_min_count` weeds is `medium`,
and everything else is `low`. The minimum counts mean a single close-up
plant can never be "dense", however much it covers. Neither count nor
coverage decides alone, because coverage mostly measures camera distance and
count ignores how much ground is covered.

**Generation.** Each image produces two questions with the same ground truth
and different `annotated` flags: a boxed one (a `plan_boxes` render job with
plain outlines, or no render if there are no boxes) and an unannotated one
(the original image). Images with weeds are skipped if the largest weed box
is below `min_largest_area_fraction`. Weed-free images are always kept as
the `none` class.

**Selection.** `select_density_questions` calls `select_questions` on the
boxed pool grouped by category (fixed-class mode). It then restricts the
unannotated pool to the *same source images* the boxed selection picked and
balances that to `unannotated_questions_per_class`. Reusing the same images
under both conditions is what makes the boxed-vs-unannotated comparison
fair.

`DENSITY_CHOICES` (`None/Low/Medium/High`) are the display strings, and
`density_category` is the lowercase ground-truth value.

### Grounded weed VQA (`grounded_vqa.py`)

One image yields one question per qualifying target species, all sharing
one rendered image.

1. Group annotations by species. Skip the image if it has fewer than
   `min_species` species.
2. `_target_species`: species whose largest box covers at least
   `min_largest_area_fraction`. If there are none, skip.
3. `_passes_selection`: the union area of **all** boxes must reach
   `min_total_area_fraction`, and their centre of mass must be within
   `max_centre_offset`.
4. Shuffle the species list with the shared RNG and number them 1..N. All
   boxes of a species share its number. This is the only link between
   number and species, and is never shown to the model.
5. Create one render job, drawing every box numbered. Small species that
   aren't targets are still drawn and numbered, because leaving real plants
   unboxed would mislead. They are just never asked about.
6. For each target species emit a question whose choices are all labels as
   strings (`"1"`..`"N"`) and whose `answer_index` is that species' label
   minus one.

Only weed annotations are used. The orchestrator balances the result by
species through `select_questions`. The render slug is `gvqa_weed`, so the
images end in `_gvqa_weed.jpg`.

### Localisation (`localisation.py`)

The simplest generator, because nothing is drawn. For each image, group by
species of the role, then:

- Skip the whole image unless its total plant count is at least
  `min_scene_weeds` and there are at least `min_other_species` species
  besides the target (`len(grouped) - 1`). This forces a busy scene.
- For each species, skip it if it has more than `max_boxes` instances or any
  instance covers less than `min_box_area_fraction` of the image.
- Emit a `LocalisationQuestion` with the original image, the image size, and
  the ground-truth boxes in `(x, y, w, h)` pixels.

`image_width` and `image_height` are stored in the question because the
scorer must convert a model's normalised coordinates back to pixels.
Selection is the species-pool mode of `select_questions`.

## Rendering

`render.py` has two halves.

**Planning**, called by generators. `plan_boxes` (plain outlines),
`plan_species_boxes` (plain outlines, with the species name in the file
name), and `plan_numbered_boxes` (numbered, coloured) each build a
`RenderJob` without touching any image. Output paths are
`<dataset>_<image_id>_<slug><ext>`, and line width scales with image size
(`max(3, min(w, h)//200)`).

**Drawing**, called after selection. `render_all` collects the render jobs
from the selected questions and **de-duplicates by output path**, since many
questions share a picture: every target species in a grounded-VQA image, or
a MC question and its open-ended ablation. It creates the output
directories, then runs `render` over the jobs with `parallel_map`.

`render` handles two modes. Unnumbered jobs draw red rectangles. Numbered
jobs draw all outlines first, then all tags, so no outline covers another
box's tag. Colours come from `LABEL_PALETTE_HEX`, indexed by label, with
black or white text chosen by luminance. `_tag_position` places the number
tag above the box, else below it, else just inside the top edge, to avoid
covering small plants. The palette is duplicated in the review app's
`shared.js` and must be kept in sync by hand.

## The orchestrator (`scripts/generate-all-questions.py`)

`main` is a straight-line script:

1. Parse CLI args, resolve the config, and merge values (CLI over config).
   A task runs if its config toggle is on and its `--skip-*` flag is absent.
2. Create one `random.Random(seed)` and load all images.
3. For each enabled task, in the order species ID, fine-grained, density,
   grounded VQA, localisation: generate, select, and store the result in a
   `tasks` dict keyed by task name. Each task catches
   `InsufficientQuestionsError` and exits with a task-labelled message.
4. Run `render_all` once across the union of all selected questions, so one
   rendered-images directory is shared and de-duplication works across
   tasks.
5. Store the questions in the configured format (below), together with the
   generation-config text: the timestamp, the full command line and the
   config file's contents, so the output's provenance survives later edits
   to the config file.

The `tasks` dict is keyed by task name (`species_id`, ...), which is what
both formats need.

- **JSON** (`storage.format: json`, the default). Each task is written to
  its file from `TASK_FILES` with `model_dump(mode="json")`, which drops
  `render` because it is excluded from the model. `all-questions.json` and
  `generation-config.yaml` are written alongside.
- **SQLite** (`storage.format: sqlite`). `database.write_benchmark` stores
  the (task, question) pairs, the images they use and the config text in one
  database ([DATABASE.md](DATABASE.md)). Because it only looks at the
  selected questions, images that were rendered for nothing never reach the
  database. The database is opened, and checked for existing responses,
  before any generation work starts, so a database that can't accept a new
  benchmark fails immediately instead of after rendering.

Because all tasks draw from one shared RNG in a fixed order, a seed makes
the whole run reproducible, and disabling an earlier task shifts the
samples of later ones. Rendering consumes no randomness, which is why
parallelism doesn't affect results.

`scripts/benchmark-composition.py` is a pure function of the config. It
multiplies the quotas (`num_species x questions_per_species`, and so on),
applies the same `round(n x fraction)` as `open_ended.py`, and reports
anything left `None` as data-dependent. It doesn't touch the datasets.
