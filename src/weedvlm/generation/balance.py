"""
Exact species/class selection for generated questions, applied as the
final step in each generate-*.py script, after the full (unrestricted)
question set for a task has already been generated. "Class" here means
a question's benchmark_class.

Two distinct situations, both handled by select_questions depending on
whether num_classes is given:

- species-id / localisation: benchmark_class is a species name drawn
  from a large, open-ended pool. num_classes picks an exact-size subset
  of that pool -- species are first filtered to the ones that can
  actually supply questions_per_class questions (so a request never
  fails just because random.sample happened to land on a low-supply
  species when higher-supply ones were available), then exactly
  num_classes are chosen from among the eligible ones.
- fine-grained / density: benchmark_class is a small, fixed, known set
  (same_species/different_species; none/low/medium/high) -- there's no
  meaningful "subset of classes" to pick, so num_classes is never
  passed for these. Every class present is kept and must independently
  supply exactly questions_per_class questions; none is silently
  dropped for coming up short, since a fine-grained/density benchmark
  missing one of its own classes isn't balanced, it's broken.

Either way, there's no cross-class backfilling: every kept class
supplies exactly questions_per_class questions on its own, so the
task's total output size is always exactly
(num_classes or every class present) * questions_per_class.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from weedvlm.types.questions import QuestionBase


class InsufficientQuestionsError(RuntimeError):
    """Raised when num_classes or questions_per_class can't be met by
    the questions generated -- i.e. the underlying reviewed data
    doesn't have enough qualifying images/species, not a bug in the
    selection logic itself. The message says what to loosen."""


def select_questions(
    questions: Sequence[QuestionBase],
    *,
    num_classes: int | None = None,
    questions_per_class: int | None = None,
    rng: random.Random | None = None,
) -> list[QuestionBase]:
    if num_classes is None and questions_per_class is None:
        return list(questions)

    rng = rng or random.Random()

    by_class: dict[str, list[QuestionBase]] = {}
    for question in questions:
        by_class.setdefault(question.benchmark_class, []).append(question)

    if num_classes is None:
        # Fixed/known class set: keep every class present and require
        # each to independently meet the quota -- never silently drop one.
        classes = list(by_class)
        if questions_per_class is not None:
            for c in classes:
                if len(by_class[c]) < questions_per_class:
                    raise InsufficientQuestionsError(
                        f"Class {c!r} has only {len(by_class[c])} qualifying questions, but "
                        f"questions_per_class={questions_per_class} requires that many from "
                        f"every class. Lower questions_per_class, or loosen this task's "
                        f"image-selection thresholds."
                    )
            return [q for c in classes for q in rng.sample(by_class[c], questions_per_class)]
        return [q for c in classes for q in by_class[c]]

    # Open-ended species pool: filter to species that can meet the quota,
    # then choose exactly num_classes from among the eligible ones.
    if questions_per_class is None:
        eligible = list(by_class)
    else:
        eligible = [c for c, pool in by_class.items() if len(pool) >= questions_per_class]

    if len(eligible) < num_classes:
        requirement = (
            f" with at least {questions_per_class} qualifying questions each"
            if questions_per_class is not None
            else ""
        )
        raise InsufficientQuestionsError(
            f"Requested exactly {num_classes} species{requirement}, but only {len(eligible)} "
            f"of the {len(by_class)} species with any qualifying questions meet that. Lower "
            f"num_species, lower questions_per_species, or loosen this task's image-selection "
            f"thresholds."
        )
    classes = rng.sample(eligible, num_classes)

    if questions_per_class is None:
        return [q for c in classes for q in by_class[c]]
    return [q for c in classes for q in rng.sample(by_class[c], questions_per_class)]
