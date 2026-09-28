"""Plain-text report of the metrics from weedvlm.evaluation.metrics."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _suite_table(suites: dict[str, Any]) -> list[str]:
    width = max(len(name) for name in suites)
    lines = [
        f"{'suite':<{width}}  {'n':>5}  {'failed':>6}  {'accuracy':>8}  {'95% CI':<13}  "
        f"{'all-Qs':>7}  {'chance':>6}  {'unsure':>6}  {'macroF1':>7}"
    ]
    for name, s in sorted(suites.items()):
        ci = s["accuracy_ci95"]
        ci_text = f"{100 * ci[0]:.1f}-{100 * ci[1]:.1f}%" if ci else "--"
        lines.append(
            f"{name:<{width}}  {s['n_total']:>5}  {s['n_failed']:>6}  {_pct(s['accuracy']):>8}  "
            f"{ci_text:<13}  {_pct(s['accuracy_all']):>7}  {_pct(s['chance_accuracy'], 0):>6}  "
            f"{_pct(s['unsure_rate']):>6}  {_num(s['macro_f1']):>7}"
        )
    return lines


def _density_section(suites: dict[str, Any]) -> list[str]:
    lines = []
    for name, s in sorted(suites.items()):
        if "density" not in s:
            continue
        d = s["density"]
        lines += [
            "",
            f"{name}: mean category distance {_num(d['mean_category_distance'])} "
            f"(0 = exact, 1 = one category off)",
            f"  weed count     MAE {_num(d['count']['mean_abs_error'], 1)}  "
            f"median AE {_num(d['count']['median_abs_error'], 1)}  "
            f"bias {_num(d['count']['mean_signed_error'], 1):>5}  (n={d['count']['n']})",
            f"  coverage (pp)  MAE {_num(d['coverage_pp']['mean_abs_error'], 1)}  "
            f"median AE {_num(d['coverage_pp']['median_abs_error'], 1)}  "
            f"bias {_num(d['coverage_pp']['mean_signed_error'], 1):>5}  (n={d['coverage_pp']['n']})",
            "  confusion (rows = truth, columns = predicted):",
        ]
        categories = list(d["confusion_truth_by_predicted"])
        lines.append("    " + " " * 8 + "".join(f"{c:>8}" for c in categories))
        for truth, row in d["confusion_truth_by_predicted"].items():
            lines.append(f"    {truth:<8}" + "".join(f"{row[c]:>8}" for c in categories))
    return lines


def _weakest_classes(suites: dict[str, Any], suite: str, count: int = 5) -> list[str]:
    if suite not in suites:
        return []
    classes = suites[suite]["by_class"]
    ranked = sorted(classes.items(), key=lambda kv: kv[1]["accuracy"])[:count]
    return [
        "",
        f"weakest classes in {suite}:",
        *(f"  {name:<28} {_pct(c['accuracy']):>6}  (n={c['n']})" for name, c in ranked),
    ]


def render_report(
    models: dict[str, Any], failures: dict[str, list[tuple[str, int]]] | None = None
) -> str:
    lines: list[str] = []
    for model, m in models.items():
        u = m["usage"]
        lines += [
            "=" * 100,
            f"{model}: {u['requests']} responses scored, {u['failed']} failed, "
            f"${u['total_cost_usd']:.4f} total (${_num(u['mean_cost_usd'], 5)}/request), "
            f"mean {_num(u['mean_total_tokens'], 0)} tokens, "
            f"{_num(u['mean_latency_seconds'], 1)}s latency, {u['retried_requests']} retried",
            "=" * 100,
            "accuracy = correct / answered; all-Qs counts failed requests as wrong. "
            "Unsure/invalid answers are wrong.",
            "",
            *_suite_table(m["suites"]),
            *_density_section(m["suites"]),
        ]
        for condition, pair in m["species_id_multiple_choice_vs_open_ended"].items():
            lines += [
                "",
                f"species ID ({condition}), same {pair['n_pairs']} questions asked both ways: "
                f"multiple choice {_pct(pair['multiple_choice_accuracy'])} vs "
                f"open-ended {_pct(pair['open_ended_accuracy'])}",
            ]
        lines += _weakest_classes(m["suites"], "species_id/multiple_choice/boxed")
        if failures and failures.get(model):
            lines += ["", "failed requests:"]
            lines += [f"  {count:>4} x {reason}" for reason, count in failures[model]]
        lines.append("")
    return "\n".join(lines)
