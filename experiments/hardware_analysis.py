"""Real-hardware study helpers: score ideal vs FakeFez vs IBM chip, and plot the comparison."""
import matplotlib.pyplot as plt
import numpy as np

from evqaoa import DEPOT, bitstring_to_schedule, is_valid
from experiments.hardware import readout_rates
from experiments.noise import correct_readout, hellinger_fidelity, keep_valid, normalise, score

SOURCES = ('ideal', 'fakefez', 'hardware', 'hardware_readout', 'hardware_valid')
LABELS = {'ideal': 'ideal simulator', 'fakefez': 'FakeFez (noisy sim)', 'hardware': 'ibm_fez raw',
          'hardware_readout': 'ibm_fez readout-corrected',
          'hardware_valid': 'ibm_fez invalid shots discarded'}
COLORS = {'ideal': '#9aa5b1', 'fakefez': '#4c78a8', 'hardware': 'tab:orange',
          'hardware_readout': 'tab:red', 'hardware_valid': '#2ca02c'}
FIDELITY_PAIRS = (('hardware', 'ideal'), ('hardware', 'fakefez'), ('fakefez', 'ideal'))
DEVICE_VARIANTS = ('raw', 'readout_corrected', 'invalid_discarded')
RUN_AXIS_LABEL = 'QAOA circuit (mixer_start_depth p)'
METRICS = (('valid_rate', 'share of shots that are valid schedules'),
           ('p_optimal', 'P(optimal schedule)'),
           ('approx_ratio', 'approximation ratio (valid shots)'))


def run_names(hw_counts):
    """Run entries are the dicts that carry counts; 'usage' / 'readout_calibration' are metadata."""
    return [k for k, v in hw_counts.items() if isinstance(v, dict) and 'counts' in v]


def options_summary(options):
    """One-line description of the sampler's error-suppression settings."""
    parts = []
    dd = options.get('dynamical_decoupling', {})
    if dd.get('enable'):
        parts.append(f"DD {dd.get('sequence_type', 'on')}")
    tw = options.get('twirling', {})
    twirled = [n for n, key in (('gate', 'enable_gates'), ('measure', 'enable_measure')) if tw.get(key)]
    if twirled:
        parts.append(f"twirling {'+'.join(twirled)}")
    return ', '.join(parts) or 'none'


def record_table(hw_counts):
    rows = []
    for name in run_names(hw_counts):
        e = hw_counts[name]
        stats = e.get('circuit_stats', {})
        rows.append(dict(run=name, backend=e.get('backend'), job_id=e.get('job_id'),
                         shots=e.get('shots'), error_suppression=options_summary(e.get('options', {})),
                         two_qubit_gates=stats.get('two_qubit_gates'), depth=stats.get('depth')))
    return rows


def analyse_run(dists, model, baselines, problem=DEPOT):
    """Score every available source plus the post-selected chip output; Hellinger between pairs.

    dists needs 'ideal', 'fakefez', 'hardware'; 'hardware_readout' is optional.
    """
    sources = dict(dists, hardware_valid=keep_valid(dists['hardware'], problem))
    # an empty post-selection (no valid shot) scores as all zeros
    scores = {src: score(sources[src], model, baselines, problem) for src in SOURCES if src in sources}
    hellinger = {f'{a}_vs_{b}': hellinger_fidelity(sources[a], sources[b]) for a, b in FIDELITY_PAIRS}
    return dict(scores=scores, hellinger=hellinger)


def top_schedules(reference, others, k=10, problem=DEPOT):
    """The k most likely bitstrings under `reference`, with their probability in every source."""
    ref = normalise(reference)
    probs = {src: normalise(d) for src, d in others.items()}
    ranked = sorted(ref, key=ref.get, reverse=True)[:k]
    return [dict(bitstring=b, valid=bool(is_valid(problem, bitstring_to_schedule(b))), ideal=ref[b],
                 **{src: p.get(b, 0.0) for src, p in probs.items() if src != 'ideal'})
            for b in ranked]


def device_dists(hw_counts, name, problem=DEPOT):
    """One chip run as raw counts, readout-corrected and post-selected (invalid shots discarded)."""
    entry = hw_counts[name]
    rates = readout_rates(hw_counts['readout_calibration'], entry['physical_qubits'])
    return dict(raw=dict(entry['counts']), readout_corrected=correct_readout(entry['counts'], rates),
                invalid_discarded=keep_valid(entry['counts'], problem))


def device_scores(hw_counts, name, model, baselines, problem=DEPOT):
    """noise.score of every device_dists variant."""
    return {variant: score(dist, model, baselines, problem)
            for variant, dist in device_dists(hw_counts, name, problem).items()}


def binomial_std(p, n):
    """Shot-noise standard error of a proportion p estimated from n shots (nan for n = 0)."""
    return float(np.sqrt(p * (1 - p) / n)) if n > 0 else float('nan')


def study05_row(rows, label, reps):
    return next((r for r in rows if r['config'] == label and r['p'] == reps), None)


def _finish(fig, path, legend_ax=None):
    """Save; one shared legend under the panels keeps it off the bars."""
    if legend_ax is None:
        fig.tight_layout()
    else:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.07 * 4.6 / fig.get_figheight(), 1, 1))
    fig.savefig(path, dpi=150)
    return fig


def plot_metric_bars(results, path, reference=None):
    """Grouped bars: valid share / P(optimal) / approx ratio for each run and source.

    reference: optional {metric: value} drawn as a dashed 'random guess' line.
    """
    reference = reference or {}
    names = list(results)
    present = [s for s in SOURCES if any(s in results[n]['scores'] for n in names)]
    w = 0.8 / len(present)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.6))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for i, src in enumerate(present):
            xs = np.arange(len(names)) + (i - (len(present) - 1) / 2) * w
            vals = [results[n]['scores'].get(src, {}).get(metric, np.nan) for n in names]
            ax.bar(xs, np.nan_to_num(vals), w, color=COLORS[src], label=LABELS[src])
        if metric in reference:
            ax.axhline(reference[metric], ls='--', color='grey', label='random guess')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_title('Valid schedules')
    axes[1].set_title('Chance of the optimal plan')
    axes[2].set_title('Approximation ratio')
    return _finish(fig, path, legend_ax=axes[0])


def plot_fidelity(results, path):
    """Hellinger fidelity of the chip vs ideal and vs FakeFez (FakeFez vs ideal for reference)."""
    names = list(results)
    w = 0.8 / len(FIDELITY_PAIRS)
    colors = ('tab:orange', '#4c78a8', '#9aa5b1')
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, ((a, b), color) in enumerate(zip(FIDELITY_PAIRS, colors)):
        xs = np.arange(len(names)) + (i - 1) * w
        ax.bar(xs, [results[n]['hellinger'][f'{a}_vs_{b}'] for n in names], w, color=color,
               label=f'{LABELS[a]} vs {LABELS[b]}')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Hellinger fidelity (1 = identical)')
    ax.set_title('How close is the real chip to the simulators?')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    return _finish(fig, path)


def plot_top_schedules(tops, path):
    """Per run: the 10 most likely ideal schedules and how often each source returned them."""
    names = list(tops)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        rows = tops[name]
        srcs = [s for s in SOURCES if rows and s in rows[0]]
        w = 0.8 / len(srcs)
        for i, src in enumerate(srcs):
            xs = np.arange(len(rows)) + (i - (len(srcs) - 1) / 2) * w
            ax.bar(xs, [r[src] for r in rows], w, color=COLORS[src], label=LABELS[src])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r['bitstring'] + ('' if r['valid'] else ' ✗') for r in rows],
                           rotation=30, ha='right', fontsize=8, family='monospace')
        ax.set_ylabel('probability')
        ax.set_title(f'{name}: top-{len(rows)} schedules of the ideal circuit (✗ = invalid)')
        ax.grid(axis='y', alpha=0.3)
    return _finish(fig, path, legend_ax=axes[0, 0])


def plot_grouped_bars(table, path, metrics, colors, title, errors=None, reference=None,
                      xlabel=RUN_AXIS_LABEL):
    """One panel per metric, one bar group per run, one bar per source.

    table: {run: {source: {metric: value}}}; errors has the same shape (std, optional);
    metrics: ((key, y-axis label, panel title), ...); reference: {metric: value} drawn as
    a dashed 'random guess' line.
    """
    errors, reference = errors or {}, reference or {}
    names = list(table)
    sources = list(dict.fromkeys(src for n in names for src in table[n]))
    w = 0.8 / len(sources)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.8), squeeze=False)
    for ax, (key, ylabel, panel_title) in zip(axes[0], metrics):
        for i, src in enumerate(sources):
            xs = np.arange(len(names)) + (i - (len(sources) - 1) / 2) * w
            vals = [table[n].get(src, {}).get(key, np.nan) for n in names]
            errs = [errors.get(n, {}).get(src, {}).get(key, np.nan) for n in names]
            has_err = not np.all(np.isnan(errs))
            ax.bar(xs, np.nan_to_num(vals), w, color=colors.get(src), label=src,
                   yerr=np.nan_to_num(errs) if has_err else None, capsize=3 if has_err else 0)
        if key in reference:
            ax.axhline(reference[key], ls='--', color='grey', label='random guess')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 5), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    return fig
