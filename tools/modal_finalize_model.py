"""Combine independent benchmark banks and package the verified model on Modal."""
from pathlib import Path
import modal

app = modal.App('alpha-holdem-final-model-report')
volume = modal.Volume.from_name('alpha-holdem-budget-results')
image = modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4')


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=120,
              retries=0, volumes={'/runs': volume}, scaledown_window=2, max_containers=1)
def finalize(run_id, previous_run, hands, seeds):
    import json, shutil, numpy as np
    volume.reload()
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two unique independent banks are required')
    ordered = sorted(seeds)
    if any(a + hands > b for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Deal seed banks overlap')
    run = Path('/runs') / run_id
    reports = [json.loads((run / f'comparison_{previous_run}_seed_{s}_hands_{hands}_mixed/evaluation.json').read_text()) for s in seeds]
    first = reports[0]
    for seed, report in zip(seeds, reports):
        if report['spec']['seed'] != seed or report['spec']['hands'] != hands:
            raise ValueError('Unexpected evaluation bank')
        for key in ('models', 'model_configs', 'config', 'opponents', 'reference'):
            if report['spec'].get(key) != first['spec'].get(key):
                raise ValueError('Reports compare different models or opponent tables')
        if report['health'] != first['health']:
            raise ValueError('Checkpoint health changed between evaluations')
    names = list(first['spec']['models'])
    cases = [c['name'] for c in first['spec']['opponents']]
    def stats(values):
        mean = float(np.mean(values))
        radius = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
        return {'mean': mean, 'ci': [float(mean-radius), float(mean+radius)]}
    rewards = {}
    for name in names:
        for case in cases:
            rows = []
            for report in reports:
                matches = [r for r in report['results'] if r['model'] == name and r['opponent'] == case]
                if len(matches) != 1 or len(matches[0]['rewards']) != hands:
                    raise ValueError('Incomplete model/table evaluation')
                rows.extend(matches[0]['rewards'])
            rewards[name, case] = np.asarray(rows)
    reference = first['spec']['reference']
    aggregate = {}
    for name in names:
        values = np.mean([rewards[name, case] for case in cases], axis=0)
        aggregate[name] = stats(values)
        if name != reference:
            ref = np.mean([rewards[reference, case] for case in cases], axis=0)
            aggregate[name]['paired_change'] = stats(values-ref)
    tables = {case: {name: stats(rewards[name, case]) for name in names} for case in cases}
    result = {'run_id': run_id, 'reference_run': previous_run, 'seeds': seeds,
              'hands_per_model_table': hands*len(seeds),
              'total_evaluation_hands': hands*len(seeds)*len(names)*len(cases),
              'aggregate': aggregate, 'tables': tables, 'health': first['health'],
              'supported_pool_improvement': bool(aggregate['structured_final']['paired_change']['ci'][0] > 0),
              'positive_all_tables': all(tables[c]['structured_final']['ci'][0] > 0 for c in cases)}
    final = run/'final_model'; final.mkdir(exist_ok=True)
    for file in ('output_weight.pkl', 'training_config.json', 'latest_checkpoint.json'):
        shutil.copy2(run/'league'/file, final/file)
    # The full optimizer checkpoint stays under league, not final_model.
    manifest = json.loads((final/'latest_checkpoint.json').read_text())
    manifest['path'] = '../league/' + manifest['path']
    (final/'latest_checkpoint.json').write_text(json.dumps(manifest, indent=2))
    (final/'benchmark_summary.json').write_text(json.dumps(result, indent=2))
    gain = aggregate['structured_final']['paired_change']
    report = '# Six-player model benchmark\n\n'
    report += f"Run: `{run_id}`. {result['total_evaluation_hands']:,} benchmark hands across two independent deal banks, computed on Modal CPU.\n\n"
    report += f"Final pool profit: {aggregate['structured_final']['mean']:+.2f} chips/hand; reference: {aggregate[reference]['mean']:+.2f}. Paired improvement: {gain['mean']:+.2f}, 95% interval [{gain['ci'][0]:+.2f}, {gain['ci'][1]:+.2f}].\n\n"
    report += '| Opponent table | Final chips/hand | 95% interval | Reference |\n|---|---:|---|---:|\n'
    for case in cases:
        final_stats = tables[case]['structured_final']
        report += f"| {case} | {final_stats['mean']:+.2f} | [{final_stats['ci'][0]:+.2f}, {final_stats['ci'][1]:+.2f}] | {tables[case][reference]['mean']:+.2f} |\n"
    report += '\nSettings: structured six-player model, entropy 0.02, ranked historical sampling, opponent refresh 0.01, frozen pool. The final continuation added one bounded T4 hour.\n\n'
    report += f"Health: {first['health']['trained_steps']:,} transitions, {first['health']['updates']:.0f} updates, {first['health']['parameters']:,} parameters; finite weights: {first['health']['all_weights_finite']}.\n\n"
    report += 'Files: `output_weight.pkl` for inference, `training_config.json` for the matching architecture, and `latest_checkpoint.json` pointing to full optimizer/league state on the Modal volume.\n\n'
    report += 'Limits: one training seed and fixed benchmark opponents. Positive benchmark profit and improvement do not establish optimal poker play or exploitability. The screen did not isolate a significant entropy effect; the final result measures the complete continuation recipe.\n'
    (final/'model_card.md').write_text(report)
    volume.commit()
    print(report, flush=True)
    return result


@app.local_entrypoint()
def main(run_id: str, previous_run: str, hands: int = 3600, seeds: str = '19300000,24700000'):
    import json
    for value in (run_id, previous_run):
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):
            raise ValueError('Unsafe run ID')
    banks = [int(s) for s in seeds.split(',')]
    if hands < 6 or hands % 6 or any(s < 0 for s in banks):
        raise ValueError('Invalid hands or seeds')
    call = finalize.spawn(run_id, previous_run, hands, banks)
    record = {'call_id': call.object_id, 'run_id': run_id, 'seeds': banks}
    (Path(__file__).resolve().parents[1]/'work/final_model_report_launch.json').write_text(json.dumps(record))
    print(record)
