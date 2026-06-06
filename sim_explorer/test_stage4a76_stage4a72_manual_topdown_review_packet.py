#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/ubuntu22/sc_explorer_ws')
OUT = ROOT / 'outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet'


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    checks = {}
    checks['output_dir_exists'] = OUT.is_dir()
    checks['summary_exists'] = (OUT / 'stage4a76_stage4a72_manual_topdown_review_summary.json').is_file()
    checks['manual_review_template_exists'] = (OUT / 'stage4a72_manual_review_template.csv').is_file() and (OUT / 'stage4a72_manual_review_template.json').is_file()
    checks['sample_index_exists'] = (OUT / 'stage4a72_review_sample_index.csv').is_file() and (OUT / 'stage4a72_review_sample_index.json').is_file()
    checks['main_html_exists'] = (OUT / 'stage4a72_topdown_review_index.html').is_file()
    checks['all_actions_floorplan_exists'] = (OUT / 'stage4a72_all_actions_floorplan.png').is_file()
    checks['by_start_or_by_step_floorplan_exists'] = (OUT / 'stage4a72_all_actions_by_start_floorplan.png').is_file() or (OUT / 'stage4a72_all_actions_by_step_floorplan.png').is_file()
    checks['review_instructions_exist'] = (OUT / 'stage4a72_manual_review_instructions.md').is_file()
    checks['no_promotion_report_exists'] = (OUT / 'stage4a72_no_promotion_report.json').is_file()
    checks['no_training_report_exists'] = (OUT / 'stage4a72_no_training_report.json').is_file()
    checks['no_runtime_report_exists'] = (OUT / 'stage4a72_no_runtime_report.json').is_file()
    checks['future_sketch_do_not_run'] = (OUT / 'future_stage4a77_review_import_and_promotion_decision_sketch.md').read_text(encoding='utf-8').splitlines()[0] == 'DO NOT RUN IN STAGE 4A-7.6.'
    with (OUT / 'stage4a72_manual_review_template.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    checks['at_least_30_review_rows'] = len(rows) >= 30
    checks['human_review_fields_default'] = all(r.get('human_review_status') == 'unreviewed' and not r.get('human_comment') and not r.get('human_review_reason') for r in rows)
    checks['no_promote_candidate_prefilled_yes'] = all(r.get('promote_candidate_yes_no') != 'yes' for r in rows)
    no_promo = load('stage4a72_no_promotion_report.json')
    no_train = load('stage4a72_no_training_report.json')
    no_runtime = load('stage4a72_no_runtime_report.json')
    summary = load('stage4a76_stage4a72_manual_topdown_review_summary.json')
    checks['no_expert_action_index_primary_created_from_stage4a72'] = no_promo.get('expert_action_index_primary_created_from_stage4a72') is False
    checks['stage4a72_candidate_expansion_only'] = no_promo.get('stage4a72_promoted') is False and summary.get('stage4a72_promoted') is False
    checks['lambda48_shadow_only'] = no_promo.get('lambda48_primary_use') is False
    checks['no_isaac_startup'] = no_runtime.get('isaac_startup') is False
    checks['no_capture'] = no_runtime.get('capture') is False
    checks['no_map_predict'] = no_runtime.get('map_predict') is False
    checks['no_sscnet_inference'] = no_runtime.get('sscnet_inference') is False
    checks['no_action'] = no_runtime.get('action_execution') is False
    checks['no_rollout'] = no_runtime.get('rollout') is False and no_runtime.get('short_rollout') is False
    checks['no_long_rollout'] = no_runtime.get('long_rollout') is False
    checks['no_training'] = no_train.get('bc_training') is False
    checks['no_optimizer_step'] = no_train.get('optimizer_step') is False
    checks['no_checkpoint'] = no_train.get('checkpoint') is False
    checks['no_rl_gdpo_ppo'] = no_train.get('rl_gdpo_ppo') is False
    checks['hashes_unchanged'] = all(v.get('unchanged') for v in summary.get('hash_report', {}).values())
    tracked = subprocess.run('git ls-files', cwd=ROOT, shell=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
    forbidden = ['outputs/', 'logs/', 'checkpoints/', '.npz', '.npy', '.png', '.mp4', '.usd', '.pth', '.tar']
    checks['git_large_artifact_policy_preserved'] = not any(any(tok in f for tok in forbidden) for f in tracked)
    blockers = [k for k, v in checks.items() if not v]
    print(json.dumps({'stage': 'Stage 4A-7.6 validator', 'all_passed': not blockers, 'checks': checks, 'blockers': blockers}, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == '__main__':
    raise SystemExit(main())
