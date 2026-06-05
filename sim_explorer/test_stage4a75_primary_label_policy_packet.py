#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/ubuntu22/sc_explorer_ws')
OUT = ROOT / 'outputs/isaac_stage4a75_no_training_primary_label_policy_next_step_packet'


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    checks = {}
    required = [
        'stage4a75_primary_label_policy_next_step_summary.json',
        'stage4a75_primary_label_policy_next_step_summary.md',
        'loaded_context_manifest.json',
        'loaded_context_manifest.md',
        'loaded_stage4a70_primary_dataset_evidence.json',
        'loaded_stage4a70_primary_dataset_evidence.md',
        'loaded_stage4a72_candidate_expansion_evidence.json',
        'loaded_stage4a72_candidate_expansion_evidence.md',
        'loaded_stage4a73_adapter_evidence.json',
        'loaded_stage4a73_adapter_evidence.md',
        'loaded_stage4a74_decision_packet_evidence.json',
        'loaded_stage4a74_decision_packet_evidence.md',
        'active_goal_completion_evidence.json',
        'active_goal_completion_evidence.md',
        'current_primary_label_rule.md',
        'current_primary_label_rule.json',
        'why_stage4a72_is_not_primary.md',
        'why_stage4a72_is_not_primary.json',
        'option_comparison_table.csv',
        'option_comparison_table.json',
        'option_comparison_table.md',
        'promotion_policy_risk_audit.json',
        'promotion_policy_risk_audit.md',
        'no_training_decision_report.json',
        'no_training_decision_report.md',
        'no_promotion_report.json',
        'no_promotion_report.md',
        'no_runtime_report.json',
        'no_runtime_report.md',
        'exact_approval_phrases_for_next_options.md',
        'exact_approval_phrases_for_next_options.json',
        'future_stage4a76_promotion_implementation_sketch.md',
        'future_stage4a76_primary_eligible_short_rollout_design_sketch.md',
        'recommended_next_faithful_step.md',
        'source_hash_report.json',
        'source_hash_report.md',
        'checkpoint_hash_report.json',
        'checkpoint_hash_report.md',
        'prior_dataset_hash_report.json',
        'prior_dataset_hash_report.md',
        'git_status_before.txt',
        'git_status_after.txt',
    ]
    checks['output_dir_exists'] = OUT.is_dir()
    checks['required_outputs_present'] = all((OUT / x).is_file() for x in required)
    summary = load('stage4a75_primary_label_policy_next_step_summary.json')
    no_train = load('no_training_decision_report.json')
    no_promo = load('no_promotion_report.json')
    no_runtime = load('no_runtime_report.json')
    phrases = load('exact_approval_phrases_for_next_options.json')
    prior_hash = load('prior_dataset_hash_report.json')
    source_hash = load('source_hash_report.json')
    ckpt_hash = load('checkpoint_hash_report.json')
    checks['summary_completed'] = summary.get('completed') is True and summary.get('blocked') is False
    checks['option_comparison_exists'] = (OUT / 'option_comparison_table.json').is_file() and (OUT / 'option_comparison_table.md').is_file()
    checks['exact_approval_phrases_exist'] = len(phrases.get('phrases', {})) == 5
    checks['no_training_report_exists'] = (OUT / 'no_training_decision_report.json').is_file()
    checks['no_promotion_report_exists'] = (OUT / 'no_promotion_report.json').is_file()
    checks['no_runtime_report_exists'] = (OUT / 'no_runtime_report.json').is_file()
    checks['future_sketches_marked_do_not_run'] = (
        (OUT / 'future_stage4a76_promotion_implementation_sketch.md').read_text(encoding='utf-8').splitlines()[0] == 'DO NOT RUN IN STAGE 4A-7.5.'
        and (OUT / 'future_stage4a76_primary_eligible_short_rollout_design_sketch.md').read_text(encoding='utf-8').splitlines()[0] == 'DO NOT RUN IN STAGE 4A-7.5.'
    )
    checks['stage4a72_not_promoted'] = no_promo.get('stage4a72_labels_promoted_to_expert_action_index_primary') is False
    checks['no_new_stage4a72_expert_action_index_primary'] = no_promo.get('new_expert_action_index_primary_from_stage4a72_created') is False
    checks['lambda48_shadow_only'] = no_promo.get('lambda48_role') == 'shadow/baseline only' and summary.get('lambda48_status') == 'shadow/baseline only'
    checks['no_isaac_startup'] = no_runtime.get('isaac_started') is False
    checks['no_capture'] = no_runtime.get('capture') is False
    checks['no_map_predict'] = no_runtime.get('map_predict') is False
    checks['no_sscnet_inference'] = no_runtime.get('sscnet_inference') is False
    checks['no_action'] = no_runtime.get('action_execution') is False
    checks['no_rollout'] = no_runtime.get('rollout') is False and no_runtime.get('short_rollout') is False
    checks['no_long_rollout'] = no_runtime.get('long_rollout') is False
    checks['no_bc_training'] = no_train.get('bc_training_executed') is False
    checks['no_optimizer_step'] = no_train.get('optimizer_step') is False
    checks['no_backward'] = no_train.get('backward_pass') is False
    checks['no_checkpoint'] = no_train.get('checkpoint_created') is False
    checks['no_rl_gdpo_ppo'] = no_train.get('rl_gdpo_ppo_executed') is False
    checks['hashes_unchanged'] = (
        source_hash['fixed_usd']['unchanged']
        and ckpt_hash['checkpoint']['unchanged']
        and all(x['unchanged'] for x in prior_hash['datasets'].values())
    )
    tracked = subprocess.run('git ls-files', cwd=ROOT, shell=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()
    forbidden_tokens = ['outputs/', 'logs/', 'checkpoints/', '.npz', '.npy', '.png', '.mp4', '.usd', '.pth', '.tar']
    checks['git_large_artifact_policy_preserved'] = not any(any(tok in f for tok in forbidden_tokens) for f in tracked)
    blockers = [k for k, v in checks.items() if not v]
    result = {'stage': 'Stage 4A-7.5 validator', 'all_passed': not blockers, 'checks': checks, 'blockers': blockers}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == '__main__':
    raise SystemExit(main())
