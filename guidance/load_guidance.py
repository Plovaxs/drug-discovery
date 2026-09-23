"""Loads the frozen affinity/synthesizability guidance models for sampling.

Returns (None, None) whenever a lambda is 0 (or unset), so sampling never
depends on a guidance checkpoint existing unless it is actually used — this
is what lets sampling_baseline.yml (both lambdas 0) run today, before Task
C/D produce any trained guidance model.

If a lambda is nonzero but the corresponding checkpoint config/file is
missing, this raises immediately with a message naming the task that
produces it, rather than silently sampling unguided (which would corrupt
the ablation in Task F by silently turning a "guided" run into an unguided
one).
"""
import os


def load_guidance_models(config, device='cuda:0', lambda_affinity=0.0, lambda_synth=0.0):
    affinity_model = None
    synth_model = None

    if lambda_affinity != 0.0:
        ckpt_path = config.sample.get('affinity_checkpoint', None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise RuntimeError(
                f'lambda_affinity={lambda_affinity} but config.sample.affinity_checkpoint '
                f'({ckpt_path!r}) does not exist. Train the affinity guidance model first '
                f'(Task C: models/property_pred/ trained on PDBbind/affinity_info.pkl-labeled '
                f'CrossDocked2020 entries), then set affinity_checkpoint in the sampling config.')
        from guidance.affinity_guidance import AffinityGuidance
        affinity_model = AffinityGuidance(ckpt_path, device=device)

    if lambda_synth != 0.0:
        ckpt_path = config.sample.get('synth_checkpoint', None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise RuntimeError(
                f'lambda_synth={lambda_synth} but config.sample.synth_checkpoint '
                f'({ckpt_path!r}) does not exist. Train the synthesizability guidance model '
                f'first (Task D), then set synth_checkpoint in the sampling config.')
        from guidance.synth_guidance import SynthGuidance
        synth_model = SynthGuidance(ckpt_path, device=device)

    return affinity_model, synth_model
