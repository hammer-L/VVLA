# Action-aware Language–Action Classifier (protocol 2.0.0)

The classifier scores local compatibility of current images, an instruction,
and a normalized future action chunk `[B,8,7]`. Protocol v2 prevents either
language or action from being ignored by training on real counterfactual actions
as well as controlled language negatives.

## Data and loss

Each training anchor yields an inseparable triplet: `(V,L+,A+)`,
`(V,L-,A+)`, and `(V,L+,A-)`. Language variants and the two donor types rotate
deterministically by epoch. The loss is fixed to

`BCE(pos_weight=2) + 0.5 language_rank + 0.5 action_rank + 0.1 VL_probe + 0.1 A_probe`,

where each rank term is `softplus(1 - (s+ - s-))`. Qwen-VL and the inherited
GR00T action policy are frozen.

## Architecture

Qwen's last-layer states remain tokens. A learned projection maps each VL token
to the classifier width. A timestep MLP plus learned positions maps the eight
7-D actions to eight action tokens. Two learned classifier queries first
cross-attend VL tokens and then action tokens; their final mean is the only input
to the deployed MLP logit head. The main head has no direct pooled `z_vl` or
`z_a` path.

`vl_audit_head(detach(z_vl))` and `action_audit_head(detach(z_a))` quantify
single-modality leakage. Their parameters train, but their input detach prevents
probe losses from updating either encoder. They never contribute to deployment
scores or checkpoint selection.

## Evaluation and selection

Every anchor expands to the strict 13-item v2 group: 3 positive, 4 language
negative, 3 wrong-phase action, and 3 hard wrong-task action instances. Reports
persist main, donor-action, source-mean-action, VL-probe, and A-probe logits and
compute language/action PRA, both donor-specific metrics, audit degradation,
micro/macro summaries, and episode-cluster 95% intervals.

Checkpoint order is lexicographic: minimum of language/action PRA, their
harmonic mean, joint AUROC, then lower validation loss. The validation threshold
is frozen before test. Protocol-v1 metadata and classifier selections are
explicitly rejected.
