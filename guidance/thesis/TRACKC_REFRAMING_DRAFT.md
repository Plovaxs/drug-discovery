# Track C reframing: lead on heavy-atom count and validity, demote ligand efficiency

Draft replacement text, not yet applied to the chapters. The thesis is rebuilt once, at the end, so this
sits here until then.

## The problem, in one paragraph

The abstract currently reads: rejection sampling "improved the raw docking score in 12 of 15 pockets but
**worsened ligand efficiency in 14 of 15** (on average 12 more heavy atoms, and 29 percentage points
lower PoseBusters validity)."

Ligand efficiency is LE = affinity / heavy-atom count. Rejection sampling made the molecules **12 heavy
atoms larger**, and affinity scales sublinearly with molecular size. So LE *must* fall, largely as an
arithmetic consequence of the size increase. Presenting "LE worsened in 14 of 15" as independent
evidence **double-counts the size effect**.

This is not a minor stylistic point. Three papers argue LE is not a sound metric at all:

* **Kenny 2018** (*J. Cheminformatics*, 95 cites): LE "has a nontrivial dependency on the concentration
  unit used to express affinity that stems from the inability of the logarithm function to take
  dimensioned arguments. Consequently, perception of efficiency varies with the choice of concentration
  unit and **it is argued that the ligand efficiency metric is not physically meaningful nor should it be
  considered to be a metric**."
* **Polanski et al. 2017** (*J. Cheminformatics*): the characteristic LE trend is a 1/MW dependency; LE
  "should not be interpreted as a molecular descriptor connected with a single molecule but as a property
  (binding per gram)", and the hyperbolic trend is "not a real increase in binding potency but a physical
  limitation".
* **Zhao 2025** (*ACS Med. Chem. Lett.*): LE's "mathematical construction embeds a **strong size bias**
  that distorts cross-size comparisons", and size-independent variants inherit sensitivity to an
  arbitrary standard-state choice.

An examiner who knows the medicinal-chemistry literature will know this. Better to pre-empt it.

## The fix, and why the conclusion survives intact

Lead on the two quantities that are direct, unnormalised measurements and carry no size bias:

* **+12 heavy atoms on average** — a raw count
* **−29 percentage points PoseBusters validity** — a pass/fail rate on physical plausibility

Both are damning on their own, and neither is a ratio of correlated quantities. The docking-score
improvement in 12 of 15 pockets then reads as what it is: the proxy went up while the molecules became
larger and less physically valid, which is reward over-optimisation, not improvement.

## Suggested replacement wording

**Abstract.** "Rejection sampling improved the raw docking score in 12 of 15 pockets while producing
molecules that were on average 12 heavy atoms larger and 29 percentage points less PoseBusters-valid in
14 of 15; ligand efficiency fell correspondingly, though we note that metric's known size bias (Kenny
2018) and do not rest the conclusion on it."

**Chapter text.** Report the heavy-atom and validity deltas first, with their statistics. Then: "Ligand
efficiency, the ratio of affinity to heavy-atom count, also declined in 14 of 15 pockets. We report this
for continuity with medicinal-chemistry practice but do not treat it as independent evidence: because the
generated molecules were on average 12 heavy atoms larger and affinity scales sublinearly with size, a
decline in LE is in part an arithmetic consequence of the size increase. The metric's construction has
been argued to embed a size bias and to depend on an arbitrary concentration unit (Kenny 2018; Polanski
et al. 2017; Zhao 2025)."

**Limitations.** Add one line: "Ligand efficiency is used descriptively only; its mathematical validity
as a metric is contested, and the conclusions of Chapter VII rest on heavy-atom count and PoseBusters
validity rather than on LE."

## What NOT to do

Do not drop LE entirely. Practising medicinal chemists use it, a reader will look for it, and removing it
would look like hiding an unfavourable number rather than handling a known metric carefully. Report it,
contextualise it, and do not let it carry the argument.
