# AlBayanSimile
We combined two worlds:

Neural – AraBERT sees the sentence and predicts the probability of a simile.

Symbolic – Classical Arabic rules check for simile particles (كأن, كما), verbs (يشبه, يماثل), nouns (مثل, شبيه), and prefixes (كـ…). Each rule adds confidence and keeps a reasoning trace.

Then we fuse them:

Hybrid score = 0.6 * neural_prob + 0.4 * symbolic_confidence

Score ≥ 0.5 → simile, else not simile.
Output includes step-by-step explanations of why the model decided that.
