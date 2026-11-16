# Response to Reviewers: PuneCon 2026, Paper ID 2150

**Few-Shot Cross-Plant Anomaly Detection in Industrial IoT** (Track 5: Cyber-Physical Systems)

We thank the reviewers for their constructive comments. All points have been addressed in the revised manuscript as follows.

## Reviewer 1

**1. Comparison with existing transfer-learning and few-shot anomaly detection approaches.**
A comparison table (Table I, Section II) now positions the work against reconstruction-based, meta-learning and few-shot methods, including USAD and TranAD, which are evaluated on the same SWaT and WADI testbeds. Section IV now states explicitly that the Static-transfer configuration is the conventional transfer-learning baseline (pre-training on the source plant followed by fine-tuning on the target, with all other settings held identical), against which meta-learning is compared directly using a paired six-seed test.

**2. Formal academic style of the abstract.**
The abstract has been rewritten in a formal style with a problem–method–result structure.

**3. Limitation of using only two industrial testbeds.**
Section VII now discusses that SWaT and WADI are related water systems from the same research centre with similar PLC-based control, so the demonstrated transfer is between related cyber-physical systems. Cross-domain evaluation is identified as future work in Sections VII and VIII.

## Reviewer 3

**1. Novelty in the abstract and distinction from conventional transfer learning.**
The abstract now states how the approach differs from conventional transfer learning (meta-learning an initialization across the operating regimes of the source plant, compared with fine-tuning weights pre-trained on a single source distribution). It also states the contributions: a label-free cross-plant setting without shared sensors, a paired comparison with conventional pre-training, and a verification procedure for a silent first-order MAML implementation fault.

**2. Motivation for the LSTM autoencoder.**
A new paragraph ("Choice of architecture", Section III-A) justifies the choice: temporal order-awareness, compactness for repeated inner-loop updates, robustness with small support sets compared with attention-based encoders, and isolation of the meta-learning effect from architectural changes. Alternative encoders are noted as future work.

**3. Discussion of Figures 1 and 2.**
Fig. 1 is now discussed further in Section III-A, covering the role of the bottleneck, the repeated code and the scoring path. Fig. 2 is now described step by step in Section III-C, following the source and target rows and stating what crosses between the plants.
