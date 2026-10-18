# Completed full-study extension: audit

All 432 planned jobs appear in the saved endpoint table: 324 architecture jobs and 108 matched-update jobs. The table has exactly 3,888 unique job/target/endpoint rows (432 × 3 targets × initialization/selected/final). The Colab verifier reports all saved predictions verified and no synthetic outputs. The executed five Python source hashes match the preserved local source exactly. Independently recomputed split-level aggregates match the saved condition table within 1e-12.

The execution environment records Tesla T4 CUDA, Python 3.13.15, PyTorch 2.11.0+cu130, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1 and transformers 4.52.4. GPU scatter operations are not guaranteed bitwise deterministic.

At class-1 share 0.80, architecture-panel in-domain selected and final collapse is 81/81 with unweighted cross-entropy and 0/81 with weighted cross-entropy. At share 0.50, both losses have 0/81 selected/final collapse. All 324 initialization evaluations collapse; the observed contrast concerns recovery, not necessarily creation of collapse.

Matched-update selected collapse is 27/27 for each batch mode at 30 updates. At 300 updates it is 11/27 full-batch and 14/27 batch-128; final collapse is 8/27 and 15/27 respectively. Full-batch selected counts by CoAID/FakeNewsNet/LIAR are 0/9, 3/9 and 8/9; batch-128 counts are 0/9, 5/9 and 9/9. These do not establish a mini-batch advantage. Number of examples processed differs between batch modes; this is not a noise-only causal test.

For weighted architecture models at share 0.80, 43/162 cross-domain selected evaluations collapse (text-only 18/54, GCN 13/54, GAT 12/54), despite 0/81 in-domain collapse. Target evaluations share trained models and splits and are not independent training runs.

Nine runs per condition comprise three training seeds within each of three overlapping grouped splits. The included split-specific table is the primary uncertainty record; pooled means are descriptive and are not nine independent population replicates. No pooled significance claims are made.

Audit boundary: this local audit checks the saved verifier report, environment/code identity, plan coverage and aggregate consistency. Individual prediction files, weights, graph records and trajectories remain in Drive and were verified by the executed Colab verifier; they have not all been independently downloaded and reverified locally. Initialization/selected/final aggregate rows do not establish transient-collapse counts. Monant article-level label validation remains unresolved; no fourth article-truth performance benchmark is claimed.
