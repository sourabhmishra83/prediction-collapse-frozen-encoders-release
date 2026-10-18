# Monant usage validation and sample audit — 17 October 2026

## Decision

The underlying data are Monant Medical Misinformation, not the separate FakeHealth HealthStory/HealthRelease resource. The local folder and derivative filename are misleading. The derivative is reproducible as an associated-claim-rating proxy, but its binary labels are not validated whole-article truth. It must not be added as a fourth article-truth performance benchmark on the strength of this audit. Monant remains suitable as a derived-label validity case, and the available expert pair annotations support separate claim-presence/stance tasks.

## What was checked

The full archived derivative has 16,576 unique articles. All saved binary labels are reproduced by majority vote over embedded claim ratings: true/mostly-true map to 0, false/mostly-false to 1, other ratings are ignored, and ties map to 1. There are 775 such ties. Every selected article-level annotation is category prediction, method 9 (aggregation), rather than an expert article-truth label. This numerical reconstruction establishes label lineage; the precise original Monant-to-derivative producing script was not located. The archived datasets_setup notebook instead describes the separate FakeHealth download/schema and cannot establish the provenance of the actual Monant-derived CSV. It was inspected as historical documentation only, not executed or adopted as new experiment code.

A diagnostic using embedded stance-aware partial-veracity votes resolves 15,999 rows, differs from the saved label on 2,730 (17.06% of resolved rows), and leaves 577 unresolved. It has 761 binary vote ties. This alternative is also an automatically derived proxy; replacing the old labels with it would not establish human article truth. The authors' descriptive notebook uses claim presence and partial-veracity fields and does not justify a stance-blind majority vote as whole-article truth. A true claim can be contradicted by an article, and a false claim can be rejected; copying the claim's rating onto the article ignores that distinction.

## Official sample checked against the local full source

Official repository commit: a3bb84ed7c3a9e7e83e837a277037f0c8726c499. The downloaded sample contains 1,000 rows per inspected table. All 1,000 article IDs and titles match the local source. None of the article bodies is an exact full-body match: all 1,000 sample bodies explicitly end with a truncation marker. This is a demonstration sample, not an appropriate substitute for full article text. The local derivative's full-text recipe was established in the earlier provenance audit, which reconstructed all 16,576 saved texts.

All 1,000 sampled claim IDs/ratings match the local claims table. All 1,000 sampled entity annotation IDs and decoded values match the full source; these are prediction annotations, not human article-truth labels. The sample relation table contains 172 predicted claim-presence annotations and 828 human-labeled fact-check-to-claim mappings. Those 828 mappings must not be mistaken for human article stance or truth labels. The sample is not a representative benchmark-validation sample.

## Existing human evidence

The raw relation table includes 312 not-determined-yet placeholder rows, which were excluded. Across the full dump, 882 completed presence/stance annotation rows remain before pair deduplication. No conflicting completed pair/type labels were found. On pairs included in the derivative's automatically selected claim associations, predicted presence agrees with expert presence on 191/318 pairs; stance agrees on 200/237 pairs. The denominators differ and the subset is conditioned on automatic association, so these descriptive agreements are not representative population error estimates. They are not measures of article-level truth accuracy.

Independent of whether the automatic aggregation retained a pair, 340 completed expert presence pairs (304 derivative articles: 192 present, 89 not-present, 59 suggestive) and 240 completed expert stance pairs (218 articles: 155 supporting, 67 contradicting, 18 neutral) were extracted locally. These are task-aligned candidate evaluation subsets, not new whole-article labels and not newly executed model experiments. Article and claim reuse must be controlled when allocating any future train/test splits; do not assume row independence. The raw sample, article texts and these restricted subsets remain local.

The 142 articles with an associated pair having completed expert presence=present and directional expert stance are still only partially covered. Pair evidence does not certify the truth of the complete article. An exploratory reading of three expert-contradicting examples confirmed why claim ratings cannot simply be inherited as article labels; this was an AI-assisted semantic check, not independent human adjudication or medical fact-checking.

## Additional consistency checks

Two embedded claim references are absent from the supplied claims table. There are 3,236 raw embedded/table rating differences: 3,232 are unknown versus the literal null marker, a missing-value convention rather than substantive disagreement; four are embedded unknown versus table false. The four remaining differences require snapshot/version interpretation before using those specific pairs as evidence. Neither should be silently rewritten. Counts refer to article-claim occurrences, not unique claims. Preserve original files and report the provenance boundary.

## Review sample and what remains

A deterministic sample of 120 articles was frozen with seed 20261017: 20 from each of six strata (saved class 0/1 crossed with agreement/disagreement/unresolved under the partial-veracity diagnostic). Inclusion probabilities and population sizes are recorded in the private sampling key. Equal stratum sample sizes are not population proportional; any overall error estimate must use sampling weights and appropriate uncertainty. No precision or power target is claimed for this initial review sample.

Local reviewer pack: work/monant_validation_private/blinded_review_articles.jsonl, reviewer_ratings_blank.csv, sampling_frame_key.csv and reviewer_key.jsonl. The blinded pack hides saved/proxy labels but includes article text and associated claim statements; reviewers should determine claim presence/stance and obtain dated evidence independently rather than assume automatic associations are correct. Reviewer IDs can be joined to the private key after ratings are locked. The sample covers all proxy strata but does not fix missing article ground truth.

Independent human article-truth review has not occurred. Before such a benchmark is claimed, define article-level assessability/label semantics and obtain two independent reviews, documented evidence, disagreement adjudication and a complete uncertainty analysis. Otherwise retain the present label-validity case and three-source truth-classification study. No extra GPU run is needed for this usage decision.

## Primary documentation

- [Official dataset repository and intended tasks](https://github.com/kinit-sk/medical-misinformation-dataset)
- [Official sample](https://github.com/kinit-sk/medical-misinformation-dataset/tree/main/sample_data)
- [Dataset record and conditions](https://zenodo.org/records/5996864)

The source documentation distinguishes human labels from model predictions and explicitly avoids whole-article true/false labeling. Raw restricted data are not included in the uploaded audit or GitHub distribution.
