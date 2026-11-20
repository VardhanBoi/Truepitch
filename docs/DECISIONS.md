# Decision log
Evidence tags: [E] established research finding, [R] our engineering recommendation, [H] hypothesis to validate.

D1 [R] Frozen encoders + cached embeddings + small heads. ~690-1200 samples cannot support full fine-tuning without high variance.
D2 [R] Text: RoBERTa-base primary, BERT-base-uncased alternative (original MUStARD baselines used BERT; gives comparability). Mean-pool token states for frozen use.
D3 [R] Speech: eGeMAPSv02 (openSMILE, 88 functionals) as handcrafted features; wav2vec 2.0 base, frozen, mid-layer mean+std pooling. [E] mid layers carry more paralinguistic information than the last layer of pretrained-only SSL models.
D4 [R] Evaluation: nested speaker-grouped CV (5 outer x inner val). Macro-F1 primary. Report mean +- std over folds and seeds, paired bootstrap CIs for model differences.
D5 [R] Acoustic valence s_a comes from a model trained on data INDEPENDENT of the sarcasm labels (see D6). Text sentiment s_t from a pretrained sentiment classifier, mapped to [-1, 1] as P(pos) - P(neg).
D6 [R] Never train s_a on MUStARD(++) valence labels and then feed it to the sarcasm model without out-of-fold prediction: all sarcastic MUStARD utterances were reported to be negative sentiment (Chauhan et al.), so valence is label-correlated => leakage path.
D7 [R] Mismatch ablation has 3 arms: (a) no mismatch, (b) raw s_t and s_a as inputs, (c) m=|s_t-s_a| (spec) [+ signed d=s_t-s_a exploratory]. H3 is only supported if (c) beats (b), not only (a).
D8 [R] MVP fusion = gated attention over two projected modality vectors (tiny param count). Late fusion stacks OUT-OF-FOLD probabilities.
D9 [H] Prosody gains may be small and inside fold noise; a null result is a valid, reportable finding.

# Known confounds (must be analysed, not ignored)
- MUStARD non-sarcastic clips come from MELD; sarcastic from other sources => acoustic/recording-condition shortcuts. Report per-show results.
- Laugh tracks and background music contaminate prosody features.
- wav2vec 2.0 encodes lexical content and speaker/channel, so "speech-only" is not purely prosody.
- Utterance-only labelling: many sarcastic items (MUStARD++ "propositional") need context, capping all models. Out of MVP scope, document as a limitation.
- Few dominant speakers => uneven speaker-grouped folds (check describe_folds output).

# Roadmap (status)
P1 Inspection/planning: DONE (this scaffold). P2 Dataset setup: loader+splits DONE, data download PENDING.
P3 Features, P4 Unimodal (E1-E3), P5 Multimodal (E4-E7), P6 App, P7 Report: NOT STARTED.
Experiment status: E1-E7 all NOT RUN.

# Phase 2-5 decisions (2-day schedule)
D10 [R] SCOPE CUTS (time): MUStARD++ dropped from this iteration (access unverified) -> future work. React/Vite frontend replaced by a FastAPI-served single HTML page in the app phase. Praat not used (eGeMAPS covers F0/loudness/jitter/shimmer). BERT alternative not run.
D11 [R] s_t = P(pos)-P(neg) from distilbert-base-uncased-finetuned-sst-2-english (movie-review domain: limitation). s_a = valence output of audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (MSP-Podcast; independent of sarcasm labels; CC BY-NC-SA 4.0, research only).
D12 [R] Calibration: s_a is moment-matched to the s_t scale using TRAIN-fold mean/std only (label-free), clipped to [-1,1]. m=|s_t-s_a| computed afterwards. Limitation: acoustic valence of acted/podcast speech vs TV dialogue.
D13 [R] Fusion speech input = concat(eGeMAPS 88, wav2vec2 768), standardised on train rows. Mid-layer wav2vec2 = mean of layers 6-9 (fixed a priori, not tuned on test).
D14 [R] Model selection/early stopping on validation LOSS (stable on ~70 val samples); LR scheduler ReduceLROnPlateau; class-weighted CE; grad clip 1.0.
D15 [R] Late fusion = w*p_text + (1-w)*p_speech, single w chosen on validation macro-F1.
D16 [R] Extra ablations: E8 (raw s_t,s_a, no explicit difference) and E9 (signed difference). H3 needs E6 > E8, not only E6 > E7.
D17 [R] Statistics: speaker-level cluster bootstrap (2000 resamples) on pooled seed-ensembled predictions; best unimodal baseline chosen on VALIDATION F1.
