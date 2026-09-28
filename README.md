# Explainable-by-Design Speech Emotion Recognition

Interpretable Speech Emotion Recognition on IEMOCAP using self-supervised speech models (HuBERT and wav2vec2) with a Concept Bottleneck Model (CBM) and a Concept Embedding Model (CEM). The model predicts 15 interpretable acoustic concepts (pitch, energy, speech rate, pauses, voice quality) and uses them to classify 4 emotions (neutral, happy, sad, angry). The project includes concept intervention experiments and a systematic XAI evaluation based on the Co-12 framework.

## Repository structure

```
code/
  xai_common.py            shared module (paths, parsing, features, models, training, metrics)
  requirements.txt         Python dependencies
  Hubert/
    01_eda_extraction_Hubert.ipynb      EDA and feature extraction
    02_weighted_cbm_cem_Hubert.ipynb    training (probes, CBM, CEM, baselines, interventions)
    03_evaluation_no_mfcc_Hubert.ipynb  XAI evaluation (Co-12 metrics)
  wav2vec/
    01_eda_extractionwav2vec2.ipynb     same pipeline, wav2vec2 backbone
    02_weighted_cbm_wav2vec2.ipynb
    03_evaluation_no_mfcc_wav2vec2.ipynb
  compare_backbones.ipynb  final comparison of the two backbones
  my_outputs_Hubert/       generated data, trained models, results (HuBERT)
  my_outputs_Wav2Vec/      generated data, trained models, results (wav2vec2)
  compare/                 comparison figures
```

Both backbone pipelines share all code through `xai_common.py`. The notebooks differ only in the `BACKBONE` variable.

## Requirements

The code was developed and tested on Google Colab with Python 3.10 or later and a GPU runtime. The GPU is needed for notebook 01 (SSL feature extraction) and helps in notebook 02.

Install dependencies with

```
pip install -r requirements.txt
```

Main packages are numpy, pandas, scikit-learn, torch, transformers, librosa, soundfile, praat-parselmouth, matplotlib, seaborn.

## Data

You need the IEMOCAP dataset (IEMOCAP_full_release), which must be requested from USC. You can find it in the data folder.

The notebooks expect this layout on Google Drive

```
MyDrive/Colab Notebooks/XAI_SER/
  code/                        this repository
  data/IEMOCAP_full_release/   Session1 ... Session5
```

If you use different paths, edit the `PROJ` and `IEMOCAP_ROOT` variables in the setup cell of each notebook.

## How to reproduce the results

Run the notebooks in order, top to bottom.

1. `Hubert/01_eda_extraction_Hubert.ipynb`. Parses IEMOCAP, runs the EDA, extracts the 12 mean-pooled HuBERT layers, temporal chunk embeddings, and acoustic concept features. Saves everything to `my_outputs_Hubert/`. This is the slowest step (the whole dataset goes through the backbone).
2. `Hubert/02_weighted_cbm_cem_Hubert.ipynb`. Trains per-layer probes, the weighted layer aggregation, the CBM variants, the CEM, and the black-box baselines. Runs the concept intervention experiments. Saves models to `my_outputs_Hubert/my_model/` and metrics to `my_outputs_Hubert/my_results/summary.json`.
3. `Hubert/03_evaluation_no_mfcc_Hubert.ipynb`. Runs the XAI evaluation (faithfulness, consistency, contrastivity, sanity checks, interventions). Saves `evaluation_summary.json` and the figures to `my_outputs_Hubert/my_results/`.
4. Repeat steps 1 to 3 with the notebooks in `wav2vec/` for the wav2vec2 backbone. Outputs go to `my_outputs_Wav2Vec/`.
5. `compare_backbones.ipynb`. Reads the result files of both runs and produces the comparison figures and the McNemar test between backbones. Figures are saved to `compare/`.

Each notebook 01 saves a checkpoint during extraction, so it can resume if the Colab session drops.

## Experimental setup

The setup follows the standard 4-class IEMOCAP protocol. Excited is merged into happy, the other labels are discarded. Sessions 1 to 4 are used for training and validation (speaker-disjoint), Session 5 is the test set. Concept binarization thresholds and all preprocessing statistics are computed on the training set only.

## Outputs

For each backbone the pipeline produces

* `metadata.json`, `layer_embeddings.npz`, `chunk_embeddings.npz`, `acoustic_features_v3.json` in `my_outputs_*/` (extracted data)
* trained models (`probes.pkl`, `aggregator.pt`, `cbm_*.pkl`, `cem.pt`, `mlp.pkl`, `temporal_attention.pt`) in `my_outputs_*/my_model*/`
* metrics and figures (`summary.json`, `evaluation_summary.json`, `test_predictions.json`, plots) in `my_outputs_*/my_results*/`

The numbers reported in the paper come from `summary.json` and `evaluation_summary.json` of each backbone plus the figures in `compare/`.
