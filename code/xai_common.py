"""
xai_common.py Shared utilities for the P2 project (XAI for Speech).

Single source of truth for: paths, constants, IEMOCAP parsing, acoustic
features, concept binarization, models (aggregator, temporal attention, CEM),
training, and metrics. Imported by all notebooks (HuBERT and wav2vec2), which
differ only in the BACKBONE variable.

Usage on Colab:
    import sys; sys.path.append('/content/drive/MyDrive/Colab Notebooks/XAI_Project')
    import xai_common as xc
"""
from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================
# Constants
# =====================================================================
SEED = 42
LABEL_NAMES = ['neutral', 'happy', 'sad', 'angry']
LABEL_MAP = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
EMOTION_MAP = {'neu': 'neu', 'hap': 'hap', 'exc': 'hap',  # exc->hap: standard merge
               'sad': 'sad', 'ang': 'ang'}
# 'fru' is NOT discarded for being rare (it's actually one of IEMOCAP's most
# numerous classes): it is discarded for comparability with the standard
# 4-class setup used in the SER literature.
DISCARD = {'fru', 'fea', 'sur', 'dis', 'oth', 'xxx'}

ACOUSTIC_CONCEPTS = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
                     'voiced_ratio', 'rms_mean', 'rms_cv', 'rms_slope',
                     'speech_rate', 'pause_ratio', 'pause_dur_mean',
                     'jitter', 'shimmer', 'hnr', 'alpha_ratio']
MFCC_CONCEPTS = ['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4']   # ablation only, never in the XAI evaluation
DIM_CONCEPTS = ['valence', 'activation', 'dominance']       # quasi-labels: kept as a separate level
ALL_CONCEPTS = ACOUSTIC_CONCEPTS + MFCC_CONCEPTS + DIM_CONCEPTS

N_LAYERS = 12
N_CHUNKS = 8            # temporal segments for the attention pooling
MODEL_NAMES = {'hubert': 'facebook/hubert-base-ls960',
               'wav2vec2': 'facebook/wav2vec2-base'}


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)



# Paths: single configuration point, with guards
def get_paths(backbone: str, proj_root: str | Path | None = None) -> dict:
    """Returns the data/model/results paths for the requested backbone.
    """
    proj = Path(proj_root or '/content/drive/MyDrive/Colab Notebooks/XAI_Project')
    sub = {'hubert': ('my_outputs_Hubert', 'my_model', 'my_results'),
           'wav2vec2': ('my_outputs_Wav2Vec', 'my_model_wav2vec2', 'my_results_wav2vec2')}
    data = proj / sub[backbone][0]
    model = data / sub[backbone][1]
    results = data / sub[backbone][2]
    for d in (model, results):
        d.mkdir(parents=True, exist_ok=True)
        assert str(d).startswith(str(data)), f'{d} is outside {data}: accidental absolute path?'
    return {'proj': proj, 'data': data, 'model': model, 'results': results}



# Atomic saves: tmp file + os.replace, never a corrupted artifact
def _json_safe(o):
    """NaN/inf -> None: json.dump would otherwise serialize them as 'NaN',
    which is only valid in Python; with allow_nan=False the file stays
    standard JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def atomic_json(path: Path, obj) -> None:
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(_json_safe(obj), f, allow_nan=False)
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    tmp = Path(str(path) + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)



# IEMOCAP parsing with a discard counter
def parse_evaluation_dir(session_path: Path) -> tuple[list[dict], Counter]:
    """Parse SessionX/dialog/{EmoEvaluation,Evaluation}/*.txt.
    """
    eval_dir = next((session_path / 'dialog' / c
                     for c in ['EmoEvaluation', 'Evaluation']
                     if (session_path / 'dialog' / c).exists()), None)
    if eval_dir is None:
        raise FileNotFoundError(f'Evaluation dir not found in {session_path}/dialog/')

    records, discards = [], Counter()
    for txt_file in sorted(eval_dir.glob('*.txt')):
        with open(txt_file, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('[') or '\t' not in line:
                    continue
                parts = line.split('\t')
                if len(parts) < 4:
                    discards['malformed_line'] += 1
                    continue
                utt_id = parts[1].strip()
                emotion = parts[2].strip().lower()
                try:
                    nums = re.findall(r'[-+]?\d*\.?\d+', parts[3].strip())
                    valence, activation, dominance = (float(x) for x in nums[:3])
                except Exception:
                    valence = activation = dominance = float('nan')

                if emotion in DISCARD:
                    discards[f'emo:{emotion}'] += 1
                    continue
                if emotion not in EMOTION_MAP:
                    discards[f'unknown_emo:{emotion}'] += 1
                    continue
                dialog_id = utt_id.rsplit('_', 1)[0]
                wav_path = session_path / 'sentences' / 'wav' / dialog_id / f'{utt_id}.wav'
                if not wav_path.exists():
                    discards['missing_wav'] += 1
                    continue
                records.append({
                    'utt_id': utt_id,
                    'session': utt_id[4],
                    'speaker': utt_id[3:5] + '_' + utt_id.split('_')[-1][0],
                    'gender': utt_id.split('_')[-1][0],
                    'emotion': EMOTION_MAP[emotion],
                    'emotion_id': LABEL_MAP[EMOTION_MAP[emotion]],
                    'valence': valence, 'activation': activation, 'dominance': dominance,
                    'wav_path': str(wav_path),
                })
    return records, discards


def print_discards(discards: Counter, n_kept: int) -> None:
    """Readable discard table + missing-wav rate (dataset warning)."""
    tot = sum(discards.values())
    print(f'Utterances kept: {n_kept}   discarded: {tot}')
    for reason, n in discards.most_common():
        print(f'  {reason:22s} {n:5d}')
    if discards['missing_wav'] > 0.05 * max(n_kept, 1):
        print('\n[WARNING] Many missing wavs: the IEMOCAP copy is probably '
              'incomplete (e.g. improvised sessions only). Check the dataset '
              'and state the subset used in the report.')



# Audio: trim edge silences and acoustic features
def trim_edges(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Removes leading/trailing silence (internal pauses are kept: they are
    informative and become the pause_ratio concept)."""
    import librosa
    ivals = librosa.effects.split(audio, top_db=top_db)
    if len(ivals) == 0:
        return audio
    return audio[ivals[0][0]:ivals[-1][1]]


def compute_acoustic(audio: np.ndarray, sr: int = 16000) -> dict:
    """Interpretable acoustic/prosodic features + MFCC (ablation).
    """
    import librosa
    from scipy.signal import find_peaks

    audio = trim_edges(audio)
    feats = {}
    dur = len(audio) / sr

    # --- pitch ---
    f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr, frame_length=2048)
    f0_v = f0[voiced] if voiced is not None and voiced.any() else np.array([])
    f0_v = f0_v[~np.isnan(f0_v)]
    feats['pitch_mean'] = float(np.mean(f0_v)) if f0_v.size else float('nan')
    feats['pitch_std'] = float(np.std(f0_v)) if f0_v.size else float('nan')

    # F0 contour dynamics (in semitones: scale-invariant with respect to the
    # speaker's register) and fraction of voiced frames
    if f0_v.size >= 2:
        p10, p90 = np.percentile(f0_v, [10, 90])
        feats['pitch_range'] = float(12 * np.log2(p90 / p10)) if p10 > 0 else float('nan')
        hop_pyin = 2048 // 4
        t_v = np.flatnonzero(voiced) * (hop_pyin / sr)
        st = 12 * np.log2(f0[voiced] / np.mean(f0_v))
        ok = ~np.isnan(st)
        feats['pitch_slope'] = (float(np.polyfit(t_v[ok], st[ok], 1)[0])
                                if ok.sum() >= 2 and np.ptp(t_v[ok]) > 0 else float('nan'))
    else:
        feats['pitch_range'] = feats['pitch_slope'] = float('nan')
    feats['voiced_ratio'] = (float(np.mean(voiced))
                             if voiced is not None and voiced.size else float('nan'))

    # --- energy ---
    hop = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std'] = float(np.std(rms))          # no longer a concept: kept for derived features
    feats['rms_cv'] = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else float('nan')
    # relative slope of the energy (>0 voice building up, <0 fading out)
    if len(rms) >= 2 and rms.mean() > 0:
        t_r = np.arange(len(rms)) * hop / sr
        feats['rms_slope'] = float(np.polyfit(t_r, rms / rms.mean(), 1)[0])
    else:
        feats['rms_slope'] = float('nan')

    # --- speech rate: peaks of the smoothed envelope (~syllables) ---
    env = np.convolve(rms, np.ones(5) / 5, mode='same')          # ~160 ms smoothing
    min_dist = max(1, int(0.10 * sr / hop))                      # >= 100 ms between peaks
    peaks, _ = find_peaks(env, distance=min_dist,
                          prominence=0.1 * (env.max() - env.min() + 1e-9))
    speech_ivals = librosa.effects.split(audio, top_db=30)
    speech_dur = sum((e - s) for s, e in speech_ivals) / sr if len(speech_ivals) else dur
    feats['speech_rate'] = float(len(peaks) / speech_dur) if speech_dur > 0 else float('nan')
    feats['pause_ratio'] = float(1.0 - speech_dur / dur) if dur > 0 else float('nan')
    # average duration of internal pauses (0 = continuous speech; distinguishes
    # few long pauses from many short ones at the same pause_ratio)
    if len(speech_ivals) >= 2:
        gaps = [(speech_ivals[i + 1][0] - speech_ivals[i][1]) / sr
                for i in range(len(speech_ivals) - 1)]
        feats['pause_dur_mean'] = float(np.mean(gaps))
    else:
        feats['pause_dur_mean'] = 0.0

    # --- voice quality (praat) ---
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        pp = call(snd, 'To PointProcess (periodic, cc)', 75, 500)
        feats['jitter'] = float(call(pp, 'Get jitter (local)', 0, 0, 1e-4, 0.02, 1.3))
        feats['shimmer'] = float(call([snd, pp], 'Get shimmer (local)',
                                      0, 0, 1e-4, 0.02, 1.3, 1.6))
        harm = call(snd, 'To Harmonicity (cc)', 0.01, 75, 0.1, 1.0)
        feats['hnr'] = float(call(harm, 'Get mean', 0, 0))
    except Exception:
        feats['jitter'] = feats['shimmer'] = feats['hnr'] = float('nan')

    # --- alpha ratio (eGeMAPS) spectral balance in dB between the low band
    # (50-1000 Hz) and the high band (1-5 kHz): high = dark/breathy voice, low
    # = bright/tense voice (anger increases high-frequency energy) ---
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop)) ** 2
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e_low = float(S[(fft_freqs >= 50) & (fft_freqs < 1000)].sum())
    e_high = float(S[(fft_freqs >= 1000) & (fft_freqs < 5000)].sum())
    feats['alpha_ratio'] = (float(10 * np.log10(e_low / e_high))
                            if e_low > 0 and e_high > 0 else float('nan'))

    # --- MFCC (ablation only, not interpretable) ---
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(4):
        feats[f'mfcc_{i + 1}'] = float(np.mean(mfcc[i]))
    return feats


def impute_nan(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Imputes NaNs with the column median computed ONLY on train
    (no sentinel values, no test statistics)."""
    X = X.copy()
    med = np.nanmedian(X[train_mask], axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X



# Concept binarization: thresholds from train ONLY, per gender
def binarize_concepts(X: np.ndarray, names: list[str], genders: np.ndarray,
                      train_mask: np.ndarray,
                      acoustic: list[str] | None = None,
                      dims: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Deployment-valid binary "high/low" concepts.

    - acoustic/MFCC: median per GENDER estimated on train only (removes the
      pitch/gender confound without using per-speaker test statistics: for a
      new utterance, knowing the speaker's gender is enough);
    - dimensional: global train median (shared 1-5 scale).
    Returns (C_bin, thresholds) so the same thresholds are reusable
    (test, LOSO folds, deployment).
    """
    acoustic = acoustic if acoustic is not None else [c for c in names if c not in DIM_CONCEPTS]
    dims = dims if dims is not None else [c for c in names if c in DIM_CONCEPTS]
    a_idx = [names.index(c) for c in acoustic]
    d_idx = [names.index(c) for c in dims]

    def _split(x_tr, x_all, med):
        for op, t in ((np.greater_equal, med), (np.greater, med),
                      (np.greater, float(np.nanmean(x_tr)))):
            b_tr = op(x_tr, t)
            if 0.02 < b_tr.mean() < 0.98:          # non-degenerate split on train
                return op(x_all, t).astype(np.int8), float(t)
        warnings.warn('binarize_concepts: degenerate split even after the fallbacks '
                      '(feature ~constant in train) the binary column stays '
                      'nearly constant and the probes may fail.')
        return op(x_all, t).astype(np.int8), float(t)   # ~constant feature: stays degenerate

    C = np.zeros(X.shape, dtype=np.int8)
    thr = {}
    for g in np.unique(genders):
        tr_g = train_mask & (genders == g)
        g_mask = genders == g
        thr[f'gender_{g}'] = {}
        for c, j in zip(acoustic, a_idx):
            med = float(np.nanmedian(X[tr_g][:, j]))
            C[g_mask, j], thr[f'gender_{g}'][c] = _split(X[tr_g][:, j], X[g_mask, j], med)
    thr['dimensional'] = {}
    for c, j in zip(dims, d_idx):
        med = float(np.nanmedian(X[train_mask][:, j]))
        C[:, j], thr['dimensional'][c] = _split(X[train_mask][:, j], X[:, j], med)
    return C, thr


def standardize_concepts(X: np.ndarray, names: list[str], genders: np.ndarray,
                         train_mask: np.ndarray,
                         acoustic: list[str] | None = None,
                         dims: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """CONTINUOUS version of binarization: per-gender z-score (acoustic)
    or global z-score (dimensional), with statistics estimated on train ONLY.
    Deployment-valid just like the per-gender medians: for a new utterance
    the speaker's gender is enough. The median-split throws away the
    concept's intensity information (~6 oracle points on the IEMOCAP data):
    this is the counterpart for the continuous CBM variant (Koh et al. 2020
    also use continuous concepts). Returns (X_z, params) with reusable
    parameters."""
    acoustic = acoustic if acoustic is not None else [c for c in names if c not in DIM_CONCEPTS]
    dims = dims if dims is not None else [c for c in names if c in DIM_CONCEPTS]
    a_idx = [names.index(c) for c in acoustic]
    d_idx = [names.index(c) for c in dims]

    Xz = np.zeros(X.shape, dtype=np.float64)
    params = {}
    for g in np.unique(genders):
        tr_g = train_mask & (genders == g)
        g_mask = genders == g
        mu = np.nanmean(X[tr_g][:, a_idx], axis=0)
        sd = np.nanstd(X[tr_g][:, a_idx], axis=0) + 1e-9
        Xz[np.ix_(g_mask, a_idx)] = (X[np.ix_(g_mask, a_idx)] - mu) / sd
        params[f'gender_{g}'] = {c: {'mean': float(m), 'std': float(s)}
                                 for c, m, s in zip(acoustic, mu, sd)}
    if d_idx:
        mu = np.nanmean(X[train_mask][:, d_idx], axis=0)
        sd = np.nanstd(X[train_mask][:, d_idx], axis=0) + 1e-9
        Xz[:, d_idx] = (X[:, d_idx] - mu) / sd
        params['dimensional'] = {c: {'mean': float(m), 'std': float(s)}
                                 for c, m, s in zip(dims, mu, sd)}
    return Xz.astype(np.float32), params



# Models
class WeightedAggregator(nn.Module):
    """Combines the 12 SSL layers with learned softmax weights.

    LayerNorm without affine parameters before the aggregation: without it,
    layers with a higher norm would dominate the gradient and receive more
    weight regardless of their content (a historical bug of the old NB3).
    """

    def __init__(self, n_layers: int = N_LAYERS, hidden_dim: int = 768, n_classes: int = 4):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def aggregate(self, x):                       # x: (B, 12, 768)
        x = self.pre_norm(x)
        alpha = torch.softmax(self.layer_weights, dim=0)
        return (x * alpha[None, :, None]).sum(dim=1)

    def forward(self, x):
        return self.classifier(self.aggregate(x))


class TemporalAttentionClassifier(nn.Module):

    def __init__(self, dim: int = 768, n_classes: int = 4, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Linear(dim, n_classes)

    def attention(self, x):                       # x: (B, T, D) -> (B, T)
        return torch.softmax(self.att(self.norm(x)).squeeze(-1), dim=1)

    def forward(self, x):
        a = self.attention(x)
        pooled = (self.norm(x) * a.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)


class ConceptEmbeddingModel(nn.Module):
    """Compact CEM (Zarlenga et al. 2022).

    For each concept k: two embeddings (active c+, inactive c-) computed from
    the input, one score p_k = sigma(s([c+, c-])); the classifier sees the
    concatenation of the mixing p_k*c+ + (1-p_k)*c-. Intervention: p_k := GT.
    """

    def __init__(self, in_dim: int = 768, n_concepts: int = 9,
                 emb_dim: int = 16, n_classes: int = 4):
        super().__init__()
        self.K, self.m = n_concepts, emb_dim
        self.ctx = nn.Sequential(nn.Linear(in_dim, n_concepts * 2 * emb_dim), nn.LeakyReLU())
        self.score = nn.Linear(2 * emb_dim, 1)                 # shared across concepts
        self.classifier = nn.Linear(n_concepts * emb_dim, n_classes)

    def forward(self, x, c_int=None, int_mask=None):
        B = x.shape[0]
        ctx = self.ctx(x).view(B, self.K, 2, self.m)
        cp, cm = ctx[:, :, 0], ctx[:, :, 1]                    # (B, K, m)
        logit_c = self.score(torch.cat([cp, cm], -1)).squeeze(-1)           # (B, K)
        p = torch.sigmoid(logit_c)
        if c_int is not None:                                  # intervention on a subset of concepts
            p = torch.where(int_mask.bool(), c_int, p)
        emb = p.unsqueeze(-1) * cp + (1 - p).unsqueeze(-1) * cm
        return self.classifier(emb.flatten(1)), p, logit_c



# Training: a single implementation shared by NB2 and the LOSO
def _early_stop_loop(model, params, X_tr, y_tr, X_va, y_va, loss_fn,
                     batch_size=256, max_epochs=2000, patience=20,
                     lr=3e-4, weight_decay=1e-4, device='cpu', verbose=False):
    """Generic minibatch loop + early stopping on validation."""
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    best, best_state, pc = float('inf'), None, 0
    hist = {'train': [], 'val': []}
    X_va, y_va = X_va.to(device), y_va.to(device)
    for epoch in range(max_epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model, xb, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        hist['train'].append(tot / len(y_tr))
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model, X_va, y_va).item()
        hist['val'].append(vl)
        if vl < best - 1e-4:
            best, best_state, pc = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= patience:
                if verbose:
                    print(f'  early stopping @ epoch {epoch + 1}')
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def train_aggregator(X_lay, y, inner_val_mask, device='cpu', verbose=False, **kw):
    """Weighted aggregator with early stopping (same hyperparameters
    everywhere: minibatch 256, max 2000 epochs, patience 20)."""
    ce = nn.CrossEntropyLoss()
    tr, va = ~inner_val_mask, inner_val_mask
    model = WeightedAggregator().to(device)
    return _early_stop_loop(
        model, model.parameters(),
        torch.tensor(X_lay[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.long),
        torch.tensor(X_lay[va], dtype=torch.float32), torch.tensor(y[va], dtype=torch.long),
        lambda m, xb, yb: ce(m(xb), yb), device=device, verbose=verbose, **kw)


def train_temporal(X_chunks, y, inner_val_mask, device='cpu', verbose=False, **kw):
    """TemporalAttentionClassifier with the same training protocol."""
    ce = nn.CrossEntropyLoss()
    tr, va = ~inner_val_mask, inner_val_mask
    model = TemporalAttentionClassifier(dim=X_chunks.shape[-1]).to(device)
    return _early_stop_loop(
        model, model.parameters(),
        torch.tensor(X_chunks[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.long),
        torch.tensor(X_chunks[va], dtype=torch.float32), torch.tensor(y[va], dtype=torch.long),
        lambda m, xb, yb: ce(m(xb), yb), device=device, verbose=verbose, **kw)


def train_cem(X_emb, y, C_bin, inner_val_mask, lam=1.0, p_int=0.25, device='cpu',
              verbose=False, **kw):
    """CEM with a joint loss: CE(task) + lam * BCEWithLogits(concepts).
    """
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    tr, va = ~inner_val_mask, inner_val_mask
    K = C_bin.shape[1]
    model = ConceptEmbeddingModel(in_dim=X_emb.shape[1], n_concepts=K).to(device)
    C_t = torch.tensor(C_bin, dtype=torch.float32)

    # the concept targets travel concatenated with y to reuse the generic loop
    y_pack_tr = torch.cat([torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1), C_t[tr]], 1)
    y_pack_va = torch.cat([torch.tensor(y[va], dtype=torch.float32).unsqueeze(1), C_t[va]], 1)

    def loss_fn(m, xb, yb_pack):
        yb, cb = yb_pack[:, 0].long(), yb_pack[:, 1:]
        if m.training and p_int > 0:
            mask = (torch.rand_like(cb) < p_int).float()       # per sample/concept RandInt
            logits, _, logit_c = m(xb, c_int=cb, int_mask=mask)
        else:
            logits, _, logit_c = m(xb)
        return ce(logits, yb) + lam * bce(logit_c, cb)

    return _early_stop_loop(
        model, model.parameters(),
        torch.tensor(X_emb[tr], dtype=torch.float32), y_pack_tr,
        torch.tensor(X_emb[va], dtype=torch.float32), y_pack_va,
        loss_fn, device=device, verbose=verbose, **kw)


class MLPHead(nn.Module):
    """Black-box MLP with the same training protocol as the other models
    (minibatch + early stopping on Session 4). Replaces sklearn's
    MLPClassifier, whose early_stopping uses an internal random split that is
    NOT speaker-independent: a uniform validation protocol across all
    compared models."""

    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, inner_val_mask, device='cpu', verbose=False, **kw):
    """Black-box MLP with the same training loop/hyperparameters."""
    ce = nn.CrossEntropyLoss()
    tr, va = ~inner_val_mask, inner_val_mask
    model = MLPHead(X.shape[1]).to(device)
    return _early_stop_loop(
        model, model.parameters(),
        torch.tensor(X[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.long),
        torch.tensor(X[va], dtype=torch.float32), torch.tensor(y[va], dtype=torch.long),
        lambda m, xb, yb: ce(m(xb), yb), device=device, verbose=verbose, **kw)


class TorchClf:
    """sklearn-like wrapper (predict/predict_proba) for an already-trained
    nn.Module: serializable with joblib, so NB3 can load and use it as before
    without depending on the underlying model type."""

    def __init__(self, model: nn.Module):
        self.model = model.to('cpu').eval()

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.model(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)



# Probes and concept scores
def fit_probes(X_tr_sc, C_bin_tr, names, max_iter=2000, C=1.0,
               inner_val=None, C_grid=(0.1, 1.0, 10.0)):
    probes = {}
    for k, name in enumerate(names):
        y = C_bin_tr[:, k]
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Concept '{name}' is degenerate: only one class ({int(y[0])}) in train. "
                "Binarization needs revisiting (ties at the median? see binarize_concepts).")
        best_C = C
        if inner_val is not None and inner_val.any() and (~inner_val).any():
            scores = {}
            for c in C_grid:
                m = LogisticRegression(max_iter=max_iter, C=c).fit(X_tr_sc[~inner_val], y[~inner_val])
                scores[c] = recall_score(y[inner_val], m.predict(X_tr_sc[inner_val]), average='macro')
            best_C = max(scores, key=scores.get)
        probes[name] = LogisticRegression(max_iter=max_iter, C=best_C).fit(X_tr_sc, y)
    return probes


def fit_cbm_head(C_tr, y_tr, inner_val=None, C_grid=(0.01, 0.1, 1.0, 10.0),
                 max_iter=2000):
    def make(c):
        return Pipeline([('scaler', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=max_iter,
                                                    class_weight='balanced', C=c))])
    best_C = 1.0
    if inner_val is not None and inner_val.any() and (~inner_val).any():
        scores = {}
        for c in C_grid:
            m = make(c).fit(C_tr[~inner_val], y_tr[~inner_val])
            scores[c] = recall_score(y_tr[inner_val], m.predict(C_tr[inner_val]), average='macro')
        best_C = max(scores, key=scores.get)
    pipe = make(best_C).fit(C_tr, y_tr)
    pipe.best_C = best_C
    return pipe


def fit_logreg(X_tr, y_tr, inner_val=None, C_grid=(0.01, 0.1, 1.0, 10.0),
               max_iter=2000, **kw):
    best_C = 1.0
    if inner_val is not None and inner_val.any() and (~inner_val).any():
        scores = {}
        for c in C_grid:
            m = LogisticRegression(max_iter=max_iter, C=c, **kw).fit(X_tr[~inner_val], y_tr[~inner_val])
            scores[c] = recall_score(y_tr[inner_val], m.predict(X_tr[inner_val]), average='macro')
        best_C = max(scores, key=scores.get)
    model = LogisticRegression(max_iter=max_iter, C=best_C, **kw).fit(X_tr, y_tr)
    model.best_C = best_C
    return model


def fit_reg_probes(X_tr_sc, Cz_tr, names, inner_val=None,
                   alpha_grid=(1.0, 10.0, 100.0)):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    probes = {}
    for k, name in enumerate(names):
        y = Cz_tr[:, k]
        best_a = alpha_grid[len(alpha_grid) // 2]
        if inner_val is not None and inner_val.any() and (~inner_val).any():
            scores = {}
            for a in alpha_grid:
                m = Ridge(alpha=a).fit(X_tr_sc[~inner_val], y[~inner_val])
                scores[a] = r2_score(y[inner_val], m.predict(X_tr_sc[inner_val]))
            best_a = max(scores, key=scores.get)
        probes[name] = Ridge(alpha=best_a).fit(X_tr_sc, y)
    return probes


def reg_concept_scores(probes, X_sc, names):
    """Continuous concept scores (z-values predicted by the regression probes)."""
    return np.column_stack([probes[c].predict(X_sc) for c in names]).astype(np.float32)


def concept_scores(probes, X_sc, names, hard=False):
    """Concept scores: probability (soft) or 0/1 (hard mitigates the
    extra-concept information leakage carried by the probabilities)."""
    P = np.column_stack([probes[c].predict_proba(X_sc)[:, 1] for c in names]).astype(np.float32)
    return (P >= 0.5).astype(np.float32) if hard else P



# Metrics and statistical tests
def metrics_report(y_true, y_pred) -> dict:
    """accuracy (WA), UA (macro recall, standard in SER) and macro-F1."""
    return {'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'ua': round(float(recall_score(y_true, y_pred, average='macro')), 4),
            'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 4)}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test on the difference between two
    classifiers evaluated on the same samples."""
    from scipy.stats import binomtest
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int((~a_ok & b_ok).sum())   # only B correct
    n10 = int((a_ok & ~b_ok).sum())   # only A correct
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    # no aggressive rounding: very small p-values were becoming exactly 0.0
    # in the summaries reports should state p < 1e-k, never p = 0
    return {'n_only_a': n10, 'n_only_b': n01, 'p_value': float(p)}


def bootstrap_diff_ci(y_true, pred_a, pred_b, metric='accuracy',
                      n_boot=2000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    fn = {'accuracy': accuracy_score,
          'ua': lambda t, p: recall_score(t, p, average='macro')}[metric]
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'diff_mean': round(float(diffs.mean()), 4),
            'ci95': [round(float(lo), 4), round(float(hi), 4)]}
