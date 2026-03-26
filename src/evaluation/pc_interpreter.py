"""Principal component interpretability analysis for PCA_SUB model.

Analyzes what the extracted principal components (factors) represent
economically by examining:
1. PC loadings on U.S. sectors — what each factor captures
2. Variance explained by each PC — relative importance
3. Regression coefficients — how PCs map to Japanese sector predictions
4. Temporal stability of PC loadings across walk-forward folds
5. Factor rotation analysis — identifying economic themes
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.pca_sub import PCASub


@dataclass
class PCLoadingSnapshot:
    """PC loadings from a single walk-forward fold."""
    fold_id: int
    train_start_date: str
    train_end_date: str
    eigvecs: np.ndarray          # (n_us, K) loadings
    eigenvalues: np.ndarray      # (K,) eigenvalues
    variance_explained: np.ndarray  # (K,) fraction of variance
    beta: np.ndarray             # (K, n_jp) regression coefficients
    intercept: np.ndarray        # (n_jp,) intercept


@dataclass
class PCInterpretationResult:
    """Full PC interpretation analysis results."""
    n_folds: int = 0
    K: int = 0
    # Aggregate loading analysis
    mean_loadings: dict = field(default_factory=dict)
    loading_stability: dict = field(default_factory=dict)
    # Variance explained
    mean_variance_explained: list = field(default_factory=list)
    cumulative_variance_explained: list = field(default_factory=list)
    # Economic interpretation
    pc_sector_associations: dict = field(default_factory=dict)
    # Regression coefficient analysis (PC -> JP sectors)
    mean_beta: dict = field(default_factory=dict)
    beta_stability: dict = field(default_factory=dict)
    # Cross-market transmission channels
    transmission_channels: list = field(default_factory=list)
    # Temporal evolution
    loading_evolution: dict = field(default_factory=dict)
    # Cosine similarity between consecutive folds
    temporal_cosine_similarity: dict = field(default_factory=dict)


class PCInterpreter:
    """Analyzes economic meaning of principal components in PCA_SUB.

    Parameters
    ----------
    K : int
        Number of principal components.
    L : int
        Lookback window for model fitting.
    lambda_decay : float
        Exponential decay rate.
    train_window : int
        Walk-forward training window size.
    test_window : int
        Walk-forward test window size.
    """

    def __init__(
        self,
        K: int = 3,
        L: int = 60,
        lambda_decay: float = 0.9,
        train_window: int = 252,
        test_window: int = 21,
    ):
        self.K = K
        self.L = L
        self.lambda_decay = lambda_decay
        self.train_window = train_window
        self.test_window = test_window

    def extract_pc_snapshots(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
    ) -> list[PCLoadingSnapshot]:
        """Extract PC loadings from each walk-forward fold.

        Fits the model on each training window and records the eigenvectors,
        eigenvalues, and regression coefficients.
        """
        T = X_us.shape[0]
        n_us = X_us.shape[1]
        snapshots = []

        fold_id = 0
        start = 0

        while start + self.train_window + self.test_window <= T:
            train_end = start + self.train_window

            X_train = X_us[start:train_end]
            Y_train = Y_jp[start:train_end]

            # Fit model to get eigenvectors and coefficients
            model = PCASub(K=self.K, L=self.L, lambda_decay=self.lambda_decay)
            model.fit(X_train, Y_train)

            # Also compute eigenvalues for variance explained
            L_eff = min(self.L, X_train.shape[0])
            X_window = X_train[-L_eff:]
            weights = model._compute_decay_weights(L_eff)
            cov = model._weighted_covariance(X_window, weights)

            from numpy.linalg import eigh
            eigenvalues_all, _ = eigh(cov)
            eigenvalues_all = np.sort(eigenvalues_all)[::-1]

            # Top-K eigenvalues
            eigenvalues_k = eigenvalues_all[:self.K]
            total_var = eigenvalues_all.sum()
            var_explained = eigenvalues_k / total_var if total_var > 0 else np.zeros(self.K)

            # Resolve sign ambiguity: ensure largest absolute loading is positive
            eigvecs = model.eigvecs_.copy()
            for k in range(self.K):
                if eigvecs[np.argmax(np.abs(eigvecs[:, k])), k] < 0:
                    eigvecs[:, k] *= -1

            snapshot = PCLoadingSnapshot(
                fold_id=fold_id,
                train_start_date=str(dates_us[start].date()),
                train_end_date=str(dates_us[train_end - 1].date()),
                eigvecs=eigvecs,
                eigenvalues=eigenvalues_k,
                variance_explained=var_explained,
                beta=model.beta_.copy(),
                intercept=model.intercept_.copy(),
            )
            snapshots.append(snapshot)
            fold_id += 1
            start += self.test_window

        return snapshots

    def _resolve_sign_across_folds(
        self, snapshots: list[PCLoadingSnapshot]
    ) -> list[PCLoadingSnapshot]:
        """Align PC signs across folds for consistent comparison.

        Uses the first fold as reference and flips signs in subsequent folds
        to maximize cosine similarity with the reference.
        """
        if len(snapshots) < 2:
            return snapshots

        ref = snapshots[0].eigvecs
        for snap in snapshots[1:]:
            for k in range(self.K):
                cos_sim = np.dot(ref[:, k], snap.eigvecs[:, k])
                if cos_sim < 0:
                    snap.eigvecs[:, k] *= -1
                    snap.beta[k, :] *= -1

        return snapshots

    def analyze_loadings(
        self,
        snapshots: list[PCLoadingSnapshot],
        us_tickers: list[str],
        us_sector_names: dict[str, str],
    ) -> tuple[dict, dict, dict]:
        """Analyze PC loadings across folds.

        Returns
        -------
        mean_loadings : dict
            Mean loading of each US sector on each PC.
        loading_stability : dict
            Std of loadings across folds for each PC.
        pc_sector_associations : dict
            Top associated US sectors for each PC (by mean absolute loading).
        """
        n_us = len(us_tickers)
        all_loadings = np.array([s.eigvecs for s in snapshots])  # (n_folds, n_us, K)

        mean_loadings = {}
        loading_stability = {}
        pc_sector_associations = {}

        for k in range(self.K):
            pc_label = f"PC{k+1}"
            loadings_k = all_loadings[:, :, k]  # (n_folds, n_us)
            mean_k = loadings_k.mean(axis=0)
            std_k = loadings_k.std(axis=0)

            # Store per-sector loadings
            sector_loadings = {}
            sector_stds = {}
            for j, ticker in enumerate(us_tickers):
                name = us_sector_names.get(ticker, ticker)
                sector_loadings[ticker] = {
                    "name": name,
                    "mean_loading": round(float(mean_k[j]), 6),
                    "std_loading": round(float(std_k[j]), 6),
                }
                sector_stds[ticker] = round(float(std_k[j]), 6)

            mean_loadings[pc_label] = sector_loadings
            loading_stability[pc_label] = {
                "mean_std_across_sectors": round(float(std_k.mean()), 6),
                "max_std": round(float(std_k.max()), 6),
                "per_sector": sector_stds,
            }

            # Top associations by absolute mean loading
            sorted_idx = np.argsort(np.abs(mean_k))[::-1]
            top_sectors = []
            for idx in sorted_idx:
                ticker = us_tickers[idx]
                name = us_sector_names.get(ticker, ticker)
                top_sectors.append({
                    "ticker": ticker,
                    "name": name,
                    "mean_loading": round(float(mean_k[idx]), 6),
                    "abs_loading": round(float(np.abs(mean_k[idx])), 6),
                })
            pc_sector_associations[pc_label] = top_sectors

        return mean_loadings, loading_stability, pc_sector_associations

    def analyze_variance_explained(
        self, snapshots: list[PCLoadingSnapshot]
    ) -> tuple[list, list]:
        """Analyze variance explained by each PC across folds."""
        all_var = np.array([s.variance_explained for s in snapshots])  # (n_folds, K)
        mean_var = all_var.mean(axis=0)
        cumulative = np.cumsum(mean_var)

        mean_var_list = [
            {
                "pc": f"PC{k+1}",
                "mean_variance_explained": round(float(mean_var[k]), 6),
                "std_variance_explained": round(float(all_var[:, k].std()), 6),
                "min": round(float(all_var[:, k].min()), 6),
                "max": round(float(all_var[:, k].max()), 6),
            }
            for k in range(self.K)
        ]

        cumulative_list = [
            {"pc": f"PC1-{k+1}", "cumulative_variance": round(float(cumulative[k]), 6)}
            for k in range(self.K)
        ]

        return mean_var_list, cumulative_list

    def analyze_regression_coefficients(
        self,
        snapshots: list[PCLoadingSnapshot],
        jp_tickers: list[str],
        jp_sector_names: dict[str, str],
    ) -> tuple[dict, dict]:
        """Analyze how PCs map to Japanese sector predictions via beta coefficients.

        Returns
        -------
        mean_beta : dict
            Mean regression coefficient from each PC to each JP sector.
        beta_stability : dict
            Stability (std) of beta across folds.
        """
        all_beta = np.array([s.beta for s in snapshots])  # (n_folds, K, n_jp)

        mean_beta = {}
        beta_stability = {}

        for k in range(self.K):
            pc_label = f"PC{k+1}"
            beta_k = all_beta[:, k, :]  # (n_folds, n_jp)
            mean_k = beta_k.mean(axis=0)
            std_k = beta_k.std(axis=0)

            sector_betas = {}
            for j, ticker in enumerate(jp_tickers):
                name = jp_sector_names.get(ticker, ticker)
                sector_betas[ticker] = {
                    "name": name,
                    "mean_beta": round(float(mean_k[j]), 6),
                    "std_beta": round(float(std_k[j]), 6),
                    "t_stat": round(float(mean_k[j] / std_k[j]) if std_k[j] > 1e-12 else 0.0, 4),
                }

            mean_beta[pc_label] = sector_betas
            beta_stability[pc_label] = {
                "mean_std": round(float(std_k.mean()), 6),
                "max_std": round(float(std_k.max()), 6),
            }

        return mean_beta, beta_stability

    def analyze_transmission_channels(
        self,
        snapshots: list[PCLoadingSnapshot],
        us_tickers: list[str],
        us_sector_names: dict[str, str],
        jp_tickers: list[str],
        jp_sector_names: dict[str, str],
    ) -> list[dict]:
        """Identify cross-market transmission channels.

        For each PC, traces the path: US sector loadings -> PC -> JP sector betas,
        identifying which US sectors drive predictions for which JP sectors.
        """
        all_loadings = np.array([s.eigvecs for s in snapshots])
        all_beta = np.array([s.beta for s in snapshots])

        mean_loadings = all_loadings.mean(axis=0)  # (n_us, K)
        mean_beta = all_beta.mean(axis=0)  # (K, n_jp)

        # Total transmission: loadings @ beta gives (n_us, n_jp) mapping
        transmission = mean_loadings @ mean_beta  # (n_us, n_jp)

        channels = []
        for k in range(self.K):
            pc_label = f"PC{k+1}"

            # Top US sector contributors (by absolute loading)
            us_sorted = np.argsort(np.abs(mean_loadings[:, k]))[::-1]
            top_us = []
            for idx in us_sorted[:5]:
                ticker = us_tickers[idx]
                top_us.append({
                    "ticker": ticker,
                    "name": us_sector_names.get(ticker, ticker),
                    "loading": round(float(mean_loadings[idx, k]), 6),
                })

            # Top JP sector recipients (by absolute beta)
            jp_sorted = np.argsort(np.abs(mean_beta[k, :]))[::-1]
            top_jp = []
            for idx in jp_sorted[:5]:
                ticker = jp_tickers[idx]
                top_jp.append({
                    "ticker": ticker,
                    "name": jp_sector_names.get(ticker, ticker),
                    "beta": round(float(mean_beta[k, idx]), 6),
                })

            channels.append({
                "pc": pc_label,
                "interpretation": self._interpret_pc(mean_loadings[:, k], us_tickers, us_sector_names),
                "top_us_contributors": top_us,
                "top_jp_recipients": top_jp,
            })

        # Direct US->JP transmission matrix (top pairs)
        n_us, n_jp = transmission.shape
        pairs = []
        for i in range(n_us):
            for j in range(n_jp):
                pairs.append({
                    "us_ticker": us_tickers[i],
                    "us_name": us_sector_names.get(us_tickers[i], us_tickers[i]),
                    "jp_ticker": jp_tickers[j],
                    "jp_name": jp_sector_names.get(jp_tickers[j], jp_tickers[j]),
                    "transmission_strength": round(float(transmission[i, j]), 8),
                    "abs_strength": round(float(np.abs(transmission[i, j])), 8),
                })

        pairs.sort(key=lambda x: x["abs_strength"], reverse=True)
        channels.append({
            "type": "direct_transmission_matrix",
            "top_20_pairs": pairs[:20],
        })

        return channels

    def _interpret_pc(
        self,
        loadings: np.ndarray,
        us_tickers: list[str],
        us_sector_names: dict[str, str],
    ) -> str:
        """Generate an economic interpretation label for a PC based on loadings."""
        sorted_idx = np.argsort(np.abs(loadings))[::-1]

        # Check if it's a "market" factor (all loadings same sign)
        signs = np.sign(loadings)
        if np.all(signs > 0) or np.all(signs < 0):
            return "Market factor (all sectors move together)"

        # Check for sector-specific patterns
        top3_tickers = [us_tickers[i] for i in sorted_idx[:3]]
        top3_names = [us_sector_names.get(t, t) for t in top3_tickers]
        top3_signs = ['+' if loadings[sorted_idx[i]] > 0 else '-' for i in range(3)]

        # Risk-on/risk-off pattern
        cyclical = {"XLK", "XLY", "XLF", "XLI", "XLB"}
        defensive = {"XLU", "XLP", "XLV", "XLRE"}

        positive_tickers = set(us_tickers[i] for i in range(len(us_tickers)) if loadings[i] > 0)
        negative_tickers = set(us_tickers[i] for i in range(len(us_tickers)) if loadings[i] < 0)

        cyclical_positive = len(positive_tickers & cyclical)
        defensive_positive = len(positive_tickers & defensive)
        cyclical_negative = len(negative_tickers & cyclical)
        defensive_negative = len(negative_tickers & defensive)

        if cyclical_positive >= 3 and defensive_negative >= 2:
            return "Risk-on/risk-off factor (cyclical vs defensive)"
        if defensive_positive >= 2 and cyclical_negative >= 3:
            return "Defensive rotation factor (defensive vs cyclical)"

        # Growth vs value pattern
        growth = {"XLK", "XLC", "XLY"}
        value = {"XLE", "XLF", "XLU"}
        if len(positive_tickers & growth) >= 2 and len(negative_tickers & value) >= 2:
            return "Growth vs value factor"
        if len(positive_tickers & value) >= 2 and len(negative_tickers & growth) >= 2:
            return "Value vs growth factor"

        # Default: describe top contributors
        desc_parts = [f"{top3_signs[i]}{top3_names[i]}" for i in range(3)]
        return f"Sector rotation factor ({', '.join(desc_parts)})"

    def analyze_temporal_evolution(
        self,
        snapshots: list[PCLoadingSnapshot],
        us_tickers: list[str],
        us_sector_names: dict[str, str],
    ) -> tuple[dict, dict]:
        """Track how PC loadings evolve over time.

        Returns
        -------
        loading_evolution : dict
            Per-PC time series of loadings for each US sector.
        cosine_similarity : dict
            Cosine similarity between consecutive fold loadings for each PC.
        """
        n_folds = len(snapshots)
        loading_evolution = {}
        cosine_similarity = {}

        for k in range(self.K):
            pc_label = f"PC{k+1}"
            loadings_over_time = np.array([s.eigvecs[:, k] for s in snapshots])  # (n_folds, n_us)

            # Per-sector evolution
            sector_evolution = {}
            for j, ticker in enumerate(us_tickers):
                name = us_sector_names.get(ticker, ticker)
                sector_evolution[ticker] = {
                    "name": name,
                    "loadings": [round(float(loadings_over_time[f, j]), 6) for f in range(n_folds)],
                }
            loading_evolution[pc_label] = {
                "fold_dates": [s.train_end_date for s in snapshots],
                "sectors": sector_evolution,
            }

            # Cosine similarity between consecutive folds
            cos_sims = []
            for f in range(1, n_folds):
                v1 = loadings_over_time[f - 1]
                v2 = loadings_over_time[f]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                cos_sims.append(round(float(cos), 6))

            cosine_similarity[pc_label] = {
                "values": cos_sims,
                "mean": round(float(np.mean(cos_sims)), 6) if cos_sims else 0.0,
                "min": round(float(np.min(cos_sims)), 6) if cos_sims else 0.0,
                "std": round(float(np.std(cos_sims)), 6) if cos_sims else 0.0,
            }

        return loading_evolution, cosine_similarity

    def run_full_analysis(
        self,
        X_us: np.ndarray,
        Y_jp: np.ndarray,
        dates_us: pd.DatetimeIndex,
        us_tickers: list[str],
        us_sector_names: dict[str, str],
        jp_tickers: list[str],
        jp_sector_names: dict[str, str],
    ) -> PCInterpretationResult:
        """Run complete PC interpretability analysis."""
        result = PCInterpretationResult(K=self.K)

        # 1. Extract PC snapshots from walk-forward folds
        print("=" * 60)
        print("EXTRACTING PC SNAPSHOTS FROM WALK-FORWARD FOLDS")
        print("=" * 60)
        snapshots = self.extract_pc_snapshots(X_us, Y_jp, dates_us)
        snapshots = self._resolve_sign_across_folds(snapshots)
        result.n_folds = len(snapshots)
        print(f"  Extracted {len(snapshots)} fold snapshots")
        print()

        # 2. Loading analysis
        print("=" * 60)
        print("PC LOADING ANALYSIS")
        print("=" * 60)
        result.mean_loadings, result.loading_stability, result.pc_sector_associations = \
            self.analyze_loadings(snapshots, us_tickers, us_sector_names)

        for k in range(self.K):
            pc = f"PC{k+1}"
            assoc = result.pc_sector_associations[pc]
            print(f"\n  {pc} top loadings:")
            for s in assoc[:5]:
                print(f"    {s['name']:30s} loading={s['mean_loading']:+.4f}")
        print()

        # 3. Variance explained
        print("=" * 60)
        print("VARIANCE EXPLAINED")
        print("=" * 60)
        result.mean_variance_explained, result.cumulative_variance_explained = \
            self.analyze_variance_explained(snapshots)

        for v in result.mean_variance_explained:
            print(f"  {v['pc']}: {v['mean_variance_explained']*100:.1f}% "
                  f"(±{v['std_variance_explained']*100:.1f}%)")
        for c in result.cumulative_variance_explained:
            print(f"  {c['pc']}: cumulative {c['cumulative_variance']*100:.1f}%")
        print()

        # 4. Regression coefficients (PC -> JP sectors)
        print("=" * 60)
        print("REGRESSION COEFFICIENTS (PC -> JP SECTORS)")
        print("=" * 60)
        result.mean_beta, result.beta_stability = \
            self.analyze_regression_coefficients(snapshots, jp_tickers, jp_sector_names)

        for k in range(self.K):
            pc = f"PC{k+1}"
            betas = result.mean_beta[pc]
            # Sort by absolute mean beta
            sorted_sectors = sorted(
                betas.items(),
                key=lambda x: abs(x[1]["mean_beta"]),
                reverse=True,
            )
            print(f"\n  {pc} top JP sector sensitivities:")
            for ticker, info in sorted_sectors[:5]:
                t_stat = info["t_stat"]
                sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
                print(f"    {info['name']:30s} β={info['mean_beta']:+.6f}  t={t_stat:+.2f} {sig}")
        print()

        # 5. Transmission channels
        print("=" * 60)
        print("CROSS-MARKET TRANSMISSION CHANNELS")
        print("=" * 60)
        result.transmission_channels = self.analyze_transmission_channels(
            snapshots, us_tickers, us_sector_names, jp_tickers, jp_sector_names
        )

        for ch in result.transmission_channels:
            if "pc" in ch:
                print(f"\n  {ch['pc']}: {ch['interpretation']}")
                print(f"    US drivers: {', '.join(c['name'] for c in ch['top_us_contributors'][:3])}")
                print(f"    JP targets: {', '.join(c['name'] for c in ch['top_jp_recipients'][:3])}")
        print()

        # Show top transmission pairs
        direct = [ch for ch in result.transmission_channels if ch.get("type") == "direct_transmission_matrix"]
        if direct:
            print("  Top 10 US->JP transmission pairs:")
            for pair in direct[0]["top_20_pairs"][:10]:
                print(f"    {pair['us_name']:25s} -> {pair['jp_name']:30s}  "
                      f"strength={pair['transmission_strength']:+.8f}")
        print()

        # 6. Temporal evolution
        print("=" * 60)
        print("TEMPORAL STABILITY OF PC LOADINGS")
        print("=" * 60)
        result.loading_evolution, result.temporal_cosine_similarity = \
            self.analyze_temporal_evolution(snapshots, us_tickers, us_sector_names)

        for k in range(self.K):
            pc = f"PC{k+1}"
            cos = result.temporal_cosine_similarity[pc]
            print(f"  {pc}: mean cosine similarity = {cos['mean']:.4f} "
                  f"(min={cos['min']:.4f}, std={cos['std']:.4f})")
        print()

        return result
