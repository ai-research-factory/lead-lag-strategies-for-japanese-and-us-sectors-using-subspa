# Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA

## Project ID
proj_2ff0333b

## Taxonomy
PCA, StatArb

## Current Cycle
16

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
この論文は、米国株式市場のセクター別リターンが日本株式市場のセクター別リターンに与える「波及効果」を捉え、統計的アービトラージ戦略を構築することを目的としています。具体的には、11の米国セクターETF（Select Sector SPDR）の前日リターンを説明変数とし、17の日本セクターETF（TOPIX-17）の翌営業日open-to-closeリターンを目的変数とします。この予測関係をモデル化するために、PCA（主成分分析）を応用した独自の「PCA_SUB（subspace regularization PCA）」アルゴリズムを提案し、米国セクター間の共分散構造から、日本市場への影響を抽出します。最終的な目標は、この予測モデルを用いてロングショート戦略を構築し、取引コストを考慮したウォークフォワード分析によってその有効性を検証することです。

### Datasets
{"name":"U.S. Select Sector SPDR ETFs (11 sectors)","source":"yfinance API","tickers":["XLE","XLF","XLU","XLI","XLK","XLV","XLY","XLP","XLB","XLRE","XLC"]}
- {"name":"TOPIX-17 Sector ETFs (17 sectors)","source":"yfinance API","tickers":["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T","1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T","1631.T","1632.T","1633.T"]}

### Targets
翌営業日の日本17業種ETFのopen-to-closeリターン

### Model
モデルの核心は、PCAを応用したPCA_SUBアルゴリズムです。このアルゴリズムは、まず米国セクターリターンの共分散行列を計算し、その主要な変動要因（主成分）を抽出します。次に、これらの主成分を説明変数、日本セクターリターンを目的変数とする回帰モデルを構築します。これにより、米国市場全体の体系的な動きが日本市場のどのセクターに、どのように影響を与えるかをモデル化します。論文では、主成分数K=3、ルックバック期間L=60、減衰率λ=0.9を主要パラメータとして使用します。

### Training
学習はウォークフォワード方式で行われます。過去の一定期間（トレーニングウィンドウ）のデータを用いてモデルを学習し、続く期間（テストウィンドウ）で予測を行います。その後、ウィンドウを1ステップずつ未来にスライドさせてこのプロセスを繰り返します。論文では、共分散行列の推定に2010年から2014年という固定期間（Cfull）を使用しており、この点が検証の重要なポイントとなります。ハイパーパラメータの最適化は、トレーニングデータ内でのネストされたウォークフォワード検証によって行われるべきです。

### Evaluation
評価は、取引コスト（手数料、スリッページ等）を考慮したバックテストによって行われます。主要な評価指標は、シャープ・レシオ、累積リターン、最大ドローダウンです。予測精度自体も、リターンの符号一致率（accuracy）などで評価します。提案モデルの有効性を示すため、単純な回帰モデルや市場のバイ・アンド・ホールド戦略などのベースラインと比較します。


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 1)


### Phase 1: コアアルゴリズム(PCA_SUB)の実装

**ゴール**: PCA_SUBモデルの基本構造を実装し、合成データで動作確認を行う。

**具体的な作業指示**:
1. `src/models/pca_sub.py` ファイルを作成する。
2. `PCASub` クラスを定義し、`__init__(self, K, L, lambda_decay)` コンストラクタを持つ。
3. `fit(self, X_us, Y_jp)` メソッドを実装する。このメソッドは、入力された米国リターン(X_us)と日本リターン(Y_jp)から、減衰率を考慮した共分散行列を計算し、主成分分析を行い、回帰係数を学習する。
4. `predict(self, X_us_new)` メソッドを実装する。このメソッドは、新しい米国リターンから日本リターンの予測値を返す。
5. `scripts/run_synthetic_test.py` を作成し、2つの相関する多変量正規分布から合成データ（X_us, Y_jp）を生成し、`PCASub`モデルの学習と予測がエラーなく実行され、期待される形状の出力が得られることを確認する。

**期待される出力ファイル**:
- src/models/pca_sub.py
- scripts/run_synthetic_test.py

**受入基準 (これを全て満たすまで完了としない)**:
- PCASubクラスが合成データでエラーなくfitとpredictを実行できる。
- predictメソッドの出力が正しい形状（サンプル数 x 日本セクター数）を持つ。








## 全体Phase計画 (参考)

→ Phase 1: コアアルゴリズム(PCA_SUB)の実装 — PCA_SUBモデルの基本構造を実装し、合成データで動作確認を行う。
  Phase 2: 実データパイプラインの構築 — yfinanceから日米のETFデータを取得し、モデルが使用できる形式に前処理する。
  Phase 3: ウォークフォワード評価フレームワークの実装 — ウォークフォワード法に基づきモデルの学習と予測を行い、基本的な予測精度を評価する。
  Phase 4: バックテストエンジンとコストモデルの実装 — モデルの予測を基にロングショート戦略を構築し、取引コストを考慮したパフォーマンスを計算する。
  Phase 5: 共分散推定期間(Cfull)の検証 — 論文のCfull=2010-2014という固定期間での共分散推定と、通常のローリング推定を比較し、影響を評価する。
  Phase 6: ハイパーパラメータ最適化 — ネストされたウォークフォワード検証を用いて、主要パラメータ(L, λ, K)の最適値を探索する。
  Phase 7: ロバスト性検証と感度分析 — モデルの安定性を評価するため、異なる市場環境やパラメータ設定でのパフォーマンスを検証する。
  Phase 8: 主成分の解釈可能性分析 — モデルが抽出した主成分（ファクター）が、経済的に何を意味しているのかを分析する。
  Phase 9: ベースラインモデルとの比較 — PCA_SUBモデルの優位性を示すため、より単純なモデルと比較評価する。
  Phase 10: 最終レポートと可視化 — すべての分析結果を統合し、再現可能なサマリーレポートを生成する。
  Phase 11: エグゼクティブサマリーとコード品質向上 — 非技術者向けの要約を作成し、コードの品質とテストカバレッジを向上させる。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_1/metrics.json` — 全メトリクスを構造化フォーマットで
- `reports/cycle_1/technical_findings.md` — 実装内容、結果、観察事項
- `docs/open_questions.md` — 未解決の疑問と仮定

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
