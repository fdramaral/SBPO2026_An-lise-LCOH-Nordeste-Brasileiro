"""
=============================================================================
MODELO ML — LCOH COM XGBOOST, LIGHTGBM, SHAP E MONTE CARLO
Artigo: Análise Techno-Econômica por ML do LCOH a partir de Curtailment
        no Nordeste Brasileiro — SBPO 2026
=============================================================================

ENTRADA:
  dados_sbpo2026/processed/dataset_ml.parquet

SAÍDAS:
  dados_sbpo2026/resultados/
    metricas_modelos.csv          — RMSE, MAE, R² por modelo e cenário
    predicoes_teste.csv           — predições do melhor modelo no hold-out
    predicoes_holdout_todos.csv   — predições de todos os modelos no hold-out
    shap_importancia.csv          — importância média de cada feature (SHAP)
    monte_carlo_lcoh.csv          — distribuições Monte Carlo do LCOH
    fig_predicoes_vs_real.png
    fig_shap_beeswarm.png
    fig_shap_barplot.png
    fig_monte_carlo.png
    holdout_rmse_mensal.png       — RMSE mensal no hold-out (Figura 3 do artigo)
    walk_forward_consolidado.png  — walk-forward CV por fold (diagnóstico complementar)

MODELOS:
  - XGBoost  (xgb.XGBRegressor)
  - LightGBM (lgb.LGBMRegressor)
  Targets: lcoh_s1_brl_kg (curtailment puro) e lcoh_s2_brl_kg (mercado spot)

VALIDAÇÃO:
  Hold-out temporal (últimos 15% dos dados, Dez/2025–Mar/2026) = métrica principal.
  Walk-forward cross-validation com 4 folds = avaliação complementar de estabilidade.

INSTALAÇÃO:
  pip install xgboost lightgbm shap scikit-learn matplotlib seaborn pandas pyarrow
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import timedelta

# ML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge           # baseline linear
from sklearn.dummy import DummyRegressor         # baseline trivial (média)
import xgboost  as xgb
import lightgbm as lgb
import shap

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÕES
# ---------------------------------------------------------------------------

PROC    = Path("dados_sbpo2026/processed")
RES     = Path("dados_sbpo2026/resultados")
RES.mkdir(parents=True, exist_ok=True)

# Features que entram no modelo (excluir IDs e targets)
FEATURES = [
    # Curtailment
    "val_coff_mwmed", "fator_coff_pct",
    "coff_lag_1h", "coff_lag_2h", "coff_lag_6h", "coff_lag_24h", "coff_lag_7d",
    "coff_media_24h", "coff_media_7d", "coff_media_30d",
    # PLD
    "val_pld_brl_mwh",
    "pld_lag_24h", "pld_lag_7d",
    "pld_media_24h", "pld_media_7d",
    # Tendência de curtailment
    "fcoff_media_24h",
    # Temporais
    "hora", "mes", "dia_semana", "ano",
    "is_fim_semana", "is_seca",
    "periodo_dia_cod",
    # Geográfico
    "estado_cod",
    # Climáticas (quando disponíveis)
    "ghi_wm2", "vento_ms",
]

TARGETS = {
    "s1": "lcoh_s1_brl_kg",   # Curtailment puro
    "s2": "lcoh_s2_brl_kg",   # Mercado spot
}

# Hiperparâmetros — calibrados para evitar overfitting em séries temporais
XGB_PARAMS = {
    "n_estimators":     800,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

LGB_PARAMS = {
    "n_estimators":     800,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples":20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":         -1,
}

N_FOLDS_WF  = 4      # folds no walk-forward CV (fold 1 seria vazio com dataset de 24 meses)
N_MC        = 1000   # simulações Monte Carlo


# ---------------------------------------------------------------------------
# 1. CARREGAR DATASET
# ---------------------------------------------------------------------------

def carregar_dataset() -> pd.DataFrame:
    p = PROC / "dataset_ml.parquet"
    if not p.exists():
        p = PROC / "dataset_ml.csv"
        if not p.exists():
            raise FileNotFoundError(
                "dataset_ml não encontrado — rode features_ml.py primeiro."
            )
        df = pd.read_csv(p, parse_dates=["dat_referencia"])
    else:
        df = pd.read_parquet(p)

    # Garantir que dat_referencia é datetime
    df["dat_referencia"] = pd.to_datetime(df["dat_referencia"])

    # Remover features ausentes do dataset (ex: clima não coletado)
    feats_disponiveis = [f for f in FEATURES if f in df.columns]
    feats_ausentes    = [f for f in FEATURES if f not in df.columns]
    if feats_ausentes:
        print(f"  ⚠  Features ausentes (ignoradas): {feats_ausentes}")

    print(f"  ✅  Dataset: {len(df):,} obs × {len(feats_disponiveis)} features")
    print(f"      Período: {df['dat_referencia'].min().date()} → "
          f"{df['dat_referencia'].max().date()}")
    return df, feats_disponiveis


# ---------------------------------------------------------------------------
# 2. WALK-FORWARD CROSS-VALIDATION
# ---------------------------------------------------------------------------

def walk_forward_cv(df: pd.DataFrame,
                    features: list,
                    target: str,
                    modelo_cls,
                    params: dict,
                    n_folds: int = N_FOLDS_WF) -> dict:
    """
    Walk-forward CV temporal:
    - Treina sempre no passado, testa sempre no futuro
    - Sem data leakage: nenhuma informação do teste vaza para o treino
    - Cada fold adiciona ~20% dos dados como novo período de treino

    Retorna dicionário com métricas por fold e predições.
    """
    df = df.dropna(subset=features + [target]).sort_values("dat_referencia")

    data_min = df["dat_referencia"].min()
    data_max = df["dat_referencia"].max()
    duracao  = (data_max - data_min) / n_folds

    metricas_folds = []
    predicoes_todas = []

    print(f"\n  Walk-forward CV — {n_folds} folds | target={target}")
    print("  " + "─" * 54)

    for fold in range(1, n_folds + 1):
        # Corte temporal: treino até (fold-1)/n_folds, teste no fold seguinte
        corte_treino = data_min + duracao * (fold - 1)
        corte_teste  = data_min + duracao * fold

        df_treino = df[df["dat_referencia"] < corte_treino]
        df_teste  = df[(df["dat_referencia"] >= corte_treino) &
                       (df["dat_referencia"] <  corte_teste)]

        if df_treino.empty or df_teste.empty:
            continue

        X_treino = df_treino[features].fillna(0)
        y_treino = df_treino[target]
        X_teste  = df_teste[features].fillna(0)
        y_teste  = df_teste[target]

        modelo = modelo_cls(**params)
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)

        rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
        mae  = mean_absolute_error(y_teste, y_pred)
        r2   = r2_score(y_teste, y_pred)

        print(f"  Fold {fold}: treino até {corte_treino.date()} | "
              f"RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        metricas_folds.append({
            "fold": fold, "rmse": rmse, "mae": mae, "r2": r2,
            "n_treino": len(df_treino), "n_teste": len(df_teste),
            "data_corte": corte_treino.date(),
        })

        # Guardar predições para análise
        df_pred = df_teste[["dat_referencia", "sig_estado"]].copy()
        df_pred["y_real"] = y_teste.values
        df_pred["y_pred"] = y_pred
        df_pred["fold"]   = fold
        predicoes_todas.append(df_pred)

    df_metricas   = pd.DataFrame(metricas_folds)
    df_predicoes  = pd.concat(predicoes_todas, ignore_index=True)

    print(f"\n  Médias CV — RMSE={df_metricas['rmse'].mean():.4f} ± "
          f"{df_metricas['rmse'].std():.4f} | "
          f"R²={df_metricas['r2'].mean():.4f}")

    return {"metricas": df_metricas, "predicoes": df_predicoes,
            "modelo_final": modelo}


# ---------------------------------------------------------------------------
# 3. TREINAR MODELO FINAL (TREINO COMPLETO)
# ---------------------------------------------------------------------------

def treinar_final(df: pd.DataFrame,
                  features: list,
                  target: str,
                  modelo_cls,
                  params: dict,
                  proporcao_treino: float = 0.85):
    """
    Treina o modelo final em 85% dos dados (ordenados temporalmente)
    e avalia no restante (hold-out temporal).
    """
    df = df.dropna(subset=features + [target]).sort_values("dat_referencia")
    corte = int(len(df) * proporcao_treino)

    df_treino = df.iloc[:corte]
    df_teste  = df.iloc[corte:]

    X_treino = df_treino[features].fillna(0)
    y_treino = df_treino[target]
    X_teste  = df_teste[features].fillna(0)
    y_teste  = df_teste[target]

    modelo = modelo_cls(**params)
    modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)
    rmse   = np.sqrt(mean_squared_error(y_teste, y_pred))
    mae    = mean_absolute_error(y_teste, y_pred)
    r2     = r2_score(y_teste, y_pred)

    print(f"  Hold-out final — RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    return modelo, df_treino, df_teste, y_pred


# ---------------------------------------------------------------------------
# 4. ANÁLISE SHAP
# ---------------------------------------------------------------------------

def calcular_shap(modelo, X_amostra: pd.DataFrame,
                  features: list, nome_modelo: str,
                  target: str) -> pd.DataFrame:
    """
    Calcula SHAP values e gera dois gráficos:
      - Beeswarm plot (distribuição de impacto por feature)
      - Bar plot (importância média absoluta)
    """
    print(f"\n  Calculando SHAP para {nome_modelo}...")
    X = X_amostra[features].fillna(0)

    # TreeExplainer apenas para modelos baseados em árvore
    # Baseline linear (Ridge) usa LinearExplainer
    try:
        explainer   = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X)
    except Exception:
        try:
            explainer   = shap.LinearExplainer(modelo, X)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"  ⚠  SHAP não disponível para {nome_modelo}: {e}")
            return pd.DataFrame({
                "feature": features,
                "shap_mean": [0.0] * len(features),
            })

    # DataFrame de importância média
    df_imp = pd.DataFrame({
        "feature":   features,
        "shap_mean": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_mean", ascending=False).reset_index(drop=True)

    # ── Fig: Beeswarm ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=features,
                      max_display=20, show=False)
    plt.title(f"SHAP Beeswarm — {nome_modelo} | Target: {target}")
    plt.tight_layout()
    plt.savefig(RES / f"shap_beeswarm_{nome_modelo}_{target}.png", dpi=150)
    plt.close()

    # ── Fig: Bar plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    top20 = df_imp.head(20)
    ax.barh(top20["feature"][::-1], top20["shap_mean"][::-1],
            color="#2196F3", edgecolor="white")
    ax.set_xlabel("Importância SHAP média (|SHAP value|)")
    ax.set_title(f"Top 20 Features — {nome_modelo} | Target: {target}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / f"shap_barplot_{nome_modelo}_{target}.png", dpi=150)
    plt.close()

    print(f"  ✅  SHAP calculado. Top 5 features:")
    print(df_imp.head(5).to_string(index=False))

    return df_imp


# ---------------------------------------------------------------------------
# 5. MONTE CARLO — INCERTEZA NOS PARÂMETROS DO ELETROLISADOR
# ---------------------------------------------------------------------------

def monte_carlo_lcoh(df_pred: pd.DataFrame,
                     n_sim: int = N_MC) -> pd.DataFrame:
    """
    Propaga incerteza dos parâmetros técnico-econômicos do eletrolisador
    via Monte Carlo (1.000 simulações).

    Distribuições dos parâmetros incertos:
      CAPEX  ~ Normal(850, 75) USD/kW    → ±8,8% (1σ)
      η      ~ Normal(0.675, 0.015)      → ±2,2%
      kWh/kg ~ Normal(52.5, 1.5)         → ±2,9%
      WACC   ~ Normal(0.12, 0.02)        → ±16,7%
      BRL/USD~ Normal(5.53, 0.30)        → ±5,4%  # média Abr/2024–Mar/2026 (BCB)
    """
    np.random.seed(42)
    n   = 20  # vida útil (fixo)

    resultados = []

    for i in range(n_sim):
        # Amostrar parâmetros
        capex_usd = np.random.normal(850,   75)
        eta       = np.clip(np.random.normal(0.675, 0.015), 0.60, 0.75)
        kwh_kg    = np.clip(np.random.normal(52.5,  1.5),   48.0, 57.0)
        wacc      = np.clip(np.random.normal(0.12,  0.02),  0.06, 0.20)
        brl_usd   = np.clip(np.random.normal(5.53,  0.30),  4.50, 7.50)
        pld_fator = np.random.normal(1.0, 0.10)   # variação no PLD (+/-10%)

        # Recalcular LCOH
        fcr              = wacc * (1 + wacc)**n / ((1 + wacc)**n - 1)
        capex_brl_kw     = capex_usd * brl_usd
        capex_anual      = capex_brl_kw * fcr
        opex_anual       = capex_brl_kw * 0.02
        # kgH₂ por kWh consumido (sem fator 1000 — unidade kW, não MW)
        kg_por_kwh_mc = eta / kwh_kg

        # FC amostrado uniformemente entre os 7 valores empíricos estaduais.
        # Essa amostragem reflete a HETEROGENEIDADE ESTRUTURAL entre estados
        # do Nordeste — não incerteza paramétrica do eletrolisador.
        # A ampla dispersão do IC 90% resultante deve ser interpretada como
        # "variação de viabilidade entre estados", não como incerteza de
        # um projeto em um estado específico.
        # Fonte dos FCs empíricos: calculados sobre o período Abr/2024–Mar/2026 (regime de curtailment estrutural)
        # a partir dos dados de constrained-off do ONS (Seção 3.2 do artigo).
        fc_estados = np.array([0.595, 0.544, 0.488, 0.313, 0.304, 0.250, 0.108])  # FCs empíricos Abr/2024–Mar/2026 (RN, BA, CE, PE, PI, PB, MA)
        fc_mc = np.random.choice(fc_estados)

        h2_anual_kw = kg_por_kwh_mc * 8760 * fc_mc
        capex_kg    = capex_anual / max(h2_anual_kw, 1e-9)
        opex_kg     = opex_anual  / max(h2_anual_kw, 1e-9)

        # Usar predições do melhor modelo como base de PLD
        pld_ajust        = df_pred["val_pld"].mean() * pld_fator
        custo_energia_s2 = pld_ajust / 1000.0 * kwh_kg / eta   # R$/kgH₂

        # Conversão única ao final: todos os custos calculados em R$/kg,
        # depois divididos pelo câmbio amostrado para obter USD/kg.
        lcoh_s1_brl = capex_kg + opex_kg
        lcoh_s2_brl = capex_kg + opex_kg + custo_energia_s2
        lcoh_s1 = lcoh_s1_brl / brl_usd    # USD/kg
        lcoh_s2 = lcoh_s2_brl / brl_usd    # USD/kg

        resultados.append({
            "sim":      i,
            "capex_usd": capex_usd,
            "eta":       eta,
            "wacc":      wacc,
            "brl_usd":   brl_usd,
            "lcoh_s1_usd_kg": lcoh_s1,
            "lcoh_s2_usd_kg": lcoh_s2,
        })

    df_mc = pd.DataFrame(resultados)

    print(f"\n  Monte Carlo ({n_sim} simulações):")
    for col, label in [
        ("lcoh_s1_usd_kg", "LCOH S1 (USD/kg)"),
        ("lcoh_s2_usd_kg", "LCOH S2 (USD/kg)"),
    ]:
        s = df_mc[col]
        print(f"    {label}: med={s.median():.2f}  "
              f"P5={s.quantile(.05):.2f}  P95={s.quantile(.95):.2f}  "
              f"IC90%=[{s.quantile(.05):.2f}, {s.quantile(.95):.2f}]")

    # ── Fig: distribuição Monte Carlo ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, cor in [
        ("lcoh_s1_usd_kg", "S1 — Curtailment puro", "#2ca02c"),
        ("lcoh_s2_usd_kg", "S2 — Mercado spot",     "#d62728"),
    ]:
        ax.hist(df_mc[col], bins=60, alpha=0.55, label=label, color=cor)
        med = df_mc[col].median()
        ax.axvline(med, color=cor, linestyle="--", linewidth=1.2,
                   label=f"Mediana {label} (USD {med:.2f}/kg)")

    ax.axvline(2.0, color="black", linestyle=":", linewidth=1.5,
               label="Meta IRENA 2030 (USD 2/kg)")
    ax.set_title(f"Distribuição LCOH — Monte Carlo ({n_sim:,} simulações)")
    ax.set_xlabel("LCOH (USD/kgH₂)")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "monte_carlo_lcoh.png", dpi=150)
    plt.close()
    print("  ✅  Figura: monte_carlo_lcoh.png")

    return df_mc


# ---------------------------------------------------------------------------
# 6. FIGURAS DE RESULTADOS
# ---------------------------------------------------------------------------

def plotar_predicoes(df_pred: pd.DataFrame,
                     nome_modelo: str, target: str):
    """Real vs predito — série temporal (amostra de 30 dias)."""
    df_pred = df_pred.sort_values("dat_referencia")

    # Amostra: últimos 30 dias do conjunto de teste
    fim   = df_pred["dat_referencia"].max()
    ini   = fim - pd.Timedelta(days=30)
    amostra = df_pred[df_pred["dat_referencia"] >= ini]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(amostra["dat_referencia"], amostra["y_real"],
            label="Real", color="#1f77b4", linewidth=1.0, alpha=0.8)
    ax.plot(amostra["dat_referencia"], amostra["y_pred"],
            label="Predito", color="#ff7f0e", linewidth=1.0,
            linestyle="--", alpha=0.8)
    ax.set_title(f"Real vs Predito — {nome_modelo} | {target} (últimos 30 dias)")
    ax.set_ylabel("R$/kgH₂")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / f"pred_vs_real_{nome_modelo}_{target}.png", dpi=150)
    plt.close()
    print(f"  ✅  Figura: pred_vs_real_{nome_modelo}_{target}.png")


# Acumulador global dos resultados walk-forward (preenchido no pipeline)
_wf_resultados = []   # lista de dicts: {modelo, target, metricas_df}


def plotar_walk_forward(resultados_cv: dict,
                        nome_modelo: str, target: str):
    """Armazena resultados para geração posterior da figura consolidada."""
    _wf_resultados.append({
        "modelo": nome_modelo,
        "target": target,
        "metricas": resultados_cv["metricas"].copy(),
    })
    print(f"  ✅  Walk-forward registrado: {nome_modelo} | {target}")


def gerar_figura_walk_forward():
    """
    Gera a Figura 4 consolidada: RMSE e R² por fold para todos os modelos
    e targets, com barras agrupadas e cores distintas por modelo/cenário.
    Uma única figura substituindo os 6 PNGs individuais anteriores.
    """
    import numpy as np

    if not _wf_resultados:
        print("  ⚠  Nenhum resultado walk-forward registrado.")
        return

    CORES = {
        ("Baseline_Ridge",  "lcoh_s1_brl_kg"): "#aec7e8",
        ("Baseline_Ridge",  "lcoh_s2_brl_kg"): "#6baed6",
        ("XGBoost",         "lcoh_s1_brl_kg"): "#fdae6b",
        ("XGBoost",         "lcoh_s2_brl_kg"): "#e6550d",
        ("LightGBM",        "lcoh_s1_brl_kg"): "#a1d99b",
        ("LightGBM",        "lcoh_s2_brl_kg"): "#31a354",
    }
    LABELS = {
        ("Baseline_Ridge",  "lcoh_s1_brl_kg"): "Ridge — S1",
        ("Baseline_Ridge",  "lcoh_s2_brl_kg"): "Ridge — S2",
        ("XGBoost",         "lcoh_s1_brl_kg"): "XGBoost — S1",
        ("XGBoost",         "lcoh_s2_brl_kg"): "XGBoost — S2",
        ("LightGBM",        "lcoh_s1_brl_kg"): "LightGBM — S1",
        ("LightGBM",        "lcoh_s2_brl_kg"): "LightGBM — S2",
    }

    n_series = len(_wf_resultados)
    n_folds  = len(_wf_resultados[0]["metricas"])
    folds    = list(range(1, n_folds + 1))

    largura  = 0.13
    offsets  = np.linspace(-(n_series - 1) / 2 * largura,
                            (n_series - 1) / 2 * largura, n_series)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for idx, entry in enumerate(_wf_resultados):
        mod   = entry["modelo"]
        tgt   = entry["target"]
        df_m  = entry["metricas"]
        cor   = CORES.get((mod, tgt), "#999999")
        label = LABELS.get((mod, tgt), f"{mod} — {tgt}")
        x     = np.array(folds) + offsets[idx]

        ax1.bar(x, df_m["rmse"], width=largura, color=cor,
                edgecolor="white", linewidth=0.5, label=label)
        ax2.bar(x, df_m["r2"],   width=largura, color=cor,
                edgecolor="white", linewidth=0.5, label=label)

    ax1.set_title("RMSE por Fold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fold", fontsize=10)
    ax1.set_ylabel("RMSE (R$/kgH₂)", fontsize=10)
    ax1.set_xticks(folds)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_title("R² por Fold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fold", fontsize=10)
    ax2.set_ylabel("R²", fontsize=10)
    ax2.set_xticks(folds)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.grid(axis="y", alpha=0.3)

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=6,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(
        "Walk-forward CV — Ridge, XGBoost e LightGBM | LCOH S1 e S2",
        fontsize=13, fontweight="bold", y=1.07
    )
    plt.tight_layout()
    plt.savefig(RES / "walk_forward_consolidado.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅  Figura: walk_forward_consolidado.png")


# ---------------------------------------------------------------------------
# 7. FIGURA HOLD-OUT TEMPORAL (métrica principal — Figura 3 do artigo)
# ---------------------------------------------------------------------------

def gerar_figura_holdout(df_preds: pd.DataFrame):
    """
    Figura 3 — Desempenho no hold-out temporal (métrica principal do artigo).

    Layout: dois painéis lado a lado (S1 esq. / S2 dir.).
    Eixo X: meses do hold-out (Dez/2025–Mar/2026).
      Nov/2025 é excluído pois o corte de 85% cai em 30/Nov, resultando
      em apenas 1 dia de dados — insuficiente para RMSE representativo.
    Eixo Y: RMSE (R$/kgH₂) calculado sobre todas as observações do mês
      e de todos os estados simultaneamente.
    Barras agrupadas por modelo (Ridge / XGBoost / LightGBM).
    """
    # --- preparação ---
    df = df_preds.copy()
    df["dat_referencia"] = pd.to_datetime(df["dat_referencia"])

    # excluir Nov/2025 (único dia — 30/Nov)
    df = df[df["dat_referencia"] >= "2025-12-01"]
    df["mes"] = df["dat_referencia"].dt.to_period("M")

    TARGETS_PLOT = [
        ("lcoh_s1_brl_kg", "LCOH S1 — Curtailment puro"),
        ("lcoh_s2_brl_kg", "LCOH S2 — Mercado spot"),
    ]
    MODELOS  = ["Baseline_Ridge", "XGBoost", "LightGBM"]
    CORES    = {
        "Baseline_Ridge": "#aec7e8",
        "XGBoost":        "#f4a460",
        "LightGBM":       "#90c890",
    }
    MESES_LABEL = {
        "2025-12": "Dez/25",
        "2026-01": "Jan/26",
        "2026-02": "Fev/26",
        "2026-03": "Mar/26",
    }

    targets_presentes = [t for t, _ in TARGETS_PLOT
                         if t in df["target"].unique()]
    n_targets = len(targets_presentes)

    fig, axes = plt.subplots(1, n_targets, figsize=(12, 4.5), sharey=False)
    if n_targets == 1:
        axes = [axes]

    width   = 0.22
    n_mod   = len(MODELOS)
    offsets = np.linspace(-(n_mod - 1) * width / 2,
                           (n_mod - 1) * width / 2, n_mod)

    for ax, target in zip(axes, targets_presentes):
        titulo = next(lbl for t, lbl in TARGETS_PLOT if t == target)
        df_t   = df[df["target"] == target]
        meses  = sorted(df_t["mes"].unique())
        x      = np.arange(len(meses))

        for idx, modelo in enumerate(MODELOS):
            df_m  = df_t[df_t["modelo"] == modelo]
            rmses = []
            for mes in meses:
                g = df_m[df_m["mes"] == mes]
                if len(g) < 10:
                    rmses.append(np.nan)
                else:
                    rmses.append(
                        np.sqrt(mean_squared_error(g["y_real"], g["y_pred"]))
                    )
            bars = ax.bar(
                x + offsets[idx], rmses,
                width=width,
                label=modelo,
                color=CORES[modelo],
                edgecolor="white",
                linewidth=0.5,
            )
            # anotar valor sobre cada barra (exceto Ridge, para não poluir)
            if modelo != "Baseline_Ridge":
                for bar, val in zip(bars, rmses):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7.5,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MESES_LABEL.get(str(m), str(m)) for m in meses],
            fontsize=9,
        )
        ax.set_xlabel("Mês", fontsize=10)
        ax.set_ylabel("RMSE (R$/kgH₂)", fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Hold-out temporal — RMSE mensal por modelo | LCOH S1 e S2  "
        "(Dez/2025–Mar/2026)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(RES / "holdout_rmse_mensal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅  Figura: holdout_rmse_mensal.png")


# ---------------------------------------------------------------------------
# 8. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  MODELO ML — LCOH  |  XGBoost + LightGBM + SHAP + MC")
    print("  Artigo SBPO 2026 — Nordeste Brasileiro")
    print("=" * 60)

    # ── Carregar dataset ──────────────────────────────────────────────────────
    print("\n📂  Carregando dataset...")
    df, features = carregar_dataset()

    # Filtrar apenas horas com curtailment efetivo (coff > 1 MWmed)
    # Sem esse filtro o target S1 seria quase constante (dominado por horas
    # sem corte, onde LCOH = apenas CAPEX/OPEX fixo) — o modelo não aprenderia nada
    if "tem_curtailment" in df.columns:
        n_antes = len(df)
        df = df[df["tem_curtailment"] == 1].copy()
        print(f"  Filtro curtailment efetivo: {n_antes:,} → {len(df):,} obs "
              f"({len(df)/n_antes*100:.1f}% do total)")
    else:
        print("  ⚠  Coluna 'tem_curtailment' ausente — usando dataset completo.")

    # ── Loop: modelos × targets ───────────────────────────────────────────────
    # Baseline: Ridge com parâmetros mínimos e sem hiperparâmetros
    # Serve como referência para verificar se o ganho dos modelos ensemble
    # é estatisticamente relevante (deve superar largamente o baseline).
    RIDGE_PARAMS = {"alpha": 1.0, "random_state": 42}

    modelos_config = [
        ("Baseline_Ridge", Ridge,            RIDGE_PARAMS),
        ("XGBoost",        xgb.XGBRegressor, XGB_PARAMS),
        ("LightGBM",       lgb.LGBMRegressor, LGB_PARAMS),
    ]

    todas_metricas = []
    todas_preds_ho = []          # acumula predições hold-out de todos os modelos/targets
    melhor_modelo  = None
    melhor_rmse    = np.inf
    melhor_preds   = None

    for target_key, target_col in TARGETS.items():
        if target_col not in df.columns:
            print(f"  ⚠  Target {target_col} não encontrado — pulando.")
            continue

        print(f"\n{'='*60}")
        print(f"  TARGET: {target_col}")
        print(f"{'='*60}")

        for nome_mod, cls_mod, params_mod in modelos_config:
            print(f"\n🤖  {nome_mod} | {target_col}")

            # Walk-forward CV
            res_cv = walk_forward_cv(
                df, features, target_col, cls_mod, params_mod
            )
            plotar_walk_forward(res_cv, nome_mod, target_col)

            # Modelo final (hold-out temporal)
            modelo_final, df_tr, df_te, y_pred_te = treinar_final(
                df, features, target_col, cls_mod, params_mod
            )

            # Métricas hold-out
            y_real_te = df_te[target_col].values
            rmse_ho   = np.sqrt(mean_squared_error(y_real_te, y_pred_te))
            mae_ho    = mean_absolute_error(y_real_te, y_pred_te)
            r2_ho     = r2_score(y_real_te, y_pred_te)

            print(f"  Hold-out — RMSE={rmse_ho:.4f}  "
                  f"MAE={mae_ho:.4f}  R²={r2_ho:.4f}")

            todas_metricas.append({
                "modelo":   nome_mod,
                "target":   target_col,
                "rmse_cv":  res_cv["metricas"]["rmse"].mean(),
                "mae_cv":   res_cv["metricas"]["mae"].mean(),
                "r2_cv":    res_cv["metricas"]["r2"].mean(),
                "rmse_ho":  rmse_ho,
                "mae_ho":   mae_ho,
                "r2_ho":    r2_ho,
            })

            # Predições no conjunto de teste
            df_pred_te = df_te[["dat_referencia", "sig_estado",
                                  target_col, "val_pld_brl_mwh"]].copy()
            df_pred_te = df_pred_te.rename(columns={target_col: "y_real",
                                                     "val_pld_brl_mwh": "val_pld"})
            df_pred_te["y_pred"]  = y_pred_te
            df_pred_te["modelo"]  = nome_mod
            df_pred_te["target"]  = target_col

            todas_preds_ho.append(df_pred_te)   # acumula para figura hold-out consolidada

            plotar_predicoes(df_pred_te, nome_mod, target_col)

            # SHAP (amostra de até 5.000 obs para performance)
            n_shap  = min(5000, len(df_te))
            amostra = df_te.sample(n_shap, random_state=42)
            df_shap = calcular_shap(
                modelo_final, amostra, features, nome_mod, target_col
            )
            df_shap["modelo"] = nome_mod
            df_shap["target"] = target_col
            df_shap.to_csv(
                RES / f"shap_importancia_{nome_mod}_{target_col}.csv",
                index=False
            )

            # Guardar melhor modelo (menor RMSE no hold-out, target S2)
            if target_col == TARGETS["s2"] and rmse_ho < melhor_rmse:
                melhor_rmse   = rmse_ho
                melhor_modelo = nome_mod
                melhor_preds  = df_pred_te

    # ── Figura 3 consolidada (hold-out temporal — métrica principal) ─────────
    df_preds_ho = pd.concat(todas_preds_ho, ignore_index=True)
    df_preds_ho.to_csv(RES / "predicoes_holdout_todos.csv", index=False)
    print(f"  ✅  Salvo: predicoes_holdout_todos.csv ({len(df_preds_ho):,} linhas)")
    gerar_figura_holdout(df_preds_ho)

    # ── Figura walk-forward (diagnóstico complementar de estabilidade) ────────
    gerar_figura_walk_forward()

    # ── Tabela de métricas ────────────────────────────────────────────────────
    df_metricas = pd.DataFrame(todas_metricas)
    df_metricas.to_csv(RES / "metricas_modelos.csv", index=False)
    print(f"\n\n{'='*60}")
    print("  RESUMO DE MÉTRICAS (HOLD-OUT TEMPORAL)")
    print(f"{'='*60}")
    print(df_metricas[["modelo","target","rmse_ho","mae_ho","r2_ho"]]
          .to_string(index=False))

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    if melhor_preds is not None:
        print(f"\n\n🎲  Monte Carlo com base nas predições do {melhor_modelo}...")
        df_mc = monte_carlo_lcoh(melhor_preds, n_sim=N_MC)
        df_mc.to_csv(RES / "monte_carlo_lcoh.csv", index=False)

    # ── Salvar predições do teste ─────────────────────────────────────────────
    if melhor_preds is not None:
        melhor_preds.to_csv(RES / "predicoes_teste.csv", index=False)

    print(f"\n✅  Pipeline ML concluído.")
    print(f"   Resultados em: {RES.resolve()}")

    return df_metricas


if __name__ == "__main__":
    df_resultados = main()
