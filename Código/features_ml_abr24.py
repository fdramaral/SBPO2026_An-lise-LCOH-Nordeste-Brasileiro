"""
=============================================================================
FEATURE ENGINEERING — DATASET PARA MODELO ML DO LCOH
Artigo: Análise Techno-Econômica por ML do LCOH a partir de Curtailment
        no Nordeste Brasileiro — SBPO 2026
=============================================================================

ENTRADAS (pasta dados_sbpo2026/processed/ ou raw/ccee_pld/):
  - coff_eolico_ne.csv      : curtailment eólico  (hora × estado)
  - coff_solar_ne.csv       : curtailment solar   (hora × estado)
  - pld_horario_ne.csv      : PLD bruto CCEE      (reconstrução de timestamp)
  - PLD_HORARIO_XXXX.csv    : arquivos brutos CCEE (fallback se processado ruim)
  - clima_openmeteo_nordeste_horario.csv : irradiação e vento (opcional)

SAÍDA:
  - dados_sbpo2026/processed/dataset_ml.parquet  (formato principal)
  - dados_sbpo2026/processed/dataset_ml.csv      (backup legível)

FEATURES GERADAS:
  Temporais   : hora, mes, dia_semana, ano, estacao, is_fim_semana
  Lag         : coff e PLD com lags de 1h, 2h, 6h, 24h, 168h
  Médias móv. : coff e PLD janelas de 24h, 7d, 30d
  Climáticas  : ghi_wm2, vento_ms (se disponível)
  Geográficas : sig_estado (one-hot ou label encoding)

TARGETS:
  lcoh_s1_brl_kg  — curtailment puro (energia a custo zero)
  lcoh_s2_brl_kg  — mercado spot (energia ao PLD)
  lcoh_s1_usd_kg  — idem em USD
  lcoh_s2_usd_kg  — idem em USD

INSTALAÇÃO:
  pip install pandas numpy pyarrow scikit-learn
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÕES
# ---------------------------------------------------------------------------

PROC = Path("dados_sbpo2026/processed")
RAW  = Path("dados_sbpo2026/raw")
PROC.mkdir(parents=True, exist_ok=True)

# Parâmetros do eletrolisador AWE (mesmos do pipeline de coleta)
ETA_MED       = 0.675
kWh_per_kg    = 52.5
CAPEX_MED_USD = 850
WACC          = 0.12
VIDA_UTIL     = 20
BRL_USD       = 5.53   # média BRL/USD Abr/2024–Mar/2026 (Banco Central do Brasil)

# Janelas de lag e média móvel (em períodos de 30 min)
# 1h=2, 2h=4, 6h=12, 24h=48, 168h=336 (7 dias), 720h=1440 (30 dias)
LAGS         = [2, 4, 12, 48, 336]       # em períodos de 30 min
JANELAS_ROLL = [48, 336, 1440]           # 24h, 7d, 30d

# ── FILTRO DE DATA: usar apenas a partir de Abr/2024 ────────────────────────
# Justificativa: o curtailment eólico antes de Abr/2024 era desprezível
# (média de 129 GWh/mês vs 1.520 GWh/mês após Abr/2024 — fator 12×).
# O período anterior não representa o fenômeno estrutural que o artigo modela.
# Os lags e médias móveis são calculados sobre toda a série histórica,
# mas apenas as observações a partir de Abr/2024 entram no dataset de ML.
DATA_CORTE_ML = pd.Timestamp("2024-04-01")

# Estações do Nordeste: seca (mai–nov) e chuvosa (dez–abr)
def estacao(mes: int) -> str:
    return "seca" if 5 <= mes <= 11 else "chuvosa"


# ---------------------------------------------------------------------------
# 1. CARREGAR CURTAILMENT (já processado pelo pipeline de coleta)
# ---------------------------------------------------------------------------

def carregar_coff() -> pd.DataFrame:
    """
    Carrega e combina curtailment eólico e solar.
    Retorna DataFrame com granularidade 30min × estado.
    """
    frames = []
    for nome in ["coff_eolico_ne.csv", "coff_solar_ne.csv"]:
        p = PROC / nome
        if not p.exists():
            print(f"  ⚠  {nome} não encontrado — rode coleta_eda_lcoh.py primeiro.")
            continue
        df = pd.read_csv(p, parse_dates=["dat_referencia"])
        frames.append(df)
        print(f"  ✅  {nome}: {len(df):,} registros")

    if not frames:
        raise FileNotFoundError("Nenhum arquivo de curtailment encontrado.")

    df_coff = pd.concat(frames, ignore_index=True)

    # Agregar eólico + solar por hora × estado (soma os dois)
    df = (df_coff
          .groupby(["dat_referencia", "sig_estado"], as_index=False)
          .agg(
              val_coff_mwmed   = ("val_coff_mwmed",   "sum"),
              val_gerref_mwmed = ("val_gerref_mwmed",  "sum"),
          ))

    df["fator_coff_pct"] = np.where(
        df["val_gerref_mwmed"] > 0,
        df["val_coff_mwmed"] / df["val_gerref_mwmed"] * 100,
        0.0
    )

    print(f"\n  Curtailment combinado: {len(df):,} registros "
          f"({df['sig_estado'].nunique()} estados)")
    print(f"  Período: {df['dat_referencia'].min()} → {df['dat_referencia'].max()}")
    return df


# ---------------------------------------------------------------------------
# 2. CARREGAR E RECONSTRUIR TIMESTAMP DO PLD
# ---------------------------------------------------------------------------

def carregar_pld() -> pd.DataFrame:
    """
    Carrega o PLD horário da CCEE e reconstrói o timestamp correto.

    Schema bruto:
      MES_REFERENCIA : "202101"  (YYYYMM)
      SUBMERCADO     : "NORDESTE"
      DIA            : "01" .. "31"
      HORA           : "00" .. "23"
      PLD_HORA       : float (R$/MWh)

    Estratégia de timestamp:
      ano  = MES_REFERENCIA[:4]
      mes  = MES_REFERENCIA[4:]
      data = f"{ano}-{mes}-{DIA} {HORA}:00:00"
    """
    # Tentar primeiro o arquivo processado (se timestamp correto)
    p_proc = PROC / "pld_horario_ne.csv"
    p_bruto_dir = RAW / "ccee_pld"

    # ── Carregar arquivos brutos (mais confiável para o timestamp) ───────────
    arquivos = sorted(p_bruto_dir.glob("PLD_HORARIO_*.csv"))
    if not arquivos:
        # Fallback: tentar arquivo processado
        if p_proc.exists():
            print("  ⚠  Usando pld_horario_ne.csv processado (timestamp pode estar incorreto)")
            df = pd.read_csv(p_proc)
            df = df[df["id_submercado"].astype(str).str.upper() == "NORDESTE"].copy()
            df["dat_referencia"] = pd.to_datetime(df["dat_referencia"], errors="coerce")
            df = df.rename(columns={"val_pld_brl_mwh": "val_pld_brl_mwh"})
            return df[["dat_referencia", "val_pld_brl_mwh"]].dropna()
        raise FileNotFoundError("Nenhum arquivo PLD encontrado.")

    print(f"\n  Carregando PLD de {len(arquivos)} arquivo(s) brutos...")
    frames = []
    for arq in arquivos:
        try:
            df = pd.read_csv(arq, sep=";", encoding="utf-8",
                             decimal=".", low_memory=False,
                             dtype=str)   # tudo como string para controlar parsing

            # Filtrar Nordeste
            df = df[df["SUBMERCADO"].str.upper().isin(
                ["NORDESTE", "NE", "NORTE/NORDESTE"]
            )].copy()
            if df.empty:
                continue

            # Reconstruir timestamp a partir de MES_REFERENCIA + DIA + HORA
            # MES_REFERENCIA: "202101" → ano=2021, mes=01
            mes_ref = df["MES_REFERENCIA"].str.strip().str.replace('"', '')
            ano_s   = mes_ref.str[:4]
            mes_s   = mes_ref.str[4:6]
            dia_s   = df["DIA"].str.strip().str.replace('"', '').str.zfill(2)
            hora_s  = df["HORA"].str.strip().str.replace('"', '').str.zfill(2)

            df["dat_referencia"] = pd.to_datetime(
                ano_s + "-" + mes_s + "-" + dia_s + " " + hora_s + ":00:00",
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce"
            )

            # Valor do PLD
            col_pld = next((c for c in df.columns
                            if "PLD" in c.upper() or "HORA" in c.upper()
                            and c != "HORA"), "PLD_HORA")
            df["val_pld_brl_mwh"] = pd.to_numeric(
                df[col_pld].str.replace(",", "."), errors="coerce"
            )

            frames.append(df[["dat_referencia", "val_pld_brl_mwh"]].dropna())

        except Exception as e:
            print(f"  ❌  {arq.name}: {e}")

    if not frames:
        raise ValueError("Nenhum dado de PLD processado com sucesso.")

    df_pld = pd.concat(frames, ignore_index=True).drop_duplicates("dat_referencia")
    df_pld = df_pld.sort_values("dat_referencia").reset_index(drop=True)

    print(f"  ✅  PLD: {len(df_pld):,} registros horários")
    print(f"      Período: {df_pld['dat_referencia'].min()} → "
          f"{df_pld['dat_referencia'].max()}")
    print(f"      PLD médio: R$ {df_pld['val_pld_brl_mwh'].mean():.2f}/MWh")
    return df_pld


# ---------------------------------------------------------------------------
# 3. CARREGAR DADOS CLIMÁTICOS (opcional)
# ---------------------------------------------------------------------------

def carregar_clima() -> pd.DataFrame:
    """Carrega dados climáticos Open-Meteo se disponíveis."""
    p = PROC / "clima_openmeteo_nordeste_horario.csv"
    if not p.exists():
        print("  ⚠  Dados climáticos não encontrados — features de clima omitidas.")
        return pd.DataFrame()

    df = pd.read_csv(p, parse_dates=["timestamp_br"])
    df = df.rename(columns={"timestamp_br": "dat_referencia"})

    # Média simples entre os 3 pontos (CE, RN, PE) por hora
    df_hora = (df.groupby("dat_referencia")
                 .agg(ghi_wm2  = ("ghi_wm2",  "mean"),
                      vento_ms = ("vento_ms", "mean"))
                 .reset_index())

    print(f"  ✅  Clima: {len(df_hora):,} registros horários")
    return df_hora


# ---------------------------------------------------------------------------
# 4. CONSTRUIR DATASET BASE (coff × PLD, por estado)
# ---------------------------------------------------------------------------

def construir_base(df_coff: pd.DataFrame,
                   df_pld:  pd.DataFrame,
                   df_clima: pd.DataFrame) -> pd.DataFrame:
    """
    Junta curtailment (30min × estado) com PLD (horário) e clima (horário).
    Resultado: uma linha por (timestamp × estado).
    """
    print("\n  Construindo dataset base...")

    # PLD: expandir para cada estado (o preço é igual para todo o NE)
    estados = df_coff["sig_estado"].unique()
    pld_exp = pd.concat(
        [df_pld.assign(sig_estado=est) for est in estados],
        ignore_index=True
    )

    # Merge coff × PLD (left join — mantém todos os períodos de curtailment)
    # Arredondar coff para hora cheia para facilitar merge com PLD horário
    df_coff["dat_ref_hora"] = df_coff["dat_referencia"].dt.floor("h")
    pld_exp["dat_ref_hora"] = pld_exp["dat_referencia"].dt.floor("h")

    df = df_coff.merge(
        pld_exp[["dat_ref_hora", "sig_estado", "val_pld_brl_mwh"]],
        on=["dat_ref_hora", "sig_estado"],
        how="left"
    )

    # Preencher PLD ausente com mediana (para períodos sem dado)
    pld_mediana = df_pld["val_pld_brl_mwh"].median()
    df["val_pld_brl_mwh"] = df["val_pld_brl_mwh"].fillna(pld_mediana)

    # Merge com clima (horário, mesmo para todos os estados)
    if not df_clima.empty:
        df_clima["dat_ref_hora"] = df_clima["dat_referencia"].dt.floor("h")
        df = df.merge(
            df_clima[["dat_ref_hora", "ghi_wm2", "vento_ms"]],
            on="dat_ref_hora",
            how="left"
        )
    else:
        df["ghi_wm2"]  = np.nan
        df["vento_ms"] = np.nan

    df = df.drop(columns=["dat_ref_hora"])
    df = df.sort_values(["sig_estado", "dat_referencia"]).reset_index(drop=True)

    print(f"  ✅  Base: {len(df):,} registros × {df.shape[1]} colunas")
    return df


# ---------------------------------------------------------------------------
# 5. FEATURES TEMPORAIS
# ---------------------------------------------------------------------------

def adicionar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona hora, mês, dia_semana, estação e flags binárias."""
    df = df.copy()
    df["hora"]         = df["dat_referencia"].dt.hour
    df["mes"]          = df["dat_referencia"].dt.month
    df["dia_semana"]   = df["dat_referencia"].dt.dayofweek   # 0=seg, 6=dom
    df["ano"]          = df["dat_referencia"].dt.year
    df["estacao"]      = df["mes"].apply(estacao)
    df["is_fim_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["periodo_dia"]  = pd.cut(
        df["hora"],
        bins=[-1, 5, 11, 17, 23],
        labels=["madrugada", "manha", "tarde", "noite"]
    )
    return df


# ---------------------------------------------------------------------------
# 6. FEATURES DE LAG E MÉDIAS MÓVEIS
# ---------------------------------------------------------------------------

def adicionar_lags_e_medias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona lags e médias móveis de curtailment e PLD por estado.
    Operações feitas dentro de cada grupo (sig_estado) para não vazar
    dados entre estados.
    """
    df = df.copy().sort_values(["sig_estado", "dat_referencia"])

    for estado, grp in df.groupby("sig_estado", sort=False):
        idx = grp.index

        # ── Lags de curtailment ──────────────────────────────────────────────
        for lag in LAGS:
            col = f"coff_lag_{lag}p"   # p = períodos de 30 min
            df.loc[idx, col] = grp["val_coff_mwmed"].shift(lag).values

        # ── Médias móveis de curtailment ─────────────────────────────────────
        for janela in JANELAS_ROLL:
            col = f"coff_roll_{janela}p"
            df.loc[idx, col] = (grp["val_coff_mwmed"]
                                .shift(1)
                                .rolling(janela, min_periods=1)
                                .mean()
                                .values)

        # ── Lags de PLD ─────────────────────────────────────────────────────
        for lag in [48, 336]:   # 24h e 7d
            col = f"pld_lag_{lag}p"
            df.loc[idx, col] = grp["val_pld_brl_mwh"].shift(lag).values

        # ── Médias móveis de PLD ─────────────────────────────────────────────
        for janela in [48, 336]:
            col = f"pld_roll_{janela}p"
            df.loc[idx, col] = (grp["val_pld_brl_mwh"]
                                .shift(1)
                                .rolling(janela, min_periods=1)
                                .mean()
                                .values)

        # ── Fator de curtailment médio (tendência) ───────────────────────────
        df.loc[idx, "fcoff_roll_48p"] = (grp["fator_coff_pct"]
                                         .shift(1)
                                         .rolling(48, min_periods=1)
                                         .mean()
                                         .values)

    # Renomear lags para nomes mais legíveis
    rename_lags = {
        "coff_lag_2p":    "coff_lag_1h",
        "coff_lag_4p":    "coff_lag_2h",
        "coff_lag_12p":   "coff_lag_6h",
        "coff_lag_48p":   "coff_lag_24h",
        "coff_lag_336p":  "coff_lag_7d",
        "coff_roll_48p":  "coff_media_24h",
        "coff_roll_336p": "coff_media_7d",
        "coff_roll_1440p":"coff_media_30d",
        "pld_lag_48p":    "pld_lag_24h",
        "pld_lag_336p":   "pld_lag_7d",
        "pld_roll_48p":   "pld_media_24h",
        "pld_roll_336p":  "pld_media_7d",
        "fcoff_roll_48p": "fcoff_media_24h",
    }
    df = df.rename(columns=rename_lags)

    n_lags = len([c for c in df.columns if "lag" in c or "media" in c])
    print(f"  ✅  Lags e médias: {n_lags} features adicionadas")
    return df


# ---------------------------------------------------------------------------
# 7. CALCULAR TARGETS (LCOH S1 e S2)
# ---------------------------------------------------------------------------

def calcular_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula LCOH para os dois cenários:
      S1 — energia a custo zero (curtailment puro)
      S2 — energia ao PLD de mercado
    """
    df = df.copy()

    # Fator de recuperação de capital
    n   = VIDA_UTIL
    fcr = WACC * (1 + WACC)**n / ((1 + WACC)**n - 1)
    capex_brl_kw       = CAPEX_MED_USD * BRL_USD
    capex_anual_brl_kw = capex_brl_kw * fcr
    opex_anual_brl_kw  = capex_brl_kw * 0.02
    # ── EFICIÊNCIA DE CONVERSÃO (kgH₂/kWh) ──────────────────────────────────
    # Sem fator 1000: unidade de potência é kW, não MW.
    kg_por_kwh = ETA_MED / kWh_per_kg           # ≈ 0.01286 kgH₂/kWh

    # ── FATOR DE CAPACIDADE ESTÁTICO POR ESTADO (FC_s) ──────────────────────
    #
    # FC(s) = fração total de intervalos do período completo com
    # constrained-off > 1 MWmed no estado s.
    #
    # Justificativa metodológica:
    #   O período 2021–2026 coincide com a aceleração estrutural do curtailment
    #   no Nordeste, gerando uma tendência monotônica que invalida o pressuposto
    #   de estacionariedade de qualquer janela rolling. Experimentos com FC
    #   rolling 12 meses resultaram em R² negativo no hold-out (modelo extrapola
    #   fora do regime de crescimento do treino). O FC estático por estado é
    #   metodologicamente mais robusto dado o período de análise disponível.
    #
    # Consequência: dentro de cada estado, LCOH S1 é constante. O modelo ML
    # aprende essencialmente a separar estados por nível de LCOH, usando
    # estado_cod e padrões de curtailment como preditores. O R² elevado do S1
    # reflete essa estrutura discreta, não capacidade preditiva generalizável.
    # Essa limitação é declarada explicitamente no artigo (Seção 5.1 e 5.4).
    #
    # MA tem FC muito baixo (~0.7%) → clampar em FC_MIN para evitar LCOH irreal.
    FC_MIN = 0.05   # 5% mínimo (≈ 438 h/ano com curtailment)
    FC_MAX = 0.60   # 60% máximo (proteção superior)

    fc_por_estado = (
        df.groupby("sig_estado")["val_coff_mwmed"]
        .apply(lambda x: float(np.clip((x > 1.0).mean(), FC_MIN, FC_MAX)))
        .rename("fc_estado")
    )

    print("  FC empírico por estado (clampado):")
    for estado, fc in fc_por_estado.sort_values(ascending=False).items():
        h2   = kg_por_kwh * 8760 * fc
        lcoh = (capex_anual_brl_kw + opex_anual_brl_kw) / h2
        print(f"    {estado}: FC={fc:.1%}  H₂={h2:.1f} kg/kW·ano  "
              f"LCOH_S1=R${lcoh:.2f}/kg (USD{lcoh/BRL_USD:.2f}/kg)")

    # Mapear FC para cada linha do dataset
    df = df.join(fc_por_estado, on="sig_estado")

    # Produção anual por kW instalado e custos por kg — específicos por estado
    df["h2_anual_por_kw"] = kg_por_kwh * 8760 * df["fc_estado"]
    df["capex_por_kg"]    = capex_anual_brl_kw / df["h2_anual_por_kw"]
    df["opex_por_kg"]     = opex_anual_brl_kw  / df["h2_anual_por_kw"]

    # ── CUSTO DE ENERGIA (R$/kgH₂) ───────────────────────────────────────────
    custo_energia_s2 = (
        df["val_pld_brl_mwh"] / 1000.0 * kWh_per_kg / ETA_MED
    )

    # ── LCOH (R$/kgH₂) ───────────────────────────────────────────────────────
    # S1: constante por estado (FC estático → CAPEX/kg fixo por estado)
    # S2: varia por hora (PLD) e por estado (FC) — genuinamente contínuo
    df["lcoh_s1_brl_kg"] = df["capex_por_kg"] + df["opex_por_kg"]
    df["lcoh_s2_brl_kg"] = df["capex_por_kg"] + df["opex_por_kg"] + custo_energia_s2

    # ── LCOH (USD/kgH₂) — conversão única ao final ───────────────────────────
    df["lcoh_s1_usd_kg"] = df["lcoh_s1_brl_kg"] / BRL_USD
    df["lcoh_s2_usd_kg"] = df["lcoh_s2_brl_kg"] / BRL_USD

    # Remover colunas auxiliares de cálculo (não são features de ML)
    df.drop(columns=["fc_estado", "h2_anual_por_kw",
                      "capex_por_kg", "opex_por_kg"],
            errors="ignore", inplace=True)

    # ── H₂ POTENCIAL (kg) — baseado no volume curtailado em MWmed ─────────────
    # val_coff_mwmed está em MWmed → × 1000 → kW·h → × η / kWh_per_kg → kgH₂
    df["h2_kg_estimado"] = (
        df["val_coff_mwmed"] * 1000.0   # MWmed × 1000 → kWmed = kW
        * kg_por_kwh                     # × kgH₂/kWh → kgH₂ por hora
    ).clip(lower=0)

    # Flag: hora com curtailment efetivo (coff > 1 MWmed)
    # Usado para filtrar o dataset antes do treinamento ML
    df["tem_curtailment"] = (df["val_coff_mwmed"] > 1.0).astype(int)

    print(f"\n  Targets calculados:")
    for col, label in [
        ("lcoh_s1_brl_kg", "LCOH S1 (R$/kg)"),
        ("lcoh_s2_brl_kg", "LCOH S2 (R$/kg)"),
        ("lcoh_s1_usd_kg", "LCOH S1 (USD/kg)"),
        ("lcoh_s2_usd_kg", "LCOH S2 (USD/kg)"),
    ]:
        s = df[col].dropna()
        print(f"    {label:25s} med={s.median():.2f}  "
              f"P10={s.quantile(.1):.2f}  P90={s.quantile(.9):.2f}")

    return df


# ---------------------------------------------------------------------------
# 8. ENCODING DE VARIÁVEIS CATEGÓRICAS
# ---------------------------------------------------------------------------

def encoding_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas para o modelo ML.
    - sig_estado  → label encoding (ordem alfabética)
    - estacao     → binária (0=chuvosa, 1=seca)
    - periodo_dia → label encoding
    """
    df = df.copy()

    # Estado: label encoding
    estados_ord = sorted(df["sig_estado"].unique())
    df["estado_cod"] = df["sig_estado"].map(
        {est: i for i, est in enumerate(estados_ord)}
    )

    # Estação: binária
    df["is_seca"] = (df["estacao"] == "seca").astype(int)

    # Período do dia: label encoding
    # Converter para string antes do map para evitar TypeError com Categorical
    periodos_ord = ["madrugada", "manha", "tarde", "noite"]
    df["periodo_dia_cod"] = (df["periodo_dia"]
                             .astype(str)
                             .map({p: i for i, p in enumerate(periodos_ord)})
                             .fillna(-1)
                             .astype(int))

    return df


# ---------------------------------------------------------------------------
# 9. LIMPEZA FINAL E EXPORTAÇÃO
# ---------------------------------------------------------------------------

def exportar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com NaN nos targets ou features críticas,
    e salva em Parquet e CSV.
    """
    # Colunas que devem existir sem NaN para o modelo funcionar
    cols_criticas = [
        "val_coff_mwmed", "val_pld_brl_mwh",
        "lcoh_s1_brl_kg", "lcoh_s2_brl_kg",
        "hora", "mes", "sig_estado",
    ]
    antes = len(df)
    df = df.dropna(subset=cols_criticas).reset_index(drop=True)
    depois = len(df)
    if antes != depois:
        print(f"  ⚠  {antes - depois:,} linhas removidas por NaN em colunas críticas.")

    # Ordenar colunas: identificadores → features → targets
    id_cols     = ["dat_referencia", "sig_estado"]
    target_cols = ["lcoh_s1_brl_kg", "lcoh_s2_brl_kg",
                   "lcoh_s1_usd_kg", "lcoh_s2_usd_kg",
                   "h2_kg_estimado"]
    drop_cols   = ["custo_energia_s1", "custo_energia_s2",
                   "estacao", "periodo_dia"]   # versões não-codificadas
    feature_cols = [c for c in df.columns
                    if c not in id_cols + target_cols + drop_cols]

    df = df[id_cols + feature_cols + target_cols]

    # Salvar
    parq = Path("dados_sbpo2026/processed/dataset_ml.parquet")
    csv  = Path("dados_sbpo2026/processed/dataset_ml.csv")

    try:
        df.to_parquet(parq, index=False)
        print(f"  💾  {parq}  ({len(df):,} linhas × {df.shape[1]} colunas)")
    except Exception as e:
        print(f"  ⚠  Parquet falhou ({e}) — salvando só CSV.")

    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"  💾  {csv}")

    return df


# ---------------------------------------------------------------------------
# 10. RELATÓRIO DE FEATURES
# ---------------------------------------------------------------------------

def relatorio_features(df: pd.DataFrame):
    """Imprime resumo do dataset final para documentação do artigo."""
    target_cols = ["lcoh_s1_brl_kg", "lcoh_s2_brl_kg",
                   "lcoh_s1_usd_kg", "lcoh_s2_usd_kg"]
    id_cols     = ["dat_referencia", "sig_estado"]
    feat_cols   = [c for c in df.columns
                   if c not in target_cols + id_cols]

    print("\n" + "=" * 60)
    print("  RELATÓRIO DO DATASET ML")
    print("=" * 60)
    print(f"  Observações : {len(df):,}")
    print(f"  Features    : {len(feat_cols)}")
    print(f"  Targets     : {len(target_cols)}")
    print(f"  Período     : {df['dat_referencia'].min().date()} → "
          f"{df['dat_referencia'].max().date()}")
    print(f"  Estados     : {sorted(df['sig_estado'].unique())}")
    print(f"\n  Features disponíveis:")
    for i, c in enumerate(feat_cols, 1):
        n_nan = df[c].isna().sum()
        pct   = n_nan / len(df) * 100
        print(f"    {i:2d}. {c:35s}  NaN={pct:.1f}%")

    print(f"\n  Targets:")
    for c in target_cols:
        s = df[c].dropna()
        print(f"    {c:30s}  med={s.median():.3f}  "
              f"std={s.std():.3f}  "
              f"[{s.min():.2f} – {s.max():.2f}]")


# ---------------------------------------------------------------------------
# 11. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  FEATURE ENGINEERING — DATASET ML")
    print("  Artigo SBPO 2026 — Nordeste Brasileiro")
    print("=" * 60)

    # Carregar dados
    print("\n📂  Carregando dados...")
    df_coff  = carregar_coff()
    df_pld   = carregar_pld()
    df_clima = carregar_clima()

    # Construir base
    df = construir_base(df_coff, df_pld, df_clima)

    # Features temporais
    print("\n⚙️   Adicionando features temporais...")
    df = adicionar_features_temporais(df)

    # Lags e médias móveis
    print("\n⚙️   Calculando lags e médias móveis (pode levar 1–2 min)...")
    df = adicionar_lags_e_medias(df)

    # ── FILTRO TEMPORAL ──────────────────────────────────────────────────────
    # Aplicado APÓS o cálculo de lags para preservar a memória histórica:
    # as médias móveis de 30 dias de Abr/2024 precisam dos dados de Mar/2024.
    # O filtro remove do dataset de ML o período de baixo curtailment (Out/21–Mar/24),
    # que representava apenas 9,6% do volume total eólico e distorcia o aprendizado.
    n_antes = len(df)
    df = df[df["dat_referencia"] >= DATA_CORTE_ML].copy()
    n_depois = len(df)
    print(f"\n🔍  Filtro temporal aplicado: {n_antes:,} → {n_depois:,} registros")
    print(f"     Período do dataset ML: {df['dat_referencia'].min().date()} "
          f"→ {df['dat_referencia'].max().date()}")

    # Targets
    print("\n⚙️   Calculando targets LCOH...")
    df = calcular_targets(df)

    # Encoding
    print("\n⚙️   Codificando variáveis categóricas...")
    df = encoding_categoricas(df)

    # Exportar
    print("\n💾  Exportando dataset...")
    df = exportar_dataset(df)

    # Relatório
    relatorio_features(df)

    print("\n✅  Feature engineering concluído.")
    print(f"   Dataset salvo em: dados_sbpo2026/processed/dataset_ml.parquet")

    return df


if __name__ == "__main__":
    df_ml = main()
