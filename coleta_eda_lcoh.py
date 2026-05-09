"""
=============================================================================
COLETA E ANÁLISE EXPLORATÓRIA DE DADOS  — v4  (21/abr/2026)
Artigo: Análise Techno-Econômica por ML do LCOH a partir de Curtailment
        no Nordeste Brasileiro — SBPO 2026
=============================================================================

FONTES:
  - ONS Dados Abertos : constrained-off eólico e fotovoltaico (mês/ano)
  - CCEE Dados Abertos: PLD horário por submercado (download manual — ver §2)

SCHEMA REAL DOS ARQUIVOS ONS (confirmado em abril/2026):
  id_subsistema, nom_subsistema, id_estado, nom_estado, nom_usina,
  id_ons, ceg, din_instante, val_geracao, val_geracaolimitada,
  val_disponibilidade, val_geracaoreferencia, val_geracaoreferenciafinal,
  cod_razaorestricao, cod_origemrestricao, dsc_restricao

  → constrained-off (MWmed) = val_geracaoreferencia - val_geracaolimitada
  → granularidade: HORÁRIA por usina

DISPONIBILIDADE DOS DADOS:
  - Eólico       : out/2021 em diante  ✅
  - Fotovoltaico : jan/2024 em diante  ✅  (404 para datas anteriores)

CCEE PLD — DOWNLOAD MANUAL NECESSÁRIO:
  O portal CCEE exige sessão de browser; as URLs com token expiram.
  Instruções no §2 abaixo.

INSTALAÇÃO:
  pip install pandas requests tqdm matplotlib seaborn openpyxl
=============================================================================
"""

# §2 ── COMO BAIXAR O PLD MANUALMENTE ────────────────────────────────────────
#
#  1. Acesse: https://dadosabertos.ccee.org.br/dataset/pld_horario
#  2. Para cada ano (2021–2026), clique em "download" ao lado do CSV.
#  3. Salve os arquivos com estes nomes exatos na pasta indicada:
#
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2021.csv
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2022.csv
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2023.csv
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2024.csv
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2025.csv
#     dados_sbpo2026/raw/ccee_pld/PLD_HORARIO_2026.csv
#
#  4. Execute o script normalmente — ele detecta os arquivos automaticamente.
# ────────────────────────────────────────────────────────────────────────────

!pip install pandas requests tqdm matplotlib seaborn openpyxl

import os
import shutil
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------------------------

ROOT = Path("dados_sbpo2026")
RAW  = ROOT / "raw"
PROC = ROOT / "processed"
FIG  = ROOT / "figuras"

for d in [RAW / "ons_eolica", RAW / "ons_fotovoltaica",
          RAW / "ccee_pld", PROC, FIG]:
    d.mkdir(parents=True, exist_ok=True)

# Migração de pastas legadas (versões anteriores do script)
for origem, destino in {
    RAW / "ons_eolico": RAW / "ons_eolica",
    RAW / "ons_solar":  RAW / "ons_fotovoltaica",
}.items():
    if origem.exists():
        for arq in origem.iterdir():
            dst = destino / arq.name
            if not dst.exists():
                shutil.move(str(arq), str(dst))
        try:
            origem.rmdir()
        except OSError:
            pass

# Estados do Nordeste
ESTADOS_NE = ["CE", "RN", "PE", "BA", "PI", "MA", "AL", "SE", "PB"]

ANO_INICIO = 2021
MES_INICIO = 10
ANO_FIM    = 2026
MES_FIM    = 3

SUBMERCADO_NE = "NE"

# Período do dataset de ML e das figuras descritivas
DATA_CORTE_FIGURAS = pd.Timestamp("2024-04-01")  # Abr/2024–Mar/2026: regime estrutural de curtailment
DATA_FIM_FIGURAS   = pd.Timestamp("2026-03-31")

# Parâmetros do eletrolisador AWE
ETA_MED       = 0.675
kWh_per_kg    = 52.5
CAPEX_MED_USD = 850
WACC          = 0.12
VIDA_UTIL     = 20
BRL_USD       = 5.53   # média BRL/USD Abr/2024–Mar/2026 (Banco Central do Brasil)


# ---------------------------------------------------------------------------
# 1. COLETA ONS
# ---------------------------------------------------------------------------

def _ons_url(tipo: str, ano: int, mes: int) -> str:
    prefixo = {
        "eolica":       "restricao_coff_eolica_tm/RESTRICAO_COFF_EOLICA",
        "fotovoltaica": "restricao_coff_fotovoltaica_tm/RESTRICAO_COFF_FOTOVOLTAICA",
    }
    base = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"
    return f"{base}/{prefixo[tipo]}_{ano}_{mes:02d}.csv"


def _periodos(ano_ini, mes_ini, ano_fim, mes_fim):
    periodos = []
    a, m = ano_ini, mes_ini
    while (a, m) <= (ano_fim, mes_fim):
        periodos.append((a, m))
        m += 1
        if m > 12:
            m, a = 1, a + 1
    return periodos


# Colunas mínimas necessárias — evita carregar todas as ~15 colunas na RAM
_COLUNAS_ONS = [
    "nom_estado", "din_instante", "nom_usina",
    "val_geracaoreferencia", "val_gerlimitada_mwmed",
    "val_geracaolimitada",   "val_geracaoreferenciafinal",
    "val_disponibilidade",   "cod_razaorestricao",
]

# Nomes dos estados NE em maiúsculas ASCII (como o ONS entrega)
_ESTADOS_NE_ONS = {
    "CEARA", "RIO GRANDE DO NORTE", "PERNAMBUCO", "BAHIA",
    "PIAUI", "MARANHAO", "ALAGOAS", "SERGIPE", "PARAIBA",
}

import unicodedata as _ud
def _normalizar(s: str) -> str:
    return _ud.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").upper()

_MAPA_SIGLA = {
    "CEARA": "CE", "RIO GRANDE DO NORTE": "RN", "PERNAMBUCO": "PE",
    "BAHIA": "BA", "PIAUI": "PI", "MARANHAO": "MA",
    "ALAGOAS": "AL", "SERGIPE": "SE", "PARAIBA": "PB",
}


def _ler_filtrar_agregar(caminho: Path, tipo: str) -> pd.DataFrame:
    """
    Lê um CSV mensal do ONS em chunks de 50k linhas,
    filtra Nordeste, calcula coff e agrega por hora×estado.
    Mantém pico de RAM em ~50 MB por arquivo independente do tamanho.
    """
    import unicodedata as ud

    CHUNK = 50_000
    # Só as colunas que realmente precisamos
    USECOLS = ["nom_estado", "din_instante",
               "val_geracaoreferencia", "val_geracaolimitada"]

    partes = []
    try:
        reader = pd.read_csv(
            caminho, sep=";", encoding="utf-8", decimal=",",
            low_memory=False, chunksize=CHUNK, usecols=USECOLS,
        )
    except Exception as e:
        tqdm.write(f"  ❌  Leitura {caminho.name}: {e}")
        return pd.DataFrame()

    for chunk in reader:
        # Normalizar e filtrar Nordeste
        chunk["_est"] = chunk["nom_estado"].apply(_normalizar)
        chunk = chunk[chunk["_est"].isin(_ESTADOS_NE_ONS)].copy()
        if chunk.empty:
            continue

        chunk["sig_estado"] = chunk["_est"].map(_MAPA_SIGLA)
        chunk["dat_referencia"] = pd.to_datetime(
            chunk["din_instante"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["dat_referencia"])

        # Coff por linha (MWmed)
        ref = pd.to_numeric(chunk["val_geracaoreferencia"], errors="coerce")
        lim = pd.to_numeric(chunk["val_geracaolimitada"],   errors="coerce")
        lim = lim.fillna(ref)   # NaN = sem restrição → coff = 0
        chunk["val_coff_mwmed"]    = (ref - lim).clip(lower=0)
        chunk["val_gerref_mwmed"]  = ref

        # Agregar por hora × estado (reduz ~120× o tamanho antes de guardar)
        chunk["hora_ref"] = chunk["dat_referencia"].dt.floor("30min")
        agg = (chunk.groupby(["hora_ref", "sig_estado"])
                    .agg(
                        val_coff_mwmed   = ("val_coff_mwmed",   "sum"),
                        val_gerref_mwmed = ("val_gerref_mwmed",  "sum"),
                    )
                    .reset_index())
        partes.append(agg)

    if not partes:
        return pd.DataFrame()

    df = pd.concat(partes, ignore_index=True)
    # Re-agregar caso o mesmo período apareça em dois chunks
    df = (df.groupby(["hora_ref", "sig_estado"])
            .agg(val_coff_mwmed   = ("val_coff_mwmed",   "sum"),
                 val_gerref_mwmed = ("val_gerref_mwmed",  "sum"))
            .reset_index())
    df["fonte"] = tipo
    return df.rename(columns={"hora_ref": "dat_referencia"})


def baixar_ons(tipo: str, forcar_download=False) -> pd.DataFrame:
    """
    Baixa CSVs mensais do ONS e retorna DataFrame JÁ FILTRADO E AGREGADO
    (hora × estado NE), consumindo <500 MB de RAM independente do volume.
    """
    # Fotovoltaico só disponível a partir de jan/2024
    ano_ini = ANO_INICIO if tipo == "eolica" else 2024
    mes_ini = MES_INICIO if tipo == "eolica" else 1

    pasta    = RAW / f"ons_{tipo}"
    periodos = _periodos(ano_ini, mes_ini, ANO_FIM, MES_FIM)
    frames   = []

    print(f"\n📥  ONS constrained-off {tipo.upper()} ({len(periodos)} meses)...")

    for ano, mes in tqdm(periodos):
        nome    = f"COFF_{tipo.upper()}_{ano}_{mes:02d}.csv"
        caminho = pasta / nome

        if not caminho.exists() or forcar_download:
            try:
                r = requests.get(_ons_url(tipo, ano, mes), timeout=30)
                r.raise_for_status()
                caminho.write_bytes(r.content)
            except requests.HTTPError:
                tqdm.write(f"  ⚠  {ano}-{mes:02d} não disponível (404)")
                continue
            except Exception as e:
                tqdm.write(f"  ❌  {ano}-{mes:02d}: {e}")
                continue

        df_mes = _ler_filtrar_agregar(caminho, tipo)
        if not df_mes.empty:
            frames.append(df_mes)

    if not frames:
        print(f"  ⚠  Nenhum dado carregado para {tipo}.")
        return pd.DataFrame()

    df_total = pd.concat(frames, ignore_index=True)
    # Agregação final (une meses consecutivos)
    df_total = (df_total
                .groupby(["dat_referencia", "sig_estado", "fonte"])
                .agg(val_coff_mwmed   = ("val_coff_mwmed",   "sum"),
                     val_gerref_mwmed = ("val_gerref_mwmed",  "sum"))
                .reset_index())
    print(f"  ✅  {tipo}: {len(df_total):,} registros (agregados hora×estado).")
    return df_total


# ---------------------------------------------------------------------------
# 2. CARREGA PLD (DOWNLOAD MANUAL)
# ---------------------------------------------------------------------------

def carregar_ccee_pld() -> pd.DataFrame:
    pasta    = RAW / "ccee_pld"
    arquivos = sorted(pasta.glob("PLD_HORARIO_*.csv"))

    if not arquivos:
        print("\n  ⚠  Nenhum arquivo PLD encontrado em raw/ccee_pld/")
        print("     Baixe manualmente conforme as instruções no topo do script.")
        return pd.DataFrame()

    print(f"\n📂  Carregando PLD CCEE ({len(arquivos)} arquivo(s))...")
    frames = []
    for arq in tqdm(arquivos):
        try:
            df = pd.read_csv(arq, sep=";", encoding="utf-8",
                             decimal=",", low_memory=False)
            frames.append(df)
        except Exception as e:
            tqdm.write(f"  ❌  {arq.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df_pld = pd.concat(frames, ignore_index=True)
    print(f"  ✅  PLD: {len(df_pld):,} registros. Colunas: {list(df_pld.columns)}")
    return df_pld


# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO ONS
# ---------------------------------------------------------------------------

def preprocessar_ons(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """
    Filtro e cálculo de coff agora feitos em _ler_filtrar_agregar() durante
    a leitura em chunks — esta função é mantida apenas por compatibilidade
    e retorna o DataFrame sem alterações.
    """
    return df


# ---------------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO CCEE PLD
# ---------------------------------------------------------------------------

def preprocessar_pld(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # Mapeamento explícito (schema real: instante | submercado | valor)
    rename = {}
    if "instante"   in df.columns: rename["instante"]   = "dat_referencia"
    if "submercado" in df.columns: rename["submercado"] = "id_submercado"
    if "valor"      in df.columns: rename["valor"]      = "val_pld_brl_mwh"

    # Fallback por palavras-chave
    if not rename:
        col_dt  = next((c for c in df.columns if any(
            p in c.lower() for p in ["instant","dat","ref","hora","time"])), None)
        col_sub = next((c for c in df.columns if any(
            p in c.lower() for p in ["subm","merc","region"])), None)
        col_val = next((c for c in df.columns if any(
            p in c.lower() for p in ["valor","pld","prec","price","value"])), None)
        if col_dt:  rename[col_dt]  = "dat_referencia"
        if col_sub: rename[col_sub] = "id_submercado"
        if col_val: rename[col_val] = "val_pld_brl_mwh"

    if len(rename) < 3:
        print(f"  ⚠  Colunas PLD não mapeadas. Disponíveis: {list(df.columns)}")
        return pd.DataFrame()

    df = df.rename(columns=rename)
    df["dat_referencia"]  = pd.to_datetime(df["dat_referencia"], errors="coerce")
    df["val_pld_brl_mwh"] = pd.to_numeric(df["val_pld_brl_mwh"], errors="coerce")

    mask = df["id_submercado"].astype(str).str.upper().isin(
        ["NE", "NORDESTE", "NORTE/NORDESTE"]
    )
    df_ne = df[mask].copy()

    if df_ne.empty:
        print(f"  ⚠  Submercado NE não encontrado.")
        print(f"     Valores encontrados: {df['id_submercado'].unique()[:10]}")
        return pd.DataFrame()

    return df_ne.dropna(subset=["dat_referencia", "val_pld_brl_mwh"])


# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING — LCOH
# ---------------------------------------------------------------------------

def calcular_lcoh(df_coff: pd.DataFrame,
                  df_pld:  pd.DataFrame) -> pd.DataFrame:
    if df_coff.empty:
        print("  ⚠  df_coff vazio — LCOH não calculado.")
        return pd.DataFrame()

    n   = VIDA_UTIL
    fcr = WACC * (1 + WACC)**n / ((1 + WACC)**n - 1)
    capex_brl_kw       = CAPEX_MED_USD * BRL_USD
    capex_anual_brl_kw = capex_brl_kw * fcr
    opex_anual_brl_kw  = capex_brl_kw * 0.02
    h2_por_kw_hora     = 1000 * ETA_MED / kWh_per_kg
    h2_anual_por_kw    = h2_por_kw_hora * 8760 * 0.45
    capex_por_kg = capex_anual_brl_kw / max(h2_anual_por_kw, 1e-9)
    opex_por_kg  = opex_anual_brl_kw  / max(h2_anual_por_kw, 1e-9)

    df = df_coff.copy()
    df["hora_ref"] = df["dat_referencia"].dt.floor("h")

    grp = ["hora_ref", "sig_estado"] if "sig_estado" in df.columns else ["hora_ref"]
    df_hora = (df.groupby(grp)
                 .agg(
                     val_coff_mwmed   = ("val_coff_mwmed",   "sum"),
                     val_gerref_mwmed = ("val_gerref_mwmed",  "sum"),
                     fonte            = ("fonte",             "first"),
                 )
                 .reset_index())

    if not df_pld.empty:
        df_pld_dia = (df_pld
                      .assign(data=df_pld["dat_referencia"].dt.date)
                      .groupby("data")["val_pld_brl_mwh"]
                      .mean().reset_index())
        df_hora["data"] = df_hora["hora_ref"].dt.date
        df_hora = df_hora.merge(df_pld_dia, on="data", how="left")
        pld_med = df_pld["val_pld_brl_mwh"].median()
        df_hora["val_pld_brl_mwh"] = df_hora["val_pld_brl_mwh"].fillna(pld_med)
    else:
        df_hora["val_pld_brl_mwh"] = np.nan

    df_hora["h2_kg_estimado"]  = (df_hora["val_coff_mwmed"] * 1000 / kWh_per_kg * ETA_MED).clip(lower=0)
    df_hora["custo_energia_s1"] = 0.0
    df_hora["custo_energia_s2"] = df_hora["val_pld_brl_mwh"] / 1000 * kWh_per_kg / ETA_MED

    df_hora["lcoh_s1_brl_kg"] = capex_por_kg + opex_por_kg + df_hora["custo_energia_s1"]
    df_hora["lcoh_s2_brl_kg"] = capex_por_kg + opex_por_kg + df_hora["custo_energia_s2"]
    df_hora["lcoh_s1_usd_kg"] = df_hora["lcoh_s1_brl_kg"] / BRL_USD
    df_hora["lcoh_s2_usd_kg"] = df_hora["lcoh_s2_brl_kg"] / BRL_USD

    df_hora["fator_coff_pct"] = np.where(
        df_hora["val_gerref_mwmed"] > 0,
        df_hora["val_coff_mwmed"] / df_hora["val_gerref_mwmed"] * 100,
        np.nan
    )
    return df_hora.rename(columns={"hora_ref": "dat_referencia"})


# ---------------------------------------------------------------------------
# 6. EDA
# ---------------------------------------------------------------------------

def eda_curtailment(df: pd.DataFrame, tipo: str):
    if df.empty:
        print(f"  ⚠  {tipo}: DataFrame vazio.")
        return

    df = df.copy()
    df["ano_mes"] = df["dat_referencia"].dt.to_period("M")
    df["ano"]     = df["dat_referencia"].dt.year
    df["mes"]     = df["dat_referencia"].dt.month
    df["hora"]    = df["dat_referencia"].dt.hour

    print(f"\n📊  EDA — {tipo.upper()}")
    print("─" * 60)
    print(f"  Período  : {df['dat_referencia'].min()} → {df['dat_referencia'].max()}")
    print(f"  Registros: {len(df):,}")
    if "val_coff_mwmed" in df.columns:
        print(f"  Coff total: {df['val_coff_mwmed'].sum()/1000:,.0f} GWh")

    if "sig_estado" in df.columns and "val_coff_mwmed" in df.columns:
        print("\n  Top estados (GWh):")
        top = df.groupby("sig_estado")["val_coff_mwmed"].sum().sort_values(ascending=False).head()
        print((top / 1000).round(0).to_string())

    # Fig 1: série mensal por estado
    if "sig_estado" in df.columns and "val_coff_mwmed" in df.columns:
        mensal = df.groupby(["ano_mes","sig_estado"])["val_coff_mwmed"].sum().reset_index()
        mensal["data"] = mensal["ano_mes"].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(14, 5))
        for est in ["CE","RN","PE","BA"]:
            sub = mensal[mensal["sig_estado"] == est]
            if not sub.empty:
                ax.plot(sub["data"], sub["val_coff_mwmed"]/1000, label=est, linewidth=1.5)
        ax.set_title(f"Curtailment {tipo.upper()} — Nordeste (GWh/mês)")
        ax.set_ylabel("GWh/mês")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / f"serie_coff_{tipo}_24m.png", dpi=150)
        plt.close()
        print(f"  ✅  serie_coff_{tipo}_24m.png")

    # Fig 2: heatmap
    if "val_coff_mwmed" in df.columns:
        pivot = df.groupby(["ano","mes"])["val_coff_mwmed"].sum().unstack("mes") / 1000
        pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun",
                         "Jul","Ago","Set","Out","Nov","Dez"][:len(pivot.columns)]
        fig, ax = plt.subplots(figsize=(12, max(3, len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.4, ax=ax, cbar_kws={"label":"GWh"})
        ax.set_title(f"Heatmap {tipo.upper()} — Nordeste (GWh)")
        plt.tight_layout()
        fig.savefig(FIG / f"heatmap_coff_{tipo}.png", dpi=150)
        plt.close()
        print(f"  ✅  heatmap_coff_{tipo}.png")

    # Fig 3: perfil horodiário
    if "val_coff_mwmed" in df.columns:
        perfil = df.groupby("hora")["val_coff_mwmed"].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        perfil.plot(kind="bar", ax=ax, color="#e67e22", edgecolor="white")
        ax.set_title(f"Perfil Horodiário — {tipo.upper()}")
        ax.set_xlabel("Hora do dia")
        ax.set_ylabel("Curtailment médio (MWmed)")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)], rotation=0)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / f"perfil_horario_coff_{tipo}.png", dpi=150)
        plt.close()
        print(f"  ✅  perfil_horario_coff_{tipo}.png")


def eda_pld(df_pld: pd.DataFrame):
    if df_pld.empty:
        print("  ⚠  PLD vazio — EDA ignorada.")
        return

    df = df_pld.copy()
    df["ano_mes"] = df["dat_referencia"].dt.to_period("M")
    df["ano"]     = df["dat_referencia"].dt.year

    print("\n📊  EDA — PLD Horário NE")
    print("─" * 60)
    print(f"  Período  : {df['dat_referencia'].min()} → {df['dat_referencia'].max()}")
    print(f"  Registros: {len(df):,}")
    print(f"\n  PLD (R$/MWh):\n{df['val_pld_brl_mwh'].describe().round(2).to_string()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="val_pld_brl_mwh", by="ano", ax=ax,
               flierprops={"marker":".","markersize":2})
    ax.set_title("PLD Horário — Submercado NE")
    plt.suptitle("")
    ax.set_ylabel("R$/MWh")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "boxplot_pld_ano.png", dpi=150)
    plt.close()
    print("  ✅  boxplot_pld_ano.png")

    df_fig = df[(df["dat_referencia"] >= DATA_CORTE_FIGURAS) &
                (df["dat_referencia"] <= DATA_FIM_FIGURAS)].copy()
    df_fig["ano_mes"] = df_fig["dat_referencia"].dt.to_period("M")
    mensal = df_fig.groupby("ano_mes")["val_pld_brl_mwh"].mean().reset_index()
    mensal["data"] = mensal["ano_mes"].dt.to_timestamp()
    media_24m = df_fig["val_pld_brl_mwh"].mean()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mensal["data"], mensal["val_pld_brl_mwh"], color="#1f77b4", linewidth=1.5)
    ax.fill_between(mensal["data"], mensal["val_pld_brl_mwh"], alpha=0.12, color="#1f77b4")
    ax.set_title("PLD Médio Mensal — Submercado NE (R$/MWh)")
    ax.set_ylabel("R$/MWh")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ax.axhline(media_24m, color="red", linewidth=1.5, linestyle="--",
           label=f"Média = R$ {media_24m:.0f}/MWh")
    ax.legend()
    fig.savefig(FIG / "serie_pld_mensal_24m.png", dpi=150)
    plt.close()
    print("  ✅  serie_pld_mensal_24m.png")


def eda_lcoh(df: pd.DataFrame):
    if df.empty or "lcoh_s1_usd_kg" not in df.columns:
        print("  ⚠  LCOH vazio — EDA ignorada.")
        return

    print("\n📊  EDA — LCOH Estimado")
    print("─" * 60)
    for col, lbl in [
        ("lcoh_s1_brl_kg","S1 Curtailment (R$/kg)"),
        ("lcoh_s2_brl_kg","S2 Spot (R$/kg)"),
        ("lcoh_s1_usd_kg","S1 Curtailment (USD/kg)"),
        ("lcoh_s2_usd_kg","S2 Spot (USD/kg)"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {lbl:30s} med={s.median():.2f} "
                  f"P10={s.quantile(.1):.2f} P90={s.quantile(.9):.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, lbl, cor in [
        ("lcoh_s1_usd_kg","S1 — Curtailment puro","#2ca02c"),
        ("lcoh_s2_usd_kg","S2 — Mercado spot","#d62728"),
    ]:
        if col in df.columns:
            df[col].dropna().clip(0, 15).hist(
                ax=ax, bins=60, alpha=0.55, label=lbl, color=cor)
    ax.axvline(2.0, color="black", linestyle="--", linewidth=1.2,
               label="Meta IRENA 2030 (~USD 2/kg)")
    ax.set_title("Distribuição LCOH Estimado — Nordeste")
    ax.set_xlabel("LCOH (USD/kgH₂)")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "dist_lcoh_s1_s2.png", dpi=150)
    plt.close()
    print("  ✅  dist_lcoh_s1_s2.png")

    if "sig_estado" in df.columns:
        df["ano_mes"] = df["dat_referencia"].dt.to_period("M")
        mensal = (df.groupby(["ano_mes","sig_estado"])["lcoh_s1_usd_kg"]
                    .median().reset_index())
        mensal["data"] = mensal["ano_mes"].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(14, 5))
        for est in ["CE","RN","PE"]:
            sub = mensal[mensal["sig_estado"] == est]
            if not sub.empty:
                ax.plot(sub["data"], sub["lcoh_s1_usd_kg"], label=est, linewidth=1.5)
        ax.set_title("LCOH Mediano Mensal (S1) — CE, RN, PE")
        ax.set_ylabel("LCOH (USD/kgH₂)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIG / "lcoh_mensal_por_estado.png", dpi=150)
        plt.close()
        print("  ✅  lcoh_mensal_por_estado.png")


# ---------------------------------------------------------------------------
# 7. EXPORTAR
# ---------------------------------------------------------------------------

def exportar(df: pd.DataFrame, nome: str):
    if df.empty:
        return
    p = PROC / f"{nome}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    try:
        df.to_parquet(PROC / f"{nome}.parquet", index=False)
    except Exception:
        pass
    print(f"  💾  {p}  ({len(df):,} linhas)")




# ---------------------------------------------------------------------------
# FIGURA COMBINADA: curtailment eólico + fotovoltaico lado a lado (24 meses)
# ---------------------------------------------------------------------------

CORES_ESTADO = {
    "CE": "#1f77b4", "RN": "#ff7f0e", "PE": "#2ca02c",
    "BA": "#d62728", "PI": "#9467bd", "PB": "#8c564b", "MA": "#e377c2"
}

def gerar_figura_curtailment_combinada(df_eol: pd.DataFrame,
                                       df_sol: pd.DataFrame):
    """
    Gera figura com 2 painéis lado a lado:
      Esq: curtailment eólico mensal por estado (Abr/2024–Mar/2026)
      Dir: curtailment fotovoltaico mensal por estado (Abr/2024–Mar/2026)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, df_raw, tipo, titulo in [
        (axes[0], df_eol, "eolica",       "Curtailment Eólico"),
        (axes[1], df_sol, "fotovoltaica", "Curtailment Fotovoltaico"),
    ]:
        if df_raw.empty:
            ax.set_title(f"{titulo} — sem dados")
            continue

        df_f = df_raw[(df_raw["dat_referencia"] >= DATA_CORTE_FIGURAS) &
                      (df_raw["dat_referencia"] <= DATA_FIM_FIGURAS)].copy()
        df_f["ano_mes"] = df_f["dat_referencia"].dt.to_period("M")
        mensal = (df_f.groupby(["ano_mes", "sig_estado"])["val_coff_mwmed"]
                      .sum().reset_index())
        mensal["gwh"]  = mensal["val_coff_mwmed"] * 0.5 / 1000
        mensal["data"] = mensal["ano_mes"].dt.to_timestamp()

        for est in sorted(mensal["sig_estado"].unique()):
            d = mensal[mensal["sig_estado"] == est].sort_values("data")
            ax.plot(d["data"], d["gwh"],
                    color=CORES_ESTADO.get(est, "#999"),
                    label=est, linewidth=1.5, marker="o", markersize=3)

        ax.set_title(f"{titulo} — Nordeste (GWh/mês)\nAbr/2024–Mar/2026",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Mês de referência")
        ax.set_ylabel("GWh/mês")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend(title="Estado", fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Curtailment Renovável — Nordeste Brasileiro (Abr/2024–Mar/2026)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / "serie_coff_combinada_24m.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅  serie_coff_combinada_24m.png")

# ---------------------------------------------------------------------------
# 8. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main(forcar_download=False):
    print("=" * 60)
    print("  COLETA E EDA — CURTAILMENT + PLD + LCOH  v4")
    print("  Artigo SBPO 2026 — Nordeste Brasileiro")
    print("=" * 60)

    # 8.1 Coleta ONS
    df_eol_raw = baixar_ons("eolica",       forcar_download=forcar_download)
    df_sol_raw = baixar_ons("fotovoltaica", forcar_download=forcar_download)

    # 8.2 PLD (manual)
    df_pld_raw = carregar_ccee_pld()

    # 8.3 Diagnóstico
    print("\n🔎  Colunas reais:")
    for nome_d, df_r in [("ONS Eólico", df_eol_raw),
                          ("ONS Solar",  df_sol_raw),
                          ("CCEE PLD",   df_pld_raw)]:
        if not df_r.empty:
            print(f"  {nome_d}: {list(df_r.columns)}")

    # 8.4 Pré-processamento
    df_eol = preprocessar_ons(df_eol_raw, "eolica")       if not df_eol_raw.empty else pd.DataFrame()
    df_sol = preprocessar_ons(df_sol_raw, "fotovoltaica")  if not df_sol_raw.empty else pd.DataFrame()
    df_pld = preprocessar_pld(df_pld_raw)                  if not df_pld_raw.empty else pd.DataFrame()

    for nome_d, df_p in [("Eólico NE", df_eol),
                          ("Solar NE",  df_sol),
                          ("PLD NE",    df_pld)]:
        if not df_p.empty:
            print(f"\n  ✅  {nome_d}: {len(df_p):,} registros")
            if "sig_estado"    in df_p.columns:
                print(f"      Estados  : {sorted(df_p['sig_estado'].unique())}")
            if "id_submercado" in df_p.columns:
                print(f"      Submercado: {df_p['id_submercado'].unique()}")

    # 8.5 LCOH
    df_coff = pd.concat([df_eol, df_sol], ignore_index=True) \
              if (not df_eol.empty or not df_sol.empty) else pd.DataFrame()
    df_lcoh = calcular_lcoh(df_coff, df_pld) if not df_coff.empty else pd.DataFrame()

    # 8.6 EDA
    eda_curtailment(df_eol, "eolica")
    eda_curtailment(df_sol, "fotovoltaica")
    eda_pld(df_pld)
    eda_lcoh(df_lcoh)

    # Figura combinada: eólico + solar lado a lado (24 meses)
    gerar_figura_curtailment_combinada(df_eol, df_sol)

    # 8.7 Exportar
    exportar(df_eol,  "coff_eolico_ne")
    exportar(df_sol,  "coff_solar_ne")
    exportar(df_pld,  "pld_horario_ne")
    exportar(df_lcoh, "lcoh_estimado_ne")

    print("\n✅  Pipeline completo.")
    print(f"   Figuras : {FIG.resolve()}")
    print(f"   Dados   : {PROC.resolve()}")
    return df_eol, df_sol, df_pld, df_lcoh


if __name__ == "__main__":
    df_eol, df_sol, df_pld, df_lcoh = main(forcar_download=False)
