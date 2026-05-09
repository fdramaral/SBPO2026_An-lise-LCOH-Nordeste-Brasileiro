"""
=============================================================================
COLETA DE DADOS CLIMÁTICOS ERA5 — NORDESTE BRASILEIRO
Artigo: Análise Techno-Econômica por ML do LCOH a partir de Curtailment
        no Nordeste Brasileiro — SBPO 2026
=============================================================================

PRÉ-REQUISITOS:
  1. Conta gratuita no Copernicus CDS: https://cds.climate.copernicus.eu => LOGAR
  2. Arquivo ~/.cdsapirc configurado com UID e API key
  3. pip install cdsapi pandas numpy

VARIÁVEIS COLETADAS:
  - ssrd : Irradiação solar de superfície descendente (J/m² → converte para W/m²)
           Proxy para GHI (Global Horizontal Irradiance)
  - u10  : Componente U do vento a 10 m (m/s)
  - v10  : Componente V do vento a 10 m (m/s)
           → velocidade = sqrt(u10² + v10²)

RESOLUÇÃO:
  - Temporal : horária (ERA5 native)
  - Espacial : 0.25° × 0.25° (~28 km)

BOUNDING BOX NORDESTE:
  - lat: -15.0° a  -1.0°  (sul a norte)
  - lon: -47.0° a -34.0°  (oeste a leste)

PERÍODO: outubro 2021 a março 2026

=============================================================================
"""

!pip install cdsapi
!pip install openmeteo-requests requests-cache retry-requests

import cdsapi
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÕES
# ---------------------------------------------------------------------------

# Gerando arquivo de API
UID     = ""
API_KEY = "51167ff4-fcc6-4ecf-9099-9bb3062481d2"

conteudo = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {API_KEY}\n"

with open("/root/.cdsapirc", "w") as f:
    f.write(conteudo)

print("Arquivo criado:")
print(conteudo)


# Diretório de saída (alinhado com o pipeline principal)
ERA5_RAW  = Path("dados_sbpo2026/raw/era5")
ERA5_PROC = Path("dados_sbpo2026/processed")
ERA5_RAW.mkdir(parents=True, exist_ok=True)

# Bounding box do Nordeste (N, W, S, E) — formato CDS
BBOX_NE = [
    -1.0,   # Norte (lat máxima)
    -47.0,  # Oeste (lon mínima)
    -15.0,  # Sul   (lat mínima)
    -34.0,  # Leste (lon máxima)
]

# Variáveis ERA5 necessárias
VARIAVEIS = [
    "surface_solar_radiation_downwards",  # ssrd → GHI proxy
    "10m_u_component_of_wind",            # u10
    "10m_v_component_of_wind",            # v10
]

# Período de análise
ANO_INICIO = 2021
MES_INICIO = 10
ANO_FIM    = 2026
MES_FIM    = 3

# Horas do dia (todas as 24h para dados horários)
HORAS = [f"{h:02d}:00" for h in range(24)]


# ---------------------------------------------------------------------------
# 1. DOWNLOAD — UM ARQUIVO NETCDF POR ANO/MÊS
# ---------------------------------------------------------------------------

def _gerar_periodos():
    """Gera lista de (ano, mês) dentro do intervalo configurado."""
    periodos = []
    a, m = ANO_INICIO, MES_INICIO
    while (a, m) <= (ANO_FIM, MES_FIM):
        periodos.append((a, m))
        m += 1
        if m > 12:
            m, a = 1, a + 1
    return periodos


def _dias_do_mes(ano: int, mes: int) -> list:
    """Retorna lista de dias ('01'..'31') para o mês."""
    import calendar
    n_dias = calendar.monthrange(ano, mes)[1]
    return [f"{d:02d}" for d in range(1, n_dias + 1)]


def baixar_era5(forcar_download: bool = False):
    """
    Baixa dados ERA5 mensais em formato NetCDF para o Nordeste.
    Cada arquivo tem ~20–80 MB dependendo do mês.
    """
    c = cdsapi.Client()
    periodos = _gerar_periodos()

    print(f"\n📥  Baixando ERA5 — {len(periodos)} meses...")
    print(f"    Variáveis: GHI (ssrd), vento u10, vento v10")
    print(f"    Região: Nordeste Brasileiro (bbox {BBOX_NE})")
    print(f"    ⚠  Tempo estimado: ~2–5 min por mês (depende da fila do CDS)\n")

    arquivos_ok = []

    for ano, mes in periodos:
        nome = f"ERA5_NE_{ano}_{mes:02d}.nc"
        caminho = ERA5_RAW / nome

        if caminho.exists() and not forcar_download:
            print(f"  ✅  {ano}-{mes:02d} já existe — pulando.")
            arquivos_ok.append(caminho)
            continue

        dias = _dias_do_mes(ano, mes)
        print(f"  ⬇  Solicitando {ano}-{mes:02d} ({len(dias)} dias × 24h)...")

        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": VARIAVEIS,
                    "year":  str(ano),
                    "month": f"{mes:02d}",
                    "day":   dias,
                    "time":  HORAS,
                    "area":  BBOX_NE,
                    "format": "netcdf",
                },
                str(caminho),
            )
            print(f"  ✅  {ano}-{mes:02d} salvo → {caminho}")
            arquivos_ok.append(caminho)

        except Exception as e:
            print(f"  ❌  Erro em {ano}-{mes:02d}: {e}")

    print(f"\n  Total baixado: {len(arquivos_ok)}/{len(periodos)} arquivos.")
    return arquivos_ok


# ---------------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO — NETCDF → DATAFRAME HORÁRIO
# ---------------------------------------------------------------------------

def processar_era5(arquivos: list = None) -> pd.DataFrame:
    """
    Converte arquivos NetCDF do ERA5 em um DataFrame horário com:
      - timestamp (UTC)
      - ghi_wm2    : Irradiação solar superficial (W/m²) — média espacial NE
      - vento_ms   : Velocidade do vento a 10 m (m/s) — média espacial NE
      - u10, v10   : Componentes do vento (para análise direcional)

    Requer: pip install netCDF4 (ou xarray)
    """
    try:
        import xarray as xr
    except ImportError:
        print("  ❌  xarray não instalado. Execute: pip install xarray netCDF4")
        return pd.DataFrame()

    if arquivos is None:
        arquivos = sorted(ERA5_RAW.glob("ERA5_NE_*.nc"))

    if not arquivos:
        print("  ⚠  Nenhum arquivo ERA5 encontrado para processar.")
        return pd.DataFrame()

    frames = []
    print(f"\n⚙️   Processando {len(arquivos)} arquivos ERA5...")

    for arq in arquivos:
        try:
            ds = xr.open_dataset(arq)

            # ssrd vem em J/m² acumulado por hora → dividir por 3600 para W/m²
            if "ssrd" in ds:
                ghi = (ds["ssrd"] / 3600.0).mean(dim=["latitude", "longitude"])
                ghi = ghi.clip(min=0)   # eliminar negativos espúrios noturnos
            else:
                ghi = None

            # Velocidade do vento: sqrt(u10² + v10²)
            if "u10" in ds and "v10" in ds:
                vel = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
                vel_mean = vel.mean(dim=["latitude", "longitude"])
                u10_mean = ds["u10"].mean(dim=["latitude", "longitude"])
                v10_mean = ds["v10"].mean(dim=["latitude", "longitude"])
            else:
                vel_mean = u10_mean = v10_mean = None

            # Montar DataFrame
            df = pd.DataFrame({"timestamp": pd.to_datetime(ds["time"].values)})

            if ghi is not None:
                df["ghi_wm2"]  = ghi.values
            if vel_mean is not None:
                df["vento_ms"] = vel_mean.values
                df["u10"]      = u10_mean.values
                df["v10"]      = v10_mean.values

            ds.close()
            frames.append(df)

        except Exception as e:
            print(f"  ❌  Erro ao processar {arq.name}: {e}")

    if not frames:
        print("  ⚠  Nenhum dado ERA5 processado.")
        return pd.DataFrame()

    df_era5 = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    df_era5 = df_era5.drop_duplicates("timestamp").reset_index(drop=True)

    # Converter UTC → horário de Brasília (UTC-3)
    df_era5["timestamp_br"] = df_era5["timestamp"] - pd.Timedelta(hours=3)

    # Features derivadas para o modelo ML
    df_era5["hora"]     = df_era5["timestamp_br"].dt.hour
    df_era5["mes"]      = df_era5["timestamp_br"].dt.month
    df_era5["dia_semana"] = df_era5["timestamp_br"].dt.dayofweek
    df_era5["ano"]      = df_era5["timestamp_br"].dt.year

    # Irradiação acumulada diária (proxy de geração solar potencial)
    df_era5["data"]     = df_era5["timestamp_br"].dt.date
    ghi_dia = (df_era5.groupby("data")["ghi_wm2"]
                      .sum()
                      .rename("ghi_diario_wh_m2")
                      .reset_index())
    df_era5 = df_era5.merge(ghi_dia, on="data", how="left")

    print(f"  ✅  ERA5 processado: {len(df_era5):,} registros horários")
    print(f"      Período: {df_era5['timestamp_br'].min()} → "
          f"{df_era5['timestamp_br'].max()}")
    print(f"\n  Estatísticas GHI (W/m²):")
    print(df_era5["ghi_wm2"].describe().round(1).to_string())
    print(f"\n  Estatísticas Vento (m/s):")
    print(df_era5["vento_ms"].describe().round(2).to_string())

    return df_era5


# ---------------------------------------------------------------------------
# 3. EXPORTAR
# ---------------------------------------------------------------------------

def exportar_era5(df: pd.DataFrame, nome: str = "era5_nordeste_horario"):
    """Salva em CSV e Parquet."""
    if df.empty:
        return
    csv_path  = ERA5_PROC / f"{nome}.csv"
    parq_path = ERA5_PROC / f"{nome}.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_parquet(parq_path, index=False)
    except Exception:
        pass
    print(f"\n  💾  ERA5 salvo: {csv_path} ({len(df):,} linhas)")


# ---------------------------------------------------------------------------
# 4. ALTERNATIVA RÁPIDA — AMOSTRA VIA OPEN-METEO (SEM CADASTRO)
# ---------------------------------------------------------------------------

def baixar_openmeteo_amostra(
        lat: float = -5.5,
        lon: float = -36.0,
        data_inicio: str = "2021-10-01",
        data_fim: str    = "2026-03-31",
) -> pd.DataFrame:
    """
    Baixa dados horários de irradiação e vento via Open-Meteo Historical API.
    NÃO requer cadastro — ideal para testes rápidos ou se o CDS estiver lento.

    Coordenadas padrão: ~centro do RN (estado com maior curtailment em 2025)
    Para CE: lat=-4.3, lon=-39.3
    Para PE: lat=-8.5, lon=-37.5

    pip install openmeteo-requests requests-cache retry-requests
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

    except ImportError:
        print("  ❌  Instale: pip install openmeteo-requests requests-cache retry-requests")
        return pd.DataFrame()

    print(f"\n📥  Baixando via Open-Meteo (lat={lat}, lon={lon})...")
    print(f"    Período: {data_inicio} → {data_fim}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": data_inicio,
        "end_date":   data_fim,
        "hourly": [
            "shortwave_radiation",      # GHI (W/m²)
            "windspeed_10m",            # vento 10m (km/h)
            "winddirection_10m",
            "direct_radiation",         # DNI (W/m²)
            "diffuse_radiation",        # DHI (W/m²)
        ],
        "timezone": "America/Fortaleza",
    }

    try:
        responses = om.weather_api(url, params=params)
        resp = responses[0]
        hourly = resp.Hourly()

        df = pd.DataFrame({
            "timestamp_br":    pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
                          .tz_convert("America/Fortaleza").tz_localize(None),
                periods = hourly.VariablesLength() * 0 + len(hourly.Variables(0).ValuesAsNumpy()),
                freq = pd.Timedelta(seconds=hourly.Interval()),
            ),
            "ghi_wm2":         hourly.Variables(0).ValuesAsNumpy(),
            "vento_ms":        hourly.Variables(1).ValuesAsNumpy() / 3.6,  # km/h → m/s
            "vento_dir_graus": hourly.Variables(2).ValuesAsNumpy(),
            "dni_wm2":         hourly.Variables(3).ValuesAsNumpy(),
            "dhi_wm2":         hourly.Variables(4).ValuesAsNumpy(),
        })

        df["lat"] = lat
        df["lon"] = lon
        df["hora"]      = df["timestamp_br"].dt.hour
        df["mes"]       = df["timestamp_br"].dt.month
        df["dia_semana"] = df["timestamp_br"].dt.dayofweek
        df["ano"]        = df["timestamp_br"].dt.year

        print(f"  ✅  Open-Meteo: {len(df):,} registros horários carregados.")
        return df

    except Exception as e:
        print(f"  ❌  Erro Open-Meteo: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main(usar_openmeteo: bool = True, forcar_download: bool = False):
    """
    usar_openmeteo=True  → coleta rápida via Open-Meteo (sem cadastro)
                           para 3 pontos representativos (CE, RN, PE)
    usar_openmeteo=False → coleta completa via ERA5/CDS (requer cadastro)
    """
    print("=" * 60)
    print("  COLETA DADOS CLIMÁTICOS — ERA5 / OPEN-METEO")
    print("  Artigo SBPO 2026 — Nordeste Brasileiro")
    print("=" * 60)

    if usar_openmeteo:
        # ── Coleta rápida: 3 pontos representativos do Nordeste ──────────────
        pontos = {
            "CE": {"lat": -4.3,  "lon": -39.3},   # Ceará — interior / Serra da Ibiapaba
            "RN": {"lat": -5.5,  "lon": -36.0},   # RN — litoral leste (maior curtailment)
            "PE": {"lat": -8.5,  "lon": -37.5},   # PE — sertão central
        }
        frames = []
        for estado, coords in pontos.items():
            df = baixar_openmeteo_amostra(
                lat = coords["lat"],
                lon = coords["lon"],
                data_inicio = "2021-10-01",
                data_fim    = "2026-03-31",
            )
            if not df.empty:
                df["estado"] = estado
                frames.append(df)

        if frames:
            df_clima = pd.concat(frames, ignore_index=True)
            exportar_era5(df_clima, "clima_openmeteo_nordeste_horario")
            return df_clima

    else:
        # ── Coleta completa via ERA5/CDS ──────────────────────────────────────
        print("\n  ℹ️   Certifique-se de que ~/.cdsapirc está configurado.")
        print("       Registro gratuito: https://cds.climate.copernicus.eu\n")
        arquivos = baixar_era5(forcar_download=forcar_download)
        if arquivos:
            df_era5 = processar_era5(arquivos)
            exportar_era5(df_era5, "era5_nordeste_horario")
            return df_era5

    return pd.DataFrame()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # usar_openmeteo=True  → rápido, sem cadastro (recomendado para começar)
    # usar_openmeteo=False → ERA5 completo (requer conta CDS)
    df_clima = main(usar_openmeteo=True, forcar_download=False)