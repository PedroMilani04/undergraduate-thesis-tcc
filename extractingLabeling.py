import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- CONFIGURAÇÕES ---
# Lista de Ativos (5 Brasileiros + 5 Americanos)
TICKERS = [
    # Brasil (B3)
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    # Estados Unidos (S&P 500)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
]

INICIO = '2015-01-01'
FIM = '2024-12-31'

# Parâmetros da Barreira Tripla (Triple Barrier)
HORIZONTE_DIAS = 180  # (k)
ALVO_RETORNO = 0.07  # (tau)

# --- DATAS DAS ELEIÇÕES PRESIDENCIAIS ---
# Brasil: 2º turno (resultado definitivo). Próxima projeção = 4 anos após 2022.
ELEICOES_BR = [
    pd.Timestamp('2018-10-28'),  # 2018 — 2º turno (Bolsonaro eleito)
    pd.Timestamp('2022-10-30'),  # 2022 — 2º turno (Lula eleito)
    pd.Timestamp('2026-10-25'),  # 2026 — projeção (último domingo de outubro)
]

# EUA: Dia da eleição (primeira terça após a primeira segunda de novembro).
# Projeção 2028 inclusa para cobrir contagem após nov/2024.
ELEICOES_USA = [
    pd.Timestamp('2016-11-08'),  # 2016 — Trump eleito
    pd.Timestamp('2020-11-03'),  # 2020 — Biden eleito
    pd.Timestamp('2024-11-05'),  # 2024 — Trump eleito
    pd.Timestamp('2028-11-07'),  # 2028 — projeção
]


def dias_para_proxima_eleicao(date, election_dates):
    """Retorna o número de dias corridos até a próxima data de eleição."""
    for election in election_dates:  # lista já está em ordem crescente
        if date <= election:
            return (election - date).days
    return 0  # fallback: além de todas as datas conhecidas


# --- SUA FUNÇÃO DE ROTULAGEM (Mantida Intacta) ---
def rotular_barreira_tripla(row, dados_futuros, horizonte, alvo):
    # Pega os preços dos próximos 'horizonte' dias
    precos_futuros = dados_futuros['Close'].iloc[0:horizonte]
    
    if len(precos_futuros) < horizonte:
        return np.nan 

    preco_inicial = row['Close']
    
    # Se preco_inicial for uma Series, pegamos o valor escalar
    if isinstance(preco_inicial, pd.Series):
        preco_inicial = preco_inicial.item()

    # Barreiras
    barreira_alta = preco_inicial * (1 + alvo)
    barreira_baixa = preco_inicial * (1 - alvo)
    
    touched_high = precos_futuros[precos_futuros >= barreira_alta].dropna().index
    touched_low = precos_futuros[precos_futuros <= barreira_baixa].dropna().index
    
    first_high = touched_high[0] if len(touched_high) > 0 else pd.Timestamp.max
    first_low = touched_low[0] if len(touched_low) > 0 else pd.Timestamp.max
    
    if first_high == pd.Timestamp.max and first_low == pd.Timestamp.max:
        return 0 
    elif first_high < first_low:
        return 1 
    else:
        return -1

# --- PROCESSAMENTO EM LOOP ---
# --- CARREGANDO VIX (uma vez, antes do loop) ---
""" print("--- CARREGANDO VIX ---")
df_vix = yf.download('^VIX', start=INICIO, end=FIM, auto_adjust=True, progress=False)
if isinstance(df_vix.columns, pd.MultiIndex):
    df_vix.columns = df_vix.columns.get_level_values(0)
df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX'})
df_vix.index = pd.to_datetime(df_vix.index).normalize()
df_vix = df_vix.reindex(pd.date_range(df_vix.index.min(), df_vix.index.max(), freq='D')).ffill()
print(f"   VIX: {len(df_vix)} dias carregados.") """

print(f"--- INICIANDO PROCESSAMENTO DE {len(TICKERS)} ATIVOS ---")

for ativo in TICKERS:
    print(f"\n>> Processando: {ativo}...")
    
    try:
        # 1. DOWNLOAD DA MATÉRIA PRIMA
        df = yf.download(ativo, start=INICIO, end=FIM, auto_adjust=True, progress=False)

        # Flatten columns if MultiIndex (removes Ticker level)
        if isinstance(df.columns, pd.MultiIndex):
            # Tenta pegar apenas o nível 0 (Price Type), ignorando o Ticker
            df.columns = df.columns.get_level_values(0)

        # Limpeza básica
        df.dropna(inplace=True)
        
        if df.empty:
            print(f"   [AVISO] Dados vazios para {ativo}. Pulando.")
            continue

        print(f"   Dados baixados: {df.shape[0]} dias.")

        # 2. ENGENHARIA DE ATRIBUTOS BÁSICA
        df['Retorno_Diario'] = df['Close'].pct_change()
        
        # Identificador (útil para saber de quem é o arquivo se abrir depois)
        df['Ticker'] = ativo

        # Dias até as próximas eleições presidenciais (contagem regressiva)
        dates = df.index.normalize()
        df['DiasParaEleicaoBR']  = [dias_para_proxima_eleicao(d, ELEICOES_BR)  for d in dates]
        df['DiasParaEleicaoUSA'] = [dias_para_proxima_eleicao(d, ELEICOES_USA) for d in dates]

        """ # VIX — índice de medo do mercado americano
        df = df.join(df_vix, how='left')
        df['VIX'] = df['VIX'].ffill().bfill() """

        # 3. ROTULAGEM (Aplicação da Função)
        print("   Calculando rótulos...")
        labels = []
        for i in range(len(df)):
            resultado = rotular_barreira_tripla(
                df.iloc[i], 
                df.iloc[i+1:], 
                HORIZONTE_DIAS, 
                ALVO_RETORNO
            )
            labels.append(resultado)

        df['Alvo'] = labels
        
        # Remove os NaNs do final (onde não deu pra calcular o futuro)
        df_final = df.dropna(subset=['Alvo'])

        os.makedirs("1-processed-data", exist_ok=True)

        # --- SALVANDO CSV INDIVIDUAL ---
        nome_arquivo = f"1-processed-data/{ativo}_rotulado.csv"
        df_final.to_csv(nome_arquivo)
        
        print(f"   SALVO: {nome_arquivo}")
        print(f"   Distribuição: {df_final['Alvo'].value_counts(normalize=True).to_dict()}")

    except Exception as e:
        print(f"   [ERRO CRÍTICO] Falha em {ativo}: {e}")

print("\n--- PROCESSO CONCLUÍDO ---")