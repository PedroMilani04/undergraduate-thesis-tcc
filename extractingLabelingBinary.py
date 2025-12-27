import yfinance as yf
import pandas as pd
import numpy as np
import os

# --- CONFIGURAÇÕES ---
TICKERS = [
    # Brasil (B3)
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    # Estados Unidos (S&P 500)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
]

INICIO = '2015-01-01'
FIM = '2024-12-31'

HORIZONTE_DIAS = 10     # (k)
ALVO_RETORNO = 0.03     # (tau) - 3% de lucro

# --- NOVA FUNÇÃO DE ROTULAGEM (MODO BINÁRIO) ---
def rotular_barreira_tripla_binaria(row, dados_futuros, horizonte, alvo):
    # Pega os preços dos próximos 'horizonte' dias
    precos_futuros = dados_futuros['Close'].iloc[0:horizonte]
    
    if len(precos_futuros) < horizonte:
        return np.nan 

    preco_inicial = row['Close']
    if isinstance(preco_inicial, pd.Series):
        preco_inicial = preco_inicial.item()

    # Barreiras
    barreira_alta = preco_inicial * (1 + alvo)
    barreira_baixa = preco_inicial * (1 - alvo)
    
    # Quando tocou em cada barreira?
    touched_high = precos_futuros[precos_futuros >= barreira_alta].dropna().index
    touched_low = precos_futuros[precos_futuros <= barreira_baixa].dropna().index
    
    # Pega a data do primeiro toque (ou data máxima se nunca tocou)
    first_high = touched_high[0] if len(touched_high) > 0 else pd.Timestamp.max
    first_low = touched_low[0] if len(touched_low) > 0 else pd.Timestamp.max
    
    # --- LÓGICA DO "SNIPER" (BINÁRIA) ---
    # A única coisa que importa: Bateu no lucro ANTES de bater no stop?
    
    if first_high < first_low:
        # Se tocou na alta antes da baixa (e antes do tempo maximo, pois first_low seria max)
        return 1  # SUCESSO (COMPRA)
    else:
        # Aqui cai tudo: 
        # - Bateu no Stop Loss (-1 antigo)
        # - Acabou o tempo (0 antigo)
        return 0  # FRACASSO (NÃO COMPRA)

# --- PROCESSAMENTO ---
print(f"--- INICIANDO ROTULAGEM BINÁRIA DE {len(TICKERS)} ATIVOS ---")

for ativo in TICKERS:
    print(f"\n>> Processando: {ativo}...")
    
    try:
        # 1. DOWNLOAD
        df = yf.download(ativo, start=INICIO, end=FIM, auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        
        if df.empty:
            print(f"   [AVISO] Dados vazios para {ativo}. Pulando.")
            continue

        print(f"   Dados baixados: {df.shape[0]} dias.")

        # 2. ENGENHARIA BÁSICA
        df['Retorno_Diario'] = df['Close'].pct_change()
        df['Ticker'] = ativo

        # 3. ROTULAGEM BINÁRIA
        print("   Aplicando Triple Barrier (Modo Binário)...")
        labels = []
        for i in range(len(df)):
            resultado = rotular_barreira_tripla_binaria(
                df.iloc[i], 
                df.iloc[i+1:], 
                HORIZONTE_DIAS, 
                ALVO_RETORNO
            )
            labels.append(resultado)

        df['Alvo'] = labels
        
        # Remove dias finais sem futuro
        df_final = df.dropna(subset=['Alvo'])

        # Garante que é inteiro (0 ou 1)
        df_final['Alvo'] = df_final['Alvo'].astype(int)

        os.makedirs("1-processed-data", exist_ok=True)

        # 4. SALVAR
        nome_arquivo = f"1-processed-data/{ativo}_rotulado.csv"
        df_final.to_csv(nome_arquivo)
        
        # Estatísticas
        contagem = df_final['Alvo'].value_counts()
        total = len(df_final)
        pct_compra = (contagem.get(1, 0) / total) * 100
        
        print(f"   SALVO: {nome_arquivo}")
        print(f"   Shape: {df_final.shape}")
        print(f"   Proporção de Compras (Alvo=1): {pct_compra:.2f}%")

    except Exception as e:
        print(f"   [ERRO CRÍTICO] Falha em {ativo}: {e}")

print("\n--- PROCESSO CONCLUÍDO ---")