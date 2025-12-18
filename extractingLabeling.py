import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURAÇÕES ---
ATIVO = 'PETR4.SA'  # Ticker da ação
INICIO = '2015-01-01'
FIM = '2024-12-31'

# Parâmetros da Barreira Tripla (Triple Barrier)
HORIZONTE_DIAS = 10     # (k) Limite de tempo vertical
ALVO_RETORNO = 0.03     # (tau) Alvo de lucro/perda (ex: 3%)

print(f"--- INICIANDO COLETA PARA {ATIVO} ---")

# 1. DOWNLOAD DA MATÉRIA PRIMA (OHLCV)
# O parâmetro 'auto_adjust=True' já ajusta dividendos e desdobramentos
df = yf.download(ATIVO, start=INICIO, end=FIM, auto_adjust=True)

# Limpeza básica: Remover dias sem negociação
df.dropna(inplace=True)
print(f"Dados baixados: {df.shape[0]} dias de pregão.")

# 2. ENGENHARIA DE ATRIBUTOS BÁSICA (Ex: Volatilidade)
# Precisamos da volatilidade para definir alvos dinâmicos (opcional, mas recomendado por Prado)
# Aqui usaremos um alvo fixo para simplificar o MVP
df['Retorno_Diario'] = df['Close'].pct_change()

# 3. IMPLEMENTAÇÃO DA BARREIRA TRIPLA (LABELING)
# Lógica: Olhamos k dias para frente.
# Se tocar +3% primeiro = COMPRA (1)
# Se tocar -3% primeiro = VENDA (-1)
# Se o tempo acabar sem tocar nenhum = ESPERA (0)

def rotular_barreira_tripla(row, dados_futuros, horizonte, alvo):
    # Pega os preços dos próximos 'horizonte' dias
    precos_futuros = dados_futuros['Close'].iloc[0:horizonte]
    
    if len(precos_futuros) < horizonte:
        return np.nan 

    preco_inicial = row['Close']
    
    # Se preco_inicial for uma Series (por causa do yfinance), pegamos o valor escalar
    if isinstance(preco_inicial, pd.Series):
        preco_inicial = preco_inicial.item()

    # Barreiras
    barreira_alta = preco_inicial * (1 + alvo)
    barreira_baixa = preco_inicial * (1 - alvo)
    
    # --- CORREÇÃO AQUI ---
    # Adicionamos .dropna() para remover os dias que NÃO tocaram na barreira
    # Caso contrário, o DataFrame mantém o índice com valores NaN
    touched_high = precos_futuros[precos_futuros >= barreira_alta].dropna().index
    touched_low = precos_futuros[precos_futuros <= barreira_baixa].dropna().index
    
    # O resto da lógica permanece igual...
    first_high = touched_high[0] if len(touched_high) > 0 else pd.Timestamp.max
    first_low = touched_low[0] if len(touched_low) > 0 else pd.Timestamp.max
    
    if first_high == pd.Timestamp.max and first_low == pd.Timestamp.max:
        return 0 
    elif first_high < first_low:
        return 1 
    else:
        return -1

print("Calculando rótulos (isso pode levar alguns segundos)...")

# Aplicando a função linha a linha (cuidado: lento em datasets gigantes, ok para este teste)
# Otimizaremos com vetorização no futuro
print(df)
labels = []
for i in range(len(df)):
    # Passamos os dados futuros para a função
    resultado = rotular_barreira_tripla(
        df.iloc[i], 
        df.iloc[i+1:], # Dados futuros
        HORIZONTE_DIAS, 
        ALVO_RETORNO
    )
    labels.append(resultado)

df['Alvo'] = labels

# --- RESULTADO FINAL ---
print("\n--- AMOSTRA DOS DADOS ---")
print(df[['Close', 'Alvo']].head(100))

print("\n--- DISTRIBUIÇÃO DAS CLASSES ---")
print(df['Alvo'].value_counts(normalize=True))
# Isso mostra se o dataset está desbalanceado (ex: muitas 'Esperas')

df[df['Alvo'] == 1.0].to_csv('PETR4_Alvo_1.csv', index=False)
df[df['Alvo'] == -1.0].to_csv('PETR4_Alvo_-1.csv', index=False)
df[df['Alvo'] == 0.0].to_csv('PETR4_Alvo_0.csv', index=False)