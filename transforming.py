import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calcular_indicadores_tecnicos(df):
    """
    Calcula os indicadores definidos no projeto:
    - Médias Móveis (SMA e EMA)
    - RSI (Índice de Força Relativa)
    - MACD (Convergência/Divergência)
    - Bandas de Bollinger
    """
    df = df.copy()
    
    # 1. Médias Móveis (Tendência)
    # SMA 20 (Curto prazo) e SMA 200 (Longo prazo/Tendência primária)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (Momentum) - Janela clássica de 14 períodos
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Tendência e Momentum)
    # EMA 12 - EMA 26
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    # Sinal: EMA 9 do MACD
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    
    # 4. Bandas de Bollinger (Volatilidade)
    # Média de 20 + 2 Desvios Padrões
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = sma_20 + (std_20 * 2)
    df['Bollinger_Lower'] = sma_20 - (std_20 * 2)
    
    # Bônus: Distância do preço para a Banda Superior (Normalizado entre 0 e 1 localmente)
    df['Bollinger_Pct'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])

    # Remove os NaNs gerados pelo cálculo (ex: os primeiros 200 dias da SMA_200 ficam vazios)
    df.dropna(inplace=True)
    
    return df


def normalizar_dados(X_train, X_test):
    """
    Normaliza os dados garantindo que a escala seja aprendida apenas no treino.
    """
    scaler = StandardScaler()
    
    # Aprende a média/desvio no TREINO e já transforma
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Aplica a mesma transformação no TESTE (sem aprender nada novo)
    X_test_scaled = scaler.transform(X_test)
    
    # Opcional: Converter de volta para DataFrame se quiser manter os nomes das colunas
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
    return X_train_scaled, X_test_scaled, scaler