import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calcular_indicadores_tecnicos(df):
    df = df.copy()
    
    # Garante DateTime no índice para extrair dia da semana
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # Se Date não for index ainda, setamos temporariamente ou usamos a coluna
        # Vamos garantir que o index seja sequencial no final, mas precisamos da data agora.
    
    # --- 1. SAZONALIDADE (NOVO) ---
    # 0=Segunda, 4=Sexta. O mercado age diferente na sexta-feira.
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # --- 2. VOLATILIDADE - ATR (NOVO) ---
    # Mede o tamanho das velas (High - Low).
    # Se o ATR é alto, o mercado está "nervoso".
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    # O True Range é o maior valor entre esses 3
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    # ATR de 14 dias
    df['ATR'] = true_range.rolling(14).mean()
    
    # ATR Relativo (O ATR de hoje é maior que a média? Estourou volatilidade?)
    df['ATR_Relativo'] = df['ATR'] / df['ATR'].rolling(20).mean()

    # --- 3. Indicadores Clássicos (Mantidos) ---
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_50 = df['Close'].rolling(window=50).mean()
    
    df['Dist_SMA_20'] = (df['Close'] - sma_20) / sma_20 
    df['Dist_SMA_Cruzamento'] = (sma_20 - sma_50) / sma_50
    df['SMA_20_Slope'] = sma_20.pct_change()
    df['SMA_50_Slope'] = sma_50.pct_change()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    sma_20_bb = df['Close'].rolling(window=20).mean()
    std_20_bb = df['Close'].rolling(window=20).std()
    upper = sma_20_bb + (std_20_bb * 2)
    lower = sma_20_bb - (std_20_bb * 2)
    df['Bollinger_Pct'] = (df['Close'] - lower) / (upper - lower)

    df['Retorno_Lag'] = df['Close'].pct_change()

    # Volume (Mantido)
    if 'Volume' in df.columns:
        df['Vol_Change'] = df['Volume'].pct_change()
        vol_sma_20 = df['Volume'].rolling(window=20).mean()
        df['Vol_Relativo'] = df['Volume'] / vol_sma_20

    # --- 4. LAGS (Incluindo as novidades) ---
    features_para_lag = [
        'Dist_SMA_20', 'Dist_SMA_Cruzamento', 
        'SMA_20_Slope', 'SMA_50_Slope',       
        'RSI', 'MACD_Hist', 'Bollinger_Pct', 'Retorno_Lag',
        'Vol_Change', 'Vol_Relativo',
        'ATR_Relativo' # <--- Adicionado
    ]
    
    # Verifica existência antes de lagar
    cols_existentes = [c for c in features_para_lag if c in df.columns]

    for col in cols_existentes:
        for lag in [1, 2, 3]: 
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    df.dropna(inplace=True)
    return df

def separar_treino_teste_temporal(df, alvo_col='Alvo', test_size=0.2):
    df = df.copy()

    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    df.sort_index(inplace=True)

    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    colunas_proibidas = [
        alvo_col, 'Ticker', 
        'Open', 'High', 'Low', 'Close', 'Volume', 'Date',
        'ATR' # Removemos o ATR valor absoluto, deixamos só o relativo
    ]
    
    cols_to_drop = [c for c in colunas_proibidas if c in df.columns]
    
    X_train = train.drop(columns=cols_to_drop)
    y_train = train[alvo_col]
    
    X_test = test.drop(columns=cols_to_drop)
    y_test = test[alvo_col]
    
    return X_train, X_test, y_train, y_test

def normalizar_dados(X_train, X_test):
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    scaler = StandardScaler()
    
    X_train_scaled_values = scaler.fit_transform(X_train)
    X_test_scaled_values = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_values, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_values, columns=X_test.columns, index=X_test.index)
        
    return X_train_scaled, X_test_scaled, scaler