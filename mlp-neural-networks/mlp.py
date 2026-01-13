import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- IMPORTAÇÃO DAS BIBLIOTECAS DE ML ---
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONEXÃO COM SEU ARQUIVO TRANSFORMING ---
# (Garante que o Python acha o transforming.py na pasta certa)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import transforming 

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = '1-processed-data'
TARGET_COL = 'Alvo'

def main():
    # ---------------------------------------------------------
    # 1. CARREGAMENTO E PREPARAÇÃO (O Trabalho Braçal)
    # ---------------------------------------------------------
    lista_X_train, lista_X_test = [], []
    lista_y_train, lista_y_test = [], []

    print(">>> Lendo arquivos...")
    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]

    for arquivo in arquivos:
        try:
            df = pd.read_csv(os.path.join(INPUT_FOLDER, arquivo))

            # [DECISÃO DE PROJETO]
            # Descomente a linha abaixo para FILTRAR NEUTROS (Modo 2 Classes)
            # Comente para usar TUDO (Modo 3 Classes)
            df = df[df[TARGET_COL] != 0] 

            if len(df) < 50: continue

            # Gera indicadores (RSI, MACD, Lags...)
            df_features = transforming.calcular_indicadores_tecnicos(df)

            # Separa tempo
            X_tr, X_te, y_tr, y_te = transforming.separar_treino_teste_temporal(
                df_features, alvo_col=TARGET_COL, test_size=0.2
            )
            
            lista_X_train.append(X_tr)
            lista_X_test.append(X_te)
            lista_y_train.append(y_tr)
            lista_y_test.append(y_te)
        except: pass

    # Junta tudo num tabelão só
    X_train = pd.concat(lista_X_train)
    X_test = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train)
    y_test = pd.concat(lista_y_test)

    # [AJUSTE DE TARGET]
    # Se for Binário (sem neutro): converte -1 para 0
    # Se for 3 Classes: converte -1->0, 0->1, 1->2 (some +1 em tudo)
    # y_train = np.where(y_train == -1, 0, 1)  <-- Exemplo Binário
    # y_test = np.where(y_test == -1, 0, 1)
    
    # --- CRUCIAL PARA REDES NEURAIS ---
    # Diferente de árvores, MLP precisa de dados na mesma escala (entre -1 e 1 ou 0 e 1)
    print(">>> Normalizando dados...")
    X_train_scaled, X_test_scaled, scaler = transforming.normalizar_dados(X_train, X_test)

    # ---------------------------------------------------------
    # 2. O CÉREBRO (AQUI VOCÊ MEXE)
    # ---------------------------------------------------------
    print(">>> Configurando a Rede Neural...")

    # [DICA DE ARQUITETURA]
    # hidden_layer_sizes=(neurônios_camada1, neurônios_camada2, ...)
    # Tente começar com (64, 32) ou (100, 50, 25)
    
    

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), # <--- DEFINA SUAS CAMADAS AQUI
        activation='relu',           # 'relu', 'tanh', 'logistic'
        solver='adam',               # Otimizador
        learning_rate_init=0.001,    # Velocidade de aprendizado
        max_iter=500,                # Quantas vezes ele vê os dados
        early_stopping=True,         # Para se não melhorar?
        random_state=42,
        verbose=True                 # Mostra o treino no terminal
    )

    # ---------------------------------------------------------
    # 3. TREINAMENTO
    # ---------------------------------------------------------
    print(">>> Treinando...")
    model.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # 4. AVALIAÇÃO
    # ---------------------------------------------------------
    print(">>> Avaliando...")
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    print(f"\n========================================")
    print(f"ACURÁCIA FINAL: {acc:.2%}")
    print(f"========================================")
    print(classification_report(y_test, preds))

    # Plota Matriz
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f"MLP - Acc: {acc:.2%}")
    plt.show()

if __name__ == "__main__":
    main()