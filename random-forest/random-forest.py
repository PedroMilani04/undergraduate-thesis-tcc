import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- IMPORTAÇÃO DAS BIBLIOTECAS DE ML ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONEXÃO COM SEU ARQUIVO TRANSFORMING ---
# (Garante que o Python acha o transforming.py na pasta certa)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import transforming 

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = os.path.join(parent_dir, '1-processed-data')
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
    
    # --- CRUCIAL PARA RANDOM FOREST? ---
    # Random Forest NÃO exige normalização, mas mantemos para consistência com o pipeline
    print(">>> Normalizando dados...")
    X_train_scaled, X_test_scaled, scaler = transforming.normalizar_dados(X_train, X_test)

    # ---------------------------------------------------------
    # 2. O CÉREBRO (AQUI VOCÊ MEXE)
    # ---------------------------------------------------------
    print(">>> Configurando a Random Forest...")

    # [DICA DE HIPERPARÂMETROS]
    # n_estimators: Número de árvores (100 a 1000 costuma ser bom)
    # max_depth: Profundidade máxima (Evita overfitting se limitar, ex: 10, 15)
    # min_samples_leaf: Mínimo de exemplos na ponta da árvore (aumentar reduz ruído)
    
    model = RandomForestClassifier(
        n_estimators=500,        # Número de árvores na floresta
        criterion='gini',        # 'gini' ou 'entropy'
        max_depth=15,            # Limita profundidade para não decorar
        min_samples_split=10,    # Mínimo para dividir um nó
        min_samples_leaf=5,      # Mínimo para ser uma folha final
        max_features='sqrt',     # Quantas features olhar por vez
        bootstrap=True,          # Usa amostragem com reposição
        random_state=42,
        n_jobs=-1,               # Usa todos os núcleos da CPU
        verbose=1                # Mostra o progresso
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
    
    # Se for 3 classes, ajustamos os nomes. Se for 2, altere a lista abaixo.
    target_names = ['Venda', 'Neutro', 'Compra'] # Ajuste conforme seu caso (2 ou 3)
    
    try:
        print(classification_report(y_test, preds, target_names=target_names))
    except:
        # Fallback se der erro nos nomes (ex: número de classes diferente)
        print(classification_report(y_test, preds))

    # Plota Matriz
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, preds), 
                annot=True, 
                fmt='d', 
                cmap='Greens', # Mudei pra verde pra diferenciar do MLP
                xticklabels=target_names if len(np.unique(y_test)) == len(target_names) else 'auto',
                yticklabels=target_names if len(np.unique(y_test)) == len(target_names) else 'auto')
    
    plt.title(f"Random Forest - Acc: {acc:.2%}")
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    
    # Crie a pasta se não existir
    if not os.path.exists('./random-forest'):
        os.makedirs('./random-forest')
        
    plt.savefig('./random-forest/rf_2-classes.png')

if __name__ == "__main__":
    main()