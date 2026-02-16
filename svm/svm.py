import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- IMPORTAÇÃO DAS BIBLIOTECAS DE ML ---
from sklearn.svm import SVC  # <--- O Motor do SVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONEXÃO COM SEU ARQUIVO TRANSFORMING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import transforming 

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = os.path.join(parent_dir, '1-processed-data')
TARGET_COL = 'Alvo'

def main():
    # ---------------------------------------------------------
    # 1. CARREGAMENTO E PREPARAÇÃO
    # ---------------------------------------------------------
    lista_X_train, lista_X_test = [], []
    lista_y_train, lista_y_test = [], []

    print(">>> Lendo arquivos para SVM...")
    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]

    for arquivo in arquivos:
        try:
            df = pd.read_csv(os.path.join(INPUT_FOLDER, arquivo))

            # [DECISÃO DE PROJETO]
            # Descomente para 2 Classes (Filtrar Neutros)
            # Comente para 3 Classes (Usar Tudo)
           #df = df[df[TARGET_COL] != 0] 

            if len(df) < 50: continue

            df_features = transforming.calcular_indicadores_tecnicos(df)

            X_tr, X_te, y_tr, y_te = transforming.separar_treino_teste_temporal(
                df_features, alvo_col=TARGET_COL, test_size=0.2
            )
            
            lista_X_train.append(X_tr)
            lista_X_test.append(X_te)
            lista_y_train.append(y_tr)
            lista_y_test.append(y_te)
        except: pass

    # Junta tudo
    X_train = pd.concat(lista_X_train)
    X_test = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train)
    y_test = pd.concat(lista_y_test)

    # [AJUSTE DE TARGET]
    # Se precisar ajustar manualmente os valores de y (ex: -1 pra 0), faça aqui.
    # O sklearn geralmente lida bem com -1, 0, 1 nativamente.
    
    # --- CRUCIAL PARA SVM ---
    # Diferente de Random Forest, o SVM É OBRIGATÓRIO ter dados normalizados.
    # Se os dados não estiverem na mesma escala, o SVM quebra.
    print(">>> Normalizando dados (CRÍTICO para SVM)...")
    X_train_scaled, X_test_scaled, scaler = transforming.normalizar_dados(X_train, X_test)

    # ---------------------------------------------------------
    # 2. O CÉREBRO (SVM CONFIGURATION)
    # ---------------------------------------------------------
    print(">>> Configurando SVM (Support Vector Machine)...")

    # [DICA DE HIPERPARÂMETROS SVM]
    # kernel: 'rbf' é o melhor para dados financeiros (não-lineares).
    # C: Controla a punição de erros. C alto = Tenta não errar nada (risco de overfitting).
    # gamma: 'scale' ou 'auto'. Controla a curvatura da decisão.
    
    model = SVC(
        kernel='rbf',        # Radial Basis Function (Padrão ouro para finanças)
        C=1.0,               # Regularização padrão
        gamma='scale',       # Ajuste automático baseado nas features
        random_state=42,
        verbose=True         # Mostra o progresso (pode ser lento)
    )

    # ---------------------------------------------------------
    # 3. TREINAMENTO
    # ---------------------------------------------------------
    print(">>> Treinando (Isso pode demorar um pouco no SVM)...")
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
    
    # Detecta automaticamente se são 2 ou 3 classes para nomear corretamente
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        target_names = ['Venda', 'Compra']
    else:
        target_names = ['Venda', 'Neutro', 'Compra']
    
    try:
        print(classification_report(y_test, preds, target_names=target_names))
    except:
        print(classification_report(y_test, preds))

    # Plota Matriz
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, preds), 
                annot=True, 
                fmt='d', 
                cmap='Oranges', # Usei Laranja pra diferenciar do resto
                xticklabels=target_names if len(unique_classes) == len(target_names) else 'auto',
                yticklabels=target_names if len(unique_classes) == len(target_names) else 'auto')
    
    plt.title(f"SVM (Support Vector Machine) - Acc: {acc:.2%}")
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    
    # Cria a pasta se não existir
    if not os.path.exists('./svm'):
        os.makedirs('./svm')
        
    plt.savefig('./svm/svm_resultado.png')
    print("Salvo em ./svm/svm_resultado.png")

if __name__ == "__main__":
    main()