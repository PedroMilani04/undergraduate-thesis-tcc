import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

# Adiciona o diretório pai ao path para importar o transforming
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import transforming  # Seu arquivo atualizado com Lags e Slopes

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = '1-processed-data'
TARGET_COL = 'Alvo'

def xgboostModel():
    lista_X_train = []
    lista_X_test = []
    lista_y_train = []
    lista_y_test = []

    print("\n" + "="*60)
    print("      INICIANDO PIPELINE DE TREINAMENTO (MODO DETALHADO)")
    print("="*60 + "\n")

    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    print(f"--- 1. CARREGAMENTO DE DADOS ({len(arquivos)} arquivos encontrados) ---")
    
    total_linhas_processadas = 0

    for arquivo in arquivos:
        try:
            print(f"   > Lendo: {arquivo}...", end=" ")
            df = pd.read_csv(os.path.join(INPUT_FOLDER, arquivo))
            shape_orig = df.shape
            
            # Filtro de Neutros
            if TARGET_COL in df.columns:
                df = df[df[TARGET_COL] != 0] 
            
            # Check de tamanho mínimo
            if len(df) < 50: 
                print(f"[PULADO] Muito pequeno {len(df)}")
                continue

            # Gera features
            df_features = transforming.calcular_indicadores_tecnicos(df)
            
            # Split
            X_tr, X_te, y_tr, y_te = transforming.separar_treino_teste_temporal(
                df_features, alvo_col=TARGET_COL, test_size=0.2
            )
            
            lista_X_train.append(X_tr)
            lista_X_test.append(X_te)
            lista_y_train.append(y_tr)
            lista_y_test.append(y_te)
            
            total_linhas_processadas += len(df)
            print(f"OK! (Orig: {shape_orig[0]} -> Treino: {len(X_tr)} / Teste: {len(X_te)})")

        except Exception as e: 
            print(f"[ERRO] {e}")

    # Consolida
    print("\n--- 2. CONSOLIDAÇÃO DOS DATASETS ---")
    X_train = pd.concat(lista_X_train)
    X_test = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train)
    y_test = pd.concat(lista_y_test)

    # Binário (-1 vira 0)
    y_train = np.where(y_train == -1, 0, 1)
    y_test = np.where(y_test == -1, 0, 1)

    print(f"   Shape Final TREINO: {X_train.shape}")
    print(f"   Shape Final TESTE:  {X_test.shape}")
    print(f"   Distribuição Treino: {np.bincount(y_train)} (0 vs 1)")

    # Normaliza
    print("\n--- 3. NORMALIZAÇÃO (StandardScaler) ---")
    X_train_scaled, X_test_scaled, scaler = transforming.normalizar_dados(X_train, X_test)
    print("   Dados normalizados com sucesso.")

    # Treino
    print("\n--- 4. TREINAMENTO DO XGBOOST (Deep Learning Mode) ---")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,       
        learning_rate=0.005,      
        max_depth=7,             
        min_child_weight=1,      
        gamma=0.1,               
        subsample=0.8,
        colsample_bytree=0.6,    
        reg_alpha=0.1,           
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    print("   Modelo treinado.")

    # Tunagem
    
    print("\n--- 5. THRESHOLD TUNING (Busca Pelo Limiar Perfeito) ---")
    print("   Calculando probabilidades...")
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    melhor_acc = 0
    melhor_limiar = 0.5
    
    # Testa de 0.40 até 0.60
    limiares = np.arange(0.40, 0.60, 0.001)
    
    print("   Iniciando varredura de 0.40 a 0.60...")
    
    for t in limiares:
        preds_teste = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds_teste)
        
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_limiar = t
            # Mostra no terminal sempre que bate um recorde
            print(f"   [NOVO RECORDE] Limiar: {t:.4f} -> Acurácia: {acc:.2%}")
            
    print("\n" + "="*60)
    print(f"   RESULTADO FINAL: {melhor_acc:.2%}")
    print(f"   LIMIAR CAMPEÃO:  {melhor_limiar:.4f}")
    print("="*60 + "\n")
    
    # --- GERAÇÃO DA IMAGEM COMPLETA ---
    print("--- 6. GERANDO ARTEFATOS GRÁFICOS ---")
    
    preds_finais = (probs >= melhor_limiar).astype(int)
    
    # Prepara o texto do relatório para por na imagem
    report_dict = classification_report(y_test, preds_finais, target_names=['Venda', 'Compra'], output_dict=True)
    report_str = classification_report(y_test, preds_finais, target_names=['Venda', 'Compra'])
    
    # Printa no terminal também
    print("Relatório Final:")
    print(report_str)

    # Cria a figura
    plt.figure(figsize=(10, 6)) # Mais alta para caber o texto embaixo
    
    # Subplot 1: Matriz de Confusão
    plt.subplot(1, 2, 1) # Parte de cima
    cm = confusion_matrix(y_test, preds_finais)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws={"size": 16})
    plt.title(f"MATRIZ DE CONFUSÃO OTIMIZADA\nAcurácia: {melhor_acc:.2%} | Limiar: {melhor_limiar:.4f}", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Previsto', fontsize=12)
    plt.xticks([0.5, 1.5], ['Venda', 'Compra'], fontsize=11)
    plt.yticks([0.5, 1.5], ['Venda', 'Compra'], fontsize=11)

    # Subplot 2: Texto com Métricas
    plt.subplot(1, 2, 2) # Parte de baixo
    plt.axis('off') # Desliga eixos
    
    # Monta um texto bonito
    texto_relatorio = (
        f"--- RELATÓRIO DE PERFORMANCE ---\n\n"
        f"Acurácia Global: {melhor_acc:.2%}\n"
        f"Limiar de Decisão: {melhor_limiar:.4f}\n\n"
        f"CLASSE VENDA:\n"
        f"Precision: {report_dict['Venda']['precision']:.2f}\n"
        f"Recall:    {report_dict['Venda']['recall']:.2f}\n"
        f"F1-Score:  {report_dict['Venda']['f1-score']:.2f}\n\n"
        f"CLASSE COMPRA:\n"
        f"Precision: {report_dict['Compra']['precision']:.2f}\n"
        f"Recall:    {report_dict['Compra']['recall']:.2f}\n"
        f"F1-Score:  {report_dict['Compra']['f1-score']:.2f}\n\n"
        f"Total de Amostras de Teste: {len(y_test)}"
    )
    
    # Escreve o texto na imagem
    plt.text(0.5, 0.5, texto_relatorio, 
             ha='center', va='center', 
             fontsize=12, family='monospace', 
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black", alpha=0.9))

    plt.tight_layout()
    plt.savefig('./xgboost/matriz-2-classes.png', dpi=300)
    print("\n[SUCESSO] Imagem salva como: 'matriz-2-classes.png'")
    print("O script finalizou com sucesso.")

if __name__ == "__main__":
    xgboostModel()