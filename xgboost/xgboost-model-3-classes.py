import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import os
import sys

# Adiciona o diretório pai ao path para importar o transforming
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import transforming  # Seu arquivo atualizado com Lags, Slopes, ATR, etc.

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = '1-processed-data'
TARGET_COL = 'Alvo'

def xgboostModel():
    lista_X_train = []
    lista_X_test = []
    lista_y_train = []
    lista_y_test = []

    print("\n" + "="*70)
    print("   INICIANDO PIPELINE BASELINE (3 CLASSES: VENDA/NEUTRO/COMPRA)")
    print("   Objetivo: Demonstrar a dificuldade de prever o ruído (Neutro)")
    print("="*70 + "\n")

    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    print(f"--- 1. CARREGAMENTO DE DADOS ({len(arquivos)} arquivos) ---")
    
    total_linhas = 0

    for arquivo in arquivos:
        try:
            print(f"   > Lendo: {arquivo}...", end=" ")
            df = pd.read_csv(os.path.join(INPUT_FOLDER, arquivo))
            shape_orig = df.shape
            
            # --- DIFERENÇA CRUCIAL: NÃO FILTRAMOS O NEUTRO ---
            # Vamos tentar prever o '0'. Isso é o que causa a confusão no modelo.
            
            if len(df) < 50: 
                print("[PULADO] Pequeno demais")
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
            
            total_linhas += len(df)
            print(f"OK! (Treino: {len(X_tr)} / Teste: {len(X_te)})")

        except Exception as e: 
            print(f"[ERRO] {e}")

    # Consolida
    print("\n--- 2. CONSOLIDAÇÃO ---")
    X_train = pd.concat(lista_X_train)
    X_test = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train)
    y_test = pd.concat(lista_y_test)

    # Ajuste de Classes para XGBoost (0, 1, 2)
    # Entrada: -1 (Venda), 0 (Neutro), 1 (Compra)
    # Saída:    0 (Venda), 1 (Neutro), 2 (Compra)
    y_train = y_train + 1
    y_test = y_test + 1

    print(f"   Shape TREINO: {X_train.shape}")
    print(f"   Shape TESTE:  {X_test.shape}")
    print(f"   Distribuição: {np.bincount(y_train.astype(int))} (Venda / Neutro / Compra)")

    # Normaliza
    print("\n--- 3. NORMALIZAÇÃO ---")
    X_train_scaled, X_test_scaled, scaler = transforming.normalizar_dados(X_train, X_test)
    print("   Normalizado.")

    # Pesos (Tentativa desesperada de fazer o modelo aprender o Neutro)
    print("\n--- 4. CALCULANDO PESOS (BALANCEAMENTO) ---")
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    print("   Pesos calculados para forçar o aprendizado das classes minoritárias.")

    # Treino
    print("\n--- 5. TREINAMENTO (MULTICLASSE) ---")
    model = xgb.XGBClassifier(
        objective='multi:softprob', # Probabilidade para N classes
        num_class=3,                # 3 Classes obrigatórias
        n_estimators=1000,       
        learning_rate=0.01,      
        max_depth=7,             
        min_child_weight=1,      
        gamma=0.1,               
        subsample=0.8,
        colsample_bytree=0.6,    
        reg_alpha=0.1,           
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Passamos os pesos aqui
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print("   Modelo treinado.")

    # Avaliação
    print("\n--- 6. AVALIAÇÃO E RELATÓRIOS ---")
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*60)
    print(f"   ACURÁCIA (3 CLASSES): {acc:.2%}")
    print("="*60 + "\n")

    # --- GERAÇÃO DA IMAGEM COMPLETA ---
    classes_nomes = ['Venda', 'Neutro', 'Compra']
    
    # Dicionário para o texto
    report_dict = classification_report(y_test, preds, target_names=classes_nomes, output_dict=True)
    report_str = classification_report(y_test, preds, target_names=classes_nomes)
    
    print("Relatório Final:")
    print(report_str)

    # Figura
    plt.figure(figsize=(10, 6)) # Bem alta para caber 3 classes de texto
    
    # 1. Heatmap (3x3 agora)
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, annot_kws={"size": 14})
    plt.title(f"MATRIZ DE CONFUSÃO (BASELINE 3 CLASSES)\nAcurácia: {acc:.2%}", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Previsto', fontsize=12)
    plt.xticks([0.5, 1.5, 2.5], classes_nomes, fontsize=11)
    plt.yticks([0.5, 1.5, 2.5], classes_nomes, fontsize=11, rotation=0)

    # 2. Texto
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    texto_relatorio = (
        f"--- RESULTADO DO MODELO BASELINE ---\n"
        f"Hipótese: Previsão Completa (Venda/Neutro/Compra)\n\n"
        f"Acurácia Global: {acc:.2%}\n"
        f"Baseline Aleatório: ~33.3%\n\n"
        f"CLASSE VENDA:\n"
        f"Prec: {report_dict['Venda']['precision']:.2f} | Rec: {report_dict['Venda']['recall']:.2f} | F1: {report_dict['Venda']['f1-score']:.2f}\n\n"
        f"CLASSE NEUTRO (O VILÃO):\n"
        f"Prec: {report_dict['Neutro']['precision']:.2f} | Rec: {report_dict['Neutro']['recall']:.2f} | F1: {report_dict['Neutro']['f1-score']:.2f}\n\n"
        f"CLASSE COMPRA:\n"
        f"Prec: {report_dict['Compra']['precision']:.2f} | Rec: {report_dict['Compra']['recall']:.2f} | F1: {report_dict['Compra']['f1-score']:.2f}\n\n"
        f"Nota: A baixa performance na classe 'Neutro'\nconfirma a dificuldade de separar ruído de sinal."
    )
    
    plt.text(0.5, 0.5, texto_relatorio, 
             ha='center', va='center', 
             fontsize=11, family='monospace', 
             bbox=dict(boxstyle="round,pad=1", fc="#fff5f5", ec="red", alpha=0.9))

    plt.tight_layout()
    plt.savefig('./xgboost/matriz-baseline-3-classes.png', dpi=300)
    print("\n[SUCESSO] Imagem salva como: 'matriz-baseline-3-classes.png'")

if __name__ == "__main__":
    xgboostModel()