import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import os
import sys
from sklearn.feature_selection import RFE

# Adiciona o diretório raiz do projeto ao path para importar o transforming
# Estrutura: tcc/4-modelos-generalistas/xgboost/  →  raiz = dois níveis acima
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)          # 4-modelos-generalistas/
root_dir = os.path.dirname(parent_dir)              # tcc/ (onde transforming.py está)
sys.path.append(root_dir)

import transforming  # Seu arquivo atualizado com Lags, Slopes, ATR, etc.

# --- CONFIGURAÇÕES ---
INPUT_FOLDER = os.path.join(root_dir, '1-processed-data')
RAW_DATA_FOLDER = os.path.join(root_dir, '0-raw-data')
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

    # --- CARREGA TAXAS DE JUROS (uma vez, fora do loop) ---
    print("--- 0. CARREGANDO TAXAS DE JUROS ---")

    # Fed Funds Rate
    fed_path = os.path.join(RAW_DATA_FOLDER, 'fed_funds_rate_COMPLETO_2015_2024.csv')
    df_fed = pd.read_csv(fed_path, parse_dates=['DATE'])
    df_fed = df_fed[['DATE', 'Taxa_Media_(% aa)']].rename(columns={'DATE': 'Date'})
    df_fed['Date'] = pd.to_datetime(df_fed['Date']).dt.normalize()
    df_fed = df_fed.set_index('Date').sort_index()
    # Preenche fins de semana / feriados com valor anterior
    df_fed = df_fed.reindex(pd.date_range(df_fed.index.min(), df_fed.index.max(), freq='D')).ffill()
    print(f"   Fed Funds Rate: {len(df_fed)} dias carregados.")

    # Selic
    selic_path = os.path.join(RAW_DATA_FOLDER, 'taxa_selic_apurada_v2.csv')
    df_selic = pd.read_csv(selic_path, parse_dates=['Data'])
    df_selic = df_selic[['Data', 'Taxa (% a.a.)']].rename(columns={'Data': 'Date', 'Taxa (% a.a.)': 'Taxa_Selic_(% aa)'})
    df_selic['Date'] = pd.to_datetime(df_selic['Date'], dayfirst=True).dt.normalize()
    df_selic = df_selic.set_index('Date').sort_index()
    df_selic['Taxa_Selic_(% aa)'] = pd.to_numeric(df_selic['Taxa_Selic_(% aa)'], errors='coerce')
    df_selic = df_selic.reindex(pd.date_range(df_selic.index.min(), df_selic.index.max(), freq='D')).ffill()
    print(f"   Selic Rate: {len(df_selic)} dias carregados.")

    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    print(f"\n--- 1. CARREGAMENTO DE DADOS ({len(arquivos)} arquivos) ---")
    
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

            # --- Adiciona taxas de juros como features ---
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df = df.join(df_fed, on='Date', how='left')
            df = df.join(df_selic, on='Date', how='left')
            # Preenche eventuais NaN (datas fora do range das taxas)
            df['Taxa_Media_(% aa)'] = df['Taxa_Media_(% aa)'].ffill().bfill()
            df['Taxa_Selic_(% aa)'] = df['Taxa_Selic_(% aa)'].ffill().bfill()

            # --- FEATURE ENGINEERING INTELIGENTE PARA JUROS (DELTAS E SPREAD) ---
            
            # 1. Deltas (Variação da taxa em relação ao passado)
            # Usando janelas financeiras clássicas: 5 dias (1 sem), 21 dias (1 mês), 63 dias (3 meses)
            for lag in [5, 21, 63]: 
                df[f'Fed_Delta_{lag}d'] = df['Taxa_Media_(% aa)'] - df['Taxa_Media_(% aa)'].shift(lag)
                df[f'Selic_Delta_{lag}d'] = df['Taxa_Selic_(% aa)'] - df['Taxa_Selic_(% aa)'].shift(lag)
                
            # 2. Aceleração da Selic (A diferença da diferença: os juros estão caindo/subindo mais rápido?)
            df['Selic_Aceleracao'] = df['Selic_Delta_21d'] - df['Selic_Delta_21d'].shift(21)
            
            # 3. Spread Brasil x EUA (A principal feature de fluxo cambial)
            df['Spread_BR_US'] = df['Taxa_Selic_(% aa)'] - df['Taxa_Media_(% aa)']
            df['Spread_BR_US_Delta_21d'] = df['Spread_BR_US'] - df['Spread_BR_US'].shift(21)

            # --- LIMPEZA: REMOVENDO O RUÍDO ABSOLUTO ---
            # Removemos a taxa pura para evitar que o modelo sofra overfitting "decorando" o ano
            df.drop(columns=['Taxa_Media_(% aa)', 'Taxa_Selic_(% aa)'], inplace=True, errors='ignore')



            

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

    # Garante que todas as 3 classes existam em y_train para evitar erro de validação do XGBoost
    for c in [0, 1, 2]:
        if c not in y_train.values:
            print(f"   [AVISO] Classe {c} ausente no treino! Injetando amostra falsa com peso baixo.")
            X_train_scaled = np.vstack([X_train_scaled, X_train_scaled[0:1]])
            y_train = pd.concat([y_train, pd.Series([c])], ignore_index=True)
            sample_weights = np.append(sample_weights, 1e-5)



    # --- NOVO PASSO: RECURSIVE FEATURE ELIMINATION (RFE) ---
    print("\n--- 4.5 SELEÇÃO DE FEATURES (RFE) ---")

    # Criamos um estimador "leve" para o RFE não demorar uma eternidade
    # Usamos os mesmos parâmetros básicos do seu XGBoost
    estimator = xgb.XGBClassifier(
        n_estimators=100, # Menos estimadores aqui para ser rápido
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        tree_method='hist' # Acelera o processamento
    )

    # Definimos quantas features queremos manter (ex: as 15 melhores)
    # Se quiser que o modelo decida sozinho, use RFECV
    n_features_to_select = 3
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)

    print(f"   Executando RFE para selecionar as {n_features_to_select} melhores features...")
    selector = selector.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # Quais features sobreviveram?
    features_selecionadas = X_train.columns[selector.support_]
    ranking = pd.DataFrame({
        'Feature': X_train.columns,
        'Ranking': selector.ranking_
    }).sort_values(by='Ranking')

    print("\n🚀 Top Features Selecionadas:")
    print(features_selecionadas.tolist())

    # Agora filtramos os dados para o treino final
    X_train_scaled = selector.transform(X_train_scaled)
    X_test_scaled = selector.transform(X_test_scaled)



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
    report_dict = classification_report(y_test, preds, labels=[0, 1, 2], target_names=classes_nomes, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, preds, labels=[0, 1, 2], target_names=classes_nomes, zero_division=0)
    
    print("Relatório Final:")
    print(report_str)

    # Figura
    plt.figure(figsize=(10, 6)) # Bem alta para caber 3 classes de texto
    
    # 1. Heatmap (3x3 agora)
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, annot_kws={"size": 14})
    plt.title(f"MATRIZ DE CONFUSÃO (BASELINE 3 CLASSES)\nAcurácia: {acc:.2%}", fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Previsto', fontsize=12)
    plt.xticks([0.5, 1.5, 2.5], classes_nomes, fontsize=11)
    plt.yticks([0.5, 1.5, 2.5], classes_nomes, fontsize=11, rotation=0)

    # 2. Texto
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    alvo_retorno = 0.07  # Threshold (tau) from extractingLabeling.py
    ev_venda = (report_dict['Venda']['precision'] * alvo_retorno) - ((1 - report_dict['Venda']['precision']) * alvo_retorno)
    ev_compra = (report_dict['Compra']['precision'] * alvo_retorno) - ((1 - report_dict['Compra']['precision']) * alvo_retorno)

    texto_relatorio = (
        f"--- RESULTADO DO MODELO BASELINE ---\n"
        f"Hipótese: Previsão Completa (Venda/Neutro/Compra)\n\n"
        f"Acurácia Global: {acc:.2%}\n"
        f"Baseline Aleatório: ~33.3%\n\n"
        f"CLASSE VENDA:\n"
        f"Prec: {report_dict['Venda']['precision']:.2f} | Rec: {report_dict['Venda']['recall']:.2f} | F1: {report_dict['Venda']['f1-score']:.2f}\n"
        f"EV-Venda: {ev_venda:.4f}\n\n"
        f"CLASSE NEUTRO (O VILÃO):\n"
        f"Prec: {report_dict['Neutro']['precision']:.2f} | Rec: {report_dict['Neutro']['recall']:.2f} | F1: {report_dict['Neutro']['f1-score']:.2f}\n\n"
        f"CLASSE COMPRA:\n"
        f"Prec: {report_dict['Compra']['precision']:.2f} | Rec: {report_dict['Compra']['recall']:.2f} | F1: {report_dict['Compra']['f1-score']:.2f}\n"
        f"EV-Compra: {ev_compra:.4f}\n\n"
        f"Nota: A baixa performance na classe 'Neutro'\nconfirma a dificuldade de separar ruído de sinal."
    )
    
    plt.text(0.5, 0.5, texto_relatorio, 
             ha='center', va='center', 
             fontsize=11, family='monospace', 
             bbox=dict(boxstyle="round,pad=1", fc="#fff5f5", ec="red", alpha=0.9))

    plt.tight_layout()
    plt.savefig('./4-modelos-generalistas/xgboost/rfe/TAXAS+RFE-matriz-baseline-3-classes-3.png', dpi=300)
    print("\n[SUCESSO] Imagem salva como: 'matriz-baseline-3-classes+TAXAS+RFE.png'")

    # --- 7. EXPORTAÇÃO DOS DADOS DE TREINO E TESTE ---
    print("\n--- 7. EXPORTANDO DADOS (TREINO/TESTE) ---")
    cols = features_selecionadas # <-- A CORREÇÃO ESTÁ AQUI
    
    if isinstance(X_train_scaled, np.ndarray):
        df_train_export = pd.DataFrame(X_train_scaled, columns=cols)
    else:
        df_train_export = X_train_scaled.copy()
        df_train_export = df_train_export.reset_index(drop=True)
        
    df_train_export['Target_Real'] = y_train.values
    df_train_export['Target_Previsto'] = model.predict(X_train_scaled)
    df_train_export['Split'] = 'Treino'

    if isinstance(X_test_scaled, np.ndarray):
        df_test_export = pd.DataFrame(X_test_scaled, columns=cols)
    else:
        df_test_export = X_test_scaled.copy()
        df_test_export = df_test_export.reset_index(drop=True)
        
    df_test_export['Target_Real'] = y_test.values
    df_test_export['Target_Previsto'] = preds
    df_test_export['Split'] = 'Teste'

    df_export = pd.concat([df_train_export, df_test_export], ignore_index=True)
    export_path = './4-modelos-generalistas/xgboost/rfe/RFE-features_treino_teste_3_classes.csv'
    df_export.to_csv(export_path, index=False)
    print(f"   [SUCESSO] Arquivo salvo em: '{export_path}'")

if __name__ == "__main__":
    xgboostModel()