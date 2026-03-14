"""
sensitivity_analysis.py
========================
Sensitivity analysis for specific HORIZONTE_DIAS periods.

For each value of HORIZONTE_DIAS:
  1. Re-runs extractingLabeling.py logic to regenerate the labeled CSVs.
  2. Runs the XGBoost 3-class model and captures its metrics.
  3. Saves all results to a summary CSV in this folder.

Run from the PROJECT ROOT:
    python 2-analise-de-sensibilidade/sensitivity_analysis.py
"""

import os
import sys
import importlib
import types
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# ---------------------------------------------------------------------------
# Path setup — we must be able to import both extractingLabeling and
# transforming, which live in the project root.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HORIZONTE_PERIODS = [15, 30, 90, 180, 360, 720, 1800] # specific horizon periods to test
THRESHOLDS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
]
INICIO = '2015-01-01'
FIM    = '2024-12-31'

INPUT_FOLDER  = os.path.join(ROOT_DIR, '1-processed-data')
OUTPUT_CSV    = os.path.join(SCRIPT_DIR, 'sensitivity_results-years.csv')


# ============================================================================
# STEP 1 — Labeling  (re-implemented inline to avoid global-state issues)
# ============================================================================

def rotular_barreira_tripla(row, dados_futuros, horizonte, alvo):
    """Triple-barrier labeling function (copy from extractingLabeling.py)."""
    precos_futuros = dados_futuros['Close'].iloc[0:horizonte]
    if len(precos_futuros) < horizonte:
        return np.nan

    preco_inicial = row['Close']
    if isinstance(preco_inicial, pd.Series):
        preco_inicial = preco_inicial.item()

    barreira_alta  = preco_inicial * (1 + alvo)
    barreira_baixa = preco_inicial * (1 - alvo)

    touched_high = precos_futuros[precos_futuros >= barreira_alta].dropna().index
    touched_low  = precos_futuros[precos_futuros <= barreira_baixa].dropna().index

    first_high = touched_high[0] if len(touched_high) > 0 else pd.Timestamp.max
    first_low  = touched_low[0]  if len(touched_low)  > 0 else pd.Timestamp.max

    if first_high == pd.Timestamp.max and first_low == pd.Timestamp.max:
        return 0
    elif first_high < first_low:
        return 1
    else:
        return -1


def run_labeling(horizonte_dias: int, alvo: float) -> None:
    """Download (or reuse cached) data and regenerate labeled CSVs."""
    import yfinance as yf

    os.makedirs(INPUT_FOLDER, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  LABELING  —  HORIZONTE_DIAS = {horizonte_dias}  |  ALVO = {alvo}")
    print(f"{'='*70}")

    for ativo in TICKERS:
        print(f"\n  >> Processando: {ativo}...")
        try:
            df = yf.download(ativo, start=INICIO, end=FIM,
                             auto_adjust=True, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.dropna(inplace=True)

            if df.empty:
                print(f"     [AVISO] Dados vazios para {ativo}. Pulando.")
                continue

            df['Retorno_Diario'] = df['Close'].pct_change()
            df['Ticker'] = ativo

            labels = []
            for i in range(len(df)):
                resultado = rotular_barreira_tripla(
                    df.iloc[i],
                    df.iloc[i + 1:],
                    horizonte_dias,
                    alvo
                )
                labels.append(resultado)

            df['Alvo'] = labels
            df_final = df.dropna(subset=['Alvo'])

            nome_arquivo = os.path.join(INPUT_FOLDER, f"{ativo}_rotulado.csv")
            df_final.to_csv(nome_arquivo)
            print(f"     SALVO: {nome_arquivo}  |  "
                  f"Dist: {df_final['Alvo'].value_counts(normalize=True).to_dict()}")

        except Exception as e:
            print(f"     [ERRO CRÍTICO] Falha em {ativo}: {e}")

    print("\n  Labeling concluído.")


# ============================================================================
# STEP 2 — XGBoost 3-class model  (returns metrics dict)
# ============================================================================

def run_xgboost_model() -> dict:
    """
    Trains and evaluates the XGBoost 3-class model.
    Returns a dict with accuracy, precision, recall, and F1 per class.
    """
    import transforming  # lives in ROOT_DIR

    TARGET_COL = 'Alvo'
    classes_nomes = ['Venda', 'Neutro', 'Compra']

    lista_X_train, lista_X_test = [], []
    lista_y_train, lista_y_test = [], []

    arquivos = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    print(f"\n  Carregando {len(arquivos)} arquivos de {INPUT_FOLDER} ...")

    for arquivo in arquivos:
        try:
            df = pd.read_csv(os.path.join(INPUT_FOLDER, arquivo))

            if len(df) < 50:
                continue

            df_features = transforming.calcular_indicadores_tecnicos(df)

            X_tr, X_te, y_tr, y_te = transforming.separar_treino_teste_temporal(
                df_features, alvo_col=TARGET_COL, test_size=0.2
            )

            lista_X_train.append(X_tr)
            lista_X_test.append(X_te)
            lista_y_train.append(y_tr)
            lista_y_test.append(y_te)

        except Exception as e:
            print(f"     [ERRO ao processar {arquivo}] {e}")

    if not lista_X_train:
        raise RuntimeError("Nenhum dado de treino disponível após o carregamento.")

    X_train = pd.concat(lista_X_train)
    X_test  = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train)
    y_test  = pd.concat(lista_y_test)

    # Remap -1/0/1 → 0/1/2
    y_train = y_train + 1
    y_test  = y_test  + 1

    X_train_scaled, X_test_scaled, _ = transforming.normalizar_dados(X_train, X_test)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Ensure all 3 classes exist in y_train to avoid XGBoost Scikit-Learn validation errors
    for c in [0, 1, 2]:
        if c not in y_train.values:
            X_train_scaled = np.vstack([X_train_scaled, X_train_scaled[0:1]])
            y_train = pd.concat([y_train, pd.Series([c])], ignore_index=True)
            sample_weights = np.append(sample_weights, 1e-5)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
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
        eval_metric='mlogloss',
        verbosity=0
    )

    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    preds = model.predict(X_test_scaled)
    acc   = accuracy_score(y_test, preds)

    report = classification_report(
        y_test, preds, labels=[0, 1, 2], target_names=classes_nomes, output_dict=True, zero_division=0
    )

    print(f"\n  Acurácia: {acc:.4f}")
    for cls in classes_nomes:
        r = report[cls]
        print(f"  {cls:8s}  Prec={r['precision']:.4f}  "
              f"Rec={r['recall']:.4f}  F1={r['f1-score']:.4f}")

    result = {
        'accuracy': acc,
    }
    for cls in classes_nomes:
        result[f'{cls.lower()}_precision'] = report[cls]['precision']
        result[f'{cls.lower()}_recall']    = report[cls]['recall']
        result[f'{cls.lower()}_f1']        = report[cls]['f1-score']

    result['macro_avg_f1']    = report['macro avg']['f1-score']
    result['weighted_avg_f1'] = report['weighted avg']['f1-score']

    return result


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    all_results = []

    horizonte_values = HORIZONTE_PERIODS
    total_iteracoes = len(horizonte_values) * len(THRESHOLDS)

    idx = 1
    for horizonte in horizonte_values:
        for alvo in THRESHOLDS:
            print(f"\n\n{'#'*70}")
            print(f"#  ITERAÇÃO {idx}/{total_iteracoes}  —  HORIZONTE = {horizonte}, ALVO = {alvo}")
            print(f"{'#'*70}")

            # 1. Regenerate labeled data
            run_labeling(horizonte, alvo)

            # 2. Train XGBoost and collect metrics
            metrics = run_xgboost_model()

            # 3. Store result
            row = {'horizonte_dias': horizonte, 'alvo_retorno': alvo, **metrics}
            all_results.append(row)

            print(f"\n  [OK] Resultado salvo em memória: {row}")
            idx += 1

    # 4. Write aggregated CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print(f"\n\n{'='*70}")
    print(f"  ANÁLISE DE SENSIBILIDADE CONCLUÍDA")
    print(f"  Resultados salvos em: {OUTPUT_CSV}")
    print(f"{'='*70}\n")
    print(df_results.to_string(index=False))


if __name__ == '__main__':
    main()
