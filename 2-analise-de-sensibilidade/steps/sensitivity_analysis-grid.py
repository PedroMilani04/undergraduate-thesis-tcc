"""
sensitivity_analysis-grid.py
==============================
Full grid search:
  ALVO_RETORNO  : 0.01 → 0.06  (step 0.01)   →   6 values
  HORIZONTE_DIAS: 3    → 20    (step 1)        →  18 values
  ──────────────────────────────────────────────────────────
  Total combinations: 108

Goal: find the (threshold, horizon) pairs that MAXIMISE Venda and
      Compra F1-scores (and precision / recall individually).

Features:
  • Incremental checkpointing — results are appended to the CSV after
    every single combination.  If the run is interrupted, just re-run:
    already-finished combos are detected and skipped automatically.
  • Final summary printed in the terminal and saved as a separate
    best-results CSV.

Run from the PROJECT ROOT:
    python 2-analise-de-sensibilidade/sensitivity_analysis-grid.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

INPUT_FOLDER  = os.path.join(ROOT_DIR, '1-processed-data')
OUTPUT_CSV    = os.path.join(SCRIPT_DIR, 'sensitivity_results_grid.csv')
BEST_CSV      = os.path.join(SCRIPT_DIR, 'sensitivity_results_grid_best.csv')

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
ALVO_VALUES      = [round(v, 4) for v in np.arange(0.01, 0.061, 0.01)]   #  6 values
HORIZONTE_VALUES = list(range(3, 21))                                       # 18 values
TOTAL_COMBOS     = len(ALVO_VALUES) * len(HORIZONTE_VALUES)                # 108

TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
]
INICIO = '2015-01-01'
FIM    = '2024-12-31'

# Column order for the output CSV
COLUMNS = [
    'alvo_retorno', 'horizonte_dias',
    'accuracy',
    'venda_precision', 'venda_recall', 'venda_f1',
    'neutro_precision', 'neutro_recall', 'neutro_f1',
    'compra_precision', 'compra_recall', 'compra_f1',
    'macro_avg_f1', 'weighted_avg_f1',
    'runtime_seconds',
]


# ============================================================================
# Helpers
# ============================================================================

def load_completed() -> set:
    """Return set of (alvo, horizonte) tuples already in the checkpoint CSV."""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV)
        return set(zip(df['alvo_retorno'].round(4), df['horizonte_dias'].astype(int)))
    except Exception:
        return set()


def append_result(row: dict) -> None:
    """Append a single result row to the checkpoint CSV."""
    df_row = pd.DataFrame([row])[COLUMNS]
    write_header = not os.path.exists(OUTPUT_CSV)
    df_row.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header)


# ============================================================================
# STEP 1 — Labeling
# ============================================================================

def rotular_barreira_tripla(row, dados_futuros, horizonte, alvo):
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


def run_labeling(horizonte_dias: int, alvo_retorno: float) -> None:
    import yfinance as yf
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    for ativo in TICKERS:
        try:
            df = yf.download(ativo, start=INICIO, end=FIM,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            if df.empty:
                continue

            df['Retorno_Diario'] = df['Close'].pct_change()
            df['Ticker'] = ativo

            labels = [
                rotular_barreira_tripla(df.iloc[i], df.iloc[i + 1:],
                                        horizonte_dias, alvo_retorno)
                for i in range(len(df))
            ]
            df['Alvo'] = labels
            df_final = df.dropna(subset=['Alvo'])
            df_final.to_csv(os.path.join(INPUT_FOLDER, f"{ativo}_rotulado.csv"))

        except Exception as e:
            print(f"     [LABELING ERROR {ativo}] {e}", flush=True)


# ============================================================================
# STEP 2 — XGBoost 3-class model
# ============================================================================

def run_xgboost_model() -> dict:
    import transforming

    TARGET_COL    = 'Alvo'
    classes_nomes = ['Venda', 'Neutro', 'Compra']

    lista_X_train, lista_X_test = [], []
    lista_y_train, lista_y_test = [], []

    for arquivo in [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]:
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
            print(f"     [MODEL LOAD ERROR {arquivo}] {e}", flush=True)

    if not lista_X_train:
        raise RuntimeError("No training data available.")

    X_train = pd.concat(lista_X_train)
    X_test  = pd.concat(lista_X_test)
    y_train = pd.concat(lista_y_train) + 1   # remap -1/0/1 → 0/1/2
    y_test  = pd.concat(lista_y_test)  + 1

    X_train_sc, X_test_sc, _ = transforming.normalizar_dados(X_train, X_test)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

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
        verbosity=0,
    )
    model.fit(X_train_sc, y_train, sample_weight=sample_weights)

    preds  = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=classes_nomes, output_dict=True
    )

    result = {'accuracy': acc}
    for cls in classes_nomes:
        result[f'{cls.lower()}_precision'] = report[cls]['precision']
        result[f'{cls.lower()}_recall']    = report[cls]['recall']
        result[f'{cls.lower()}_f1']        = report[cls]['f1-score']
    result['macro_avg_f1']    = report['macro avg']['f1-score']
    result['weighted_avg_f1'] = report['weighted avg']['f1-score']
    return result


# ============================================================================
# MAIN — Grid Search
# ============================================================================

def print_progress(done: int, total: int, elapsed: float) -> None:
    pct  = done / total * 100
    rate = done / elapsed if elapsed > 0 else 0
    eta  = (total - done) / rate if rate > 0 else float('inf')
    eta_str = f"{eta/3600:.1f}h" if eta != float('inf') else "?"
    bar_len = 40
    filled  = int(bar_len * done / total)
    bar     = '█' * filled + '░' * (bar_len - filled)
    print(f"\r  [{bar}] {done}/{total} ({pct:.1f}%)  "
          f"elapsed={elapsed/3600:.2f}h  ETA={eta_str}",
          end='', flush=True)


def main():
    completed = load_completed()
    remaining = [
        (alvo, horizonte)
        for alvo in ALVO_VALUES
        for horizonte in HORIZONTE_VALUES
        if (round(alvo, 4), int(horizonte)) not in completed
    ]

    print(f"\n{'='*70}")
    print(f"  GRID SEARCH  —  ALVO_RETORNO × HORIZONTE_DIAS")
    print(f"  Total combinations : {TOTAL_COMBOS}")
    print(f"  Already completed  : {len(completed)}")
    print(f"  Remaining          : {len(remaining)}")
    print(f"  Output CSV         : {OUTPUT_CSV}")
    print(f"{'='*70}\n")

    if not remaining:
        print("  Nothing left to compute. Jumping to summary.")
    
    global_start = time.time()
    done_this_run = 0

    for alvo, horizonte in remaining:
        combo_num = len(completed) + done_this_run + 1
        t0 = time.time()

        print(f"\n\n{'─'*70}")
        print(f"  Combo {combo_num}/{TOTAL_COMBOS}  │  "
              f"ALVO={alvo:.2%}  │  HORIZONTE={horizonte} days", flush=True)
        print(f"{'─'*70}", flush=True)

        try:
            # 1. Relabel
            run_labeling(horizonte, alvo)

            # 2. Train & evaluate
            metrics = run_xgboost_model()

            # 3. Save checkpoint immediately
            elapsed_combo = time.time() - t0
            row = {
                'alvo_retorno':    alvo,
                'horizonte_dias':  horizonte,
                'runtime_seconds': round(elapsed_combo, 1),
                **metrics,
            }
            append_result(row)
            done_this_run += 1

            print(f"\n  ✓ acc={metrics['accuracy']:.4f}  "
                  f"venda_f1={metrics['venda_f1']:.4f}  "
                  f"compra_f1={metrics['compra_f1']:.4f}  "
                  f"({elapsed_combo/60:.1f} min)", flush=True)

        except Exception as e:
            print(f"\n  [ERROR] Combo ({alvo}, {horizonte}) failed: {e}", flush=True)
            continue

        elapsed_total = time.time() - global_start
        print_progress(len(completed) + done_this_run, TOTAL_COMBOS, elapsed_total)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  GRID SEARCH COMPLETE — Loading full results for summary...")

    df = pd.read_csv(OUTPUT_CSV)
    df['alvo_retorno'] = df['alvo_retorno'].round(4)

    print(f"\n  Total results: {len(df)} / {TOTAL_COMBOS} combinations\n")

    metrics_of_interest = {
        'Venda F1'         : 'venda_f1',
        'Venda Precision'  : 'venda_precision',
        'Venda Recall'     : 'venda_recall',
        'Compra F1'        : 'compra_f1',
        'Compra Precision' : 'compra_precision',
        'Compra Recall'    : 'compra_recall',
        'Accuracy'         : 'accuracy',
        'Macro F1'         : 'macro_avg_f1',
    }

    best_rows = []
    for label, col in metrics_of_interest.items():
        best = df.loc[df[col].idxmax()]
        best_rows.append({
            'Best metric'   : label,
            'Value'         : round(best[col], 4),
            'ALVO_RETORNO'  : f"{best['alvo_retorno']:.2%}",
            'HORIZONTE_DIAS': int(best['horizonte_dias']),
            'accuracy'      : round(best['accuracy'], 4),
            'venda_f1'      : round(best['venda_f1'], 4),
            'compra_f1'     : round(best['compra_f1'], 4),
            'macro_avg_f1'  : round(best['macro_avg_f1'], 4),
        })

    df_best = pd.DataFrame(best_rows)
    df_best.to_csv(BEST_CSV, index=False)

    print("  ┌─ TOP COMBINATIONS PER METRIC ─────────────────────────────────┐")
    print(df_best.to_string(index=False))
    print(f"  └────────────────────────────────────────────────────────────────┘")

    # Also print top-5 overall by average of venda_f1 + compra_f1
    df['venda_compra_avg_f1'] = (df['venda_f1'] + df['compra_f1']) / 2
    top5 = df.nlargest(5, 'venda_compra_avg_f1')[
        ['alvo_retorno', 'horizonte_dias', 'accuracy',
         'venda_f1', 'compra_f1', 'macro_avg_f1', 'venda_compra_avg_f1']
    ].round(4)

    print(f"\n  ┌─ TOP 5 by avg(Venda F1, Compra F1) ───────────────────────────┐")
    print(top5.to_string(index=False))
    print(f"  └────────────────────────────────────────────────────────────────┘")

    print(f"\n  Full results : {OUTPUT_CSV}")
    print(f"  Best results : {BEST_CSV}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
