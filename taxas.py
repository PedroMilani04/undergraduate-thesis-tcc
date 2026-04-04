import pandas as pd
import pandas_datareader.data as web

def extrair_fed_funds_completo():
    data_inicio = '2015-01-01'
    data_fim = '2024-12-31'
    
    # Códigos oficiais do banco de dados do FRED
    series_fred = {
        'DFF': 'Taxa_Media_(% aa)',
        'EFFRVOL': 'Volume_Financeiro_(Bilhoes_USD)',
        'EFFR1': 'Taxa_Minima_1_Percentil',
        'EFFR25': 'Taxa_25_Percentil',
        'EFFR75': 'Taxa_75_Percentil',
        'EFFR99': 'Taxa_Maxima_99_Percentil'
    }

    print("⏳ Conectando aos servidores do Federal Reserve...")
    
    try:
        # Baixa todas as colunas de uma vez só
        df_completo = web.DataReader(list(series_fred.keys()), 'fred', data_inicio, data_fim)
        
        # Renomeia as colunas para o nosso padrão legível
        df_completo.rename(columns=series_fred, inplace=True)
        
        # Como as negociações só ocorrem em dias úteis, preenchemos os finais de semana 
        # com os dados de sexta-feira (forward fill) para manter a série contínua
        df_completo.fillna(method='ffill', inplace=True)
        
        # Salva o arquivo CSV
        nome_arquivo = 'fed_funds_rate_COMPLETO_2015_2024.csv'
        df_completo.to_csv(nome_arquivo)
        
        print(f"✅ Sucesso! Base completa salva como '{nome_arquivo}'.")
        print(f"Total de dias: {len(df_completo)}")
        print("\nVeja as colunas extraídas:")
        print(df_completo.head())
        
    except Exception as e:
        print(f"❌ Erro ao extrair os dados: {e}")

if __name__ == "__main__":
    extrair_fed_funds_completo()