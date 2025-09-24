from sqlalchemy import create_engine
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


url_nasdaq = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
nasdaq_symbol_df = pd.read_csv(url_nasdaq, sep="|")
nasdaq_symbols = nasdaq_symbol_df['Symbol'].tolist()

start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

historical_data = pd.DataFrame()

for i, name in enumerate(nasdaq_symbols[:200]):
    try:
        test_data = yf.download(name, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if test_data.empty:
            continue
        test_data_resampled = test_data.resample('ME').agg({'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'})
        test_data_resampled['Symbol'] = name
        historical_data = pd.concat([historical_data,test_data_resampled], ignore_index=True)
        print(historical_data)
    except Exception:
        continue
historical_data.reset_index(inplace=True)
print(historical_data)

