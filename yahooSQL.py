from sqlalchemy import create_engine
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO


url_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url_SP500, headers=headers)

SP500_table = pd.read_html(StringIO(response.text))
SP500_symbols_df = SP500_table[0]
SP500_symbols = SP500_symbols_df['Symbol'].tolist()
SP500_symbols = [s.replace('.', '-') for s in SP500_symbols]

start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

historical_data = pd.DataFrame()
all_data = []

for name in SP500_symbols:
    try:
        ticker = yf.Ticker(name)
        test_data = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
        print(f"{name} raw data:")
        if test_data.empty:
            continue
        test_data = test_data.reset_index()
        
        test_data['Symbol'] = name
        all_data.append(test_data)
    except Exception:
        continue

if all_data:
    historical_data = pd.concat(all_data, ignore_index=True)
    print(historical_data)
else:
    print("empty")

engine = create_engine('postgresql://postgres:Donrules21_@localhost/signachronodata')
historical_data.to_sql('S&P500_price_history', engine, if_exists='replace')