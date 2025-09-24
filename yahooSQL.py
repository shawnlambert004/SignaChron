from sqlalchemy import create_engine
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


test_data = yf.download('GOOGL', start='2023-01-01', end='2023-12-31')
print(test_data)
