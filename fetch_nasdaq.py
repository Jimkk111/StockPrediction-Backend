import akshare as ak
import pandas as pd
from datetime import date

df = ak.index_us_stock_sina(symbol='.IXIC')
df_recent = df[df['date'] >= date(2020, 1, 1)].copy()
df_recent = df_recent.sort_values('date').reset_index(drop=True)
df_out = df_recent[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
df_out.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df_out['Date'] = pd.to_datetime(df_out['Date']).dt.strftime('%d-%m-%y')
out_path = 'data/nasdaq.csv'
df_out.to_csv(out_path, index=False)
print(f'Saved {len(df_out)} rows to {out_path}')
print(df_out.head())
print('...')
print(df_out.tail())
