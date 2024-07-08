import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import itertools
import warnings
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Configure warnings to ignore specific messages
warnings.filterwarnings('ignore', message=".*findfont: Font family.*not found.*")

# Configure inline plotting
# %matplotlib inline

# Initialize dictionary to store stock data
sp_data_dict = dict()

# Read symbols from file and load data
with open('symbol_list.txt', 'r') as file:
    for line in file:
        symbol = line.strip()
        df = pd.read_excel(symbol+"_sp_data.xlsx")
        sp_data_dict[symbol] = df

rsi_data_dict = dict()

# RSI 열 추가
for symbol, sp_data in sp_data_dict.items():
    rsi_data = sp_data[['Close']].copy()
    price = rsi_data["Close"].values # 종가 데이터
    rise = price[1:] - price[:-1] # 당일 종가가 전일종가보다 높은 리스트
    rise[rise < 0] = 0
    fall = price[:-1] - price[1:] # 당일 종가가 전일종가보다 낮은 리스트
    fall[fall < 0] = 0

    rsi_data.drop(0, inplace=True)
    rsi_data['Up'] = rise
    rsi_data['Down'] = fall

    for n in [5,10,14,28]:
        sum_rise_n = rsi_data["Up"].rolling(n).sum()
        sum_fall_n = rsi_data["Down"].rolling(n).sum()
        rsi_data["RSI_"+str(n)] = sum_rise_n / (sum_rise_n + sum_fall_n)
    rsi_data_dict[symbol] = rsi_data

result = pd.DataFrame()
for n in [5,10,14,28]:
    record = []
    for symbol, sp_data in sp_data_dict.items():
        rsi_data = rsi_data_dict[symbol]
        RSI = rsi_data["RSI_"+str(n)].values

        buy_point_list = (RSI[1:]<0.3) & (RSI[:-1]>=0.3)
        buy_point_list = np.insert(buy_point_list, 0, False)
        sell_point_list = (RSI[1:]>0.7) & (RSI[:-1]<=0.7)
        sell_point_list = np.insert(sell_point_list, 0, False)

        buy_point_list = rsi_data.index[buy_point_list]
        sell_point_list = rsi_data.index[sell_point_list]

        for bp in buy_point_list:
            if (sum(bp < sell_point_list)>0) and (bp+1<=rsi_data.index[-1]):
                buy_price = rsi_data.loc[bp+1, "Close"]
                sp = sell_point_list[sell_point_list>bp][0] + 1
                if sp <= rsi_data.index[-1]:
                    sell_price = rsi_data.loc[sp, "Close"]
                    profit = (sell_price-buy_price) / buy_price * 100
                    record.append(profit)
            else:
                break
    result = pd.concat([result, pd.Series(record).describe()], axis = 1)
result.columns = ['5일', '10일', '14일', '28일']
display(result)
                
