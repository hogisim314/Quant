# !pip install finance-datareader
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install pykrx

# 미국주식 골든 크로스 데드 크로스 기반 - 정보 저장
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import itertools
import warnings
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 미국 NASDAQ 상장 기업 목록을 불러오기
US_data_list = fdr.StockListing('NASDAQ')

# Excel 파일로 저장
US_data_list.to_excel('US_Stock_list.xlsx', index=False)

# Excel 파일에서 불러오기
loaded_excel = pd.read_excel('US_Stock_list.xlsx')

df = pd.DataFrame(loaded_excel)

symbols = df['Symbol'].tolist()[:20] # 시총 상위 20개 종목

with open('symbol_list.txt', 'w') as file:
    for symbol in symbols:
        file.write(f"{symbol}\n")

for symbol in symbols:
    try:
        data = fdr.DataReader(symbol, '2020-01-01', '2024-07-07')
        data.to_excel(f'{symbol}_sp_data.xlsx')
        data['symbol'] = symbol
    except:
        pass

# Configure warnings to ignore specific messages
warnings.filterwarnings('ignore', message=".*findfont: Font family.*not found.*")

# Configure inline plotting
# %matplotlib inline

# Initialize dictionary to store stock data
sp_data_dict = dict()

with open('symbol_list.txt', 'r') as file:
    for line in file:
        symbol = line.strip()
        df = pd.read_excel(symbol+"_sp_data.xlsx")
        sp_data_dict[symbol] = df

# 이평선 계산
for symbol, sp_data in sp_data_dict.items():
    sp_data['MA_5'] = sp_data['Close'].rolling(window=5).mean()
    sp_data['MA_20'] = sp_data['Close'].rolling(window=10).mean()
    sp_data['MA_60'] = sp_data['Close'].rolling(window=60).mean()
    sp_data['MA_120'] = sp_data['Close'].rolling(window=120).mean()

# 골든 크로스 및 데드 크로스 신호 생성
for symbol, sp_data in sp_data_dict.items():
    for cross, st, lt in itertools.product(['Golden', 'Dead'], ['MA_5', 'MA_20'], ['MA_60', 'MA_120']):
        output_col = "{} {} {}".format(cross, st, lt)
        st_data = sp_data[st].values
        lt_data = sp_data[lt].values
        if cross == 'Golden':
            output = (st_data[1:] >= lt_data[1:]) & (st_data[:-1] < lt_data[:-1])
        elif cross == 'Dead':
            output = (st_data[1:] < lt_data[1:]) & (st_data[:-1] >= lt_data[:-1])
        output = np.insert(output, 0, False)

        # Check if output length matches DataFrame length
        if len(output) == len(sp_data):
            sp_data[output_col] = output
        else:
            print(f"Length mismatch for symbol: {symbol}, output_col: {output_col}")
            print(f"Output length: {len(output)}, DataFrame length: {len(sp_data)}")

# 애플 주식 데이터를 불러온 후 20일 이동평균선과 120일 이동평균선을 그래프로 그리기
plt.figure(figsize=(15, 8))
apple = sp_data_dict['AAPL']
plt.plot(apple['Date'], apple['MA_20'], label='20 MAG')
plt.plot(apple['Date'], apple['MA_120'], label='120 MAG')

# 골든 크로스와 데드 크로스를 그래프로 표시
golden_cross_data = apple.loc[apple['Golden MA_20 MA_120']]
dead_cross_data = apple.loc[apple['Dead MA_20 MA_120']]
plt.scatter(golden_cross_data['Date'], golden_cross_data['MA_120'], label="golden cross", color="red", s=60)
plt.scatter(dead_cross_data['Date'], dead_cross_data['MA_120'], label="dead cross", color="blue", s=60)

# 범례 표시
plt.legend()
plt.ylabel("Price")
plt.xlabel("Date")
plt.show()

# 골든크로스 매수, 데드 크로스 매도 전략을 이용한 수익률 계산
def calc_ror_using_gd_cross(sp_data_dict, symbol, st, lt):
    money = 10 ** 8
    sp_data = sp_data_dict[symbol]
    gc_idx_list = sp_data.loc[sp_data['Golden MA_{} MA_{}'.format(st, lt)]].index # 골크가 일어난 인덱스를 저장
    dc_idx_list = sp_data.loc[sp_data['Dead MA_{} MA_{}'.format(st, lt)]].index # 데크가 일어난 인덱스를 저장

    for buy_idx in gc_idx_list:
        if sum(dc_idx_list > buy_idx) == 0: # buy_idx가 dc_idx_list보다 다 작으면 마지막 날에 매도
            sell_idx = sp_data.index.max()
        else:
            sell_idx = dc_idx_list[dc_idx_list > buy_idx].min() # 큰 값 중에서 가장 최소
        buy_price = sp_data.loc[buy_idx, 'Close']
        sell_price = sp_data.loc[sell_idx, 'Close']
        num_stocks = money / buy_price
        money -= num_stocks * buy_price
        money += num_stocks * sell_price
    ror =(money - 10 ** 8) / 10 ** 8 * 100
    return round(ror, 3)

for symbol, sp_data in sp_data_dict.items():
    for st, lt in itertools.product(['5', '20'], ['60', '120']):
        ror = calc_ror_using_gd_cross(sp_data_dict, symbol, st, lt)
        print(f"{symbol} {st} {lt} : {ror}")

# 데이터 파싱
data_list = []

# ROR.txt 파일 읽기
with open('ROR.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        symbol = parts[0]
        st = int(parts[1])
        lt = int(parts[2])
        value = float(parts[4].replace(':', ''))
        data_list.append((symbol, st, lt, value))

# 데이터프레임 생성
df = pd.DataFrame(data_list, columns=['Symbol', 'ST', 'LT', 'Value'])

# 각 Symbol에 대한 수익률 시각화
plt.figure(figsize=(15, 10))

# Symbol별로 그룹화
for symbol, group_data in df.groupby('Symbol'):
    plt.plot(group_data['ST'].astype(str) + '-' + group_data['LT'].astype(str), group_data['Value'], marker='o', label=symbol)

plt.xlabel('ST-LT')
plt.ylabel('ROR (%)')
plt.title('Symbol ror comparison')
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
