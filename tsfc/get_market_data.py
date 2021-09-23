#在coinmarketcap爬取数据
import pandas as pd
from selenium import webdriver

URLs = 'https://coinmarketcap.com/currencies/'
def get_market_data(market):
    driver = webdriver.Chrome()
    driver.get(URLs+market+'/historical-data/?start=20130429&end=')

    html = driver.page_source
    driver.close()

    return pd.read_html(html)

# # **爬取数据
bitcoin_data = gmd.get_market_data('bitcoin')[0]

# 数据清洗，去掉美元符号，转换为数值形式
ds.dl(bitcoin_data,'Open*')
ds.dl(bitcoin_data,'High')
ds.dl(bitcoin_data,"Low")
ds.dl(bitcoin_data,'Close**')
ds.dl(bitcoin_data,'Volume')
ds.dl(bitcoin_data,'Market Cap')

bitcoin_data = bitcoin_data.assign(Date=pd.to_datetime(bitcoin_data[
                                                           'Date']))  #改写时间格式
bitcoin_data.loc[bitcoin_data['Close**']=="-",'Close**']=0      #将'-'赋值为0
bitcoin_data['Close**'] = bitcoin_data['Close**'].astype('float64')   #转为数字类型
bitcoin_data_clean = bitcoin_data.sort_index(axis=0,
                                            ascending=False).reset_index(
    drop=True)          #倒排，并重新索引

#保存数据
bitcoin_data_clean.to_csv('bitcoin_data(_' + time.strftime("%Y%m%d") +').csv',sep=',',header=True, index=True)