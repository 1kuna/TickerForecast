import requests
import json

url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&time_from=20220101T0000&sort=RELEVANCE&apikey=7V18CQ34ZDK1M3SJ'

r = requests.get(url)
data = r.json()

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)