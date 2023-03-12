import requests

url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=7V18CQ34ZDK1M3SJ'
r = requests.get(url)
data = r.json()

print(data)