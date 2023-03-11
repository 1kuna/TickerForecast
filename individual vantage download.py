import csv
import requests

ticker = 'SPY'

CSV_URL = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&interval=5min&slice=year1month10&adjusted=true&apikey=7V18CQ34ZDK1M3SJ'

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    
    # Create a new CSV file and write the rows to it
    with open(f'10.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(my_list)