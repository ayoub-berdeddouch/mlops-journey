#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup as soup
import requests, zipfile, io


# Download all csv's from the s3 bucket
bucket_url = 'https://s3.amazonaws.com/tripdata/'
xml = soup(requests.get(bucket_url).text)
#dest_file = open('citibike.csv', 'w')

for zip_name in xml.find_all('key')[1:-1]:
    zip_name = zip_name.get_text()
    print(zip_name)
    data = requests.get(bucket_url + zip_name, stream=True)
    z = zipfile.ZipFile(io.BytesIO(data.content))
    z.extractall('citibike_data')


