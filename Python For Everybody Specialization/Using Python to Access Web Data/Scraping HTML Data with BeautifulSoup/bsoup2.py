import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl
import re

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter URL: ')
count = int(input('Enter count: '))
pos = int(input('Enter position: '))
idx = None

for name in range(count):
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')

    if idx is None:
        idx = pos - 1
    tags = soup('a')
    lst = list()
    for tag in tags:
        lst.append(tag.get("href",None))
    url = lst[idx]
    toprint = re.findall('known_by_(.*).html',url)
    for line in toprint:
        print(line)
