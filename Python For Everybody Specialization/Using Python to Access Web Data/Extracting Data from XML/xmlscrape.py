import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter location: ')
address = urllib.request.urlopen(url,context=ctx)
data = address.read()
print('Retrieved', len(data), 'characters')
tree = ET.fromstring(data)
num = tree.findall('.//count')
print("Count:",len(num))
lst = list()
for line in num:
    lst.append(int(line.text))
print("Sum:",sum(lst))
