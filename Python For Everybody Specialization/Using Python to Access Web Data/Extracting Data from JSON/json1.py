import json
import urllib.request, urllib.parse, urllib.error
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter location: ')
print("Retrieving ", url)
address = urllib.request.urlopen(url,context=ctx)
data = address.read()
print("Retrieved",len(data))
js = json.loads(data)
lst = list()
for item in js["comments"]:
    lst.append(item['count'])

print("Count", len(lst))
print("Sum", sum(lst))
