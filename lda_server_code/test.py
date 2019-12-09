import re
# str = "\"encode\"	\"data capacity length string builder bit index bits extract append\""
s = []
f = open('C://Users/Thinkpad/Desktop/methnametokens.txt','r')
lines = f.readlines()
for line in lines:
    k = re.sub("\"", '', line)
    s.append(k)
# .replace('"', '')
f.close()

f1 = open('C://Users/Thinkpad/Desktop/tokenresult.txt','w')
for j in s:
    f1.write(j)
f1.close()
# print(k)
