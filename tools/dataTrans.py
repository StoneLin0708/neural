import os
import ntpath

filedir = input('>>>dir: ')
trans = input('>>>to: ')
files = os.listdir(filedir)

ft  = trans
ft = open(ft, 'w')

for fn in files:
    fp  = filedir + fn
    print(fp)
    f = open(fp)
    for line in f.readlines():
        sline = line.splitlines()[0].split(',');
        ft.write(sline[6] + ',' + sline[7] +':' + sline[2] + ',' + sline[3]+'\n')
    f.close()

ft.close()

