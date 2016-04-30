#!/usr/bin/env python
# -*- coding: utf-8 -*-


data = []
f = open('./iris', 'r')

for line in f:
    if not line.startswith('@'):
        rowdata = line.split(', ')
        newrowdata = []
        print rowdata
        newrowdata.append(rowdata[0])
	newrowdata.append(',')

        newrowdata.append(rowdata[1])
	newrowdata.append(',')

        newrowdata.append(rowdata[2])
	newrowdata.append(',')

        newrowdata.append(rowdata[3])
	newrowdata.append(',')

        if rowdata[4] == 'Iris-setosa\n':
            newrowdata.append(0)
        elif rowdata[4] == 'Iris-versicolor\n':
            newrowdata.append(1)
        elif rowdata[4] == 'Iris-virginica\n':
            newrowdata.append(2)

        data.append(newrowdata)

f.close()

f = open('iris.raw', 'w')
for row in data:
    for data in row:
        f.write(str(data))
        f.write(' ')
    f.write('\n')

f.close()

