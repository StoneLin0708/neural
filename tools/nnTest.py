import os
import subprocess


testdir = input('>>>testdir: ')
files = os.listdir(testdir)
nnfiles = []

# processMax = 4
# processNow = 0

for name in files:
    if '.nn' in name:
        nnfiles.append(name)

for n in nnfiles:
    sname = os.path.splitext(n)
    command = './neural '+testdir+n+' '+testdir+sname[0]+' >> '+ testdir+sname[0] + '_r &'
    os.system(command)
    # if(processNow < processMax):
        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        # process.wait()
        # print process.returncode
