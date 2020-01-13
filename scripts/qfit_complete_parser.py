import pandas as pd
import os
import datetime
import argparse
import sys
import fileinput

def parse_refine_log(file):
    #file = open(file, 'r') #take this in as an arguement
    finished_pdbs = []
    with open(file) as myFile:
        lines = myFile.readlines()
    with open(file) as myFile:
        for num, line in enumerate(myFile, 1):
            #print(line)
            if '[SUCCESS] qFit has been run successfully.' in line:
                #print('found at line:', num)
                pdb_line = num - 4
                #print(lines[pdb_line])
                finished_pdbs.append(lines[pdb_line])
    with open('apo_190905_complete.txt', 'w') as file:
        for i in range(0,len(finished_pdbs)):
            file.write(str(finished_pdbs[i]))

parse_refine_log('apo_190905_toparse.out')



