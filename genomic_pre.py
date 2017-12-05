# -*- coding: utf-8 -*-
"""
CS281, final project, preliminary analysis on genomic part of MM data.

This script is for preprocessing of the genomic data. 
All the vcfs are merged into one, just keeping variant type, and allelic frequency.


Created on Wed Nov 08 17:05:04 2017

@author: tingt
"""
import os
import glob
import gzip
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def is_nan(x):
    return (x is np.nan or x != x)


## Loop through gz files...
dataDir   = "E:\\MM\\TrainingData\\Genomic Data\\MMRF IA9 CelgeneProcessed\\"
mainFile = "E:\\MM\\TrainingData\\Clinical Data\\mmrf.clinical.csv"
mutectDir = "filtered MuTect2 SnpSift Annotated vcfs\\"
strelkaDir = "filtered Strelka SnpSift Annotated vcfs\\snvs\\" # No informative indels found.

drop_col = ["Study", "Disease_Status", "Disease_Type", "Cell_Type"]
mmrf_clinical = pd.read_csv(mainFile, index_col = 1).drop(drop_col, axis=1).dropna(subset=['D_PFS'])
# Mannually add some files present but not listed. might be one of a pair..
#mmrf_clinical.WES.loc["MMRF_1137"] = "MMRF_1137_4"
#mmrf_clinical.WES.loc["MMRF_1285"] = "MMRF_1285_3"



#condition_col = [x for x in mmrf_clinical.columns if x.startswith("CBC") or x.startswith("CHEM") or x.startswith("DIAG")]
#mmrf_clinical_grouped = mmrf_clinical.groupby('D_PFS_FLAG')



### mutect and strelka are first read in separately, then merge.
mutect_parts = [0,1,3,4,9]
mutect_merge = {}


# check for data listed in the master file but not present.
for patient in mmrf_clinical.index:
    if is_nan(mmrf_clinical.WES.loc[patient]):
        continue
    pathname = glob.glob(dataDir + mutectDir + mmrf_clinical.WES.loc[patient].lower() + "*.gz")
    if len(pathname) != 1:
        #print(mmrf_clinical.loc[patient, "WES"].lower(), len(pathname))
        mmrf_clinical.loc[patient, "WES"] = math.nan # mark it. 

# Check for data present but not listed in the master file.
#for name in glob.glob(dataDir + mutectDir + "*.gz"):
#    patient = os.path.basename(name)[:9].upper()
#    if patient in mmrf_clinical.index:
#        if is_nan(mmrf_clinical.WES.loc[patient]):
#            print(patient)

# First, load in mutect.
count = 0
for patient in mmrf_clinical.index:
    if is_nan(mmrf_clinical.WES.loc[patient]):
        continue
    pathname = glob.glob(dataDir + mutectDir + mmrf_clinical.WES.loc[patient].lower() + "*.gz")
    if len(pathname) != 1:
        print("Something wrong with this file: " + pathname)
        continue
    new_row = pd.DataFrame(index = [patient])
    with gzip.open(pathname[0], 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [line.strip().split()[idx] for idx in mutect_parts]
            SNV = '|'.join(parts[:4])
            AF = float(parts[4].split(":")[2])
            new_row[SNV] = [AF]
    mutect_merge[patient] = new_row
    count += 1
#    if count == 50:
#        break
print("Finished loading mutect.. now merging.")


# First see if any mutation has occurred at least twice?... ~ 30k occurred only once. and maximum occurence is 45. mostly 2 and 3.
mutect_merge_index = [list(x.columns) for x in mutect_merge.values()]
mutect_merge_index = [x for l in mutect_merge_index for x in l]
mutect_merge_index_uniq = pd.Series(mutect_merge_index).unique()
mutect_merge_index_multiple = [(a, mutect_merge_index.count(a)) for a in mutect_merge_index_uniq if mutect_merge_index.count(a) > 1]
mutect_merge_index_dict = defaultdict(lambda: [])
for snv in mutect_merge_index_uniq:
    mutect_merge_index_dict[snv[:snv.find('|')]] += [snv[snv.find('|') + 1:]]

print("Mutect_merge snvs with multiple occurrences..." + str(len(mutect_merge_index_multiple)))

# Go through strelka, fill in AF if such mutation has occurred elsewhere in the mutect2 call. Otherwise, discard.
# Strelka is too sensitive.. no need to preserve everything. 
# Comb through strelka, keep ones with normal/alt count <= 1, and tumor/alt count >1. <--- gai gai gai!
count = 0
strelkaDict = 'ACGT'
strelka_merge = {}
strelka_parts = [0,1,3,4,9,10]
for patient in mmrf_clinical.index:
    if is_nan(mmrf_clinical.WES.loc[patient]):
        continue
    pathname = glob.glob(dataDir + strelkaDir + mmrf_clinical.WES.loc[patient].lower() + "*.gz")
    if len(pathname) != 1:
        print("Something wrong with this file: " + "\n".join(pathname))
        continue
    #new_row = pd.DataFrame(index = [patient])
    with gzip.open(pathname[0], 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [line.strip().split()[idx] for idx in strelka_parts]
            chrN = parts[0]
            formatStr_normal = parts[4].split(':')
            formatStr_tumor  = parts[5].split(':')
            refCol = strelkaDict.find(parts[2]) + 4
            refCnt_normal = float(formatStr_normal[refCol].split(',')[0])
            refCnt_tumor  = float(formatStr_tumor[refCol].split(',')[0])
            
            new_row = pd.DataFrame(index = [patient])
            
            for altB in parts[3].split(','):
                altCol = strelkaDict.find(altB) + 4
                altCnt_normal = float(formatStr_normal[altCol].split(',')[0])
                altCnt_tumor  = float(formatStr_tumor[altCol].split(',')[0])
                
                if altCnt_tumor < 2 or altCnt_normal > 1:
                    continue
                SNV = '|'.join(parts[:3] + [altB])
                SNV_later = '|'.join(parts[1:3] + [altB])
                AF = altCnt_tumor / (altCnt_tumor + refCnt_tumor)
                if SNV not in mutect_merge[patient].columns:
                    if SNV_later in mutect_merge_index_dict[chrN]:
                        mutect_merge[patient][SNV] = [AF]
                        count += 1
                    else:
                        new_row[SNV] = [AF]
        strelka_merge[patient] = new_row
print("Finished loading strelka...")

# After filling with strelka calling, now use mutations occurred at least twice now to get the clean mutations.
mutect_merge_index = [list(x.columns) for x in mutect_merge.values()]
mutect_merge_index = [x for l in mutect_merge_index for x in l]
mutect_merge_index_multiple = [a for a in mutect_merge_index_uniq if mutect_merge_index.count(a) > 1]

mutect_merge_subset = [x.loc[:, mutect_merge_index_multiple].dropna(1)  if len([y for y in x.columns if y in mutect_merge_index_multiple]) > 0 else pd.DataFrame(index=x.index) for x in mutect_merge.values()]
mutect_merge_final = pd.concat(mutect_merge_subset).fillna(0)


strelka_merge_index = [list(x.columns) for x in strelka_merge.values()]
strelka_merge_index = [x for l in strelka_merge_index for x in l]
strelka_merge_index_uniq = pd.Series(strelka_merge_index).unique()
strelka_merge_index_multiple = [(a, strelka_merge_index.count(a)) for a in strelka_merge_index_uniq if strelka_merge_index.count(a) > 1]
strelka_merge_index_multiple = [a for (a,b) in strelka_merge_index_multiple]
print("Strelka_merge snvs with multiple occurrences..." + str(len(strelka_merge_index_multiple)))


strelka_merge_subset = [x.loc[:, strelka_merge_index_multiple].dropna(1)  if len([y for y in x.columns if y in strelka_merge_index_multiple]) > 0 else pd.DataFrame(index=x.index) for x in strelka_merge.values()]
strelka_merge_final = pd.concat(strelka_merge_subset).fillna(0)

genomic_clean = pd.concat([mutect_merge_final, strelka_merge_final], axis = 1)
genomic_clean.to_csv("genomic_clean.csv")
