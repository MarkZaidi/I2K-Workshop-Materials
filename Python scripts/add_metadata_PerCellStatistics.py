# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:26:08 2021

@author: Mark Zaidi

Goal is to add external patient data to per cell or per annotation statistics, by looking up what ROI a cell came from and adding the respective metadata.
Note, the metadata added during this demo is just a random categorical assignment I made to divide the dataset into two groups. Control/experimental have no
biological meaning, nor does patient A and patient B (also random assignments).

Multiple categorical groupings can be appended, simply by increasing the number of values for each dictionary key in metadata_dict, and spedifying the appropriate new column name
"""
#%% load libraries
import pandas
import pandas as pd

import math
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation
import time
from scipy.stats import spearmanr
import winsound
import umap
#%% Read data
#Location of the measurement file generated from QuPath
csv_path=r'C:\Users\Mark Zaidi\Documents\work stuff\Conferences\I2K workshop\dataset_demo\Demo_project\annotation_measurements.csv'
#Path of where to write the processed .csv file. Set equal to csv_path if you want to overwrite it (not recommended)
output_path=r'C:\Users\Mark Zaidi\Documents\work stuff\Conferences\I2K workshop\dataset_demo\Demo_project\processed_annotation_measurements.csv'
data=pandas.read_csv(csv_path)
col_names=data.columns
data['Image'].unique()
#%% Specify metadata in dict format to include. Should be in format {sampleID:[grouping ]}
metadata_dict={'SampleA':['Ctrl'], #PANEL 1
               'SampleB':['Exp'],

               }
#%% append new columns to dataframe from dict
data['Group']='placeholder'
# data['Group2']='placeholder' 
# data['Group3']='placeholder' 

for dict_entry in metadata_dict:
    data['Group'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][0], data['Group'])
    # data['Group2'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][1], data['Group2'])
    # data['Group3'] = np.where(data['Image'].str.contains(dict_entry), metadata_dict[dict_entry][2], data['Group3'])
#Add patient and ROI columns in case it's needed
data[['Patient','IMC_ROI']] = data.Image.str.split('_',expand=True)
#%% Write out new csv
data.to_csv(output_path,index=False)
#test=pandas.read_csv(output_path)