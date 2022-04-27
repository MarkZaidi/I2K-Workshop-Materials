# -*- coding: utf-8 -*-
"""
V2 contains less hardcoded values throughout the code, and performs statistical tests on violin plots

To do:
    - add vessel distance analysis
    - calculate microvessel density across areas (MVD)



@author: Mark Zaidi
Edited by: Phoebe Lombard

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
#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\work stuff\Conferences\I2K workshop\dataset_demo\Demo_project\processed_cell_measurements.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\work stuff\Conferences\I2K workshop\dataset_demo\Demo_project\figures'

data=pandas.read_csv(csv_path)
data=data.rename(columns={'Yb(173)_173Yb-TMHistone: Nucleus: Mean':'Yb(173)_173Yb-H3K27Me3: Nucleus: Mean','Nd(146)_146Nd-Nestin: Cell: Mean':'Nd(146)_146Nd-NESTIN: Cell: Mean'})
#annotation_data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\annotation_measurements.csv')
col_names=data.columns
data=data.astype({'Patient': 'str'})
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
p_thresh=0.05,1e-28,1e-70 #p value thresholds for drawing *,**,*** on plots, respectively
measures_to_drop=['DNA193','DNA191'] #remove these from any percent positive or intensity comparisons


seed=69
#For intensity comparisons, specify number of standard deviations above mean to include intensities below it. Default is 2
num_std_to_include=2


#measurements names for demo batch
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-NESTIN: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Mean','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-H3K27Me3: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Lu(175)_175Lu-CXCR4: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#OVERWRITE: USE RENAMED TMH AND NESTIN columns
#drop specific measurements
for measure in measures_to_drop:
    measurements_of_interest[:] = [x for x in measurements_of_interest if measure not in x]
#Reset plot styles, if running this script multiple times. CURRENTLY DISABLED AS THIS PREVENTS FIGURE WINDOW POP UP
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.close('all')
plt.style.use('default')
#sort data such that PIMO negative cells show up first, gives a consistent order to violin plots
data.sort_values(param_Parent,inplace=True)

#%%Optional filtering of data
#Filter by patient
# slide_IDs=pandas.unique(data["Patient"])
# #slide_IDs.sort()
#data=data[data['Group'].str.contains("Ctrl")]

#%% get unique IHC marker names
#split class name by :, listing all classes a cell belongs to
df2=data[param_Name].str.split(':',expand=True)
#Identify unique names, filtering out the unused class name and any NoneType cases
df3=pandas.unique(df2[df2.columns].values.ravel('K'))
df3=df3[(df3 != param_UnusedClass)]
marker_list = [x for x in df3 if x != None]
#drop specific markers
for measure in measures_to_drop:
    marker_list[:] = [x for x in marker_list if measure not in x]
#remove spaces, cast to list
marker_list=[x.strip(' ') for x in marker_list]
marker_list=list(set(marker_list))
#get unique annotation class names. Should be equal to the value set by param_pos_kwd and param_neg_kwd
annotation_list=data[param_Parent].unique().tolist()
#get shortened marker name (for figure plotting)
marker_short=[i.split('-', 1)[1] for i in marker_list]
#%% iteratively get percent positive scores
#define variables
count=0
pct_name=list()
marker_count=0
pct_pos=[]
pos_in_pimo_neg=[]
pos_in_pimo_pos=[]
for marker in marker_short:
    #identify full marker name
    full_marker=marker_list[marker_count]
    marker_count=marker_count+1
    for annotation in annotation_list:
        #label current entry as the current iterators for marker and annotation
        pct_name.append(['Percent ' + marker + ' positive in ' + annotation])
        #create filters to include cells from original data that belong to current annotation and marker classification
        cond1= data[param_Name].str.contains(full_marker,regex=False)
        cond2= data[param_Parent].str.contains(annotation,regex=False)
        #If conditions are met, calculate the total number of cells that meet cond1&cond2, divide by total cells in cond2(annotation) and multiply by 100
        #pct_name corresponds to pct_pos value
        pct_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        if param_pos_kwd in annotation:
            pos_in_pimo_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        elif param_neg_kwd in annotation:
            pos_in_pimo_neg.append((cond1&cond2).sum()/cond2.sum()*100)

        count=count+1
#%% visualize percent positive as clustered bar chart
pct_pos_df=pandas.DataFrame([marker_short,pos_in_pimo_pos,pos_in_pimo_neg,[i / j for i, j in zip(pos_in_pimo_pos, pos_in_pimo_neg)]]).transpose().sort_values(3,ascending=False)
#write dataframe to csv. columns are: marker, percent positive in pimo pos, percent positive in pimo neg, and pos to neg ratio
pct_pos_df.to_csv(figpath + r'\Percent positive scores in PIMO + vs - regions.csv')

#labels = marker_short
#neg_bar = pos_in_pimo_pos
#pos_bar = pos_in_pimo_neg
labels=pct_pos_df[0]
pos_bar=pct_pos_df[1]
neg_bar=pct_pos_df[2]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
plt.xticks(rotation=90)
rects1 = ax.bar(x - width/2, neg_bar, width, label='PIMO Negative')
rects2 = ax.bar(x + width/2, pos_bar, width, label='PIMO Positive')

#add data labels
ratios = pct_pos_df[3]
for rect1, rect2, ratio in zip(rects1, rects2, ratios):
    ratio="{:.2f}".format(ratio)
    #height = rect2.get_height()
    height=(max(rect1.get_height(),rect2.get_height()))
    ax.text(rect2.get_x() + rect2.get_width() / 2, height + 5, ratio,
            ha='center', va='bottom',rotation='vertical')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Positive')
ax.set_title('Percent positive scores in PIMO +/- regions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.ylim([0,125])
fig.tight_layout()
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

#fig.tight_layout()
#plt.savefig(r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\Percent positive scores in PIMO + vs - regions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.savefig(figpath + r'\Percent positive scores in PIMO + vs - regions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.close()

# Make some labels.
#ratios = ["label%d" % i for i in xrange(len(rects))]
#%% Visualize IHC marker in PIMO positive vs negative as a violin plot
#Identify cells belonging to pos or neg Parent annotations
pos_data= data[data['Parent'].str.contains('pimo positive',regex=False)]
neg_data= data[data['Parent'].str.contains('pimo negative',regex=False)]
# Pair up marker names with measurement (not all are cell mean, some are nucleus median, etc.)
#maybe made dataframe with columns: marker, measurement, and hmmmmmmmm
col_names=data.columns

#%% now do the above, but iteratively
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Median','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

#2 lines below are violin code
# fig = plt.figure(figsize=(30, 30))
# gs = fig.add_gridspec(3, 6)

#fig = plt.figure(figsize=(6, 6))
count=0
index=0,0
#figpath=r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\figures'
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]
#%% make individual violin plot
for measure in measurements_of_interest:
    #set up conditions for filtering cells that belong to a given annotation and are less than the mean + 2 std for the measurement
    cond1_pos= data[param_Parent].str.contains(param_pos_kwd,regex=False)
    cond1_neg= data[param_Parent].str.contains(param_neg_kwd,regex=False)
    cond2=data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())
    pos_selection=data[cond1_pos&cond2][measure]
    neg_selection=data[cond1_neg&cond2][measure]

    ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")
    #Calculate stats. NOTE: NOT SURE HOW RELIABLE THESE ARE; GETTING INSANELY SMALL P VALUES
    test_results = add_stat_annotation(ax, data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], x='Parent', y=measure,
                                   box_pairs=[(param_neg_kwd, param_pos_kwd)],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=0,comparisons_correction=None)
    #print(measure,stats.ttest_ind(neg_selection,pos_selection,equal_var=False),'\n')

    plt.savefig(os.path.join(figpath,measure_short_for_fname[count]+'.png'),dpi=800,pad_inches=0.1,bbox_inches='tight')
    count=count+1
    plt.close()
#%% create overall violinplot figure
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('default')

fig = plt.figure(figsize=(30, 30))

gs = fig.add_gridspec(5, 6)
count=0
index=0,0

#figpath=r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\figures'
for measure in measurements_of_interest:
    
    #ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<0.9*data[measure].max()], palette="muted",scale='width',cut=0)
    # ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")
    # plt.savefig(os.path.join(figpath,measure_short[count]+'.png'),dpi=800,pad_inches=0.1,bbox_inches='tight')
    # count=count+1
    # plt.close()
    ##Violin subplot code below
    #print(count)
    if count<6:
        index=0,count
    elif (count>=6)&(count<12):
        index=1,count-6
    elif (count>=12)&(count<18):
        index=2,count-12
    elif (count>=18)&(count<24):
        index=3,count-18
    elif (count>=24)&(count<30):
        index=4,count-24
    sns.set(font_scale=0.5)

    ax = fig.add_subplot(gs[index])

    ax.set_ylabel(measure_short[count])

    
    ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")

    test_results = add_stat_annotation(ax, data=data[data[measure]<(data[measure].mean()+num_std_to_include*data[measure].std())], x='Parent', y=measure,
                                   box_pairs=[(param_neg_kwd, param_pos_kwd)],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=0,comparisons_correction=None)

    count=count+1

matplotlib.pyplot.pause(1)
fig.tight_layout()
plt.savefig(os.path.join(figpath,'OverallV5.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
plt.close()
#%% Calculate select double-positive combinations for bar charts in PIMO +/- areas
#Define marker pairs
pair1=["IBA1","ICAM"]
pair2=["CD68","ICAM"]
pair_list=[pair1,pair2]
pct_dp_PIMO_pos=[]
pct_dp_PIMO_neg=[]
pair_name=[]
for pair in pair_list:
    print(pair)
    
    #slice data to find cells that contain both strings in pair1 as substrings in their "Name" classification parameter
    dp_cells=data[param_Name].str.contains(pair[0],regex=False)&data[param_Name].str.contains(pair[1],regex=False)
    #Find number of cells that meet the double positive criteria above AND fall in a PIMO classification. Divide by total number of cells in PIMO classification. Multiply by 100
    pct_dp_PIMO_pos.append((dp_cells&data[param_Parent].str.contains(param_pos_kwd,regex=False)).sum()/(data[param_Parent].str.contains(param_pos_kwd,regex=False).sum())*100)
    pct_dp_PIMO_neg.append((dp_cells&data[param_Parent].str.contains(param_neg_kwd,regex=False)).sum()/(data[param_Parent].str.contains(param_neg_kwd,regex=False).sum())*100)
    pair_name.append(pair[0]+' & '+pair[1])
#Merge all into a dataframe
if pct_dp_PIMO_neg.count(0)>0:
    raise ValueError('Denominator pct_dp_PIMO_neg is 0, meaning there are no double negative cells. Please verify pair1 and pair2 are present in dataset')
pair_df=pandas.DataFrame(list(zip(pair_name,pct_dp_PIMO_pos,pct_dp_PIMO_neg,[i / j for i, j in zip(pct_dp_PIMO_pos, pct_dp_PIMO_neg)])),columns =['pair_name', 'pct_dp_PIMO_pos','pct_dp_PIMO_neg','ratio']).sort_values('ratio',ascending=False)
#%% visualize percent double positives from above as clustered bar chart
#labels = marker_short
#neg_bar = pos_in_pimo_pos
#pos_bar = pos_in_pimo_neg
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('default')

labels=pair_df["pair_name"]
pos_bar=pair_df["pct_dp_PIMO_pos"]
neg_bar=pair_df["pct_dp_PIMO_neg"]
ratios = pair_df["ratio"]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
plt.xticks(rotation=90)
rects1 = ax.bar(x - width/2, neg_bar, width, label='PIMO Negative')
rects2 = ax.bar(x + width/2, pos_bar, width, label='PIMO Positive')

#add data labels
for rect1, rect2, ratio in zip(rects1, rects2, ratios):
    ratio="{:.2f}".format(ratio)
    #height = rect2.get_height()
    height=(max(rect1.get_height(),rect2.get_height()))
    ax.text(rect2.get_x() + rect2.get_width() * 0, height + 5, ratio,
            ha='center', va='bottom',rotation='vertical')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Double Positive')
ax.set_title('Percent double positive scores in PIMO +/- regions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.ylim([0,125])
fig.tight_layout()
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

#fig.tight_layout()
plt.savefig(figpath + r'\Percent double positive scores.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.close()
#%% TEMPORARY PRUNING OF DATA
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]


testvar_measures=measurements_of_interest
testvar_measures_short=measure_short_for_fname
testvar=data
colors=sns.color_palette("tab10")
#%%PAIRPLOTS
# only pass data less than 2 standard deviations above the mean 
# Code below is wrong. measure isn't being iterated upon, meaning its filtering out all measurements such that its the mean + stdev of the last `measure` iterand.
#Output will still be generated, but technically not correct. Potential solution is to either include outliers and perform no filtering, OR  find a way to modify `data` such that it labels outliers as NaN
#pair = sns.pairplot(data=testvar,vars=testvar_measures, hue='Parent',diag_kind='kde',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01))
start_time = time.time()
plt.style.use('default')
plt.rcParams.update({'font.size': 5})
pair = sns.pairplot(data=testvar.sample(n=1000,random_state=seed).sort_values([param_Parent]),vars=testvar_measures, hue='Parent',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=0.01),diag_kws=dict(common_norm=False))
pair._legend.remove()
print('It took', time.time()-start_time, 'seconds for the pairplot.')
#Variants for plotting only positive or negative data
#pair = sns.pairplot(data=testvar[testvar["Parent"]==param_neg_kwd],vars=testvar_measures, hue='Parent',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01),palette=[colors[0]])
#pair = sns.pairplot(data=testvar[testvar["Parent"]==param_pos_kwd],vars=testvar_measures, hue='Parent',plot_kws=dict(marker=".", linewidth=1,edgecolor=None,alpha=.01),palette=[colors[1]])
# #%%
# #pair.map_lower(sns.kdeplot, levels=4, color=".2")

xlabels=testvar_measures_short
ylabels=testvar_measures_short

for i in range(len(xlabels)):
    for j in range(len(ylabels)):
        pair.axes[j,i].xaxis.set_label_text(xlabels[i])
        pair.axes[j,i].yaxis.set_label_text(ylabels[j])
        


matplotlib.pyplot.pause(10)
pair.fig.set_figheight(20)
pair.fig.set_figwidth(20)
pair.tight_layout()
#plt.savefig(os.path.join(figpath,'PairPlotV2_TESTFIGURE.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
#%% spearmann correlation and p value calculation
#PIMO negative
PIMO_neg = testvar[testvar["Parent"]==param_neg_kwd]
PIMOneg_corr = []
PIMOneg_p = []
all_corr = []
all_p = []
PIMOpos_corr = []
PIMOpos_p = []

for i in range(0,len(testvar_measures)):
    data1 = PIMO_neg[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_neg[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            PIMOneg_corr.append(round(corr,3))
            PIMOneg_p.append(pval)


spearmann_corr = pd.DataFrame(PIMOneg_corr,columns=['PIMOneg'])
p_values = pd.DataFrame(PIMOneg_p,columns=['PIMOneg'])

PIMO_pos = testvar[testvar["Parent"]==param_pos_kwd]


for i in range(0,len(testvar_measures)):
    data1 = PIMO_pos[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = PIMO_pos[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            PIMOpos_corr.append(round(corr,3))
            PIMOpos_p.append(pval)


            
spearmann_corr = spearmann_corr.assign(PIMOpos= PIMOpos_corr)
p_values = p_values.assign(PIMOpos = PIMOpos_p)



for i in range(0,len(testvar_measures)):
    data1 = testvar[testvar_measures[i]].tolist()
    for j in range(0,len(testvar_measures)):
            data2 = testvar[testvar_measures[j]].tolist()
            corr, pval = spearmanr(data1, data2)
            all_corr.append(round(corr,3))
            all_p.append(pval)


spearmann_corr = spearmann_corr.assign(All= all_corr)
p_values = p_values.assign(All = all_p)

#MARK NOTE: disabled rounding of p value to 3 significant figures. P values can be on the order of 10^-23 or smaller, and rounding will just make everything 0.
#Might also need to reevaluate thresholds for what is *, **, and *** because these p values are so small.

#%% write correlation coefficients on subplot


##MARK NOTE: See if you can merge the *s with the original text from the first for loop in this block, rather than write it out separately.
#Makes it a bit cleaner and reduces ax.write operations, which is more apparent when processing the full ~18 measurements.

# ADD CASE FOR NO STATISTICAL SIGNIFICANCE***********************************************************************************************************************

for neg_coeff,all_coeff,pos_coeff,ax,neg_ps,all_ps,pos_ps in zip(spearmann_corr['PIMOneg'],spearmann_corr['All'],spearmann_corr['PIMOpos'],pair.axes.flatten(),p_values['PIMOneg'],p_values['All'],p_values['PIMOpos']):
    xmin, xmax, ymin, ymax = ax.axis()
    if (pos_ps<p_thresh[0])&(pos_ps>=p_thresh[1]):
        ax.text(xmax*0.6, ymax*0.6,str(pos_coeff) + '*', fontsize=5,color=colors[1]) #pimo positive
    elif (pos_ps<p_thresh[1])&(pos_ps>=p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.6,str(pos_coeff) + '**', fontsize=5,color=colors[1]) #pimo positive
    elif (pos_ps<p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.6,str(pos_coeff) + '***', fontsize=5,color=colors[1]) #pimo positive
    else:
        ax.text(xmax*0.6, ymax*0.6,str(pos_coeff), fontsize=5,color=colors[1])
        
    if (all_ps<p_thresh[0])&(all_ps>=p_thresh[1]):
        ax.text(xmax*0.6, ymax*0.7,str(all_coeff) + '*', fontsize=5,color='black') #all cells
    elif (all_ps<p_thresh[1])&(all_ps>=p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.7,str(all_coeff) + '**', fontsize=5,color='black') #all cells
    elif (all_ps<p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.7,str(all_coeff) + '***', fontsize=5,color='black') #all cells
    else:
        ax.text(xmax*0.6, ymax*0.7,str(all_coeff), fontsize=5,color='black')
        
       
    if (neg_ps<p_thresh[0])&(neg_ps>=p_thresh[1]):
        ax.text(xmax*0.6, ymax*0.8,str(neg_coeff) + '*', fontsize=5,color=colors[0]) #pimo negitive
    elif (neg_ps<p_thresh[1])&(neg_ps>=p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.8,str(neg_coeff) + '**', fontsize=5,color=colors[0]) #pimo negitive
    elif (neg_ps<p_thresh[2]):
        ax.text(xmax*0.6, ymax*0.8,str(neg_coeff) + '***', fontsize=5,color=colors[0]) #pimo negitive
    else:
        ax.text(xmax*0.6, ymax*0.8,str(neg_coeff), fontsize=5,color=colors[0]) #pimo negitive
        
pair.fig.set_figheight(20)
pair.fig.set_figwidth(20)
plt.tight_layout()
#%% Write pairplot
plt.savefig(os.path.join(figpath,'PairPlotV4_test.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')    
## TO DO:
    #Calculate P values from spearman correlation. spearmanr function already does this, just need to store output in a 2nd variable instead of an underscore
    #If P <0.05, add 1 * next to number plotted on pairplot. If p <0.01, add **. If <0.001, add ***. May change p thresholds depending on how it looks
    #Create a matrix for each column in spearmann_corr, reshaped to match the shape of the pairplot (should be len(measurements_of_interest) for both the length and height)
    #Make a seaborn.matrix plot, with each square's color intensity proportional to the spearmann coefficient. Write the coefficient inside the square and add * corresponding to p values, just as it was done for the pairplot.
    #In total, you should have 3 matrix plots: pimo positive, pimo negative, and all cells
    
#winsound.Beep(440, 1000)
plt.close()

#%% HEATMAPS
##heatmap dimensions:
    #use testvar_measures for the temp short data
    #use measurements_of_interest for real deal
PIMO_kwd='PIMO'
#PIMO_kwd='Pimo'

dims = len(testvar_measures)

#reshape the spearrmann correlations (each column in spearmann_corr) into 2D arrays to be used by seaborn heatmaps
PIMOneg_2D = np.reshape(PIMOneg_corr, (dims, dims))
PIMOpos_2D = np.reshape(PIMOpos_corr, (dims, dims))
allcorr_2D = np.reshape(all_corr, (dims, dims))

#Convert 2D heatmap arrays into dataframe with labelled axes
df_PIMOneg_2D = pandas.DataFrame(data=PIMOneg_2D,index=testvar_measures_short,columns=testvar_measures_short)
df_PIMOpos_2D = pandas.DataFrame(data=PIMOpos_2D,index=testvar_measures_short,columns=testvar_measures_short)
df_allcorr_2D = pandas.DataFrame(data=allcorr_2D,index=testvar_measures_short,columns=testvar_measures_short)
#stick PIMO to the start
temp=testvar_measures_short[:]
temp.remove(PIMO_kwd)
temp.insert(0, PIMO_kwd)
#reorder dataframes with PIMO at the start
df_allcorr_2D=df_allcorr_2D.reindex(temp)
df_allcorr_2D=df_allcorr_2D.reindex(columns=temp).sort_values(PIMO_kwd,ascending=False)
#now, sort the columns by the same way the rows were sorted
new_order=df_allcorr_2D.index
#OVERWRITE: DROP SPECIFIC GENES FROM HEATMAP
#new_order=new_order.drop(['GFAP','Tau','S100','GPG95','Nestin','NGFR','CD31','CD57','Ki67','Iba1','Synaptophysin','aSMA'])
#OVERWRITE: REINDEX BY ORDER PROVIDED BY SHEILA FROM FIGURE
#new_order=new_order.reindex(['PIMO','CA9','PD-L1','CD163','CD3','CD8a','CD11b','CD4','CD20','GZMB','pS6','CD45','CD45RO','CD68'])
new_order=new_order.drop(['ICAM','VCAM','aSMA','CD31'])
df_allcorr_2D=df_allcorr_2D.reindex(columns=new_order).reindex(index=new_order)
#apply this order to the df_PIMOneg_2D and df_PIMOpos_2D dataframes
df_PIMOneg_2D=df_PIMOneg_2D.reindex(columns=new_order).reindex(index=new_order)
df_PIMOpos_2D=df_PIMOpos_2D.reindex(columns=new_order).reindex(index=new_order)
#Write out corr .csv
df_allcorr_2D.to_csv(figpath + r'\all_corr.csv')
df_PIMOneg_2D.to_csv(figpath + r'\PIMOneg_corr.csv')
df_PIMOpos_2D.to_csv(figpath + r'\PIMOpos_corr.csv')
#%% repeat above, but for creating an array of the p values
#reshape the p values (each column in p_values) into 2D arrays to be used by seaborn heatmaps
p_PIMOneg_2D = np.reshape(PIMOneg_p, (dims, dims))
p_PIMOpos_2D = np.reshape(PIMOpos_p, (dims, dims))
p_allcorr_2D = np.reshape(all_p, (dims, dims))
#Convert 2D heatmap arrays into dataframe with labelled axes
p_df_PIMOneg_2D = pandas.DataFrame(data=p_PIMOneg_2D,index=testvar_measures_short,columns=testvar_measures_short)
p_df_PIMOpos_2D = pandas.DataFrame(data=p_PIMOpos_2D,index=testvar_measures_short,columns=testvar_measures_short)
p_df_allcorr_2D = pandas.DataFrame(data=p_allcorr_2D,index=testvar_measures_short,columns=testvar_measures_short)
#stick PIMO to the start
temp=testvar_measures_short[:]
temp.remove(PIMO_kwd)
temp.insert(0, PIMO_kwd)
#reorder dataframes with PIMO at the start              #This was commented because we want to use the original sort order of the spearman corr array, else p values will not match up to their respective spearman corr
# df_allcorr_2D=df_allcorr_2D.reindex(temp)
# df_allcorr_2D=df_allcorr_2D.reindex(columns=temp).sort_values(PIMO_kwd,ascending=False)
#now, sort the columns by the same way the rows were sorted
new_order=df_allcorr_2D.index
p_df_allcorr_2D=p_df_allcorr_2D.reindex(columns=new_order)
#apply this order to the df_PIMOneg_2D and df_PIMOpos_2D dataframes
p_df_PIMOneg_2D=p_df_PIMOneg_2D.reindex(columns=new_order).reindex(index=new_order)
p_df_PIMOpos_2D=p_df_PIMOpos_2D.reindex(columns=new_order).reindex(index=new_order)
p_df_allcorr_2D=p_df_allcorr_2D.reindex(columns=new_order).reindex(index=new_order)

p_df_allcorr_2D.to_csv(figpath + r'\all_p.csv')
p_df_PIMOneg_2D.to_csv(figpath + r'\PIMOneg_p.csv')
p_df_PIMOpos_2D.to_csv(figpath + r'\PIMOpos_p.csv')
#%% plot sorted heatmaps
sns.set(font_scale=1.5,font='Times New Roman')

fig, axes = plt.subplots(1, 3, figsize=(50,50))
#fig.suptitle('Heatmaps')
# PIMO negative
sns.heatmap(ax=axes[0],data=df_allcorr_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8,fmt='.2f')
axes[0].set_title('All')
axes[0].set_yticklabels(new_order,rotation=0)

# PIMO positive
sns.heatmap(ax=axes[1],data=df_PIMOneg_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8,fmt='.2f')
axes[1].set_title('PIMO Negative')
axes[1].set_yticklabels(new_order,rotation=0)

# All
sns.heatmap(ax=axes[2],data=df_PIMOpos_2D,annot=True, xticklabels=new_order, yticklabels=new_order,cmap='bwr',center=0,vmin=-0.8,vmax=0.8,fmt='.2f')
axes[2].set_title('PIMO Positive')
axes[2].set_yticklabels(new_order,rotation=0)
sns.set(font_scale=1.5,font='Times New Roman')

#write heatmaps
plt.tight_layout()
matplotlib.pyplot.pause(1)
plt.savefig(os.path.join(figpath,'Heatmaps_v2.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')
plt.close()
 