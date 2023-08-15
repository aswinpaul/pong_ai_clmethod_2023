#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


#%% Data from simulations

with open('data_clmethod_1_M3', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M3', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)
    
mean_1 = []
for i in range(len(score_length_1)):
    p = np.count_nonzero(score_length_1[i] == 0)
    p /= len(score_length_1[i])
    p *= 100
    mean_1.append(p)
    
mean_2 = []
for i in range(len(score_length_2)):
    p = np.count_nonzero(score_length_2[i] == 0)
    p /= len(score_length_2[i])
    p *= 100
    mean_2.append(p)
    
x = [mean_1, mean_2]

with open('data_clmethod_1_M7', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M7', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

mean_1 = []
for i in range(len(score_length_1)):
    p = np.count_nonzero(score_length_1[i] == 0)
    p /= len(score_length_1[i])
    p *= 100
    mean_1.append(p)
    
mean_2 = []
for i in range(len(score_length_2)):
    p = np.count_nonzero(score_length_2[i] == 0)
    p /= len(score_length_2[i])
    p *= 100
    mean_2.append(p)
    
y = [mean_1, mean_2]

with open('data_clmethod_1_M12', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M12', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)
    
mean_1 = []
for i in range(len(score_length_1)):
    p = np.count_nonzero(score_length_1[i] == 0)
    p /= len(score_length_1[i])
    p *= 100
    mean_1.append(p)
    
mean_2 = []
for i in range(len(score_length_2)):
    p = np.count_nonzero(score_length_2[i] == 0)
    p /= len(score_length_2[i])
    p *= 100
    mean_2.append(p)
    
z = [mean_1, mean_2]

#%% Data from paper

df = pd.read_csv('aces_data_kagan22.csv', sep=',',header=None)

mcc_1 = np.array(df[1][2:8])
mcc_2 = np.array(df[3][2:8])
hcc_1 = np.array(df[5][2:8])
hcc_2 = np.array(df[7][2:8])
is_1 = np.array(df[9][2:8])
is_2 = np.array(df[11][2:8])
     
min_val_1 = float(mcc_1[0])
p10_1 = float(mcc_1[1])
median_1 = float(mcc_1[2])
average_1 = float(mcc_1[3])
p90_1 = float(mcc_1[4])
max_val_1 = float(mcc_1[5])

min_val_2 = float(mcc_2[0])
p10_2 = float(mcc_2[1])
median_2 = float(mcc_2[2])
average_2 = float(mcc_2[3])
p90_2 = float(mcc_2[4])
max_val_2 = float(mcc_2[5])

min_val_3 = float(hcc_1[0])
p10_3 = float(hcc_1[1])
median_3 = float(hcc_1[2])
average_3 = float(hcc_1[3])
p90_3 = float(hcc_1[4])
max_val_3 = float(hcc_1[5])

min_val_4 = float(hcc_2[0])
p10_4 = float(hcc_2[1])
median_4 = float(hcc_2[2])
average_4 = float(hcc_2[3])
p90_4 = float(hcc_2[4])
max_val_4 = float(hcc_2[5])

min_val_5 = float(is_1[0])
p10_5 = float(is_1[1])
median_5 = float(is_1[2])
average_5 = float(is_1[3])
p90_5 = float(is_1[4])
max_val_5 = float(is_1[5])

min_val_6 = float(is_2[0])
p10_6 = float(is_2[1])
median_6 = float(is_2[2])
average_6 = float(is_2[3])
p90_6 = float(is_2[4])
max_val_6 = float(is_2[5])

stats1 = [{'mean': average_1, 'med': median_1, 'q1': p10_1, 'q3': p90_1, 'whislo': min_val_1, 'whishi': max_val_1},
          {'mean': average_2, 'med': median_2, 'q1': p10_2, 'q3': p90_2, 'whislo': min_val_2, 'whishi': max_val_2},
          {'mean': average_3, 'med': median_3, 'q1': p10_3, 'q3': p90_3, 'whislo': min_val_3, 'whishi': max_val_3},
          {'mean': average_4, 'med': median_4, 'q1': p10_4, 'q3': p90_4, 'whislo': min_val_4, 'whishi': max_val_4},
          {'mean': average_5, 'med': median_5, 'q1': p10_5, 'q3': p90_5, 'whislo': min_val_5, 'whishi': max_val_5},
          {'mean': average_6, 'med': median_6, 'q1': p10_6, 'q3': p90_6, 'whislo': min_val_6, 'whishi': max_val_6}]



# %% Box Plot

#%% Plotting

# Paper 
fig, ax = plt.subplots()
p = [0.3,0.7, 1.3, 1.7, 2.3, 2.7]
ax.bxp(stats1, showfliers=False, showmeans=True, positions=p)


# Simulations
# xy = [x[0], x[1], y[0], y[1], z[0], z[1], zz[0], zz[1]]
xy = [x[0], x[1], y[0], y[1], z[0], z[1]]
p1 = [3.5, 4.0, 4.9, 5.4, 6.4, 6.9]

ax.boxplot(xy, showfliers=False, showmeans=True, positions=p1)
ax.set_xticks([0.5, 1.5, 2.5, 3.75, 5.25, 6.65])
ax.set_xticklabels(['MCC', 'HCC', 'IS', 'CL(3)', 'CL(7)', 'CL(12)'])


ax.set_ylabel("% Aces")
ax.set_title("Game play performance over time")
#ax.set_ylim((0.2,1.8))
ax.set_xlim((-0,7.5))
line1 = ax.hlines(y=average_4, xmin=0, xmax=8.8, linestyles = '--', 
                  color='black', label = 'HCC level')
line2 = ax.hlines(y=average_6, xmin=0, xmax=8.8, linestyles = '--', 
                  color='red', label = 'Random (IS) level')
ax.legend(handles=[line1, line2])

fig.savefig('graph_2.png', dpi=500, bbox_inches='tight')

#%% T-tests

cl_t_test_3 = scipy.stats.ttest_ind(x[0], x[1])
cl_t_test_7 = scipy.stats.ttest_ind(y[0], y[1])
cl_t_test_12 = scipy.stats.ttest_ind(z[0], z[1])

print("t-test cl-3", cl_t_test_3)
print("t-test cl-7", cl_t_test_7)
print("t-test cl-12", cl_t_test_12)