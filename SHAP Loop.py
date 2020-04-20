# %% Libraries
import shap
import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import Clean_and_Prepare as cp
import Model_Building_RF as rf

# %% Initiate DataFrames
coal_ash_al2o3 = pd.DataFrame()
slag_volume = pd.DataFrame()
bf4_flame_temp = pd.DataFrame()
coke_sulphur = pd.DataFrame()
flxd_perc_k2o = pd.DataFrame()
blast_moist = pd.DataFrame()
coke_ash_sio2 = pd.DataFrame()

# %% SHAP Loop
start_time = time.time()

x = 1
while x < 11:
    importlib.reload(cp)
    importlib.reload(rf)
    
    shap_values = pd.DataFrame(shap.TreeExplainer(rf.rf).shap_values(rf.cp.bf4r_inputs))
    shap_values.columns = rf.cp.bf4r_list
    
    tbf4m = rf.cp.bf4r_inputs.reset_index(drop=True)
           
    caa = pd.DataFrame({'In': tbf4m['coal_ash_al2o3'],
                       'SHAP': shap_values['coal_ash_al2o3']})
    coal_ash_al2o3 = coal_ash_al2o3.append(caa, ignore_index=True)

    sv = pd.DataFrame({'In': tbf4m['slag_volume'],
                       'SHAP': shap_values['slag_volume']})
    slag_volume = slag_volume.append(sv, ignore_index=True)
   
    bft = pd.DataFrame({'In': tbf4m['bf4_flame_temp'],
                       'SHAP': shap_values['bf4_flame_temp']})
    bf4_flame_temp = bf4_flame_temp.append(bft, ignore_index=True)
    
    cs = pd.DataFrame({'In': tbf4m['coke_sulphur'],
                       'SHAP': shap_values['coke_sulphur']})
    coke_sulphur = coke_sulphur.append(cs, ignore_index=True)
    
    fpk = pd.DataFrame({'In': tbf4m['flxd_perc_k2o'],
                       'SHAP': shap_values['flxd_perc_k2o']})
    flxd_perc_k2o = flxd_perc_k2o.append(fpk, ignore_index=True)
    
    bm = pd.DataFrame({'In': tbf4m['blast_moist'],
                       'SHAP': shap_values['blast_moist']})
    blast_moist = blast_moist.append(bm, ignore_index=True)
    
    cas = pd.DataFrame({'In': tbf4m['coke_ash_sio2'],
                       'SHAP': shap_values['coke_ash_sio2']})
    coke_ash_sio2 = coke_ash_sio2.append(cas, ignore_index=True)

    x += 1
    
print("--- %s seconds ---" % (time.time()-start_time))

# %% SHAP Value extraction
def pop_std(x):
    return x.std(ddof=0)

coal_ash_al2o3 = coal_ash_al2o3.round({'In': 4})
coal_ash_al2o3 = coal_ash_al2o3.groupby('In').agg({'SHAP': ['mean', pop_std]})
coal_ash_al2o3 = coal_ash_al2o3.reset_index(drop=False)
coal_ash_al2o3.columns = ['In','SHAP','Std Dev']

slag_volume = slag_volume.round({'In': 4})
slag_volume = slag_volume.groupby('In').agg({'SHAP': ['mean', pop_std]})
slag_volume = slag_volume.reset_index(drop=False)
slag_volume.columns = ['In','SHAP','Std Dev']

bf4_flame_temp = bf4_flame_temp.round({'In': 4})
bf4_flame_temp = bf4_flame_temp.groupby('In').agg({'SHAP': ['mean', pop_std]})
bf4_flame_temp = bf4_flame_temp.reset_index(drop=False)
bf4_flame_temp.columns = ['In','SHAP','Std Dev']

coke_sulphur = coke_sulphur.round({'In': 4})
coke_sulphur = coke_sulphur.groupby('In').agg({'SHAP': ['mean', pop_std]})
coke_sulphur = coke_sulphur.reset_index(drop=False)
coke_sulphur.columns = ['In','SHAP','Std Dev']

flxd_perc_k2o = flxd_perc_k2o.round({'In': 4})
flxd_perc_k2o = flxd_perc_k2o.groupby('In').agg({'SHAP': ['mean', pop_std]})
flxd_perc_k2o = flxd_perc_k2o.reset_index(drop=False)
flxd_perc_k2o.columns = ['In','SHAP','Std Dev']

blast_moist = blast_moist.round({'In': 4})
blast_moist = blast_moist.groupby('In').agg({'SHAP': ['mean', pop_std]})
blast_moist = blast_moist.reset_index(drop=False)
blast_moist.columns = ['In','SHAP','Std Dev']

coke_ash_sio2 = coke_ash_sio2.round({'In': 4})
coke_ash_sio2 = coke_ash_sio2.groupby('In').agg({'SHAP': ['mean', pop_std]})
coke_ash_sio2 = coke_ash_sio2.reset_index(drop=False)
coke_ash_sio2.columns = ['In','SHAP','Std Dev']

# %% ColourMap
fig, ax = plt.subplots(figsize=(6.77164, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.RdYlGn
cmap.set_under(color=(0.6470588235294118, 0.0, 0.14901960784313725))
cmap.set_over(color=(0.0, 0.40784313725490196, 0.21568627450980393))
norm = mpl.colors.Normalize(vmin=0, vmax=0.002)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal', label='SHAP Scale', extend='both')

# %% Force Plot
#explainer = shap.TreeExplainer(rf.rf)
#shap_values2 = explainer.shap_values(rf.cp.train_bf4_mdl)
shap.force_plot(explainer.expected_value, shap_values2[500,:], rf.cp.bf4r_inputs.iloc[500],
                show=False, matplotlib=True).savefig('force.png')

# %% Scatter Graphs
#cmap = mpl.cm.RdYlGn
#cmap.set_under(color=(0.6470588235294118, 0.0, 0.14901960784313725))
#cmap.set_over(color=(0.0, 0.40784313725490196, 0.21568627450980393))
#norm = mpl.colors.Normalize(vmin=0, vmax=0.002)
#
#plt.figure(2)
#plt.scatter(coal_ash_al2o3['In'], coal_ash_al2o3['SHAP'], c=coal_ash_al2o3['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Coal Ash Al2O3')
#plt.ylabel('SHAP Value')
#plt.title('Coal Ash Al2O3 SHAPs')
#plt.ylim(-0.003, 0.003)
#plt.colorbar(extend='both')
#
#plt.figure(3)
#plt.scatter(slag_volume['In'], slag_volume['SHAP'], c=slag_volume['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Slag Volume')
#plt.ylabel('SHAP Value')
#plt.title('Slag Volume SHAPs')
#plt.ylim(-0.004, 0.004)
#
#plt.figure(4)
#plt.scatter(bf4_flame_temp['In'], bf4_flame_temp['SHAP'], c=bf4_flame_temp['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('BF4 Flame Temp')
#plt.ylabel('SHAP Value')
#plt.title('BF4 Flame Temp SHAPs')
#plt.ylim(-0.004, 0.004)
#
#plt.figure(5)
#plt.scatter(coke_sulphur['In'], coke_sulphur['SHAP'], c=coke_sulphur['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Coke Sulphur')
#plt.ylabel('SHAP Value')
#plt.title('Coke Sulphur SHAPs')
#plt.ylim(-0.004, 0.004)
#
#plt.figure(6)
#plt.scatter(flxd_perc_k2o['In'], flxd_perc_k2o['SHAP'], c=flxd_perc_k2o['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Flxd Perc K2O')
#plt.ylabel('SHAP Value')
#plt.title('Flxd Perc K2O SHAPs')
#plt.ylim(-0.005, 0.002)
#
#plt.figure(7)
#plt.scatter(blast_moist['In'], blast_moist['SHAP'], c=blast_moist['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Blast Moist')
#plt.ylabel('SHAP Value')
#plt.title('Blast Moist SHAPs')
#plt.ylim(-0.005, 0.002)
#
#plt.figure(8)
#plt.scatter(coke_ash_sio2['In'], coke_ash_sio2['SHAP'], c=coke_ash_sio2['SHAP'], cmap=cmap, norm=norm)
#plt.grid(b=True, which='both', axis='both')
#plt.xlabel('Coke Ash SiO2')
#plt.ylabel('SHAP Value')
#plt.title('Coke Ash SiO2 SHAPs')
#plt.ylim(-0.005, 0.002)
#
#plt.figure(9)
#shap.summary_plot(shap_values, rf.cp.train_bf4_mdl, plot_type="bar")

# %% Scatter Chart dataFrames with applied criteria
caa = coal_ash_al2o3.loc[(coal_ash_al2o3['SHAP'] > 0.00075)]
sv = slag_volume.loc[(slag_volume['SHAP'] > 0.00075)]
bft = bf4_flame_temp.loc[(bf4_flame_temp['SHAP'] > 0.00075)]
cs = coke_sulphur.loc[(coke_sulphur['SHAP'] > 0.00075)]
fpk = flxd_perc_k2o.loc[(flxd_perc_k2o['SHAP'] > 0.00075)]
bm = blast_moist.loc[(blast_moist['SHAP'] > 0.00075)]
cas = coke_ash_sio2.loc[(coke_ash_sio2['SHAP'] > 0.00075)]

# %% Graphs
# First Figure
fig, ax = plt.subplots(4, 2, figsize=(6.771654, 9.9448819))

cmap = mpl.cm.RdYlGn
cmap.set_under(color=(0.6470588235294118, 0.0, 0.14901960784313725))
cmap.set_over(color=(0.0, 0.40784313725490196, 0.21568627450980393))
norm = mpl.colors.Normalize(vmin=0, vmax=0.002)
#fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[:, 0], orientation='vertical', shrink=0.4, label='SHAP Scale', extend='both')

# SHAP Scatter
ax[0, 0].scatter(coal_ash_al2o3['In'], coal_ash_al2o3['SHAP'], c=coal_ash_al2o3['SHAP'], cmap=cmap, norm=norm)
ax[0, 0].grid(b=True, which='both', axis='both')
ax[0, 0].set_xlabel('Coal Ash Al2O3')
ax[0, 0].set_ylabel('SHAP Value')
ax[0, 0].set_ylim(-0.003, 0.003)
ax[0, 0].set_title('(A1)')

ax[1, 0].scatter(slag_volume['In'], slag_volume['SHAP'], c=slag_volume['SHAP'], cmap=cmap, norm=norm)
ax[1, 0].grid(b=True, which='both', axis='both')
ax[1, 0].set_xlabel('Slag Volume')
ax[1, 0].set_ylabel('SHAP Value')
ax[1, 0].set_ylim(-0.0035, 0.0025)
ax[1, 0].set_title('(B1)')

ax[2, 0].scatter(bf4_flame_temp['In'], bf4_flame_temp['SHAP'], c=bf4_flame_temp['SHAP'], cmap=cmap, norm=norm)
ax[2, 0].grid(b=True, which='both', axis='both')
ax[2, 0].set_xlabel('BF4 Flame Temp')
ax[2, 0].set_ylabel('SHAP Value')
ax[2, 0].set_ylim(-0.002, 0.003)
ax[2, 0].set_title('(C1)')

ax[3, 0].scatter(coke_sulphur['In'], coke_sulphur['SHAP'], c=coke_sulphur['SHAP'], cmap=cmap, norm=norm)
ax[3, 0].grid(b=True, which='both', axis='both')
ax[3, 0].set_xlabel('Coke Sulphur')
ax[3, 0].set_ylabel('SHAP Value')
ax[3, 0].set_ylim(-0.004, 0.004)
ax[3, 0].set_title('(D1)')

# Criteria Scatter
avg_line = mpl.patches.Patch(color='steelblue', label='Avg SHAP Value')
stddev_line = mpl.lines.Line2D([], [], color='plum', label='SHAP Std Dev')

ax[0, 1].scatter(caa['In'], caa['SHAP'], color='steelblue')
ax[0, 1].grid(b=True, which='both', axis='both')
ax[0, 1].set_xlabel('Coal Ash Al2O3')
ax[0, 1].set_ylabel('SHAP Value')
ax[0, 1].set_ylim(0.00075, 0.002)
ax[0, 1].legend(handles=[avg_line, stddev_line])
ax[0, 1].set_title('(A2)')

ax[1, 1].scatter(sv['In'], sv['SHAP'], color='steelblue')
ax[1, 1].grid(b=True, which='both', axis='both')
ax[1, 1].set_xlabel('Slag Volume')
ax[1, 1].set_ylabel('SHAP Value')
ax[1, 1].set_ylim(0.00075, 0.0025)
ax[1, 1].legend(handles=[avg_line, stddev_line])
ax[1, 1].set_title('(B2)')

ax[2, 1].scatter(bft['In'], bft['SHAP'], color='steelblue')
ax[2, 1].grid(b=True, which='both', axis='both')
ax[2, 1].set_xlabel('BF4 Flame Temp')
ax[2, 1].set_ylabel('SHAP Value')
ax[2, 1].set_ylim(0.00075, 0.003)
ax[2, 1].legend(handles=[avg_line, stddev_line])
ax[2, 1].set_title('(C2)')

ax[3, 1].scatter(cs['In'], cs['SHAP'], color='steelblue')
ax[3, 1].grid(b=True, which='both', axis='both')
ax[3, 1].set_xlabel('Coke Sulphur')
ax[3, 1].set_ylabel('SHAP Value')
ax[3, 1].set_ylim(0.00075, 0.0035)
ax[3, 1].legend(handles=[avg_line, stddev_line])
ax[3, 1].set_title('(D2)')

# Std Dev Scatter
ax21 = ax[0, 1].twinx()
ax21.plot(caa['In'], caa['Std Dev'], color='plum')
ax21.set_ylabel('SHAP Std Dev')
ax21.set_ylim(0, 0.001)

ax22 = ax[1, 1].twinx()
ax22.plot(sv['In'], sv['Std Dev'], color='plum')
ax22.set_ylabel('SHAP Std Dev')
ax22.set_ylim(0, 0.001)

ax23 = ax[2, 1].twinx()
ax23.plot(bft['In'], bft['Std Dev'], color='plum')
ax23.set_ylabel('SHAP Std Dev')
ax23.set_ylim(0, 0.001)

ax24 = ax[3, 1].twinx()
ax24.plot(cs['In'], cs['Std Dev'], color='plum')
ax24.set_ylabel('SHAP Std Dev')
ax24.set_ylim(0, 0.001)

##############################################################################
##############################################################################
# Second Figure
fig1, ax1 = plt.subplots(3, 2, figsize=(6.771654, 7.458661425))

#fig1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1[:, 0], orientation='vertical', shrink=0.4, label='SHAP Scale', extend='both')

# SHAP Scatter
ax1[0, 0].scatter(flxd_perc_k2o['In'], flxd_perc_k2o['SHAP'], c=flxd_perc_k2o['SHAP'], cmap=cmap, norm=norm)
ax1[0, 0].grid(b=True, which='both', axis='both')
ax1[0, 0].set_xlabel('Flxd Perc K2O')
ax1[0, 0].set_ylabel('SHAP Value')
ax1[0, 0].set_ylim(-0.004, 0.002)
ax1[0, 0].set_title('(C1)')

ax1[1, 0].scatter(blast_moist['In'], blast_moist['SHAP'], c=blast_moist['SHAP'], cmap=cmap, norm=norm)
ax1[1, 0].grid(b=True, which='both', axis='both')
ax1[1, 0].set_xlabel('Blast Moist')
ax1[1, 0].set_ylabel('SHAP Value')
ax1[1, 0].set_ylim(-0.0045, 0.0015)
ax1[1, 0].set_title('(D1)')

ax1[2, 0].scatter(coke_ash_sio2['In'], coke_ash_sio2['SHAP'], c=coke_ash_sio2['SHAP'], cmap=cmap, norm=norm)
ax1[2, 0].grid(b=True, which='both', axis='both')
ax1[2, 0].set_xlabel('Coke Ash SiO2')
ax1[2, 0].set_ylabel('SHAP Value')
ax1[2, 0].set_ylim(-0.005, 0.002)
ax1[2, 0].set_title('(G1)')

# Criteria Scatter
ax1[0, 1].scatter(fpk['In'], fpk['SHAP'], color='steelblue')
ax1[0, 1].grid(b=True, which='both', axis='both')
ax1[0, 1].set_xlabel('Flxd Perc K2O')
ax1[0, 1].set_ylabel('SHAP Value')
ax1[0, 1].set_ylim(0.00075, 0.0018)
ax1[0, 1].legend(handles=[avg_line, stddev_line])
ax1[0, 1].set_title('(C2)')

ax1[1, 1].scatter(bm['In'], bm['SHAP'], color='steelblue')
ax1[1, 1].grid(b=True, which='both', axis='both')
ax1[1, 1].set_xlabel('Blast Moist')
ax1[1, 1].set_ylabel('SHAP Value')
ax1[1, 1].set_ylim(0.00075, 0.0014)
ax1[1, 1].legend(handles=[avg_line, stddev_line])
ax1[1, 1].set_title('(D2)')

ax1[2, 1].scatter(cas['In'], cas['SHAP'], color='steelblue')
ax1[2, 1].grid(b=True, which='both', axis='both')
ax1[2, 1].set_xlabel('Coke Ash SiO2')
ax1[2, 1].set_ylabel('SHAP Value')
ax1[2, 1].set_ylim(0.00075, 0.0014)
ax1[2, 1].legend(handles=[avg_line, stddev_line])
ax1[2, 1].set_title('(G2)')

# Std Dev Scatter
ax31 = ax1[0, 1].twinx()
ax31.plot(fpk['In'], fpk['Std Dev'], color='plum')
ax31.set_ylabel('SHAP Std Dev')
ax31.set_ylim(0, 0.001)

ax32 = ax1[1, 1].twinx()
ax32.plot(bm['In'], bm['Std Dev'], color='plum')
ax32.set_ylabel('SHAP Std Dev')
ax32.set_ylim(0, 0.001)

ax33 = ax1[2, 1].twinx()
ax33.plot(cas['In'], cas['Std Dev'], color='plum')
ax33.set_ylabel('SHAP Std Dev')
ax33.set_ylim(0, 0.001)

plt.show()

# %% Suggested Ranges
minmax = {'Coal Ash Al2O3': [coal_ash_al2o3.loc[(coal_ash_al2o3['SHAP'] > 0.00075)]['In'].min(), coal_ash_al2o3.loc[(coal_ash_al2o3['SHAP'] > 0.00075)]['In'].max()],
          'Slag Volume': [slag_volume.loc[(slag_volume['SHAP'] > 0.00075)]['In'].min(), slag_volume.loc[(slag_volume['SHAP'] > 0.00075)]['In'].max()],
          'Flxd Perc K2O': [flxd_perc_k2o.loc[(flxd_perc_k2o['SHAP'] > 0.00075)]['In'].min(), flxd_perc_k2o.loc[(flxd_perc_k2o['SHAP'] > 0.00075)]['In'].max()],                          
          'Blast Moist': [blast_moist.loc[(blast_moist['SHAP'] > 0.00075)]['In'].min(), blast_moist.loc[(blast_moist['SHAP'] > 0.00075)]['In'].max()],
          'Coke Sulphur': [coke_sulphur.loc[(coke_sulphur['SHAP'] > 0.00075)]['In'].min(), coke_sulphur.loc[(coke_sulphur['SHAP'] > 0.00075)]['In'].max()],                                      
          'BF4 Flame Temp': [bf4_flame_temp.loc[(bf4_flame_temp['SHAP'] > 0.00075)]['In'].min(), bf4_flame_temp.loc[(bf4_flame_temp['SHAP'] > 0.00075)]['In'].max()],
          'Coke Ash SiO2': [coke_ash_sio2.loc[(coke_ash_sio2['SHAP'] > 0.00075)]['In'].min(), coke_ash_sio2.loc[(coke_ash_sio2['SHAP'] > 0.00075)]['In'].max()]
          }
          
ranges = pd.DataFrame(minmax, index=['Min', 'Max']).T
ranges.to_excel("ranges.xlsx") 

 
