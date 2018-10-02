
# coding: utf-8

# In[185]:


#Import modules needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from scipy import optimize
from sklearn import metrics
from matplotlib import rc
rc('mathtext', default='regular')
get_ipython().run_line_magic('matplotlib', 'inline')

def dataset_func():
    dataset = []
    for f in os.listdir(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master'):
        data = np.genfromtxt(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master\%s' % (f,))
        data = data.astype(str)
        data = np.insert(data,0,f[:-4],axis=0)
        dataset.append(data)
    dataset = pd.DataFrame(dataset)
    dataset = dataset.T
    for index in dataset.columns:
        dataset[dataset[index][0]] = dataset[index]
        del dataset[index]
    dataset = dataset.drop(0)
    dataset = dataset.astype('float')
    return dataset
def func_guass(r,a,b,c,d):
    return a + b * np.exp(-c * (r-d)**2)
def func_exp(r,a,b):
    return a * np.exp(-b * r)
def curve_fit_guass(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_guass,xdata,ydata,(0,0,0,0))
    (a,b,c,d) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_guass(xdata,a,b,c,d) #generate function with shifted back range
    return a,b,c,d,pcovt
def func_lin(r,a,b):
    return a*r+b
def curve_fit_lin(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_lin,xdata,ydata,(0,0))
    (a,b) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_lin(xdata,a,b) #generate function with shifted back range
    return a,b,pcovt
def curve_fit_exp(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_exp,xdata,ydata,(0,0))
    (a,b) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_exp(xdata,a,b) #generate function with shifted back range
    return a,b,pcovt
def chi_sq(data_1, data_true):
    x=[]
    bin = len(data_1)
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/data_true[i]
        x.append(z)
    s = sum(x)/(bin-1)
    if s > 1.:
        print('Bad fit: Chi_sq = %f' % (s,))
    if np.isclose(s,1.):
        print('Okay fit: Chi_sq = %f' % (s,))
    if s < 1.:
        print('Great fit: Chi_sq = %f' % (s,))
def sigma(data):
    x = []
    N = len(data)
    x_bar = sum(data)/len(data)
    for i in data.index:
        d = (data[i] - x_bar)**2
        x.append(d)
    return np.sqrt(sum(x)/(N-1))
def func_quad(r,a,b,c):
    return a*r**2 + b*r + c
def curve_fit_quad(xdata, ydata):
    popt, pcovt = optimize.curve_fit(func_quad,xdata,ydata,(0,0,0), sigma = np.sqrt(ydata))
    (a,b,c) = popt
    #print('a =',a,'b =',b,'c=',c,'d=',d)
    func = func_lin(xdata,a,b) #generate function with shifted back range
    return a,b,c,pcovt
def chi_sq_sig(data_1, data_true, sigma, bin):
    x=[]
    for i in range(len(data_1)):
        z = (data_1[i] - data_true[i])**2/sigma[i]
        x.append(z)
    chi2 = sum(x)/(bin-1)
    if chi2 > 1.0:
        print('Bad fit: %lf' % (chi2,))
    elif np.isclose(chi2,1.0,0.1):
        print('Okay fit %lf' % (chi2,))
    else :
        print('Great fit: %lf' % (chi2,))
        return chi2
def std_err(sigma, bin):
    return sigma/np.sqrt(bin)


# In[186]:


dataset = dataset_func()
#decay = pd.read_csv('~/Documents/SLab/Data/Lab_1_data/MASTER.csv', sep=',', header=None)
#decay = pd.DataFrame(decay)
#decay.columns
#canidates = decay['Isotope'].unique()

cal_d = pd.read_csv(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\calibration.csv', sep=',', header=None)
cal_d[0] = cal_d[0].drop(0)
cal_d[0] = cal_d[0].drop(8)
cal_d[0] = cal_d[0].drop(15)
cal_d[0] = cal_d[0].drop(18)
cal_d[0] = cal_d[0].dropna()

cal_d[0]=cal_d[0].astype(float) #post del isotopes from column
#calibration_data column 0 = Energy


# In[187]:


min = 0
max = 1024
dataset['channel'] = pd.DataFrame(np.linspace(min,max,1024))
noise = dataset['BKG_1']
for index in dataset.columns:
    dataset[index + ' pure signal'] = dataset[index] - noise
    dataset[index + ' pure signal'] = dataset[index + ' pure signal'].replace(0.0, np.nan)
del dataset['BKG_1 pure signal']
del dataset['BKG_2 pure signal']
del dataset['channel pure signal']
dataset = dataset.dropna()
cal_d[0] = cal_d.dropna()


# In[188]:


dataset


# plt.figure()
# plt.plot(dataset['Energy'][:200], dataset['Bi_207 pure signal'][:200], c = 'r',label= 'Bi-207')
# plt.plot(dataset['Energy'][:220], dataset['Cs_137 pure signal'][:220], c = 'b',label= 'Cs-137')
# plt.plot(dataset['Energy'][:220], dataset['Co_60 pure signal'][:220], c = 'g',label= 'Co-60')
# plt.plot(dataset['Energy'][:220], dataset['Ra_226 pure signal'][:220], c = 'k',label= 'Ra-226')
# plt.plot(dataset['Energy'][:220], dataset['Ore_A pure signal'][:220], c = 'c',label= 'Unknown Sample A')
# plt.legend()
# plt.ylim(0,5000)
# plt.xlabel('Energy (keV)')
# plt.ylabel('Counts per Energy')
# #plt.title('Signals for Unknown sample Ore A plotted with known samples')
# #plt.savefig('/Users/lucas/Documents/SLab/SampleA.png')

# In[189]:


plt.figure()
plt.plot(dataset['Energy'][100:220], dataset['Ra_226 pure signal'][100:220], c = 'k',label= 'Ra-226')
plt.plot(dataset['Energy'][:220], dataset['Ore_A pure signal'][:220], c = 'b',label= 'Unknown Sample A')
plt.legend()
plt.ylim(0,5000)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts per Energy')
#plt.title('Signals for Unknown sample Ore A plotted with known samples')
#plt.savefig('/Users/lucas/Documents/SLab/SampleA_Ra226.png')


# In[190]:


#Ra226 6 k [8:16],[29:37],[37:43],[43:53],[75:95]
#Bi207 3 r [7:15],[70:90],[140:172]
#Cs137 2 b [:8],[8:15],[80:108]
#Co60 2 g [162:188],[188:220]
plt.plot(dataset['Energy'][75:95], dataset['Ra_226 pure signal'][75:95], c = 'k',label= 'Ra-226')


# In[191]:


#9/21/18
#to do
#get calibrations==done?, curvfit, get sigma== done, chi2 for lin fit==done?, error,
#Attenuation
################################################################################
#Slice the data and get the local maximums for each signal
ra2265 = dataset['Ra_226 pure signal'][75:95]
ra2264 = dataset['Ra_226 pure signal'][41:53]
ra2263 = dataset['Ra_226 pure signal'][37:43]
ra2262 = dataset['Ra_226 pure signal'][29:37]
ra2261 = dataset['Ra_226 pure signal'][8:16]

bi2073 = dataset['Bi_207 pure signal'][140:172]
bi2072 = dataset['Bi_207 pure signal'][70:90]
bi2071 = dataset['Bi_207 pure signal'][7:15]

cs1373 = dataset['Cs_137 pure signal'][80:108]
cs1372 = dataset['Cs_137 pure signal'][8:15]
cs1371 = dataset['Cs_137 pure signal'][:8]

co602 = dataset['Co_60 pure signal'][188:220]
co601 = dataset['Co_60 pure signal'][162:188]


# In[192]:


#Get the sigma for each peak
sra1 = sigma(ra2261)
sra2 = sigma(ra2262)
sra3 = sigma(ra2263)
sra4 = sigma(ra2264)
sra5 = sigma(ra2265)
sbi1 = sigma(bi2071)
sbi2 = sigma(bi2072)
sbi3 = sigma(bi2073)
scs1 = sigma(cs1371)
scs2 = sigma(cs1372)
scs3 = sigma(cs1373)
sco1 = sigma(co601)
sco2 = sigma(co602)

sigma_cal = np.array([scs3,sco1,sco2,sbi2,sbi3,sra1,sra2,sra3,sra4,sra5])


# In[193]:


#get the max for each peak
max_ra226_1 = np.max(ra2261)
max_ra226_2 = np.max(ra2262)
max_ra226_3 = np.max(ra2263)
max_ra226_4 = np.max(ra2264)
max_ra226_5 = np.max(ra2265)


max_bi207_1 = np.max(bi2071)
max_bi207_2 = np.max(bi2072)
max_bi207_3 = np.max(bi2073)

max_cs137_1 = np.max(cs1371)
max_cs137_2 = np.max(cs1372)
max_cs137_3 = np.max(cs1373)

max_co60_1 = np.max(co601)
max_co60_2 = np.max(co602)


# In[194]:


#Find which channel those maximums occur at
cs1 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_1)[0])
cs2 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_2)[0])
cs3 = int(np.where(dataset['Cs_137 pure signal'] == max_cs137_3)[0])

co1 = int(np.where(dataset['Co_60 pure signal'] == max_co60_1)[0][0])
co2 = int(np.where(dataset['Co_60 pure signal'] == max_co60_2)[0][0])

bi1 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_1)[0])
bi2 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_2)[0])
bi3 = int(np.where(dataset['Bi_207 pure signal'] == max_bi207_3)[0])

ra1 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_1)[0])
ra2 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_2)[0])
ra3 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_3)[0])
ra4 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_4)[0])
ra5 = int(np.where(dataset['Ra_226 pure signal'] == max_ra226_5)[0])

dataset['Co_60 pure signal'][175]
dataset['Co_60 pure signal'][203]
dataset['Cs_137 pure signal'][93]
dataset['Bi_207 pure signal'][79]
dataset['Bi_207 pure signal'][158]
dataset['Ra_226 pure signal'][12]
dataset['Ra_226 pure signal'][34]
dataset['Ra_226 pure signal'][41]
dataset['Ra_226 pure signal'][48]
dataset['Ra_226 pure signal'][85]

dataset['Energy'][175]
dataset['Energy'][203]
dataset['Energy'][93]
dataset['Energy'][79]
dataset['Energy'][158]
dataset['Energy'][12]
dataset['Energy'][34]
dataset['Energy'][41]
dataset['Energy'][48]
dataset['Energy'][85]


# In[195]:


#Build the arrays for calibrations, bin is the array with the channel No where each peak was measured,
#the Energy is the accepted data from ENSDF Nuclear data sheets.
bins = np.array([cs3,co1,co2,bi2,bi3,ra1,ra2,ra3,ra4,ra5])+1
Energy = np.array([cal_d[0][17],cal_d[0][21],cal_d[0][22], cal_d[0][10],cal_d[0][12], cal_d[0][2],cal_d[0][4],cal_d[0][5],cal_d[0][6],cal_d[0][7]])
#cal_d
#error = np.array([1.0,2.987,13.421, 18.118])
#curvfit with lin function to find the parameters
scale = curve_fit_lin(bins,Energy)
#apply the scale to bin to find where they hit
E = scale[0]*bin + scale[1]
#get the uncertainty in the parameters
err = np.sqrt(np.diag(scale[2]))
t = np.linspace(0,230,1024)
y = scale[0] * t + scale[1]

#plot
plt.scatter(bins,Energy,label = 'Measured Values', c='r')
plt.plot(t,y,label = 'Fitted Values')
plt.xlabel('Channel Number')
plt.ylabel('Energy (keV)')
plt.title('Calibration of MCA')
plt.savefig('Calibration of MCA.png')
plt.legend()
#plt.text(150,200, "$Chi^{2}$ = %f" % (chi2s,))
#########plt.savefig('/Users/lucas/Documents/SLab/Data/Lab_1_data/Calibration/Calibration of MCA.png')
#get chi2 with sigma
#chi2s = chi_sq_sig(ypt, Energy, sigma_cal, len(Energy))
#err


# In[196]:


#err =array([ 0.17565474, 19.57569468])
#build the Energy column in dataset
bins = np.linspace(0,1024,1024)
lst = scale[0]*bins + scale[1]
dataset['Energy'] = pd.DataFrame(lst)


# In[197]:


#chi_sq
vals = [93,175,203,79,158,12,34,48]
measured_E = []
for index in vals:
    measured_E.append(dataset['Energy'][index])
dataset['Energy']
measured_E
Energy

chi_sq(measured_E, Energy)
#x = dataset['Energy'][4] - dataset['Energy'][3]

#each channel covers 6.715586093213975  keV
array([ 0.17565474, 19.57569468])
dm =0.17565474
db = 19.57569468
x = 5.793904113376704
dx = 1./np.sqrt(1024.)
b = 4.547237072688481
dE = np.sqrt(dm**2 + (db*x)**2 + (dx*b)**2)
dE


dataset[70:120]

#plot evereything to see it makes sense
plt.plot(dataset['Energy'][:170], dataset['Cs_137 pure signal'][:170], c = 'r',label= 'Bi-207')
plt.plot(dataset['Energy'][:170], dataset['Bi_207 pure signal'][:170], c = 'b',label= 'Cs-137')
plt.plot(dataset['Energy'][:170], dataset['Co_60 pure signal'][:170], c = 'g',label= 'Co-60')
plt.plot(dataset['Energy'][:170], dataset['Ra_226 pure signal'][:170], c = 'k',label= 'Ra-226')
plt.legend()
plt.ylim(0,4000)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')


# In[198]:


dataset['Energy']


# In[199]:


dataset.columns


# In[200]:


################################################################################
################################################################################
#curvfit portion

#To be fitted

bi2071 = dataset['Bi_207 pure signal'][65:82]

bi2071am = dataset['Bi_207A_med pure signal'][65:82]

bi2071as = dataset['Bi_207A_small pure signal'][65:82]



cs1371 = dataset['Cs_137 pure signal'][80:108]

cs1371am = dataset['Cs_137A_med pure signal'][80:108]

cs1371as = dataset['Cs_137A_small pure signal'][80:108]



co601 = dataset['Co_60 pure signal'][165:188]
co602 = dataset['Co_60 pure signal'][150:167]

co601am = dataset['Co_60A_med pure signal'][165:188]
co602am = dataset['Co_60A_med pure signal'][150:167]

co601as = dataset['Co_60A_small pure signal'][165:188]
co602as = dataset['Co_60A_small pure signal'][150:167]


# In[201]:


bi2071


# In[202]:



dataset['Energy'][93]


# In[203]:


#Bi207 1
popt_bi1, pcovt_bi1 = optimize.curve_fit(func_guass,dataset['Energy'][65:82],bi2071[:],(400.,3000.,0.002,620.))
err_bi1 = np.sqrt(np.diag(pcovt_bi1))
y_bi1 = func_guass(dataset['Energy'],*popt_bi1)
sigma_bi1 = np.sqrt(1/(2*popt_bi1[2]))
plt.plot(dataset['Energy'][65:82],bi2071)
plt.plot(dataset['Energy'][65:82],y_bi1[65:82])
err_bi1
sigma_bi1


# #Bi207 2
# popt_bi2, pcovt_bi2 = optimize.curve_fit(func_guass,dataset['Energy'][70:90],bi2072[:],(450.,4500.,0.002,590.))
# err_bi2 = np.sqrt(np.diag(pcovt_bi2))
# y_bi2 = func_guass(dataset['Energy'],*popt_bi2)
# sigma_bi2 = np.sqrt(1/(2*popt_bi2[2]**2))
# plt.plot(dataset['Energy'][70:90],bi2072)
# plt.plot(dataset['Energy'][60:100],y_bi2[60:100])
# err_bi2
# sigma_bi2
# popt_bi2

# In[207]:


#Cs137 1
popt_cs1, pcovt_cs1 = optimize.curve_fit(func_guass,dataset['Energy'][80:108],cs1371,(600.,2000.,0.002,670))
err_cs1 = np.sqrt(np.diag(pcovt_cs1))
y_cs1 = func_guass(dataset['Energy'],*popt_cs1)
sigma_cs1 = np.sqrt(1/(2*popt_cs1[2]))
plt.plot(dataset['Energy'][80:108],cs1371)
plt.plot(dataset['Energy'][50:140],y_cs1[50:140])
err_cs1
sigma_cs1


# oa= dataset['Co_60 pure signal']
# E= dataset['Energy']
# plt.plot(E[150:167],oa[150:167])

# In[205]:


#Co60 1
popt_co1, pcovt_co1 = optimize.curve_fit(func_guass,dataset['Energy'][165:188],co601,(0.,400.,0.002,1500))
err_co1 = np.sqrt(np.diag(pcovt_co1))
y_co1 = func_guass(dataset['Energy'],*popt_co1)
sigma_co1 = np.sqrt(1/(2*popt_co1[2]))
plt.plot(dataset['Energy'][165:188],co601)
plt.plot(dataset['Energy'][165:188],y_co1[165:188])
err_co1
sigma_co1


# In[206]:


#Co60 2
popt_co2, pcovt_co2 = optimize.curve_fit(func_guass,dataset['Energy'][150:167],co602,(50.,500.,0.002,1300))
err_co2 = np.sqrt(np.diag(pcovt_co2))
y_co2 = func_guass(dataset['Energy'],*popt_co2)
sigma_co2 = np.sqrt(1/(2*popt_co2[2]))
plt.plot(dataset['Energy'][150:167],co602)
plt.plot(dataset['Energy'][150:167],y_co2[150:167])
err_co2
sigma_co2


# In[135]:


#Bi207am 1
popt_bi1, pcovt_bi1 = optimize.curve_fit(func_guass,dataset['Energy'][65:82],bi2071am[:],(400.,3000.,0.002,620.))
err_bi1 = np.sqrt(np.diag(pcovt_bi1))
y_bi1am = func_guass(dataset['Energy'],*popt_bi1)
sigma_bi1 = np.sqrt(1/(2*popt_bi1[2]**2))
plt.plot(dataset['Energy'][65:82],bi2071)
plt.plot(dataset['Energy'][65:82],y_bi1[65:82])
err_bi1
sigma_bi1
popt_bi1


# In[136]:


#Bi207am 2
popt_bi2, pcovt_bi2 = optimize.curve_fit(func_guass,dataset['Energy'][70:90],bi2072am[:],(450.,4500.,0.002,590.))
err_bi2 = np.sqrt(np.diag(pcovt_bi2))
y_bi2am = func_guass(dataset['Energy'],*popt_bi2)
sigma_bi2 = np.sqrt(1/(2*popt_bi2[2]**2))
plt.plot(dataset['Energy'][70:90],bi2072)
plt.plot(dataset['Energy'][60:100],y_bi2[60:100])
err_bi2
sigma_bi2
popt_bi2


# In[137]:


#Cs137am 1
popt_cs1, pcovt_cs1 = optimize.curve_fit(func_guass,dataset['Energy'][80:108],cs1371am,(600.,2000.,0.002,670))
err_cs1 = np.sqrt(np.diag(pcovt_cs1))
y_cs1am = func_guass(dataset['Energy'],*popt_cs1)
sigma_cs1 = np.sqrt(1/(2*popt_cs1[2]**2))
plt.plot(dataset['Energy'][80:108],cs1371)
plt.plot(dataset['Energy'][50:140],y_cs1[50:140])
err_cs1
sigma_cs1
popt_cs1


# In[138]:


#Co60am 1
popt_co1, pcovt_co1 = optimize.curve_fit(func_guass,dataset['Energy'][165:188],co601am,(0.,400.,0.002,1500))
err_co1 = np.sqrt(np.diag(pcovt_co1))
y_co1am = func_guass(dataset['Energy'],*popt_co1)
sigma_co1 = np.sqrt(1/(2*popt_co1[2]**2))
plt.plot(dataset['Energy'][165:188],co601)
plt.plot(dataset['Energy'][165:188],y_co1[165:188])
err_co1
sigma_co1
popt_co1


# In[139]:


#Co60am 2
popt_co2, pcovt_co2 = optimize.curve_fit(func_guass,dataset['Energy'][150:167],co602am,(50.,500.,0.002,1300))
err_co2 = np.sqrt(np.diag(pcovt_co2))
y_co2am = func_guass(dataset['Energy'],*popt_co2)
sigma_co2 = np.sqrt(1/(2*popt_co2[2]**2))
plt.plot(dataset['Energy'][150:167],co602)
plt.plot(dataset['Energy'][150:167],y_co2[150:167])
err_co2
sigma_co2
popt_co2


# In[140]:


#Bi207as 1
popt_bi1, pcovt_bi1 = optimize.curve_fit(func_guass,dataset['Energy'][65:82],bi2071as,(400.,3000.,0.002,620.))
err_bi1 = np.sqrt(np.diag(pcovt_bi1))
y_bi1as = func_guass(dataset['Energy'],*popt_bi1)
sigma_bi1 = np.sqrt(1/(2*popt_bi1[2]**2))
plt.plot(dataset['Energy'][65:82],bi2071)
plt.plot(dataset['Energy'][65:82],y_bi1[65:82])
err_bi1
sigma_bi1
popt_bi1


# In[141]:


#Bi207as 2
popt_bi2, pcovt_bi2 = optimize.curve_fit(func_guass,dataset['Energy'][70:90],bi2072as,(450.,4500.,0.002,590.))
err_bi2 = np.sqrt(np.diag(pcovt_bi2))
y_bi2as = func_guass(dataset['Energy'],*popt_bi2)
sigma_bi2 = np.sqrt(1/(2*popt_bi2[2]**2))
plt.plot(dataset['Energy'][70:90],bi2072)
plt.plot(dataset['Energy'][60:100],y_bi2[60:100])
err_bi2
sigma_bi2
popt_bi2


# In[142]:


#Cs137as 1
popt_cs1, pcovt_cs1 = optimize.curve_fit(func_guass,dataset['Energy'][80:108],cs1371as,(600.,2000.,0.002,670))
err_cs1 = np.sqrt(np.diag(pcovt_cs1))
y_cs1as = func_guass(dataset['Energy'],*popt_cs1)
sigma_cs1 = np.sqrt(1/(2*popt_cs1[2]**2))
plt.plot(dataset['Energy'][80:108],cs1371)
plt.plot(dataset['Energy'][50:140],y_cs1[50:140])
err_cs1
sigma_cs1
popt_cs1


# In[143]:


#Co60as 1
popt_co1, pcovt_co1 = optimize.curve_fit(func_guass,dataset['Energy'][165:188],co601as,(0.,400.,0.002,1500))
err_co1 = np.sqrt(np.diag(pcovt_co1))
y_co1as = func_guass(dataset['Energy'],*popt_co1)
sigma_co1 = np.sqrt(1/(2*popt_co1[2]**2))
plt.plot(dataset['Energy'][165:188],co601)
plt.plot(dataset['Energy'][165:188],y_co1[165:188])
err_co1
sigma_co1
popt_co1


# In[144]:


#Co60as 2
popt_co2, pcovt_co2 = optimize.curve_fit(func_guass,dataset['Energy'][150:167],co602as,(50.,500.,0.002,1300))
err_co2 = np.sqrt(np.diag(pcovt_co2))
y_co2as = func_guass(dataset['Energy'],*popt_co2)
sigma_co2 = np.sqrt(1/(2*popt_co2[2]**2))
plt.plot(dataset['Energy'][150:167],co602)
plt.plot(dataset['Energy'][150:167],y_co2[150:167])
err_co2
sigma_co2
popt_co2


# In[145]:


#Find max of gauss fit

max_Bi207_1 = np.max(y_bi1)
max_Bi207_2 = np.max(y_bi2)
max_Cs137_1 = np.max(y_cs1)
max_Co60_1 = np.max(y_co1)
max_Co60_2 = np.max(y_co1)

max_Bi207_1am = np.max(y_bi1am)
max_Bi207_2am = np.max(y_bi2am)
max_Cs137_1am = np.max(y_cs1am)
max_Co60_1am = np.max(y_co1am)
max_Co60_2am = np.max(y_co1am)

max_Bi207_1as = np.max(y_bi1as)bmax_Bi207_2as = np.max(y_bi2as)
max_Cs137_1as = np.max(y_cs1as)
max_Co60_1as = np.max(y_co1as)
max_Co60_2as = np.max(y_co1as)


# In[ ]:


Bi1


# In[ ]:


popt_ra4, pcovt_ra4 = optimize.curve_fit(func_guass,dataset['Energy'][29:37],ra2264,(1000.,1400.,0.002,320.))
err_ra4 = np.sqrt(np.diag(pcovt_ra4))
sigma_ra4 = np.sqrt(1/(2*popt_ra4[2]**2))
y_ra4 = func_guass(dataset['Energy'], *popt_ra4)
plt.plot(dataset['Energy'][28:40],y_ra4[28:40])
err_ra4
sigma_ra4
popt_ra4


# In[105]:


#Ra226 4
#peak_ra4 = curve_fit_guass(dataset['Energy'][29:37], ra2264)

plt.plot(dataset['Energy'][29:37],ra2264)


# In[98]:


################################################################################
#curvfit unknown sample

dataset.columns

OA = dataset['Ore_A pure signal']
E = dataset['Energy']

#peak 1 = OA[75:95]
#peak 2 = OA[43:53]
#peak 2 = OA[43:53]
#peak 2 = OA[43:53]




plt.plot(E[:53],OA[:53])


# In[82]:


################################################################################
#plot the energy vs the counts vs the intensities to cross check.
dataset.columns
cal_d[0][10], cal_d[0][12],cal_d[0][14] #Bi
cal_d[0][17]#Cs
cal_d[0][21], cal_d[0][22] #Co

cal_d
spyr = np.pi*10**7 #s/yr
t12 = 1600. #yr
run_time = 180. #s
t12_s = spyr * t12
sample_left = run_time/t12_s
sample_left

counts = sample_left * cal_d[1]
counts = counts*10**(14)
counts
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(dataset['Energy'][:120],dataset['Ore_A pure signal'][:120],c='r', label = 'Unknown Source - Ore_A')
#ax2.scatter(80.574,i[86],c='k',s=5,label = 'Ho166')
ax2.scatter(186.221, cal_d[1][2],c='b',s=5,label = 'Ra266')
ax2.scatter(262.270, cal_d[1][4],c='b',s=5)
ax2.scatter(414.600, cal_d[1][5],c='b',s=5)
ax2.scatter(449.370, cal_d[1][6],c='b',s=5)
ax2.scatter(600.660, cal_d[1][7],c='b',s=5)
ax2.scatter(cal_d[0][21], cal_d[1][21],c='k',s=5,label = 'Co60')
ax2.scatter(cal_d[0][22], cal_d[1][22],c='k',s=5)
ax2.scatter(cal_d[0][10], cal_d[1][10],c='g',s=5,label = 'Bi207')
ax2.scatter(cal_d[0][12], cal_d[1][12],c='g',s=5)
ax2.scatter(cal_d[0][14],cal_d[1][14],c='g',s=5)
ax2.scatter(cal_d[0][17], cal_d[1][17],c='c',s=5,label = 'Cs137')
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"Counts")
ax2.set_ylabel(r"Number of Decays per 100 isotopes")
ax.set_xlim(0,750)
ax2.set_ylim(0, 7500)
ax.set_ylim(0,700)
ax.legend(loc=0)
ax2.legend(loc=2)
plt.show()


# In[157]:


################################################################################
#Attenuation
#get the I and I0s for each isotpe and size, use all info to get mu
I0 = np.array([max_Bi207_1, max_Bi207_2, max_Cs137_1, max_Co60_1, max_Co60_2])
I = np.array([max_Bi207_1am, max_Bi207_2am, max_Cs137_1am, max_Co60_1am, max_Co60_2am])
Is = np.array([max_Bi207_1as, max_Bi207_2as, max_Cs137_1as, max_Co60_1as, max_Co60_2as])
x1 =(25,10)


def mu(x1,x2,d):
    return -np.log(x1/x2)/d

for x in x1:
    if x== x1[0]:
        dudI = -1./(I*x)
        dI = np.sqrt(I)
        dudI0 = 1./(I0*x)
        dI0 = np.sqrt(I0)
        dudx = np.log(I/I0)/x**2
        dx = 0.05

        du = ((dudI*dI)**2 + (dudI0*dI0)**2 + (dudx*dx)**2)**(1/2)
        u = mu(I,I0,x)
        print(u, du, x)
    else:
        dudI = -1./(Is*x)
        dI = np.sqrt(Is)
        dudI0 = 1./(I0*x)
        dI0 = np.sqrt(I0)
        dudx = np.log(Is/I0)/x**2
        dx = 0.05

        du = ((dudI*dI)**2 + (dudI0*dI0)**2 + (dudx*dx)**2)**(1/2)
        u = mu(Is,I0,x)
        print(u, du, x)


# In[25]:


#Import modules needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
from scipy import optimize
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
#To do list
#error Analysis
#find the unknown sample
#plot our data vs known data
#make graph without sub bkg

################################################################################
#pull datasets and store them in a pandas dataframe
#has to be a cleaner way to do this
#The Calibration datasets

def dataset_0_func():
    dataset_0 = []
    for file in os.listdir(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master'):
        data = np.genfromtxt(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master\%s' % (file,))
        data = data.astype(str)
        data = np.insert(data,0,file,axis=0)
        dataset_0.append(data)
    dataset_0 = pd.DataFrame(dataset_0)
    dataset_0 = dataset_0.T
    for index in dataset_0.columns:
        dataset_0[dataset_0[index][0]] = dataset_0[index]
        del dataset_0[index]
    dataset_0 = dataset_0.drop(0)
    dataset_0 = dataset_0.astype('float')
    return dataset_0
#the measurements datasets
def dataset_1_func():
    dataset_1 = []
    for file in os.listdir(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master'):
        data = np.genfromtxt(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master\%s' % (file,))
        data = data.astype(str)
        data = np.insert(data,0,file,axis=0)
        dataset_1.append(data)
    dataset_1 = pd.DataFrame(dataset_1)
    dataset_1 = dataset_1.T
    for index in dataset_1.columns:
        dataset_1[dataset_1[index][0]] = dataset_1[index]
        del dataset_1[index]
    dataset_1 = dataset_1.drop(0)
    dataset_1 =dataset_1.astype('float')
    return dataset_1
dataset_1 = dataset_1_func()
dataset_0 = dataset_0_func()


# In[26]:


dataset_0


# In[2]:


pa1 = dataset_1['SPECRTUM_4_ATTN_1.txt pure signal']
pa2 = dataset_1['SPECRTUM_4_ATTN_2.txt pure signal']
p1 = dataset_1['SPECRTUM_4_ATTN_1.txt']
p2 = dataset_1['SPECRTUM_4_ATTN_2.txt']

pa = (pa1+pa2)/2.
plt.plot(dataset_1['Energy'][:120], p[:120], label = 'p')
plt.plot(dataset_1['Energy'][:120], pa[:120], label = 'pa') 
plt.plot(dataset_1['Energy'][:120], pa1[:120], label = 'pa1')
plt.plot(dataset_1['Energy'][:120], pa2[:120], label = 'pa2')
plt.plot(dataset_1['Energy'][:120], p1[:120], label = 'p1')
plt.plot(dataset_1['Energy'][:120], p2[:120], label = 'p2')
plt.plot(dataset_1['Energy'][:120], p3[:120], label = 'p3')
plt.plot(dataset_1['Energy'][:120], p1[:120] - noise[:120], label = 'clean')
plt.plot(dataset_1['Energy'][:120], noise[:120], label = 'noise')
plt.legend()


# In[8]:



test= []

for file in os.listdir(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master'):
    x = np.genfromtxt(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master\'%)


# In[23]:


np.genfromtxt(r'C:\Users\arr10\Documents\Fall2018\Expirimental II\SLab-Data-master\Ra_226.txt')

