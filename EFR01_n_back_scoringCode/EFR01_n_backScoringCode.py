
#Zizu 02/22/20 & Diego 02/26/2020
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', required = True)
inputs = parser.parse_args()
sub = inputs.subject

root = ET.parse('msmri522_2vs0_back.xml').getroot() #read template xxml file
scorelabel=root.getchildren()[5].getchildren() #the stimuli scores is in index 5


bb=pd.read_csv('exampleEFLogfile/{}-frac2B_1.00_no1B.log'.format(sub),skiprows=6,sep='\t',header=None) #read logfile for a particular subjects
bb.columns=['Subject','Trial','EventType','Code','Time','TTime','Uncertainty0','Duration','Uncertainty1',
       'ReqTime','ReqDur','StimType','PairIndex']

back0=[] #0back
back2=[] #2back
for i in range(0,len(scorelabel)):
    if scorelabel[i].get('category') == '0BACK':
        back0.append([scorelabel[i].get('expected'),scorelabel[i].get('index')])
    elif scorelabel[i].get('category') == '2BACK':
        back2.append([scorelabel[i].get('expected'),scorelabel[i].get('index')])

# each list consists of both results with NR means No result and Macth means correct result as it is on xml
# how to comppute final score? maye be (number of  Match/( number of NR + number of Match))!!
# you combine all the output in one file may be  in json
scoresummary={'0BACK':back0,'2BACK':back2}
c=list(scoresummary.items())


allback=[]

templateback0=c[0][1]
templateback2=c[1][1]
for i in range(0,len(templateback0)):
    a1=bb[bb['Trial'] >= np.int(templateback0[i][1])-2]
    a2=bb[bb['Trial'] <= np.int(templateback0[i][1])]
    aa=np.array(pd.merge(a1, a2,how='inner')['TTime'].to_list())
    if len(aa) > 6 :
        if aa[0] > 0 :
            response=aa[0]/10
        else :
            res = next((i for i, j in enumerate(aa[range(0,len(aa),2)]) if j), None)
            ste=res-1
            centr=2*res-1
            response=aa[centr]/10+ste*800
    else:
        response=0
    allback.append([c[0][0],templateback0[i][1],templateback0[i][0],response])

for i in range(0,len(templateback2)):
    a1=bb[bb['Trial'] >= np.int(templateback2[i][1])-2]
    a2=bb[bb['Trial'] <= np.int(templateback2[i][1])]
    aa=np.array(pd.merge(a1, a2,how='inner')['TTime'].to_list())
    if len(aa) > 6 :
        if aa[0] > 0 :
            response=aa[0]/10
        else :
            res = next((i for i, j in enumerate(aa[range(0,len(aa),2)]) if j), None)
            ste=res-1
            centr=2*res-1
            response=aa[centr]/10+ste*800
    else:
        response=0

    allback.append([c[1][0],templateback2[i][1],templateback2[i][0],response])


# output
dfallback=pd.DataFrame(allback)
dfallback.columns=['task','Index','Results','ResponseTime']
dfallback

zeroback = dfallback[dfallback.task=="0BACK"]
twoback = dfallback[dfallback.task=="2BACK"]

twoback_accuracy = []
twoback_speed_fp = []
twoback_speed_tp = []
for i in range(len(twoback["task"])):

    if (twoback.iloc[i,2] == 'NR') & (twoback.iloc[i,3] == 0.0):
        # this is a true negative
        twoback_accuracy.append("TN")
    if (twoback.iloc[i,2] == 'NR') & (twoback.iloc[i,3] != 0.0):
        # this is a false positive
        twoback_accuracy.append("FP")
        twoback_speed_fp.append(twoback.iloc[i,3])
    if (twoback.iloc[i,2] == 'Match') & (twoback.iloc[i,3] == 0.0):
        #this is a false negative
        twoback_accuracy.append("FN")
    if (twoback.iloc[i,2] == 'Match') & (twoback.iloc[i,3] != 0.0):
        # this is a true positive
        twoback_accuracy.append("TP")
        twoback_speed_tp.append(twoback.iloc[i,3])
twoback_speed_fp = np.mean(twoback_speed_fp)
twoback_speed_tp = np.mean(twoback_speed_tp)

zeroback_accuracy = []
zeroback_speed_fp = []
zeroback_speed_tp = []
for i in range(len(zeroback["task"])):

    if (zeroback.iloc[i,2] == 'NR') & (zeroback.iloc[i,3] == 0.0):
        # this is a true negative
        zeroback_accuracy.append("TN")
    if (zeroback.iloc[i,2] == 'NR') & (zeroback.iloc[i,3] != 0.0):
        # this is a false positive
        zeroback_accuracy.append("FP")
        zeroback_speed_fp.append(zeroback.iloc[i,3])
    if (zeroback.iloc[i,2] == 'Match') & (zeroback.iloc[i,3] == 0.0):
        #this is a false negative
        zeroback_accuracy.append("FN")
    if (zeroback.iloc[i,2] == 'Match') & (zeroback.iloc[i,3] != 0.0):
        # this is a true positive
        zeroback_accuracy.append("TP")
        zeroback_speed_tp.append(zeroback.iloc[i,3])
zeroback_speed_fp = np.mean(zeroback_speed_fp)
zeroback_speed_tp = np.mean(zeroback_speed_tp)

twobackNumFP = twoback_accuracy.count("FP")
twobackNumTP = twoback_accuracy.count("TP")
twobackNumFN = twoback_accuracy.count("FN")
twobackNumTN = twoback_accuracy.count("TN")

zerobackNumFP = zeroback_accuracy.count("FP")
zerobackNumTP = zeroback_accuracy.count("TP")
zerobackNumFN = zeroback_accuracy.count("FN")
zerobackNumTN = zeroback_accuracy.count("TN")

from scipy.stats import norm
import math
Z = norm.ppf

def dp(hits, misses, fas, crs):
    #
    # hits = True Positive
    # misses = False Negative
    # fas = False Positive
    # crs = True Negative
    #
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    # Return d'
    dprime = Z(hit_rate) - Z(fa_rate)
    return(dprime)


twoback_dPrime = dp(twobackNumTP,twobackNumFN,twobackNumFP,twobackNumTN)
zeroback_dPrime = dp(zerobackNumTP,zerobackNumFN,zerobackNumFP,zerobackNumTN)

data = {
    "twobackNumFN": twobackNumFN,
    "twobackNumFP": twobackNumFP,
    "twobackNumTN": twobackNumTN,
    "twobackNumTP": twobackNumTP,
    "zerobackNumFN":zerobackNumFN,
    "zerobackNumFP": zerobackNumFP,
    "zerobackNumTN": zerobackNumTN,
    "zerobackNumTP": zerobackNumTP,
    "twoback_speed_fp": twoback_speed_fp,
    "twoback_speed_tp": twoback_speed_tp,
    "zeroback_speed_fp": zeroback_speed_fp,
    "zeroback_speed_tp": zeroback_speed_tp,
    "twoback_dPrime": twoback_dPrime,
    "zeroback_dPrime": zeroback_dPrime
}
output = pd.DataFrame(data, columns = ["twobackNumFN",
    "twobackNumFP",
    "twobackNumTN",
    "twobackNumTP",
    "zerobackNumFN",
    "zerobackNumFP",
    "zerobackNumTN",
    "zerobackNumTP",
    "twoback_speed_fp",
    "twoback_speed_tp",
    "zeroback_speed_fp",
    "zeroback_speed_tp",
    "twoback_dPrime",
    "zeroback_dPrime"], index = ["{}".format(sub)])
output.index.name = 'bblid'

output.to_csv("{}_nBackScore.csv".format(sub))
