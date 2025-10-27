#functions for use in models

import pandas as pd #if need to install, use 'pip install pandas' in terminal
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import DistanceMetric
import math


#late probability
def probVol(w, f1, f2 = 0, f3 = 0):
    if f2 == 0:
        #1-leg probability
        return max(0,1 - w/(7/f1))
    elif f3 == 0:
        #2-leg probability
        return max(0,((1/2)*((max(0,7/f1+7/f2-w))**2-(max(0,(7/f1)-w))**2-(max(0,(7/f2)-w))**2))/((7/f1)*(7/f2)))
    else:
        #3-leg probability
        return max(0,((1/6)*((max(0,7/f1+7/f2+7/f3-w))**3-(max(0,(7/f1)+(7/f2)-w))**3-(max(0,(7/f2)+(7/f3)-w))**3-(max(0,(7/f1)+(7/f3)-w))**3
                     +(max(0,(7/f1)-w))**3+(max(0,(7/f2)-w))**3+(max(0,(7/f3)-w))**3)/((7/f1)*(7/f2)*(7/f3))))

def findC(w_hat, f1, f2, f3 = 0):
    if f3 == 0:
        return (w_hat/7)*(1/(1/f1+1/f2))
    else:
        return (w_hat/7)*(1/(1/f1+1/f2+1/f3))
    


def probChk(w_hat, legs, prob, transMode, maxLTL, maxTL):
    cval = 1
    cval2 = 0
    minSum = 0
    if prob == 0.5:
        return 0.5
    elif legs == 1:
        return prob
    elif legs == 2:
        if transMode == 'LTL':
            l2MF = maxLTL
        else:
            l2MF = maxTL
        for f2 in range(1,l2MF+1):
            for f1 in range(f2,maxTL+1):
                c = min(1,findC(w_hat,f1,f2))
                if probVol(w_hat,f1,f2) < 1-prob:
                    if c < cval:
                        cval = c
                else:
                    if c > cval2:
                        cval2 = c
        if cval == 1:
            return 1
        else:
            return max(0.001,cval2+0.001)
    else:
        if transMode == 'LTL':
            l3MF = maxLTL
        else:
            l3MF = maxTL
        for f3 in range(1,l3MF+1):
            for f2 in range(f3,maxTL+1):
                for f1 in range(f2,maxTL+1):
                    c = min(1,findC(w_hat,f1,f2,f3))
                    if probVol(w_hat,f1,f2,f3) < 1-prob:
                        if c < cval:
                            cval = c
                    else: 
                        if c > cval2:
                            cval2 = c
                        
        if cval == 1:
            return 1
        else:
            return max(0.001,cval2+0.001)
        
        
#Creating class for preparing dataframes for model
##Assumes: weekly demand and sheets are ordered: Month, Lanes, Rts
class monthData:
    def __init__(self, dfL, dfR, timeConstr, month):
        self.LegMode, self.Legs, self.laneParams, self.LegRts = self.laneRtParams(dfL, dfR)
        self.RtCost, self.laneCosts = self.defObjParams(dfL, dfR)
        if timeConstr:
            self.constrRt, self.constrRts = self.timeConstrRts(dfR, dfL)
            self.finalLeg = self.timeConstrParams(self.constrRt, self.constrRts)
        self.modeDict, self.rts, self.dmdRts, self.bigMVals = self.dictionaries(dfR, dfL)
        ## for older files
        self.constants = []

    def laneRtParams(self, dfLanes, dfRts):
        #creating lists of indices, where LegMode are all non-vendor direct leg/mode combinations,
        ## Legs just includes non-vendor direct legs
        #collecting unique leg IDs - excluding direct lanes
        dfLExcludeDir = dfLanes[~((dfLanes['ORIGIN_TYPE']=='VND')&(dfLanes['DEST_TYPE']=='LMD'))].copy()
        Legs = dfLExcludeDir['LEG_ID'].unique()
        laneParams = (dfLExcludeDir[['LEG_ID', 'MODE', 'MAX_WEIGHT', 
                                      'MIN_WEIGHT']].set_index(['LEG_ID', 'MODE']).to_dict('index')) 
        LegMode = list(laneParams)  
        
        #collecting the routes that contain leg l
        #removing vendor direct routes
        dfR = dfRts[~(dfRts['LEG1_TYPE']=='VND->LMD')].copy()
        #creating a dictionary, where each key is a leg and the values are the associated routes
        l1 = dfR[['ROUTE_NBR','LEG1']].rename(columns={'LEG1':'LEG'})
        l2 = dfR[['ROUTE_NBR','LEG2']][dfR['LEG2']!=0].rename(columns={'LEG2':'LEG'})
        l3 = dfR[['ROUTE_NBR','LEG3']][dfR['LEG3']!=0].rename(columns={'LEG3':'LEG'})
        l4 = dfR[['ROUTE_NBR','LEG4']][dfR['LEG4']!=0].rename(columns={'LEG4':'LEG'})
        df = pd.concat([l1,l2,l3,l4])
        LegRts = df.groupby('LEG')['ROUTE_NBR'].apply(list).to_dict()
            
        return LegMode, Legs, laneParams, LegRts
    
    def defObjParams(self, dfLanes, dfRts):
        #RtCost: dictionary to collect total route cost -> sum of Handling and transportation costs
        #Rt cost is the sum of direct and handling (one will always be 0)
        RtCost = dict(zip(dfRts.ROUTE_NBR.values,dfRts.DR_COST.values
                                          +dfRts.HANDLING_PER_CUBE.values*dfRts.WGT.values/10))
        #going through each lane to collect fixed costs and price per pound
        laneCosts = dfLanes[['LEG_ID', 'MODE', 'FIXED_COST', 'COST_PER_POUND']].set_index(['LEG_ID', 'MODE']).to_dict('index')
        return  RtCost, laneCosts
    
    def timeConstrRts(self, dfRts, dfLanes):
        #collecting routes that will be time-constrained
        constrRt = dfRts[~(dfRts['LEG1_TYPE']=='VND->LMD')].copy()
        return constrRt, constrRt['ROUTE_NBR'].tolist()
    
    def timeConstrParams(self, constrRt, constrRts):

        #finalLeg - the last leg in the route
        constrRt['finalLeg'] = constrRt.apply(lambda x: x['LEG'+str(x['LEGS'])], axis = 1)
        finalLeg = dict(zip(constrRt['ROUTE_NBR'],constrRt['finalLeg']))
            

        return finalLeg
    
    def dictionaries(self, dfRts, dfLanes):
        #collecting modes for each leg
        modeDict = dfLanes.groupby('LEG_ID')['MODE'].apply(list).to_dict()  
        
        #collecting cube and weight for each route other than vendor direct routes
        rts = dfRts[~(dfRts['LEG1_TYPE']=='VND->LMD')][['ROUTE_NBR','WGT']].set_index('ROUTE_NBR').to_dict('index')
        
        #collecting routes for each demand
        dmdRts = dfRts.groupby('DEMAND_ID')['ROUTE_NBR'].apply(list).to_dict() 
        
        dfL = dfLanes[dfLanes['MODE']==0].copy()
        dfL['bigM'] = dfL.apply(lambda x: math.ceil(dfRts[(dfRts['LEG1']==x['LEG_ID'])|(dfRts['LEG2']==x['LEG_ID'])|
                                                (dfRts['LEG3']==x['LEG_ID'])|(dfRts['LEG4']==x['LEG_ID'])].drop_duplicates(subset='DEMAND_ID')['WGT'].sum()/x['MAX_WEIGHT']), axis=1)
        bigMVals = dict(zip(zip(dfL['LEG_ID'],dfL['MODE']),dfL['bigM']))
        
        return modeDict, rts, dmdRts, bigMVals

def defObjParams(dfLanes, dfRts):
        #RtCost: dictionary to collect total route cost -> sum of Handling and transportation costs
        #Rt cost is the sum of direct and handling (one will always be 0)
        RtCost = dict(zip(dfRts.ROUTE_NBR.values,dfRts.DR_COST.values
                                          +dfRts.HANDLING_PER_CUBE.values*dfRts.WGT.values/10))
        return  RtCost
    
class dataPM:
    def __init__(self, dfR, dfC, maxLT, minLT):
        self.PPCosts, self.sales, self.cogs = self.defPMObj(dfR)
        (self.conRates, self.conDmds, self.nonConDmds, self.conRts, 
         self.nonConRts, self.conLTs, self.conRMax, self.dirConRates, self.nonConLTs) = self.converData(dfC, dfR, maxLT, minLT)
        self.rtFix, self.rtDmd, self.dirRts = self.addData(dfR)
    
    def defPMObj(self, dfRts):
        #RtCost: dictionary to collect total route cost -> sum of Handling and transportation costs
        #Rt cost is the sum of direct and handling (one will always be 0)
        # DRCost = dict(zip(dfRts.ROUTE_NBR.values,dfRts.DR_COST.values))
        PPCost = dict(zip(dfRts.ROUTE_NBR.values,dfRts.HANDLING_PER_CUBE.values/10))
        dfR = dfRts.drop_duplicates(subset = 'DEMAND_ID')
        sales = dict(zip(dfR.DEMAND_ID.values,dfR.SALES.values))
        cogs = dict(zip(dfR.DEMAND_ID.values,dfR.COGS.values))
        return  PPCost, sales, cogs
    
    def converData(self, dfCon, dfRts, maxLT, minLTRed):
        dfM = dfCon[['DEMAND_ID', 'LT', 'PREDICTION']].copy().rename(columns={"LT": "LT_UPPER_BD", 
                                                                              "PREDICTION": "nomRate"})
        dfR = dfRts[dfRts['LEG1_TYPE']!='VND->LMD'].copy()
        dfCR = dfR[['DEMAND_ID', 'LT_UPPER_BD']].drop_duplicates(subset='DEMAND_ID').merge(dfM,how="inner", 
                                                                                     on=['DEMAND_ID','LT_UPPER_BD'])
        #calculating minimum possible LTs for consolidation routes to filter conRates later
        dfR['minTime'] = dfR.apply(lambda x: math.ceil(x['FIXED']+0.5*x['LEGS']), axis = 1)
        dfR = dfR.sort_values(by=['DEMAND_ID','minTime']).drop_duplicates(subset='DEMAND_ID').copy()
        
        minLT = dict(zip(dfR.DEMAND_ID,dfR.minTime))
        nomLT = dict(zip(dfCR.DEMAND_ID,dfCR.LT_UPPER_BD))
        nomR = dict(zip(dfCR.DEMAND_ID,dfCR.nomRate))
        dfCon['nomRates'] = dfCon['DEMAND_ID'].apply(lambda x: nomR[x])
        dfCon['chg'] = dfCon['PREDICTION']/dfCon['nomRates']
        dirConRates = dict(zip(zip(dfCon.DEMAND_ID.values,dfCon.LT.values),dfCon.chg.values))
        dfCon['keep'] = dfCon.apply(lambda x: 'yes' if x['LT'] >= minLT[x['DEMAND_ID']] 
                                               else 'no', axis = 1)
        dfCon = dfCon[dfCon['keep']=='yes'].drop(columns=['keep']).copy()
        conRates = dict(zip(zip(dfCon.DEMAND_ID.values,dfCon.LT.values),dfCon.chg.values))
        
        conDmds = dfCR['DEMAND_ID'].tolist()
        dfNonCon = dfRts[~(dfRts['DEMAND_ID'].isin(conDmds))].copy()
        nonConDmds = dfNonCon['DEMAND_ID'].unique().tolist()
        nonConLTs = dict(zip(dfNonCon['ROUTE_NBR'],dfNonCon['LT_UPPER_BD']))
        conRts = dfRts[(dfRts['DEMAND_ID'].isin(conDmds))
                       &(dfRts['LEG1_TYPE']!='VND->LMD')]['ROUTE_NBR'].tolist()
        nonConRts = dfRts[(dfRts['DEMAND_ID'].isin(nonConDmds))
                         |(dfRts['LEG1_TYPE']=='VND->LMD')]['ROUTE_NBR'].tolist()
        conLTs = dfCon.groupby('DEMAND_ID')['LT'].apply(list).to_dict() 
        dfCM = dfCon.sort_values(by=['DEMAND_ID','chg'], ascending=False).drop_duplicates(subset='DEMAND_ID')
        conRMax = dict(zip(dfCM['DEMAND_ID'],dfCM['chg']))
        return conRates, conDmds, nonConDmds, conRts, nonConRts, conLTs, conRMax, dirConRates, nonConLTs
    
    def addData(self, dfRts):
        rtFix = dict(zip(dfRts['ROUTE_NBR'],dfRts['FIXED']))
        rtDmd = dict(zip(dfRts['ROUTE_NBR'],dfRts['DEMAND_ID']))
        dRts = dfRts[dfRts['LEG1_TYPE']=='VND->LMD'].copy()
        dirRts = dict(zip(dRts['DEMAND_ID'],dRts['ROUTE_NBR']))
        return rtFix, rtDmd, dirRts
    
    
#function to prep heuristic inputs
def heurPrep(dfRts, dfLanes, dfMonth):
    #for selecting routes in heuristic
    #creating a list of FCs
    listFC = dfLanes[dfLanes['ORIGIN_TYPE']=='FC']['ORIGIN_ID'].unique()
    # #creating a dictionary for each of the FCs
    # dmdsFC = {}
    #collecting demands that only have one route to not include in the route subsets later used
    dfRtCt = dfRts[['DEMAND_ID','ROUTE_NBR']].groupby('DEMAND_ID').agg({'ROUTE_NBR':'count'}).reset_index().copy()

    #creating a new column for origin IDs and origin Zips
    dfRts['orgID_orgZIP'] = list(zip(dfRts.ORIGIN_ID.values, dfRts.ORIGIN_ZIP.values))
    #collecting routes to use in the heuristic iterations
    dfChk = dfRts[~(dfRts['LEG1_TYPE']=='VND->LMD')].copy()
    fcRts, fcVnds, fcVndWts = {}, {}, {}
    vndCons = {}
    for fc in listFC:
        dfCopy = dfChk[(dfChk['ORIGIN_ID'].astype(str)==str(fc))
                        |(dfChk['DEST1'].astype(str)==str(fc))
                        |(dfChk['DEST2'].astype(str)==str(fc))
                        |(dfChk['DEST3'].astype(str)==str(fc))
                        |(dfChk['DEST4'].astype(str)==str(fc))].copy()
        fcRts[fc] = dfCopy.groupby('DEMAND_ID')['ROUTE_NBR'].apply(list).to_dict()
        dfCopy = dfCopy.drop_duplicates(subset = 'DEMAND_ID').groupby('orgID_orgZIP').agg({'WGT':'sum'}).reset_index()
        dfCopy['weights'] = dfCopy['WGT']/dfCopy['WGT'].sum()
        fcVnds[fc] = dfCopy['orgID_orgZIP'].tolist()
        fcVndWts[fc] = dfCopy['weights'].tolist()
        dfCons = dfChk[(dfChk['DEST1'].astype(str)==str(fc))].copy()
        vndCons[fc] = dfCons['orgID_orgZIP'].unique().tolist()
    
    rtList = dfRts['ROUTE_NBR'].tolist()
    
    #dictionary of demands for each origin location
    dmdsVDF = dfRts.groupby(['ORIGIN_ID'])['DEMAND_ID'].apply(list).to_dict() 
    #list of demands for each vendor location
    dfVDFsum = dfRts.groupby(['ORIGIN_ID']).agg({'WGT':'sum'}).reset_index()
    dfVDFsum = dfVDFsum.sort_values(by = 'WGT', ascending = False)
    vndList = dfVDFsum['ORIGIN_ID'].tolist()
    dfVDFsum['weights'] = dfVDFsum['WGT'].values/dfVDFsum['WGT'].sum()
    vndWgts = dfVDFsum['weights'].tolist()

    #collecting rts for each dmd ID
    dmdRts = dfRts.groupby('DEMAND_ID')['ROUTE_NBR'].apply(list).to_dict()
    return rtList, listFC, fcVnds, fcVndWts, dmdsVDF, fcRts, vndList, vndWgts, dmdRts

def output(MonthOut, LanesOut, RtsOut, 
           objVal, solveTime,
           laneUsed, laneFreq, dirRtsU, routes, totalTime):
    laneCosts = []
    handling = []
    
    #removing unused lanes from lanes df
    LanesOut = LanesOut[LanesOut[['LEG_ID','MODE']].apply(tuple, 1).isin(laneUsed)].copy()
    #adding a column to the lanes df for the frequency
    LanesOut['LANE_FREQ'] = LanesOut.apply(lambda x: laneFreq[laneUsed.index((int(x['LEG_ID']),
                                                             int(x['MODE'])))],axis = 1)
    LanesOut['EXP_WAIT'] = (1/2)*(7/LanesOut['LANE_FREQ'])
    #labeling lanes with TL or LTL
    LanesOut['BIN_MODE'] = 'LTL'
    LanesOut.loc[LanesOut['MODE'].isin([0,100,200]),'BIN_MODE'] = 'TL'


    #calculating the total wait across a route
    RtsOut['EXP_WAIT'] = RtsOut.apply(lambda x: sum(LanesOut.loc[LanesOut['LEG_ID']
                                                            ==x['LEG'+str(l+1)]]['EXP_WAIT'].values[0]
                                                        for l in range(x['LEGS'])), axis = 1)
    
    RtsOut['EXP_LT'] =  RtsOut['FIXED'] + RtsOut['EXP_WAIT']

    #adding handling cost to modelStats
    handling.append(np.sum(RtsOut['WGT'].values/10*RtsOut['HANDLING_PER_CUBE'].values))

    #collecting final lane weight and cube values
    LanesOut = laneTotals(RtsOut, LanesOut)
    #adding lane costs to modelStats
    laneCosts.append(LanesOut['LANE_COST'].sum())

    #adding the routes used to the monthly demand sheet
    MonthOut = MonthOut.merge(RtsOut[['DEMAND_ID', 'ROUTE_NBR']], on='DEMAND_ID')

    #creating a data frame for the optimization model outputs
    zipped = list(zip([objVal], laneCosts, handling, 
                     [solveTime], [totalTime], [dirRtsU]))
    ModelStats = pd.DataFrame(zipped, columns = ['OBJVAL ($)', 'LANE_COSTS', 'HANDLING_COSTS', 
                                                 'HEUR_TIME (sec)',
                                                 'TOTAL_TIME (sec)','DIRECT_RTS_USED'])

            
    return MonthOut, LanesOut, RtsOut, ModelStats

def outputPM(MonthOut, LanesOut, RtsOut, 
           objVal, solve, 
           laneUsed, laneFreq, dirRtsU, routes, totalTime, ltUsed, convRates,
            # wsObj, wsPM, wsSales, wsCOGS, wsLogCost,
             percRtsOrig, percRtsRts, percRtsLT, percRts):
    
    #removing unused lanes from lanes df
    LanesOut = LanesOut[LanesOut[['LEG_ID','MODE']].apply(tuple, 1).isin(laneUsed)].copy()
    #adding a column to the lanes df for the frequency
    LanesOut['LANE_FREQ'] = LanesOut.apply(lambda x: laneFreq[laneUsed.index((int(x['LEG_ID']),
                                                             int(x['MODE'])))],axis = 1)
    LanesOut['EXP_WAIT'] = (1/2)*(7/LanesOut['LANE_FREQ'])
    #labeling lanes with TL or LTL
    LanesOut['BIN_MODE'] = 'LTL'
    LanesOut.loc[LanesOut['MODE']==0,'BIN_MODE'] = 'TL'


    #tracking LTs selected by the model
    RtsOut['LT_USED'] = RtsOut.apply(lambda x: ltUsed[x['DEMAND_ID']] if x['DEMAND_ID'] in ltUsed
                            else x['LT_UPPER_BD'], axis = 1)
    RtsOut['LT_USED'] = RtsOut.apply(lambda x: x['DR_LT'] if x['DR_LT'] != 0
                                    else x['LT_USED'], axis = 1)
    #setting conversion rates used
    ##HSE is 1 
    RtsOut['conv'] = RtsOut.apply(lambda x: convRates[x['LT_USED']]/convRates[x['LT_UPPER_BD']]
                                              if x['LEG1_TYPE'] not in ['FC->FC','FC->LMD']
                                             else 1, axis = 1)
    # RtsOut['conv'] = RtsOut.apply(lambda x: x['DR_CONR'] if x['DR_CONR'] != 0
    #                                          else x['conv'], axis = 1)
    RtsOut['convSALES'] = RtsOut['SALES']*RtsOut['conv']
    RtsOut['convCOGS'] = RtsOut['COGS']*RtsOut['conv']
    #updating weights
    RtsOut['origWGT'] = RtsOut['WGT']
    RtsOut['WGT'] = RtsOut['origWGT']*RtsOut['conv']
    #finding new rev (sales-cogs)*conv
    RtsOut['REV'] = RtsOut['convSALES'] - RtsOut['convCOGS']

    #calculating the total wait across a route
    RtsOut['EXP_WAIT'] = RtsOut.apply(lambda x: sum(LanesOut.loc[LanesOut['LEG_ID']
                                                            ==x['LEG'+str(l+1)]]['EXP_WAIT'].values[0]
                                                        for l in range(x['LEGS'])), axis = 1)
    RtsOut['EXP_LT'] =  RtsOut['FIXED'] + RtsOut['EXP_WAIT']
    
    #adding handling cost to modelStats
    handling = np.sum(RtsOut['WGT'].values/10*RtsOut['HANDLING_PER_CUBE'].values)

    #collecting final lane weight and cube values
    LanesOut = laneTotals(RtsOut, LanesOut)
    #adding lane costs to modelStats
    laneCosts = LanesOut['LANE_COST'].sum()

    #adding the routes used to the monthly demand sheet
    MonthOut = MonthOut.merge(RtsOut[['DEMAND_ID', 'ROUTE_NBR']], on='DEMAND_ID')

    
    #creating a data frame for the optimization model outputs
    zipped = list(zip([objVal], [laneCosts], [handling], 
                       [solve], [totalTime], [dirRtsU]))
    ms = pd.DataFrame(zipped, columns = ['OBJVAL ($)', 'LANE_COSTS', 'HANDLING_COSTS', 
                                                 'HEUR_TIME (sec)',
                                                 'TOTAL_TIME (sec)','DIRECT_RTS_USED'])

    
    ms['sales'] = RtsOut['convSALES'].sum()
    ms['cogs'] = RtsOut['convCOGS'].sum()
    ms['logCost'] = ms.apply(lambda x: x['sales'] - x['cogs'] - x['OBJVAL ($)'], axis = 1)
    ms['profMarg'] = ms.apply(lambda x: (x['sales'] - x['cogs'] - x['logCost'])/x['sales'], axis = 1)
    
    ms['origPercRts'] = percRtsOrig
    ms['percRts'] = percRts
    ms['percRtsRts'] = percRtsRts
    ms['percRtsLT'] = percRtsLT
    
    RtsOut['LT_CHG'] = RtsOut['LT_USED'] - RtsOut['LT_UPPER_BD']
    
            
    return MonthOut, LanesOut, RtsOut, ms

def outputPMMIP(MonthOut, LanesOut, RtsOut, 
           objVal,
           laneUsed, laneFreq, dirRtsU, routes, totalTime, ltUsed, convRates, 
               mipGap, lwrBd):

    #removing unused lanes from lanes df
    LanesOut = LanesOut[LanesOut[['LEG_ID','MODE']].apply(tuple, 1).isin(laneUsed)].copy()
    #adding a column to the lanes df for the frequency
    LanesOut['LANE_FREQ'] = LanesOut.apply(lambda x: laneFreq[laneUsed.index((int(x['LEG_ID']),
                                                             int(x['MODE'])))],axis = 1)
    LanesOut['EXP_WAIT'] = (1/2)*(7/LanesOut['LANE_FREQ'])
    #labeling lanes with TL or LTL
    LanesOut['BIN_MODE'] = 'LTL'
    LanesOut.loc[LanesOut['MODE'].isin([0,100,200]),'BIN_MODE'] = 'TL'

    #tracking LTs selected by the model
    RtsOut['LT_USED'] = RtsOut.apply(lambda x: ltUsed[x['DEMAND_ID']] if x['DEMAND_ID'] in ltUsed
                            else x['LT_UPPER_BD'], axis = 1)
    RtsOut['LT_USED'] = RtsOut.apply(lambda x: x['DR_LT'] if x['DR_LT'] != 0
                                    else x['LT_USED'], axis = 1)
    #setting conversion rates used
    ##HSE is 1 
    RtsOut['conv'] = RtsOut.apply(lambda x: convRates[x['LT_USED']]/convRates[x['LT_UPPER_BD']]
                                              if x['LEG1_TYPE'] not in ['FC->FC','FC->LMD']
                                             else 1, axis = 1)

    RtsOut['convSALES'] = RtsOut['SALES']*RtsOut['conv']
    RtsOut['convCOGS'] = RtsOut['COGS']*RtsOut['conv']
    #updating weights
    RtsOut['origWGT'] = RtsOut['WGT']
    RtsOut['WGT'] = RtsOut['origWGT']*RtsOut['conv']
    #finding new rev (sales-cogs)*conv
    RtsOut['REV'] = (RtsOut['SALES'] - RtsOut['COGS'])*RtsOut['conv']

    #calculating the total wait across a route
    RtsOut['EXP_WAIT'] = RtsOut.apply(lambda x: sum(LanesOut.loc[LanesOut['LEG_ID']
                                                            ==x['LEG'+str(l+1)]]['EXP_WAIT'].values[0]
                                                        for l in range(x['LEGS'])), axis = 1)
    RtsOut['EXP_LT'] =  RtsOut['FIXED'] + RtsOut['EXP_WAIT']

    #adding handling cost to modelStats
    handling = np.sum(RtsOut['WGT'].values/10*RtsOut['HANDLING_PER_CUBE'].values)

    #collecting final lane weight and cube values
    LanesEx = laneTotals(RtsOut, LanesOut)
    #adding lane costs to modelStats
    laneCosts = LanesEx['LANE_COST'].sum()

    #adding the routes used to the monthly demand sheet
    MonthEx = MonthOut.merge(RtsOut[['DEMAND_ID', 'ROUTE_NBR']], on='DEMAND_ID')


    #creating a data frame for the optimization model outputs
    zipped = list(zip([objVal], [lwrBd], [mipGap], [laneCosts], [handling], 
                       [totalTime], [dirRtsU]))
    ms = pd.DataFrame(zipped, columns = ['OBJVAL ($)','lwrBd', 'mipGap','LANE_COSTS', 'HANDLING_COSTS', 
                                                 'TOTAL_TIME (sec)','DIRECT_RTS_USED'])

    
    ms['sales'] = RtsOut['convSALES'].sum()
    ms['cogs'] = RtsOut['convCOGS'].sum()
    ms['logCost'] = ms.apply(lambda x: x['sales'] - x['cogs'] - x['OBJVAL ($)'], axis = 1)
    ms['profMarg'] = ms.apply(lambda x: (x['sales'] - x['cogs'] - x['logCost'])/x['sales'], axis = 1)
    
    RtsOut['LT_CHG'] = RtsOut['LT_USED'] - RtsOut['LT_UPPER_BD']
    
            
    return MonthEx, LanesEx, RtsOut, ms

def outputMIP(MonthOut, LanesOut, RtsOut, 
           objVal, solveTime, 
           laneUsed, laneFreq, dirRts, routes, totalTime,
             mipGap, lwrBd):
    
    laneCosts = []
    handling, sales = [], []
    
    #removing unused lanes from lanes df
    LanesOut = LanesOut[LanesOut[['LEG_ID','MODE']].apply(tuple, 1).isin(laneUsed)].copy()
    #adding a column to the lanes df for the frequency
    LanesOut['LANE_FREQ'] = LanesOut.apply(lambda x: laneFreq[laneUsed.index((int(x['LEG_ID']),
                                                             int(x['MODE'])))],axis = 1)
    LanesOut['EXP_WAIT'] = (1/2)*(7/LanesOut['LANE_FREQ'])
    #labeling lanes with TL or LTL
    LanesOut['BIN_MODE'] = 'LTL'
    LanesOut.loc[LanesOut['MODE']==0,'BIN_MODE'] = 'TL'


    #calculating the total wait across a route
    RtsOut['EXP_WAIT'] = RtsOut.apply(lambda x: sum(LanesOut.loc[LanesOut['LEG_ID']
                                                            ==x['LEG'+str(l+1)]]['EXP_WAIT'].values[0]
                                                        for l in range(x['LEGS'])), axis = 1)

    RtsOut['EXP_LT'] =  RtsOut['FIXED'] + RtsOut['EXP_WAIT']

    #adding handling cost to modelStats
    handling.append(np.sum(RtsOut['WGT'].values/10*RtsOut['HANDLING_PER_CUBE'].values))

    #adding sales
    sales.append(RtsOut['SALES'].sum()-RtsOut['COGS'].sum())

    #collecting final lane weight and cube values
    LanesOut = laneTotals(RtsOut, LanesOut)
    #adding lane costs to modelStats
    laneCosts.append(LanesOut['LANE_COST'].sum())

    #adding the routes used to the monthly demand sheet
    MonthOut = MonthOut.merge(RtsOut[['DEMAND_ID', 'ROUTE_NBR']], on='DEMAND_ID')

    
    #creating a data frame for the optimization model outputs
    zipped = list(zip([objVal], [lwrBd], [mipGap], laneCosts, handling, sales,
                       [solveTime], [totalTime], [dirRts]))
    ModelStats = pd.DataFrame(zipped, columns = ['OBJVAL ($)', 'lowerBd','mipGap', 'LANE_COSTS', 'HANDLING_COSTS', 'rev',
                                                 'HEUR_TIME (sec)',
                                                 'TOTAL_TIME (sec)','DIRECT_RTS_USED'])

            
    return MonthOut, LanesOut, RtsOut, ModelStats

def laneTotals(dfRts, dfLanes):
    weight = []
    for row in dfLanes.itertuples():
        weight.append(dfRts.loc[(dfRts['LEG1']==row.LEG_ID)|
                                (dfRts['LEG2']==row.LEG_ID)|
                                (dfRts['LEG3']==row.LEG_ID),'WGT'].sum())
    dfLanes['LANE_WGT'] = weight
    dfLanes['LANE_COST'] = (dfLanes['COST_PER_POUND'].values*dfLanes['LANE_WGT'].values 
                            +dfLanes['FIXED_COST'].values*dfLanes['LANE_FREQ'].values)
    
    return dfLanes

def statsModes(dfL, origins = [], dests = []):
    if len(origins) == 0: origins = dfL['ORIGIN_TYPE'].unique().tolist()
    if len(dests) == 0: dests = dfL['DEST_TYPE'].unique().tolist()
    #reducing dataframe to specific lane type(s)
    dfL = dfL[(dfL['ORIGIN_TYPE'].isin(origins))
              &(dfL['DEST_TYPE'].isin(dests))].copy()
    dfLTL = dfL[~(dfL['MODE'].isin([0,100,200]))].copy()
    dfTL = dfL[(dfL['MODE'].isin([0,100,200]))].copy()
    
    if len(dfLTL) > 0 and len(dfTL) > 0:
        groupedLTL = dfLTL.groupby(['FSCL_YR_WK']).agg({'LEG_ID':'count','ton-miles':sum,
                                                   'LANE_COST':sum}).reset_index()
        groupedTL = dfTL.groupby(['FSCL_YR_WK']).agg({'LEG_ID':'count','ton-miles':sum,
                                                   'LANE_COST':sum}).reset_index()
        return (groupedLTL['LEG_ID'].tolist(), groupedLTL['ton-miles'].tolist(), groupedLTL['LANE_COST'].tolist(),
                groupedTL['LEG_ID'].tolist(), groupedTL['ton-miles'].tolist(), groupedTL['LANE_COST'].tolist())
    elif len(dfTL) > 0:
        groupedTL = dfTL.groupby(['FSCL_YR_WK']).agg({'LEG_ID':'count','ton-miles':sum,
                                                   'LANE_COST':sum}).reset_index()
        empty = [0 for i in dfL['FSCL_YR_WK'].unique()]
        return (empty, empty, empty,
                groupedTL['LEG_ID'].tolist(), groupedTL['ton-miles'].tolist(), groupedTL['LANE_COST'].tolist())
    elif len(dfLTL) > 0:
        groupedLTL = dfLTL.groupby(['FSCL_YR_WK']).agg({'LEG_ID':'count','ton-miles':sum,
                                                   'LANE_COST':sum}).reset_index()
        empty = [0 for i in dfL['FSCL_YR_WK'].unique()]
        return (groupedLTL['LEG_ID'].tolist(), groupedLTL['ton-miles'].tolist(), groupedLTL['LANE_COST'].tolist(),
                empty, empty, empty)
    else:
        empty = [0 for i in dfL['FSCL_YR_WK'].unique()]
        return (empty, empty, empty,
                empty, empty, empty)
    
def tonMiles(dfLanes, ModelStats):
    #adding ton-miles
    dfLanes['ton-miles'] = dfLanes['DISTANCE']*dfLanes['LANE_WGT']/2000
    
    #collecting stats
    #all lane-types
    (ModelStats['LTL_Lane_Count'], ModelStats['LTL_TON_MI'], ModelStats['LTL_LANE_COSTS'],
     ModelStats['TL_Lane_Count'], ModelStats['TL_TON_MI'], ModelStats['TL_LANE_COSTS']) = statsModes(dfLanes)
    #direct lanes
    laneT = 'VNDtoFinDest_'
    (ModelStats[str(laneT)+'LTL_Lane_Count'], ModelStats[str(laneT)+'LTL_TON_MI'], ModelStats[str(laneT)+'LTL_COSTS'],
     ModelStats[str(laneT)+'TL_Lane_Count'], ModelStats[str(laneT)+'TL_TON_MI'], ModelStats[str(laneT)+'TL_COSTS']
     ) = statsModes(dfLanes, origins=['VND'], dests=['LMD'])
    #VND to consolidation point lanes
    laneT = 'VNDtoConPt_'
    (ModelStats[str(laneT)+'LTL_Lane_Count'], ModelStats[str(laneT)+'LTL_TON_MI'], ModelStats[str(laneT)+'LTL_COSTS'],
     ModelStats[str(laneT)+'TL_Lane_Count'], ModelStats[str(laneT)+'TL_TON_MI'], ModelStats[str(laneT)+'TL_COSTS']
     ) = statsModes(dfLanes, origins=['VND'], dests=['FC'])
    #FC to FC lanes
    laneT = 'FCtoFC_'
    (ModelStats[str(laneT)+'LTL_Lane_Count'], ModelStats[str(laneT)+'LTL_TON_MI'], ModelStats[str(laneT)+'LTL_COSTS'],
     ModelStats[str(laneT)+'TL_Lane_Count'], ModelStats[str(laneT)+'TL_TON_MI'], ModelStats[str(laneT)+'TL_COSTS']
     ) = statsModes(dfLanes, origins=['FC'], dests=['FC'])
    #FC to final dests lanes
    laneT = 'FCtoFinDest_'
    (ModelStats[str(laneT)+'LTL_Lane_Count'], ModelStats[str(laneT)+'LTL_TON_MI'], ModelStats[str(laneT)+'LTL_COSTS'],
     ModelStats[str(laneT)+'TL_Lane_Count'], ModelStats[str(laneT)+'TL_TON_MI'], ModelStats[str(laneT)+'TL_COSTS']
     ) = statsModes(dfLanes, origins=['FC'], dests=['LMD'])
    
    return ModelStats


def dirRtCostsLT(wgt, leg, dfL, minFreq, maxLTL):
    minFreq = math.ceil(minFreq)
    la = dfL[dfL['LEG_ID']==leg].copy()
    la['freq'] = la['MAX_WEIGHT'].apply(lambda x: max(minFreq, math.ceil(wgt/x)))
    la = la[~((la['MODE']!=0)&(la['freq']>maxLTL))].copy()
    la['shpAmt'] = la['freq'].apply(lambda x: wgt/x)
    la['cost'] = la.apply(lambda x: round(wgt*x['COST_PER_POUND'] + x['freq']*x['FIXED_COST'],2) 
                              if x['MIN_WEIGHT'] < x['shpAmt'] <= x['MAX_WEIGHT'] else 9999999, axis=1)
    la = la.sort_values(by=['cost','MODE']).drop_duplicates(subset = ['LEG_ID']).copy()
    mode = int(la['MODE'])
    cost = la['cost'].min()
    freq = la['freq'].min()
    return pd.Series({'DR_MODE':mode, 'DR_COST':cost, 'DR_FREQ':freq})

def dirRtCostsPM(comm, wgt, l, dfL, conserv, sales, cogs, conRates, conLTsC, laneCosts, laneParams, ltUpperBd, bigMTL = 7, 
                maxLTLShip = 5):
    prof = sales[comm] - cogs[comm]
    la = dfL[dfL['LEG_ID']==l].copy()
    #Initiating the model
    thd = gp.Model(name = 'PM_MMCW_dir')
    thd.params.OutputFlag = 0
    # thd.params.MIPfocus = 1
    thd.params.presolve = 0
    
    #max time should be LTL transit time plus max wait
    maxLT = max(math.ceil(laneParams[(l,1)]['transitTime'] + (7/2)), ltUpperBd)

    #removing LTs greater than the max allowed
    lts = conLTsC[comm].copy()
    add = {}
    for lt in lts:
        if lt > maxLT:
            # print('removing '+str(lt))
            conLTsC[comm].remove(lt)
            if comm in add:
                add[comm].append(lt)
            else:
                add[comm] = [lt]
    ltMax = max(conLTsC[comm])
    
    #setting frequency sets for modes
    fVal = {}
    for m in la['MODE'].unique():
        if m == 0:
            fVal[m] = [(f+1) for f in range(bigMTL)]
        else:
            fVal[m] = [(f+1) for f in range(maxLTLShip)]
    #for disjunctive constraints
    disjZ = []
    for m in la['MODE'].unique():
        for f in fVal[m]:
            disjZ.append((m,f))
            
    #creating the variables
    ##v_{m} are continuous variables that capture the amount of weight flowing on leg l using mode m
    v_vars = thd.addVars(la['MODE'].tolist(), lb = 0, name = 'v') 
    #adding frequency variables f_m
    z_vars = thd.addVars(disjZ, lb = 0, name = 'z', vtype = GRB.BINARY)
    #w_{t} are binary variables indicating the lead time t that commodity k chooses
    w_vars = thd.addVars(conLTsC[comm], lb = 0, name = 'w', vtype = GRB.BINARY)
    
    #Objective Function
    #cost of using the lanes 
    costLanes = gp.quicksum(laneCosts[(l,m)]['FIXED_COST']*gp.quicksum(f*z_vars[m,f] for f in fVal[m])
                            + laneCosts[(l,m)]['COST_PER_POUND']*v_vars[m] for m in la['MODE'].tolist())

    #rev for commodities
    ##for conversion demands
    profCon = prof*gp.quicksum(conRates[comm,t]*w_vars[t] for t in conLTsC[comm])

    #combining the costs
    ObjFunct = - profCon + costLanes
    #assigning objective to minimize
    thd.setObjective(ObjFunct, GRB.MINIMIZE)

    #determining weight flowing along conversion-demand route r
    thd.addConstr(gp.quicksum(v_vars[m] for m in la['MODE'].tolist()) == wgt*gp.quicksum(conRates[comm,t]*w_vars[t] 
                                                          for t in conLTsC[comm]))


    #calculating frequency of trips based on maximum weight - constraint (1i)
    thd.addConstrs(v_vars[m] <= laneParams[(l,m)]['MAX_WEIGHT']*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['MODE'].tolist())

    #calculating frequency of trips based on minimum weight - constraint (1j)
    thd.addConstrs(v_vars[m] >= max(0.1, laneParams[(l,m)]['MIN_WEIGHT'])*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['MODE'].tolist())
    
    #only one binary for frequency and mode
    thd.addConstr(gp.quicksum(gp.quicksum(z_vars[m,f] for f in fVal[m]) for m in la['MODE'].tolist()) == 1)
    
    #only one binary for lead time 
    thd.addConstr(gp.quicksum(w_vars[t] for t in conLTsC[comm]) == 1)
    
    #lead time constraint
    thd.addConstrs(7*conserv[1]*gp.quicksum(1/f*z_vars[m,f] for f in fVal[m]) <=
                     gp.quicksum(t*w_vars[t] for t in conLTsC[comm]) 
                       - gp.quicksum(laneParams[(l,m)]['transitTime']*z_vars[m,f] for f in fVal[m])
                   for m in la['MODE'].tolist())

    
    thd.optimize()
    
    for (m,f) in disjZ:
        if round(z_vars[m,f].x) == 1:
            mode = m
            freq = f
            fixed = laneParams[(l,m)]['transitTime']
    cost = round(thd.objVal, 2)
    for t in conLTsC[comm]:
        if round(w_vars[t].x) == 1:
            lt = t
            ltWgt = conRates[comm,t]*wgt  
            conRt = conRates[comm,t]

    cogs = round(cogs[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    sales = round(sales[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    logCost = round(sum(laneCosts[(l,m)]['FIXED_COST']*sum(f*z_vars[m,f].x for f in fVal[m])
                            + laneCosts[(l,m)]['COST_PER_POUND']*v_vars[m].x for m in la['MODE'].tolist()),2)

    for comm in add:
        for ltAdd in add[comm]:
            conLTsC[comm].append(ltAdd)
        
    
    return pd.Series({'DR_MODE':mode, 'DR_COGS':cogs, 'DR_SALES':sales , 'DR_COST':logCost , 'DR_FREQ':freq, 'DR_LT':lt, 'DR_WGT': ltWgt,
                     'DR_CONR':conRt,'LT_MAX':ltMax,'FIXED':fixed, 'W_hat':lt-fixed})

def dirRtCostsPMFixLn(comm, wgt, l, dfL, conserv, sales, cogs, conRates, conLTsC, laneCosts, laneParams, ltUpperBd, mode, freq, bigMTL = 7, 
                maxLTLShip = 5):
    prof = sales[comm] - cogs[comm]
    la = dfL[(dfL['LEG_ID']==l)].copy()
    #Initiating the model
    thd = gp.Model(name = 'PM_MMCW_dir')
    thd.params.OutputFlag = 0
    # thd.params.MIPfocus = 1
    thd.params.presolve = 0
    
    #max time should be LTL transit time plus max wait
    maxLT = max(math.ceil(laneParams[(l,1)]['transitTime'] + (7/2)), ltUpperBd)

    #removing LTs greater than the max allowed
    lts = conLTsC[comm].copy()
    add = {}
    for lt in lts:
        if lt > maxLT:
            # print('removing '+str(lt))
            conLTsC[comm].remove(lt)
            if comm in add:
                add[comm].append(lt)
            else:
                add[comm] = [lt]
    ltMax = max(conLTsC[comm])
    
    #setting frequency sets for modes
    fVal = {}
    fVal[mode] = [freq]

    disjZ = [(mode,freq)]
            
    #creating the variables
    ##v_{m} are continuous variables that capture the amount of weight flowing on leg l using mode m
    v_vars = thd.addVar(lb = 0, name = 'v') 
    # #adding frequency variables f_m
    #w_{t} are binary variables indicating the lead time t that commodity k chooses
    w_vars = thd.addVars(conLTsC[comm], lb = 0, name = 'w', vtype = GRB.BINARY)
    
    #Objective Function
    #cost of using the lanes 
    costLanes = (laneCosts[(l,mode)]['FIXED_COST']*freq
                            + laneCosts[(l,mode)]['COST_PER_POUND']*v_vars)

    #rev for commodities
    ##for conversion demands
    profCon = prof*gp.quicksum(conRates[comm,t]*w_vars[t] for t in conLTsC[comm])

    #combining the costs
    ObjFunct = - profCon + costLanes
    #assigning objective to minimize
    thd.setObjective(ObjFunct, GRB.MINIMIZE)

    #determining weight flowing along conversion-demand route r
    thd.addConstr(v_vars  == wgt*gp.quicksum(conRates[comm,t]*w_vars[t] 
                                                          for t in conLTsC[comm]))


    #calculating frequency of trips based on maximum weight - constraint (1i)
    thd.addConstr(v_vars <= laneParams[(l,mode)]['MAX_WEIGHT']*freq)

    #calculating frequency of trips based on minimum weight - constraint (1j)
    thd.addConstr(v_vars >= max(0.1, laneParams[(l,mode)]['MIN_WEIGHT'])*freq)
    
    # #only one binary for frequency and mode
    # thd.addConstr(gp.quicksum(gp.quicksum(z_vars[m,f] for f in fVal[m]) for m in la['MODE'].tolist()) == 1)
    
    #only one binary for lead time 
    thd.addConstr(gp.quicksum(w_vars[t] for t in conLTsC[comm]) == 1)
    
    #lead time constraint
    thd.addConstr(7*conserv[1]/freq <=
                     gp.quicksum(t*w_vars[t] for t in conLTsC[comm]) 
                       - laneParams[(l,mode)]['transitTime'])
    
    thd.optimize()

    fixed = laneParams[(l,mode)]['transitTime']
            # print(f'mode {mode}, freq {freq}, fixed {fixed}')
    cost = round(thd.ObjVal, 2)
    for t in conLTsC[comm]:
        if round(w_vars[t].x) == 1:
            lt = t
            ltWgt = conRates[comm,t]*wgt  
            conRt = conRates[comm,t]
            # print(f'lt {lt}, ltWgt {ltWgt}, conRt {conRt}')

    cogs = round(cogs[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    sales = round(sales[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    logCost = round(laneCosts[(l,mode)]['FIXED_COST']*freq
                            + laneCosts[(l,mode)]['COST_PER_POUND']*v_vars.x,2)

    for comm in add:
        for ltAdd in add[comm]:
            conLTsC[comm].append(ltAdd)
        
    
    return pd.Series({'DR_MODE':mode, 'DR_COGS':cogs, 'DR_SALES':sales , 'DR_COST':logCost , 'DR_FREQ':freq, 'DR_LT':lt, 'DR_WGT': ltWgt,
                     'DR_CONR':conRt,'LT_MAX':ltMax,'FIXED':fixed, 'W_hat':lt-fixed})

def dirRtCostsSM(comm, wgt, l, dfL, conserv, sales, cogs, conRates, conLTsC, laneCosts, laneParams, bigMTL = 7, 
                maxLTLShip = 5):
    prof = sales[comm] - cogs[comm]
    la = dfL[dfL['LEG_ID']==l].copy()
    #Initiating the model
    thd = gp.Model(name = 'PM_MMCW_dir')
    thd.params.OutputFlag = 0
    thd.params.MIPfocus = 1
    thd.params.presolve = 0
    

    #setting frequency sets for modes
    fVal = {}
    for m in la['MODE'].unique():
        if m == 0:
            fVal[m] = [(f+1) for f in range(bigMTL)]
        else:
            fVal[m] = [(f+1) for f in range(maxLTLShip)]
    #for disjunctive constraints
    disjZ = []
    for m in la['MODE'].unique():
        for f in fVal[m]:
            disjZ.append((m,f))
            
    #creating the variables
    ##v_{m} are continuous variables that capture the amount of weight flowing on leg l using mode m
    v_vars = thd.addVars(la['MODE'].tolist(), lb = 0, name = 'v') 
    #adding frequency variables f_m
    z_vars = thd.addVars(disjZ, lb = 0, name = 'z', vtype = GRB.BINARY)
    #w_{t} are binary variables indicating the lead time t that commodity k chooses
    w_vars = thd.addVars(conLTsC[comm], lb = 0, name = 'w', vtype = GRB.BINARY)
    
    #Objective Function
    #cost of using the lanes 
    costLanes = gp.quicksum(laneCosts[(l,m)]['FIXED_COST']*gp.quicksum(f*z_vars[m,f] for f in fVal[m])
                            + laneCosts[(l,m)]['COST_PER_POUND']*v_vars[m] for m in la['MODE'].tolist())

    #rev for commodities
    ##for conversion demands
    profCon = prof*gp.quicksum(conRates[comm,t]*w_vars[t] for t in conLTsC[comm])

    #combining the costs
    ObjFunct = - profCon + costLanes
    #assigning objective to minimize
    thd.setObjective(ObjFunct, GRB.MINIMIZE)

    #determining weight flowing along conversion-demand route r
    thd.addConstr(gp.quicksum(v_vars[m] for m in la['MODE'].tolist()) == wgt*gp.quicksum(conRates[comm,t]*w_vars[t] 
                                                          for t in conLTsC[comm]))


    #calculating frequency of trips based on maximum weight - constraint (1i)
    thd.addConstrs(v_vars[m] <= laneParams[(l,m)]['MAX_WEIGHT']*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['MODE'].tolist())

    #calculating frequency of trips based on minimum weight - constraint (1j)
    thd.addConstrs(v_vars[m] >= max(0.1, laneParams[(l,m)]['MIN_WEIGHT'])*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['MODE'].tolist())
    
    #only one binary for frequency and mode
    thd.addConstr(gp.quicksum(gp.quicksum(z_vars[m,f] for f in fVal[m]) for m in la['MODE'].tolist()) == 1)
    
    #only one binary for lead time 
    thd.addConstr(gp.quicksum(w_vars[t] for t in conLTsC[comm]) == 1)
    
    #lead time constraint
    thd.addConstrs(7*conserv[1]*gp.quicksum(1/f*z_vars[m,f] for f in fVal[m]) <=
                     gp.quicksum(t*w_vars[t] for t in conLTsC[comm]) 
                       - gp.quicksum(laneParams[(l,m)]['transitTime']*z_vars[m,f] for f in fVal[m])
                   for m in la['MODE'].tolist())


    
    thd.optimize()
    
    try:
        for (m,f) in disjZ:
            if round(z_vars[m,f].x) == 1:
                mode = m
                freq = f
                fixed = laneParams[(l,m)]['transitTime']

        cost = round(thd.objVal, 2)
        for t in conLTsC[comm]:
            if round(w_vars[t].x) == 1:
                lt = t
                ltWgt = conRates[comm,t]*wgt  
                conRt = conRates[comm,t]

        cogs = round(cogs[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
        sales = round(sales[comm]*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
        logCost = round(sum(laneCosts[(l,m)]['FIXED_COST']*sum(f*z_vars[m,f].x for f in fVal[m])
                                + laneCosts[(l,m)]['COST_PER_POUND']*v_vars[m].x for m in la['MODE'].tolist()),2)

        
    except:
        print(comm)
        print(comm, conRates[comm,conLTsC[comm][0]], conLTsC[comm])
        
    
    return pd.Series({'DR_MODE':mode, 'DR_COGS':cogs, 'DR_SALES':sales , 'DR_COST':logCost , 'DR_FREQ':freq, 'DR_LT':lt, 'DR_WGT': ltWgt,
                     'DR_CONR':conRt,'FIXED':fixed, 'W_hat':lt-fixed})

def calcLTLTrans(miles, delayMiles = 350, delay = 4):
    return max(1,0.5+(math.ceil((24*miles/550+delay*miles/delayMiles-12)/24)))

def dirRtCosts(wgt, leg, dfL, maxLTL):
    la = dfL[dfL['LEG_ID']==leg].copy()
    la['freq'] = la['MAX_WEIGHT'].apply(lambda x: math.ceil(wgt/x))
    la = la[~((la['MODE']!=0)&(la['freq']>maxLTL))].copy()
    if len(la) == 0: 
        print(leg)
        stop
    la['shpAmt'] = la['freq'].apply(lambda x: wgt/x)
    la['cost'] = la.apply(lambda x: round(wgt*x['COST_PER_POUND'] + x['freq']*x['FIXED_COST'],2) 
                              if x['MIN_WEIGHT'] < x['shpAmt'] <= x['MAX_WEIGHT'] else 9999999, axis=1)
    la = la.sort_values(by=['cost','MODE']).drop_duplicates(subset = ['LEG_ID']).copy()
    mode = int(la['MODE'])
    cost = la['cost'].min()
    freq = int(la['freq'])
    return pd.Series({'DR_MODE':mode, 'DR_COST':cost, 'DR_FREQ':freq})


def listCreate(newList, rmStopImpr, listFC, ct, fcVnds, fcVndWts, 
               vndList, vndWgts, dmdsVDF, dmdRts, vndSel, rtList, 
               vnd, fcSel, percRts, fcRts, dfRts):
    if newList:
        # stNL = time.time()
        print('newList')
        #resetting vendor count
        vnd = 0
        
        #if improvement using random has stopped, use fc-free
        if rmStopImpr:
        
            #resetting the ct value if >= # of fcs
            if ct >= len(listFC):
                ct = 0
            #select one fc from the list
            fcSel = listFC[ct]
            #updating ct to pick the next fc next time
            ct += 1
            #selecting vendors with demands that flow through the selected fc
            if len(fcVnds[fcSel])!=0:
                #selecting a random group of origins to free their routes
                order = np.random.choice(len(fcVnds[fcSel]), len(fcVnds[fcSel]), p=fcVndWts[fcSel], replace=False)
                vndSel = []
                for o in order:
                    vndSel.append(fcVnds[fcSel][o])
            else:
                vndSel = []
        else:
            #selecting a random group of vendors to free their routes
            #selecting a random group of origins to free their routes
            order = np.random.choice(len(vndList), len(vndList), p=vndWgts, replace=False)
            vndSel = []
            for o in order:
                vndSel.append(fcVnds[fcSel][o])
        # print('new list time: '+str(time.time()-stNL))     
    #selecting routes 
    selected = []   
    newList = False     
    # stSel = time.time()
    if len(vndSel) != 0:
        vndSt = vnd
        #collecting demands until route ct met
        while len(selected) < percRts*len(rtList):
            v = vndSel[vnd]
            vnd+=1
            if rmStopImpr:
                selected += dfRts.loc[(dfRts['DEMAND_ID'].isin(dmdsVDF[v]))&(dfRts['ROUTE_NBR'].isin(rtList))
                     &(dfRts['DEMAND_ID'].isin(fcRts[fcSel])), 'ROUTE_NBR'].tolist()
            else:
                selected += dfRts.loc[(dfRts['DEMAND_ID'].isin(dmdsVDF[v]))&(dfRts['ROUTE_NBR'].isin(rtList)), 'ROUTE_NBR'].tolist()

            if vnd == len(vndSel): 
                newList = True
                vnd = 0
                if vndSt == 0:
                    break
            elif vndSt == vnd:
                break
    # print('selecting time: '+str(time.time()-stSel))
    
    return newList, selected, vndSel, vnd, fcSel, ct

#function to calculate a distance matrix between all origins
def distMatrix(dfM):
    #converting lat and long into radians for scipy functions
    dfM['lat'] = np.radians(dfM['ORIGIN_LAT'])
    dfM['lon'] = np.radians(dfM['ORIGIN_LONG'])
    #choosing the haversine distance metric
    dist = DistanceMetric.get_metric('haversine')
    #creating a distance matrix dataframe
    dfD = pd.DataFrame(dist.pairwise(np.radians(dfM[['lat','lon']]))*3959,  columns=dfM.orgID_orgZip.unique(), index=dfM.orgID_orgZip.unique())
    cols = dfD.columns.tolist()
    distDict = {}
    for c in cols:
        dfDc = dfD[[c]].sort_values(by=c).copy()
        dfDc = dfDc[dfDc[c]!=0].copy().reset_index()
        distDict[c] = dfDc['index'].tolist()

    #returns a dictionary where the keys are origin IDs and values are the list of remaining origins ordered by distance to the key
    return distDict

#function to prep heuristic inputs when using the random neighborhood selection method
def heurPrepRNS(dfRts, dfLanes):

    fcs = [1001,1002,1003,1004,1005,1006,1007,1008]
    rtList = dfRts[~(dfRts['ORIGIN_ID'].isin(fcs))]['ROUTE_NBR'].tolist()
    
    #dictionary of demands for each origin location
    dfMonth = dfMonth[~(dfMonth['ORIGIN_ID'].isin(fcs))].copy()
    #creating a new column for origin IDs and origin Zips
    dmdsVDF = dfMonth.groupby(['ORIGIN_ID','ORIGIN_ZIP'])['DEMAND_ID'].apply(list).to_dict() 
    #aggregating origin volume to calculate weights
    dfVDFsum = dfMonth.groupby(['ORIGIN_ID','ORIGIN_ZIP']).agg({'WGT':'sum'}).reset_index()
    dfVDFsum = dfVDFsum.sort_values(by = 'WGT', ascending = False)
    vndList = list(zip(dfVDFsum['ORIGIN_ID'],dfVDFsum['ORIGIN_ZIP'])) 
    dfVDFsum['weights'] = dfVDFsum['WGT'].values/dfVDFsum['WGT'].sum()
    vndVolWgts = dfVDFsum['weights'].tolist()
    
    # #creating list of equal weights for vendors
    # dfVDFsum['eqWgt'] = 1/len(dfVDFsum)
    # vndEqWgts = dfVDFsum['eqWgt'].tolist()
    
    #creating dictionary where keys are origin IDs and values are lists of other origins ordered by distance to the key
    dfMonth['orgID_orgZip'] = list(zip(dfMonth['ORIGIN_ID'],dfMonth['ORIGIN_ZIP']))
    distDict = distMatrix(dfMonth[['orgID_orgZip','ORIGIN_LAT','ORIGIN_LONG']].drop_duplicates(subset='orgID_orgZip').copy())
    
    #dictionary of demands for each destination location
    dmdsLMD = dfMonth.groupby(['DEST_ID','DEST_ZIP'])['DEMAND_ID'].apply(list).to_dict() 
    # #creating list of equal weights for LMDs
    # lmdEqWgts = [1/len(dmdsLMD) for lmd in dmdsLMD]
    
    return rtList, dmdsVDF, vndList, vndVolWgts, distDict, dmdsLMD

#given a vendor list and weights per vendor, 
## this function outputs a list of origins - ordered randomly or randomly weighted by volume or randomly, if equal weight
def randOriginList(vndList, vndWgts):
    
    #selecting a random group of origins to free their routes
    order = np.random.choice(len(vndList), len(vndList), p=vndWgts, replace=False)
    originList = [vndList[o] for o in order]  
    
    return originList

#given an LMD list and weights per LMD, 
## this function outputs a list of LMDs - ordered randomly by equal weight
def randLMDList(lmdList):
    
    #selecting a random group of origins to free their routes
    order = np.random.choice(len(lmdList), len(lmdList), replace=False)
    selList = [lmdList[o] for o in order]  
    
    return selList


## given an vendor list, weights per vendor, and ordered list of origins by distance to selected origin,
## this function returns a list of origins to use to define a neighborhood
def distOriginList(vndList, vndWgts, distDict, prevVnds):
    tabuLength = int(math.ceil(0.75*len(vndList)))
    #selecting an origin at random (weighted by vndWgts)
    origins = np.random.choice(len(vndList), tabuLength+1, p=vndWgts, replace=False)
    for i in origins:
        if vndList[i] not in prevVnds:
            origin = i
            prevVnds.insert(0,vndList[i])
            prevVnds = prevVnds[0:tabuLength]
            break
    # print('wtdDist origin: '+str(vndList[origin]))
    originList = [vndList[origin]] + distDict[vndList[origin]].copy()
    # print(prevVnds)
    return originList, prevVnds

def vndSelRts(origOrd, dfR, dmdsVDF, rtLen):
    #selecting routes 
    selectedRts = []
    #collecting demands until route ct met
    for v in origOrd:
        selectedRts += dfR.loc[(dfR['DEMAND_ID'].isin(dmdsVDF[v])), 'ROUTE_NBR'].tolist()
        if len(selectedRts) >= rtLen:
            break
    return selectedRts

def select(newList, selNH, origList, origVolWgts, dmdsVDF, rtLen, dfR, distDict, prevVnds, 
            lmdOrd, dmdsLMD, lmd):

    #if improvement using random has stopped, use fc-free
    ## getting origin list
    if selNH == 'wtdDist':
        origOrd, prevVnds = distOriginList(origList, origVolWgts, distDict, prevVnds)
        selectedRts = vndSelRts(origOrd, dfR, dmdsVDF, rtLen)
    elif selNH == 'rand':
        if newList:
            lmdOrd = randLMDList(list(dmdsLMD))
            # set newList to false so entire list is used
            newList = False
        #selecting routes 
        selectedRts = []
        lmdSt = lmd
        #collecting demands until route ct met
        while len(selectedRts) < rtLen:
            l = lmdOrd[lmd]
            lmd += 1
            selectedRts += dfR.loc[(dfR['DEMAND_ID'].isin(dmdsLMD[l])), 'ROUTE_NBR'].tolist()
            if lmd == len(dmdsLMD): 
                newList = True
                lmd = 0
                if lmdSt == 0:
                    break
            elif lmdSt == lmd:
                break
    else:
        origOrd = randOriginList(origList, origVolWgts)
        selectedRts = vndSelRts(origOrd, dfR, dmdsVDF, rtLen)     
    
    return newList, selectedRts, prevVnds, lmdOrd, lmd

