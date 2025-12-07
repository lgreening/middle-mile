#functions for use in Cost Min Model and all associated

import pandas as pd #if need to install, use 'pip install pandas' in terminal
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
import copy
import random as rm
import math
from sklearn.metrics import DistanceMetric
from math import radians



def probVol(w, f1, f2 = 0, f3 = 0):
    if f2 == 0:
        #1-arc probability
        return max(0,1 - w/(7/f1))
    elif f3 == 0:
        #2-arc probability
        return max(0,((1/2)*((max(0,7/f1+7/f2-w))**2-(max(0,(7/f1)-w))**2-(max(0,(7/f2)-w))**2))/((7/f1)*(7/f2)))
    else:
        #3-arc probability
        return max(0,((1/6)*((max(0,7/f1+7/f2+7/f3-w))**3-(max(0,(7/f1)+(7/f2)-w))**3-(max(0,(7/f2)+(7/f3)-w))**3-(max(0,(7/f1)+(7/f3)-w))**3
                     +(max(0,(7/f1)-w))**3+(max(0,(7/f2)-w))**3+(max(0,(7/f3)-w))**3)/((7/f1)*(7/f2)*(7/f3))))

def findC(w_hat, f1, f2, f3 = 0):
    if f3 == 0:
        return (w_hat/7)*(1/(1/f1+1/f2))
    else:
        return (w_hat/7)*(1/(1/f1+1/f2+1/f3))
    
def probChk(w_hat, arcs, prob, lmdMode, maxLTL, maxTL):
    cval = 1
    cval2 = 0
    if prob == 0.5:
        return 0.5
    elif arcs == 1:
        return prob
    elif arcs == 2:
        if lmdMode == 'LTL':
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
        if lmdMode == 'LTL':
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

#used to prepare dataframes for models
class optData:
    def __init__(self, arcData, pathData, maxLTLShip = 5, bigMTime = 7, bigMTL = 40, Q = 12000):
        self.dfA, self.dfP = self.readFiles(arcData, pathData)
        self.arcs, self.arcMode, self.arcParams, self.arcPaths, self.bigMVals, self.modeDict, self.fcArcs = self.arcParams(self.dfA, self.dfP, 
                                                                                                                         bigMTL, maxLTLShip, bigMTime, Q)
        self.paths, self.dmdPaths, self.pathParams, self.conPaths, self.arcsConstrPaths = self.pathParams(self.dfP)
        self.pathCost, self.arcCosts = self.defObjParams(self.dfA, self.dfP)
        self.arcTL, self.arcLTL, self.maxF = self.timeConstrData(self.dfP, self.dfA, maxLTLShip, bigMTime, bigMTL)

    def readFiles(self, arcData, pathData):
        dfA = pd.read_csv(arcData)
        dfP = pd.read_csv(pathData).replace(np.nan, '', regex=True)
            
        # #keeping directs and removing LTL consolidation routes (i.e., those that use an FC)
        # dfP = dfP[(dfP['dirFreq']>0)|(dfP['transitMode']!='LTL')].copy()
        #removing arcs not in this path set
        dfA = dfA[dfA['arcID'].isin(pd.unique(dfP[['arc1','arc2','arc3']].values.ravel()))].copy()
        # #removing LTL from consolidation arcs, but not direct arcs
        # dfA = dfA[(dfA['mode']==0)|((dfA['originType']=='VND')&(dfA['destType']=='LMD'))].copy()
        
        # #removing other modes from direct arcs - reduces size of dfA
        # dfDir = dfP[dfP['dirFreq']>=1].copy()
        # dirArcModes = list(zip(dfDir['arc1'],dfDir['dirMode']))
        # dfA['arcMode'] = list(zip(dfA['arcID'],dfA['mode']))
        # dfA = dfA[(dfA['arcMode'].isin(dirArcModes))|
        #                     ~((dfA['originType']=='VND')&(dfA['destType']=='LMD'))].copy()

        return dfA, dfP
    
    def arcParams(self, dfArcs, dfP, bigMTL, maxLTLShip, bigMTime, Q):
        #creating lists of indices, where arcMode are all arc/mode combinations excluding VND->LMD arcs,
        ## arcs just includes consolidation arcs
        #collecting unique arc IDs - excluding direct arcs
        dfAExDir = dfArcs[~((dfArcs['originType']=='VND')&(dfArcs['destType']=='LMD'))].copy()
        arcs = dfAExDir['arcID'].unique()
        arcMode = list(zip(dfAExDir['arcID'],dfAExDir['mode']))
        arcParams = (dfArcs[['arcID','mode', 'maxWgt','minWgt']].set_index(['arcID','mode']).to_dict('index'))          
        
        #creating a dictionary, where each key is an arc and the values are the associated paths
        ## first removing vendor direct paths - all will have a dirFreq >= 1
        dfP = dfP[dfP['dirFreq']<=0].copy()
        ## collecting all arcs for each path in a separate df
        arcList = []
        maxArcs = int(dfP['arcCt'].max())
        for a in range(maxArcs):
            arcList.append(dfP[['demandID','pathID','wgt',f'arc{a+1}']][~(dfP[f'arc{a+1}'].isin(['','0',0]))].rename(columns={f'arc{a+1}':'arc'}))
        ## concat all dfs
        df = pd.concat(arcList)
        ## create the dictionary
        arcPaths = df.groupby('arc')['pathID'].apply(list).to_dict()
        # calculating the bigM value for each arc
        df = df.drop_duplicates(subset=['demandID','arc'])
        dfW = df[['arc','wgt']].groupby(by='arc').sum().reset_index()
        totWgts = dict(zip(dfW['arc'],dfW['wgt']))
        dfAExDir['bigM'] = dfAExDir.apply(lambda x: max(bigMTime,min(bigMTL,math.ceil(totWgts[x['arcID']]/Q))) if x['mode'] == 0
                                                     else maxLTLShip, axis = 1)
        
        bigMVals = dict(zip(zip(dfAExDir['arcID'],dfAExDir['mode']),dfAExDir['bigM']))
        # mode dictionary
        modeDict = dfArcs.groupby('arcID')['mode'].apply(list).to_dict()
        
        fcArcs = dfArcs[(dfArcs['originType']=='FC')&(dfArcs['destType']=='FC')]['arcID'].tolist()
            
        return arcs, arcMode, arcParams, arcPaths, bigMVals, modeDict, fcArcs
    
    def pathParams(self, dfP):
        # all paths including directs
        paths = dfP['pathID'].tolist()
        #collecting paths for each demand ID
        dmdPaths = dfP.groupby('demandID')['pathID'].apply(list).to_dict()
        #path parameters
        pathParams = (dfP[['pathID','wgt','w_hat','arcCt','fixedTT']].set_index(['pathID']).to_dict('index'))
        #removing vendor direct paths - all will have a dirFreq >= 1
        dfP = dfP[dfP['dirFreq']<=0].copy()
        #all consolidation (non-direct) paths
        conPaths = dfP['pathID'].tolist()
        #getting arcs in the time-constrained paths
        arcsConstrPaths = dict(zip(dfP['pathID'],zip(dfP['arc1'],dfP['arc2'],dfP['arc3'])))
        for p in conPaths:
            arcsConstrPaths[p] = [i for i in arcsConstrPaths[p] if i != '' and str(i) != '0']
        
        return paths, dmdPaths, pathParams, conPaths, arcsConstrPaths
        
    def defObjParams(self, dfArcs, dfP):
        #pathCost: dictionary to collect total path cost -> sum of Handling and transportation costs
        #path cost is the sum of direct and handling (one will always be 0)
        pathCost = dict(zip(dfP.pathID.values,dfP.dirCost.values
                                          +dfP.handling_per_lb*dfP.wgt.values))
        #going through each arc to collect fixed costs and price per pound
        arcCosts = dfArcs[['arcID', 'mode', 'fixedCost', 'varCost']].set_index(['arcID', 'mode']).to_dict('index')
    
        return  pathCost, arcCosts
    
    def timeConstrData(self, dfP, dfA, maxLTLShip, bigMTime, bigMTL):
        #time-constrained paths df
        dfCP = dfP[dfP['dirFreq']<=0].copy()
        dfCP['lastArc'] = dfCP.apply(lambda x: x['arc'+str(x['arcCt'])], axis = 1)
        #specifying paths which require truckload or LTL on their final arc
        arcTL = dict(zip(dfCP[dfCP['transitMode']=='TL']['pathID'],dfCP[dfCP['transitMode']=='TL']['lastArc']))
        arcLTL = dict(zip(dfCP[dfCP['transitMode']=='LTL']['pathID'],dfCP[dfCP['transitMode']=='LTL']['lastArc']))
        
        maxF = {}
        for m in dfA['mode'].unique():
            if m == 0:
                maxF[m] = bigMTime
            else:
                maxF[m] = maxLTLShip
                
        return arcTL, arcLTL, maxF
    
def dirPathCosts(wgt, arc, dfA, maxLTL):
    la = dfA[dfA['arcID']==arc].copy()
    la['freq'] = la['maxWgt'].apply(lambda x: math.ceil(wgt/x))
    la = la[~((la['mode']!=0)&(la['freq']>maxLTL))].copy()
    if len(la) == 0: 
        print(arc)
        raise RuntimeError("No valid arc found for the given weight and arc.")
    la['shpAmt'] = la['freq'].apply(lambda x: wgt/x)
    la['cost'] = la.apply(lambda x: round(wgt*x['varCost'] + x['freq']*x['fixedCost'],2) 
                              if x['minWgt'] < x['shpAmt'] <= x['maxWgt'] else 9999999, axis=1)
    la = la.sort_values(by=['cost','mode']).drop_duplicates(subset = ['arcID']).copy()
    mode = int(la['mode'])
    cost = la['cost'].min()
    freq = int(la['freq'])
    return pd.Series({'dirMode':mode, 'dirCost':cost, 'dirFreq':freq})



class odt_data:
    """
    ODT data container.
    Inputs:
      dfP : DataFrame of path/ath-level info
      dfC : DataFrame of conversion-rate info 
      maxODT, minODT : passed through 
    Produces attributes (examples):
      PPCosts, sales, cogs,
      conRates, conDmds, nonConDmds, conPaths, nonConPaths, conODTs, conRMax, dirConRates, nonConODTs,
      pathFix, pathDmd, dirPaths
    """
    def __init__(self, dfP, conversion_data, maxODT, minODT, conserv = 0.8, chgcurve=1, maxLTLShip=5, bigMTime=7):
        # conversion data df
        dfC, self.nomODT = self.build_conversion_df(conversion_data, dfP, minODT, maxODT)
        # conversion-related dictionaries (renamed outputs)
        (self.conRates, self.conDmds, self.nonConDmds, self.conPaths,
         self.nonConPaths, self.conODTs, self.conRMax,
         self.dirConRates, self.nonConODTs) = self.converData(dfC, dfP, maxLTLShip, bigMTime, chgcurve)
        # path-level attributes (originally rtFix, rtDmd, dirPaths)
        self.pathDmd, self.dirPaths = self.addData(dfP)
        # conservatism values for consolidation routes
        self.cVals, self.cValsFC = self.conservatism(self.conRates, dfP, conserv, maxLTLShip, bigMTime)

    
    def build_conversion_df(self, conv_rate_file, dfP, minODT, maxODT):

        # Read conversion rate data
        df_con = pd.read_csv(conv_rate_file)

        # get current ODT (nomODT) for each demand from dfP
        nomODT = dict(zip(dfP['demandID'], dfP['current_ODT']))

        # calculate minimum possible ODTs for consolidation routes to filter conRates later
        dfPmin = dfP[dfP['arc1_type']!='VND->LMD'].copy()
        dfPmin['minTime'] = dfPmin.apply(lambda x: math.ceil(x['fixedTT']+0.5*x['arcCt']), axis = 1)
        dfPmin = dfPmin.sort_values(by=['demandID','minTime']).drop_duplicates(subset='demandID').copy()
        minODTs = dict(zip(dfPmin.demandID,dfPmin.minTime))

        # Filter and expand demandIDs
        df_cr = dfP[dfP['arc1_type'] == 'VND->FC'][['demandID']].drop_duplicates()
        df_cr['key'] = 1
        df_con['key'] = 1

        # Outer join and sort
        df_con = (
            pd.merge(df_con, df_cr, how='outer', on='key')
            .sort_values(by=['demandID', 'ODT'])
            .drop(columns='key')
        )

        # Filter based on ODT range
        def should_keep(row):
            lower_bound = max(nomODT[row['demandID']] - minODT, minODTs[row['demandID']])
            upper_bound = nomODT[row['demandID']] + maxODT
            return lower_bound <= row['ODT'] <= upper_bound

        df_con = df_con[df_con.apply(should_keep, axis=1)].copy()
        return df_con, nomODT

    def converData(self, dfCon, dfP, maxLTLShip, bigMTime, chgCurve):
        """
        Build conversion rate dictionaries and lists.
        Returns (in this order):
          conRates, conDmds, nonConDmds, conPaths, nonConPaths, conODTs, conRMax, dirConRates, nonConODTs
        """
        # rename columns from dfCon to match original logic
        dfM = dfCon[['demandID', 'ODT', 'prediction']].copy().rename(
            columns={"ODT": "current_ODT", "prediction": "nomRate"}
        )

        # keep only consolidation candidate rows in dfP (exclude vendor-directs)
        dfP = dfP[dfP['arc1_type'] != 'VND->LMD'].copy()

        # find nominal ODT rows for each demand and merge with predictions
        dfCR = dfP[['demandID', 'current_ODT']].drop_duplicates(subset='demandID').merge(
            dfM, how="inner", on=['demandID', 'current_ODT']
        )

        # calculate minimum possible ODTs for consolidation routes to filter conRates later
        hMinLTL = (7/maxLTLShip)/2
        hMinTL = (7/bigMTime)/2
        dfP['minTime'] = dfP.apply(lambda x: np.ceil(hMinLTL+hMinTL*(x['arcCt']-1) + x['fixedTT']) 
                                                if x['transitMode']=='LTL'
                                                else np.ceil(hMinTL*x['arcCt'] + x['fixedTT']), 
                                                axis = 1)
        dfMinP = dfP.sort_values(by=['demandID', 'minTime']).drop_duplicates(subset='demandID').copy()

        minODTs = dict(zip(dfMinP.demandID, dfMinP.minTime))
        nomR = dict(zip(dfCR.demandID, dfCR.nomRate))

        # attach nominal rates onto dfCon and compute change factor 'chg'
        dfCon['nomRates'] = dfCon['demandID'].apply(lambda x: nomR[x])
        dfCon['chg'] = dfCon['prediction'] / dfCon['nomRates']

        # dirConRates (mapping (demandID, ODT) -> chg) before filtering by min ODT
        dirConRates = dict(zip(zip(dfCon.demandID.values, dfCon.ODT.values), dfCon.chg.values))

        # filter out conversion rows whose ODT < minimum feasible minODT for that demand
        dfCon['keep'] = dfCon.apply(lambda x: 'yes' if x['ODT'] >= minODTs[x['demandID']] else 'no', axis=1)
        dfCon = dfCon[dfCon['keep'] == 'yes'].drop(columns=['keep']).copy()

        # conRates mapping (demandID, ODT) -> chg for feasible ODTs
        conRates = dict(zip(zip(dfCon.demandID.values, dfCon.ODT.values), dfCon.chg.values))

        #adjusting the conversion rates for sensitivity experiments
        if chgCurve != 1:
            for key in conRates:
                if conRates[key] > 1:
                    conRates[key] = 1+(conRates[key]-1)*chgCurve
                elif conRates[key] < 1:
                    conRates[key] = 1-(1-conRates[key])*chgCurve
            for key in dirConRates:
                if dirConRates[key] > 1:
                    dirConRates[key] = 1+(dirConRates[key]-1)*chgCurve
                elif dirConRates[key] < 1:
                    dirConRates[key] = 1-(1-dirConRates[key])*chgCurve


        # conversion-demand list (demands that have consolidation options)
        conDmds = dfCR['demandID'].tolist()

        # non-conversion demands = routes/demands that are not in conDmds
        dfNonCon = dfP[~(dfP['demandID'].isin(conDmds))].copy()
        nonConDmds = dfNonCon['demandID'].unique().tolist()

        # nonConODTs mapping for non-consolidation routes
        nonConODTs = dict(zip(dfNonCon['pathID'], dfNonCon['current_ODT']))

        # conPaths = list of pathID that are consolidation candidates (keeps original identifier)
        conPaths = dfP[(dfP['demandID'].isin(conDmds)) & (dfP['arc1_type'] != 'VND->LMD')]['pathID'].tolist()

        # nonConPaths = routes that are non-conversion OR vendor-direct
        nonConPaths = dfP[(dfP['demandID'].isin(nonConDmds)) | (dfP['arc1_type'] == 'VND->LMD')]['pathID'].tolist()

        # conODTs: demand -> list of possible ODTs
        conODTs = dfCon.groupby('demandID')['ODT'].apply(list).to_dict()

        # conRMax: demand -> maximum change factor among feasible ODTs (used as big-M multiplier)
        dfCM = dfCon.sort_values(by=['demandID', 'chg'], ascending=False).drop_duplicates(subset='demandID')
        conRMax = dict(zip(dfCM['demandID'], dfCM['chg']))


        return conRates, conDmds, nonConDmds, conPaths, nonConPaths, conODTs, conRMax, dirConRates, nonConODTs

    def addData(self, dfP):
        """
        Route -> path-level attributes:
          pathDmd: mapping pathID -> demandID
          dirPaths: mapping demandID -> direct path pathID (vendor-directs)
        """
        pathDmd = dict(zip(dfP['pathID'], dfP['demandID']))
        dPaths = dfP[dfP['arc1_type'] == 'VND->LMD'].copy()
        dirPaths = dict(zip(dPaths['demandID'], dPaths['pathID']))

        return pathDmd, dirPaths
    
    def conservatism(self, conRates, dfP, conserv, maxLTLShip, bigMTime):
        dfLTs = pd.Series(conRates).reset_index()   
        dfLTs.columns = ['demandID', 'current_ODT', 'conv']

        dfConP = dfP[dfP['arc1_type']=='VND->FC'].copy()
        dfConP = dfConP.drop(columns='current_ODT')
        dfConP = pd.merge(dfConP,dfLTs, on=['demandID'])
        dfConP = dfConP[dfConP['current_ODT'] >= dfConP['fixedTT']+dfConP['arcCt']*0.5].copy()
        dfConP['w_hat'] = dfConP['current_ODT']-dfConP['fixedTT']

        #adjusting conservatism values for each route
        #adjusting conservatism values for each route if c != 0.5
        if conserv == 0.5:
            dfConP['cVal'] = 0.5
        else:
            dfConP['cVal'] = dfConP.apply(lambda x: probChk(x['w_hat'],x['arcCt'],conserv,x['transitMode'],maxLTLShip,bigMTime), axis = 1)
        dfConP = dfConP[(dfConP['cVal']!=1)].copy()
        #pulling the conservatism values 
        cVals = dict(zip(zip(dfConP['pathID'],dfConP['current_ODT']), dfConP['cVal']))

        dfFCP = dfP[dfP['arc1_type'].isin(['FC->FC','FC->LMD'])].copy()
        if conserv == 0.5:
            dfFCP['cVal'] = 0.5
        else:
            dfFCP['cVal'] = dfFCP.apply(lambda x: probChk(x['w_hat'],x['arcCt'],conserv,x['transitMode'],maxLTLShip,bigMTime), axis = 1)
        dfFCP = dfFCP[(dfFCP['cVal']!=1)].copy()

        cValsFC = dict(zip(dfFCP['pathID'],dfFCP['cVal']))

        return cVals, cValsFC

def defPMObj(data, odt_data, conserv):
    """
    Profit maximization objective parameters.
    """
    dfA = data.dfA.copy()
    dfP = data.dfP.copy()
    ##adjusting direct route costs for conservatism 
    DRLaneParams = (dfA[(dfA['originType']=='VND')
                            &(dfA['destType']=='LMD')][['arcID', 'mode', 'maxWgt', 
                                        'minWgt', 'transitTime']].set_index(['arcID', 'mode']).to_dict('index'))


    dfP[['dirMode', 'dirCOGS', 'dirSales', 'dirCost', 'dirFreq', 'dirODT', 
                'dirWgt', 'dirConR','ODT_max','fixedTT',
                    'w_hat']] = dfP.apply(lambda x: dirPathCostsPM(x['demandID'], x['wgt'], x['arc1'], 
                                                                        dfA, conserv, x['sales'], x['cogs'], odt_data.dirConRates, odt_data.conODTs, 
                                                                        data.arcCosts, DRLaneParams, x['current_ODT'])
                                                    if x['arc1_type']=='VND->LMD' 
                                                    else pd.Series({'dirMode':0, 'dirCOGS':0, 'dirSales':0, 'dirCost':0, 'dirFreq':0, 
                                                                    'dirODT':0, 'dirWgt':0, 'dirConR':0,'ODT_max':0,
                                                                    'fixedTT':x['fixedTT'], 'w_hat':x['w_hat']}),
                                                            axis = 1)

    # Handling cost per pound
    ppCost = dict(zip(dfP.pathID.values,dfP.handling_per_lb.values))
    # drop duplicates by demand to form sales / cogs per demand
    dfP = dfP.drop_duplicates(subset='demandID')
    sales = dict(zip(dfP.demandID.values, dfP.sales.values))
    cogs = dict(zip(dfP.demandID.values, dfP.cogs.values))

    # for directs
    dirPathList = dfP[dfP['arc1_type']=='VND->LMD']['pathID'].tolist()
    dirSales = dict(zip(dfP[dfP['pathID'].isin(dirPathList)]['demandID'],dfP[dfP['pathID'].isin(dirPathList)]['dirSales']))
    dirCOGS = dict(zip(dfP[dfP['pathID'].isin(dirPathList)]['demandID'],dfP[dfP['pathID'].isin(dirPathList)]['dirCOGS']))
    dirPathCost = dict(zip(dfP[dfP['pathID'].isin(dirPathList)]['pathID'],dfP[dfP['pathID'].isin(dirPathList)]['dirCost']))

    return ppCost, dirPathCost, sales, cogs, dirSales, dirCOGS



def dirPathCostsPM(comm, wgt, l, dfL, conserv, sales, cogs, conRates, conLTsC, laneCosts, laneParams, ltUpperBd, bigMTL = 7, 
                maxLTLShip = 5):
    prof = sales - cogs
    la = dfL[dfL['arcID']==l].copy()
    #Initiating the model
    cnd = gp.Model(name = 'PM_MMCW_dir')
    cnd.params.OutputFlag = 0
    # cnd.params.MIPfocus = 1
    cnd.params.presolve = 0
    
    #max time should be LTL transit time plus max wait
    maxLT = max(math.ceil(laneParams[(l,1)]['transitTime'] + (7/2)), ltUpperBd)

    #removing LTs greater than the max allowed
    lts = conLTsC[comm].copy()
    add = {}
    for lt in lts:
        if lt > maxLT:
            conLTsC[comm].remove(lt)
            if comm in add:
                add[comm].append(lt)
            else:
                add[comm] = [lt]
    ltMax = max(conLTsC[comm])
    
    #setting frequency sets for modes
    fVal = {}
    for m in la['mode'].unique():
        if m == 0:
            fVal[m] = [(f+1) for f in range(bigMTL)]
        else:
            fVal[m] = [(f+1) for f in range(maxLTLShip)]
    #for disjunctive constraints
    disjZ = []
    for m in la['mode'].unique():
        for f in fVal[m]:
            disjZ.append((m,f))
            
    #creating the variables
    ##v_{m} are continuous variables that capture the amount of weight flowing on leg l using mode m
    v_vars = cnd.addVars(la['mode'].tolist(), lb = 0, name = 'v') 
    #adding frequency variables f_m
    z_vars = cnd.addVars(disjZ, lb = 0, name = 'z', vtype = GRB.BINARY)
    #w_{t} are binary variables indicating the lead time t that commodity k chooses
    w_vars = cnd.addVars(conLTsC[comm], lb = 0, name = 'w', vtype = GRB.BINARY)
    
    #Objective Function
    #cost of using the lanes 
    costLanes = gp.quicksum(laneCosts[(l,m)]['fixedCost']*gp.quicksum(f*z_vars[m,f] for f in fVal[m])
                            + laneCosts[(l,m)]['varCost']*v_vars[m] for m in la['mode'].tolist())

    #rev for commodities
    ##for conversion demands
    profCon = prof*gp.quicksum(conRates[comm,t]*w_vars[t] for t in conLTsC[comm])

    #combining the costs
    ObjFunct = - profCon + costLanes
    #assigning objective to minimize
    cnd.setObjective(ObjFunct, GRB.MINIMIZE)

    #determining weight flowing along conversion-demand route r
    cnd.addConstr(gp.quicksum(v_vars[m] for m in la['mode'].tolist()) == wgt*gp.quicksum(conRates[comm,t]*w_vars[t] 
                                                          for t in conLTsC[comm]))


    #calculating frequency of trips based on maximum weight - constraint (1i)
    cnd.addConstrs(v_vars[m] <= laneParams[(l,m)]['maxWgt']*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['mode'].tolist())

    #calculating frequency of trips based on minimum weight - constraint (1j)
    cnd.addConstrs(v_vars[m] >= max(0.1, laneParams[(l,m)]['minWgt'])*(gp.quicksum(f*z_vars[m,f] for f in fVal[m]))
                       for m in la['mode'].tolist())
    
    #only one binary for frequency and mode
    cnd.addConstr(gp.quicksum(gp.quicksum(z_vars[m,f] for f in fVal[m]) for m in la['mode'].tolist()) == 1)
    
    #only one binary for lead time 
    cnd.addConstr(gp.quicksum(w_vars[t] for t in conLTsC[comm]) == 1)
    
    #lead time constraint
    cnd.addConstrs(7*conserv*gp.quicksum(1/f*z_vars[m,f] for f in fVal[m]) <=
                     gp.quicksum(t*w_vars[t] for t in conLTsC[comm]) 
                       - gp.quicksum(laneParams[(l,m)]['transitTime']*z_vars[m,f] for f in fVal[m])
                   for m in la['mode'].tolist())

    
    cnd.optimize()
    
    for (m,f) in disjZ:
        if round(z_vars[m,f].x) == 1:
            mode = m
            freq = f
            fixed = laneParams[(l,m)]['transitTime']
    cost = round(cnd.objVal, 2)
    for t in conLTsC[comm]:
        if round(w_vars[t].x) == 1:
            lt = t
            ltWgt = conRates[comm,t]*wgt  
            conRt = conRates[comm,t]

    cogs = round(cogs*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    sales = round(sales*sum(conRates[comm,t]*w_vars[t].x for t in conLTsC[comm]),2)
    logCost = round(sum(laneCosts[(l,m)]['fixedCost']*sum(f*z_vars[m,f].x for f in fVal[m])
                            + laneCosts[(l,m)]['varCost']*v_vars[m].x for m in la['mode'].tolist()),2)

    for comm in add:
        for ltAdd in add[comm]:
            conLTsC[comm].append(ltAdd)
        
    
    return pd.Series({'dirMode':mode, 'dirCOGS':cogs, 'dirSales':sales , 'dirCost':logCost , 'dirFreq':freq, 'dirODT':lt, 'dirWgt': ltWgt,
                     'dirConR':conRt,'ODT_max':ltMax,'fixedTT':fixed, 'w_hat':lt-fixed})
