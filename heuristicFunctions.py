import pandas as pd
import numpy as np
import math


#function to convert distance dataframes into distance dictionaries with distances ordered
def distDF_toDict(dfD, origin=True):
    distDict = {}
    #a dictionary where the keys are origin IDs and values are the list of remaining origins ordered by distance to the key
    if origin:
        dfD = dfD[dfD['locationType1'].str.contains('V')&(dfD['locationType2'].str.contains('V'))].copy()
        for o in dfD['locationID1'].unique():
            df_temp = dfD[dfD['locationID1']==o].copy()
            df_temp = df_temp.sort_values(by='distance')
            distDict[o] = df_temp['locationID2'].tolist()
    else:
        dfD = dfD[dfD['locationType1'].str.contains('L')&(dfD['locationType2'].str.contains('L'))].copy()
        for d in dfD['locationID1'].unique():
            df_temp = dfD[dfD['locationID1']==d].copy()
            df_temp = df_temp.sort_values(by='distance')
            distDict[d] = df_temp['locationID2'].tolist()

    #returns dictionary
    return distDict

#Creating class for heuristic params
class heuristicData:
    def __init__(self, data, distance_data):
        self.pathList, self.dfDmds, self.dfDist = self.necessaryData(data, distance_data)
        self.dmdsOrig, self.origList, self.origVolWgts, self.distDict  = self.originParams(self.dfDmds, self.dfDist)
        self.dmdsLMD, self.destList, self.destVolWgts, self.destDistDict = self.destParams(self.dfDmds, self.dfDist)
    def necessaryData(self, data, distance_data):
        pathList = data.dfP[~(data.dfP['arc1_type'].str.contains('FC->'))]['pathID'].tolist()
        #dictionary of demands for each origin location
        dfDmds = data.dfP[~(data.dfP['arc1_type'].str.contains('FC->'))][['demandID','originID','finalDest','wgt']].drop_duplicates(subset=['demandID']).copy() 
        # read in distance data dataframe
        dfDist = pd.read_csv(distance_data)
        
        return pathList, dfDmds, dfDist
    
    def originParams(self, dfDmds, dfDist):
        #grouping dmds by originID
        dmdsOrig = dfDmds.groupby('originID')['demandID'].apply(list).to_dict() 
        #aggregating origin volume to calculate weights
        dfWgtSum = dfDmds.groupby('originID').agg({'wgt':'sum'}).reset_index()
        dfWgtSum = dfWgtSum.sort_values(by = 'wgt', ascending = False)
        origList = dfWgtSum['originID'].tolist() 
        dfWgtSum['weights'] = dfWgtSum['wgt'].values/dfWgtSum['wgt'].sum()
        origVolWgts = dfWgtSum['weights'].tolist()

        #getting vendor-only distance dictionary from dfDist
        dfDist = dfDist[(dfDist['locationID1'].isin(origList))
                        &(dfDist['locationID2'].isin(origList))].copy()
        distDict = distDF_toDict(dfDist, origin=True)
        
        return dmdsOrig, origList, origVolWgts, distDict
    
    def destParams(self, dfDmds, dfDist):
        #dictionary of demands for each destination location
        dmdsLMD = dfDmds.groupby(['finalDest'])['demandID'].apply(list).to_dict() 
        #aggregating destination volume to calculate weights
        dfWgtSum = dfDmds.groupby('finalDest').agg({'wgt':'sum'}).reset_index()
        dfWgtSum = dfWgtSum.sort_values(by = 'wgt', ascending = False)
        destList = dfWgtSum['finalDest'].tolist() 
        dfWgtSum['weights'] = dfWgtSum['wgt'].values/dfWgtSum['wgt'].sum()
        destVolWgts = dfWgtSum['weights'].tolist()
        
        #creating dictionary where keys are dest IDs and values are lists of other dests ordered by distance to the key
        dfDist = dfDist[(dfDist['locationID1'].isin(destList))
                        &(dfDist['locationID2'].isin(destList))].copy()
        distDict = distDF_toDict(dfDist, origin=False)
        
        return dmdsLMD, destList, destVolWgts, distDict
    

#given a list and weights per item, 
## this function outputs a weighted-random ordered list
def randWtdList(inputList, wgts):
    
    #selecting a random group of origins to free their paths
    order = np.random.choice(len(inputList), len(inputList), p=wgts, replace=False)
    rndList = [inputList[o] for o in order]  
    
    return rndList

#given a list, 
## this function outputs a random ordering of the list (equally-weighted)
def randList(inputList):
    
    #selecting a random group of origins to free their paths
    order = np.random.choice(len(inputList), len(inputList), replace=False)
    rndList = [inputList[o] for o in order]  
    
    return rndList

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
    originList = [vndList[origin]] + distDict[vndList[origin]].copy()
    return originList, prevVnds

def selPaths(listOrd, dfP, dmds, pathLen, ct = 0, newList = True):
    #selecting paths 
    selectedPaths = []
    listSt = ct
    #collecting demands until path ct met
    while len(selectedPaths) < pathLen:
        l = listOrd[ct]
        ct += 1
        selectedPaths += dfP.loc[(dfP['demandID'].isin(dmds[l])), 'pathID'].tolist()
        if ct == len(dmds): 
            newList = True
            ct = 0
            if listSt == 0:
                break
        elif listSt == ct:
            break
    return selectedPaths, newList, ct

def select(data, heurData, newList, selNH, prevVnds, lmdOrd, lmd, fc, pathLen, iterCount):
    np.random.seed(iterCount)
    ##  NH2
    if selNH == 'vndWtdDist':
        origOrd, prevVnds = distOriginList(heurData.origList, heurData.origVolWgts, heurData.distDict, prevVnds)
        selectedPaths, newListVndPH, ctPH = selPaths(origOrd, data.dfP, heurData.dmdsOrig, pathLen)
    ##  NH3
    elif selNH == 'randLMD':
        origOrd = []
        if newList:
            lmd = 0
            lmdOrd = randList(list(heurData.dmdsLMD))
            # set newList to false so entire list is used
            newList = False
        selectedPaths, newList, lmd = selPaths(lmdOrd, data.dfP, heurData.dmdsLMD, pathLen, lmd, newList)
    ##  NH1
    elif selNH == 'randVndWtd':
        origOrd = randWtdList(heurData.origList, heurData.origVolWgts)
        selectedPaths, newListVndPH, ctPH = selPaths(origOrd, data.dfP, heurData.dmdsOrig, pathLen)      
    else:
        raise Exception('selNH not set properly')
    return newList, selectedPaths, prevVnds, lmdOrd, lmd, fc, origOrd
