import sys
import argparse
import distutils
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import time
import math

from optFunctions import optData, odt_data
from heuristicFunctions import heuristicData, select
from mmtc_odt_bin import build as build_mmtc_odt_bin
from mmtc_odt_pw import build as build_mmtc_odt_pw

parser = argparse.ArgumentParser()
parser.add_argument('--inst', type=str, help='instance - size and iteration')
parser.add_argument('--instFolder', type=str, help='directory where instances are stored')
parser.add_argument('--timeLimit', type=int, help='time limit of mip in seconds')
parser.add_argument('--resultsFolder', type=str, help='directory where results are stored')
parser.add_argument('--wsFolder', type=str, help='directory where warm start files are stored')
parser.add_argument('--percPaths', type=float, help='minimum percentage of paths to use for neighborhood')
parser.add_argument('--loc',type=str,help='location of files')
parser.add_argument('--minODT', type=int, help='number of units ODT can be reduced by', default=0)
parser.add_argument('--maxODT', type=int, help='number of units ODT can be increased by', default=0)
parser.add_argument('--modelType', type=str, help='model type: bin or piecewise', default='bin')
args = parser.parse_args()

## maximum number of LTL shipments per week
maxLTLShip = 5
## maximum number of truckload shipments for time constraints
bigMTime = 14
## maximum number of truckload shipments for volume constraints
bigMTL = 40
## maximum volume (in lbs) per truckload
Q = 12000

## time limit for single LNS iteration
itTime = 300

## defining the data
arcData = f'{args.instFolder}/arcs_{args.inst}.csv'
pathData = f'{args.instFolder}/paths_{args.inst}.csv'
dmdData = f'{args.instFolder}/dmds_{args.inst}.csv'
convRateData = f'{args.instFolder}/conversion_rates.csv'
distanceData = f'{args.instFolder}/distance_matrix.csv'

## class of data objects for model
data = optData(dmdData, arcData, pathData, convRateData, maxLTLShip, bigMTime, bigMTL, args.singleMode, Q, args.outsource)
## class of data objects for ODT-specific data
odt_data = odt_data(data.dfP, convRateData, args.maxODT, args.minODT)

## reading in the warm start file
if args.warmStart:
    print('using warm start')
    dfWS = pd.read_csv(f'{args.wsFolder}/ws_{args.inst}.csv')
else:
    print('no warm start specified, using direct path')
    # use direct path data as warm start
    dfWS = data.dfP[data.dfP['DR_COST'] > 0].copy()

## generating the model
## and instance name
model = f'mmtc_odt_{args.modelType}'
if args.modelType == 'bin':
    mmtc_odtMip, vars = build_mmtc_odt_bin
else:
    mmtc_odtMip, vars = build_mmtc_odt_pw

instNm = f'{model}_Heur_{args.inst}_min{args.minODT}_max{args.maxODT}_{args.timeLimit}s'
print(instNm)

## set the warm start solution
### fixing all paths in the warm start to 1
[mmtc_odtMip.setAttr('LB', vars['x'][p], 1) for p in dfWS['pathID']]
### allowing a long time limit to give sufficient time to generate WS         
mmtc_odtMip.Params.timelimit = 600
mmtc_odtMip.optimize()
### resetting the path variables
[mmtc_odtMip.setAttr('LB', vars['x'][p], 0) for p in dfWS['pathID']]

#to set a time limit (seconds) for individual heuristic iterations
mmtc_odtMip.Params.timelimit = itTime
## setting a smaller MIP gap to not get stuck trying to prove optimality
mmtc_odtMip.Params.mipGap = 0.005
## turning off gurobi logs
mmtc_odtMip.Params.OutputFlag = 0


###LNS HEURISTIC#####################################

## class for heuristic-related data
heurData = heuristicData(data, distanceData)

## used for print statement
spacer = '--------------------'
## initializing the iteration count (used for random seed)
### and number of non-improving iterations for the current neighborhood
iterCount, currNeigh = 0, 0
## initializing the number of overall non-improving iterations
### and the current objective value
curr, currObj = 0, mmtc_odtMip.objVal
## initializing the cumulative gurobi solve time and heuristic solve time
cumulaTime, currTime = 0, 0
## initializing neighborhoods - adjust list to neighborhoods you want to use (ordered)
neighborhoods = ['randVndWtd', 'vndWtdDist', 'randLMD']
## sets the first neighborhood as the first in the list of NHs
selNH = neighborhoods[0]
## initializes the variable to select the next NH in the list
sel = 1
## for the 'vndWtdDist' NH - list of previously selected vendors
prevVnds = []
## for 'fcVnd' NH - fc is the position in the list of FCs; selects first fc with 0
fc = 0
## for 'randLMD' NH - once given a list of randomly ordered LMDs, starts with first; 
### continues through the list with this counter and then resets at the end (goes across iterations)
lmd = 0
## for 'randLMD' NH - the randomly ordered list of LMDs the heuristic is cycling through
lmdOrd = []
## for 'randLMD' NH - if true, need to generated a new lmdOrd list
newList = True
## if true, LNS stops; happens after stopIt non-improving iterations
quitRunning = False
stopIt = 30
## initialize to optimize paths first
chooseODT = False
## counts the number of times we optimized for ODTs and aths
choosePCt = 0
chooseODTCt = 0
## counts the number of iterations with mip gaps >= 2%
mipCtP = 0
mipCtODT = 0
## counts the number of iterations with mip gaps < 2%
mipRedCtP = 0
mipRedCtODT = 0
## only for paper 2 (not adj) 
### multiplier/step size adjuster for increasing/decreasing neighborhood size
mipRedIncrP = 1
mipRedIncrODT = 1
mipIncrP = 1
mipIncrODT = 1
## initializing selected to fix all paths to the current solution
### paths originating at FCs are always free to change (these are not in pathList)
selected = heurData.pathList.copy()
pathsList = heurData.pathList.copy()
## collecting paths used in the solution and fixing to 0 if not used
pathsUsed = []
for p in selected:
    if round(vars['x'][p].x) == 0:
        vars['x'][p].ub = 0
    else:
        pathsUsed.append(p)
        
## initializing the objective and time lists
objList = [round(mmtc_odtMip.objVal,1)]
lbList = [round(mmtc_odtMip.objBound,1)]
gapList = [mmtc_odtMip.MIPGap]
iterList = [0]
timeList = [0]

## percentage of paths defines (minimum) neighborhood size
percPaths = args.percPaths
percPathsPaths = percPaths/2
percPathsODT = percPaths
## minimum number of paths to collect for each neighborhood
percPaths = percPathsPaths
pathLen = percPaths*len(pathsList)

heurStart = time.time()
## while the total heuristic time is less than the timeLimit, continue to iterate
while currTime <= args.timeLimit*2/3:
    if args.maxLT == 0 and args.minLT == 0:
        print('alternating heuristic not needed')
        break
    ############################
    # just for print statement #
    preSolu = mmtc_odtMip.objVal
    print(' ')
    print(spacer)
    iterCount += 1
    print('Iteration '+str(iterCount))
    print('percPaths: '+str(percPaths))
    print('neighborhood: '+str(selNH))
    print('Single focus: path '+str(chooseODT))
    print('Heuristic running time (sec): '+str((currTime)))
    print('Cumulative solve time (sec): '+str(cumulaTime))
    stL = time.time()
    ############################
    
    ## selecting paths in neighborhood
    newList, selected, prevVnds, lmdOrd, lmd, fc = select(data, heurData, newList, selNH, prevVnds,
                                                         lmdOrd, lmd, fc, pathLen, iterCount)
    
    ## freeing selected paths by setting upper bound to 1
    [mmtc_odtMip.setAttr('UB', vars['x'][p], 1) for p in selected]
    ## more print statements for checking heuristic
    print(str(len(selected))+' paths selected')
    dmdsCh = data.dfP[data.dfP['pathID'].isin(selected)]['demandID'].unique().tolist()
    print(str(len(dmdsCh))+' commodities selected')
    ## freeing ODTs if optimizing ODTs
    if chooseODT:
        dmdODTSelPl = [(k,t) for (k,t) in vars['w'].keys() if k in dmdsCh]
        [mmtc_odtMip.setAttr('UB', vars['w'][k,t], 1) for (k,t) in dmdODTSelPl]
        print(str(len(dmdODTSelPl))+' additional lead time variables freed')

    ## solving the (restricted) model
    mmtc_odtMip.optimize()
    heurObj = round(mmtc_odtMip.objVal,2)
    print('Heuristic objective: '+str(heurObj))
    print('Heuristic gap: '+str(round(100*mmtc_odtMip.MIPGap,2))+'%')
    print('Absolute improvement: '+str(round(preSolu - heurObj,1)))
    print('Percent improvement: '+str(round(preSolu - heurObj,1)/preSolu *100))
    objList.append(heurObj)
    lbList.append(round(mmtc_odtMip.objBound,1))
    gapList.append(mmtc_odtMip.MIPGap)
    iterList.append(iterCount)
    ## adding runtime for the cumulative gurobi solve time
    cumulaTime += mmtc_odtMip.Runtime
    timeList.append(round(cumulaTime))
    
    ## fixing all paths to 0 if they previously were not selected
    [mmtc_odtMip.setAttr('UB', vars['x'][p], 0) for p in selected if vars['x'][p].x < 0.1]
    if chooseODT:
        ## fixing all ODT lead time vars to 0 if they previously were not selected
        [mmtc_odtMip.setAttr('UB', vars['w'][k,t], 0) for (k,t) in dmdODTSelPl if vars['w'][k,t].x < 0.1]

    ## collecting paths used in the solution
    pathsUsed = [p for p in list(set(pathsUsed + selected)) if vars['x'][p].x > 0.9]
    
    ## if the percentage of paths to collect is over 75% of total 
    ### and the solution is not changing and the gap is below 0.55%
    ### increase curr (the count of non-improving iterations) by 5
    if percPaths > 0.9 and currObj - mmtc_odtMip.objVal <= 5 and mmtc_odtMip.MIPGap <= 0.0055:
        print('no improvement at max percPaths')
        curr += 6 - max(ctFoc,0)
        ctFoc = 10

    ## increasing the percent of paths that are optimized if previous IPs have been solving well
    if round(mmtc_odtMip.MIPGap,3) < 0.02:
        ## count of consecutive iterations with small mip gaps
        mipCt += 1
        mipRedIncr = 1
        ## resetting count of iterations with large mip gaps to 0
        mipRedCt = 0
        ## if mipCt is 6 or more, increase the percentage of paths by 2%
        if mipCt >= 6: 
            mipCt = 0
            percPaths += 0.02*mipIncr
            ## percPaths cannot exceed 100%
            percPaths = min(1,percPaths)
            print('increasing percPaths to '+str(percPaths))
            mipIncr += 1
            pathLen = percPaths*len(pathsList)
    else:
        ## count of consecutive iterations with large mip gaps
        mipRedCt += 1
        mipIncr = 1
        ## resetting count of iterations with small mip gaps to 0
        mipCt = 0
        ## if mipRedCt is 3 or more, decrease the percentage of paths by 2%
        if mipRedCt >= 3: 
            mipRedCt = 0
            percPaths -= 0.02*mipRedIncr
            ## percPaths cannot be less than 0.01
            percPaths = max(0.01, percPaths)
            print('decreasing percPaths to '+str(percPaths))
            mipRedIncr += 1
            pathLen = percPaths*len(pathsList)


    ## if the improvement is less than 1% of the current objective, considered "non-improving" for NHs
    if mmtc_odtMip.objVal - currObj <= currObj*0.0001:
        ## increases count of non-improving iterations for the current NH
        currNeigh += 1
        print('nonImproving neighborhood iterations: '+str(currNeigh))
        ## if current NH has had 5 or more non-improving NH iterations, switch to the next in the list
        if currNeigh >=5:
            print('currNeigh >= 5')
            currNeigh = 0
            ## selecting new NH from list
            selNH = neighborhoods[sel]
            print('using '+selNH)
            sel += 1
            ## resets sel if using the last NH in the list
            if sel == len(neighborhoods):
                sel = 0
            ## resets randLMD to create a new list 
            newList = True
    else:
        ## if the current NH found an improving solution, reset the non-improving count
        currNeigh = 0
    ## if the improvement is less than 0.1, considered "non-improving" for the overall heuristic
    if mmtc_odtMip.objVal - currObj <= currObj*0.00005:
        ## increases count of non-improving iterations
        curr += 1
        print('nonImproving iterations: '+str(curr))
        ## if the number of non-improving iterations exceeds stopIt, heuristic will stop running
        if curr >=1+stopIt:
            print('curr >= 1 + stopIt')
            quitRunning = True
    ## if improvement exceeds 0.1, number of non-improving iterations is reset and heuristic will continue to run
    else:
        currObj = mmtc_odtMip.objVal
        curr = 0  
        quitRunning = False
            
    if (ctFoc >= 6) and chooseODT == True:
        print('ctFoc >= 6; switching to ODT focus')
        #setting route list to physical route chosen and all avalaible ODTs
        pathsList = pathsUsed.copy()
        dfP = data.dfP[data.dfP['pathID'].isin(pathsList)].copy()
        if quitRunning == True:
            chooseODTCt += 1
            print('chooseODTCt = '+str(chooseODTCt))
        chooseODT = False
        runTime = 0
        print('before: pr '+str(percPaths)+', mC '+str(mipCt)+', mRC '+str(mipRedCt)+', mI'+str(mipIncr)+', mRI'+str(mipRedIncr))
        
        percPathsPaths = percPaths
        mipCtPaths = mipCt
        mipIncrPaths = mipIncr
        mipRedCtPaths = mipRedCt
        mipRedIncrPaths = mipRedIncr

        percPaths = percPathsODT
        mipCt = mipCtODT
        mipRedCt = mipRedCtODT
        mipIncr = mipIncrODT
        mipRedIncr = mipRedIncrODT
        print('after: pr ' +str(percPaths)+', mC '+str(mipCt)+', mRC ' +str(mipRedCt)+', mI'+str(mipIncr)+', mRI'+str(mipRedIncr))
        ctFoc = 0
        
        rtLen = percPaths*len(pathsList)
            
    elif (ctFoc >= 6):
        print('ctFoc >= 6; switching to Route focus')
        #setting route list to ODT chosen and physical paths that can meet that ODT
        pathsList = heurData.pathList.copy()
        dfR = data.dfP.copy()
        if quitRunning == True:
            chooseRtCt += 1
            print('chooseRtCt = '+str(chooseRtCt))
        chooseODT = True
        runTime = 0
        print('before: pr ' +str(percPaths)+', mC '+str(mipCt)+', mRC ' +str(mipRedCt)+', mI'+str(mipIncr)+', mRI'+str(mipRedIncr))

        percPathsODT = percPaths
        mipCtODT = mipCt
        mipIncrODT = mipIncr
        mipRedCtODT = mipRedCt
        mipRedIncrODT = mipRedIncr

        percPaths = percPathsPaths
        mipCt = mipCtPaths
        mipRedCt = mipRedCtPaths
        mipIncr = mipIncrPaths
        mipRedIncr = mipRedIncrPaths
        
        rtLen = percPaths*len(pathsList)


        print('after: pr ' +str(percPaths)+', mC '+str(mipCt)+', mRC ' +str(mipRedCt)+', mI'+str(mipIncr)+', mRI'+str(mipRedIncr))
        ctFoc = 0

    ## if number of non-improving iterations > stopIt 
    ### and the total gurobi solve time exceeds 5 minutes (used for small instances)
    ### stops running LNS 
    if quitRunning and cumulaTime > 300:
        print('quitRunning')
        break

    print(spacer)
    currTime = time.time()-heurStart

print(spacer)    
print(' ')
print(' ')

curr = 0
currNeigh = 0
quitRunning = False
newList = True
#initializing neighborhoods
selNH = neighborhoods[0]
sel = 1
prevVnds = []
vnd = 0
lmd = 0
print(spacer)
print(spacer)
print('switching to joint')
print(spacer)
print(spacer)
print(' ')
# reducing percPaths since optimizing both paths and ODTs
percPaths = args.percPaths/3
mipRedCt = 0
mipRedIncr = 1
mipCtPaths = 0
mipIncr = 1
incrPR = False
stopIt = 30
pathsList = heurData.pathList.copy()
pathLen = percPaths*len(pathsList)

## while the total heuristic time is less than the timeLimit, continue to iterate
while currTime <= args.timeLimit:
    ############################
    # just for print statement #
    preSolu = mmtc_odtMip.objVal
    print(' ')
    print(spacer)
    iterCount += 1
    print('Iteration '+str(iterCount))
    print('percPaths: '+str(percPaths))
    print('neighborhood: '+str(selNH))
    print('Single focus: path '+str(chooseODT))
    print('Heuristic running time (sec): '+str((currTime)))
    print('Cumulative solve time (sec): '+str(cumulaTime))
    stL = time.time()
    ############################
    
    ## selecting paths in neighborhood
    newList, selected, prevVnds, lmdOrd, lmd, fc = select(data, heurData, newList, selNH, prevVnds,
                                                         lmdOrd, lmd, fc, pathLen, iterCount)
    
    ## freeing selected paths by setting upper bound to 1
    [mmtc_odtMip.setAttr('UB', vars['x'][p], 1) for p in selected]
    ## more print statements for checking heuristic
    print(str(len(selected))+' paths selected')
    dmdsCh = data.dfP[data.dfP['pathID'].isin(selected)]['demandID'].unique().tolist()
    print(str(len(dmdsCh))+' commodities selected')
    ## freeing ODTs 
    dmdODTSelPl = [(k,t) for (k,t) in vars['w'].keys() if k in dmdsCh]
    [mmtc_odtMip.setAttr('UB', vars['w'][k,t], 1) for (k,t) in dmdODTSelPl]
    print(str(len(dmdODTSelPl))+' additional lead time variables freed')

    ## solving the (restricted) model
    mmtc_odtMip.optimize()
    heurObj = round(mmtc_odtMip.objVal,2)
    print('Heuristic objective: '+str(heurObj))
    print('Heuristic gap: '+str(round(100*mmtc_odtMip.MIPGap,2))+'%')
    print('Absolute improvement: '+str(round(preSolu - heurObj,1)))
    print('Percent improvement: '+str(round(preSolu - heurObj,1)/preSolu *100))
    objList.append(heurObj)
    lbList.append(round(mmtc_odtMip.objBound,1))
    gapList.append(mmtc_odtMip.MIPGap)
    iterList.append(iterCount)
    ## adding runtime for the cumulative gurobi solve time
    cumulaTime += mmtc_odtMip.Runtime
    timeList.append(round(cumulaTime))
    
    ## fixing all paths to 0 if they previously were not selected
    [mmtc_odtMip.setAttr('UB', vars['x'][p], 0) for p in selected if vars['x'][p].x < 0.1]
    ## fixing all ODT lead time vars to 0 if they previously were not selected
    [mmtc_odtMip.setAttr('UB', vars['w'][k,t], 0) for (k,t) in dmdODTSelPl if vars['w'][k,t].x < 0.1]

    ## collecting paths used in the solution
    pathsUsed = [p for p in list(set(pathsUsed + selected)) if vars['x'][p].x > 0.9]
    
    ## if the percentage of paths to collect is over 75% of total 
    ### and the solution is not changing and the gap is below 0.55%
    ### increase curr (the count of non-improving iterations) by 5
    if percPaths > 0.75 and currObj - mmtc_odtMip.objVal <= 0.1 and mmtc_odtMip.MIPGap <= 0.0055:
        print('no improvement at max percPaths')
        curr += 5

    ## increasing the percent of paths that are optimized if previous IPs have been solving well
    if round(mmtc_odtMip.MIPGap,3) < 0.02:
        ## count of consecutive iterations with small mip gaps
        mipCt += 1
        mipRedIncr = 1
        ## resetting count of iterations with large mip gaps to 0
        mipRedCt = 0
        ## if mipCt is 6 or more, increase the percentage of paths by 2%
        if mipCt >= 6: 
            mipCt = 0
            percPaths += 0.02*mipIncr
            ## percPaths cannot exceed 100%
            percPaths = min(0.8,percPaths)
            print('increasing percPaths to '+str(percPaths))
            mipIncr += 1
            pathLen = percPaths*len(pathsList)
    else:
        ## count of consecutive iterations with large mip gaps
        mipRedCt += 1
        mipIncr = 1
        ## resetting count of iterations with small mip gaps to 0
        mipCt = 0
        ## if mipRedCt is 3 or more, decrease the percentage of paths by 2%
        if mipRedCt >= 3: 
            mipRedCt = 0
            percPaths -= 0.02*mipRedIncr
            ## percPaths cannot be less than 0.01
            percPaths = max(0.01, percPaths)
            print('decreasing percPaths to '+str(percPaths))
            mipRedIncr += 1
            pathLen = percPaths*len(pathsList)


    ## if the improvement is less than 1% of the current objective, considered "non-improving" for NHs
    if mmtc_odtMip.objVal - currObj <= currObj*0.0001:
        ## increases count of non-improving iterations for the current NH
        currNeigh += 1
        print('nonImproving neighborhood iterations: '+str(currNeigh))
        ## if current NH has had 5 or more non-improving NH iterations, switch to the next in the list
        if currNeigh >=5:
            print('currNeigh >= 5')
            currNeigh = 0
            ## selecting new NH from list
            selNH = neighborhoods[sel]
            print('using '+selNH)
            sel += 1
            ## resets sel if using the last NH in the list
            if sel == len(neighborhoods):
                sel = 0
            ## resets randLMD to create a new list 
            newList = True
    else:
        ## if the current NH found an improving solution, reset the non-improving count
        currNeigh = 0
    ## if the improvement is less than 0.1, considered "non-improving" for the overall heuristic
    if mmtc_odtMip.objVal - currObj <= 0.1:
        ## increases count of non-improving iterations
        curr += 1
        print('nonImproving iterations: '+str(curr))
        ## if the number of non-improving iterations exceeds stopIt, heuristic will stop running
        if curr >=1+stopIt:
            print('curr >= 1 + stopIt')
            quitRunning = True
    ## if improvement exceeds 0.1, number of non-improving iterations is reset and heuristic will continue to run
    else:
        currObj = mmtc_odtMip.objVal
        curr = 0  
            
    


        print('after: pr ' +str(percPaths)+', mC '+str(mipCt)+', mRC ' +str(mipRedCt)+', mI'+str(mipIncr)+', mRI'+str(mipRedIncr))
        ctFoc = 0

    ## if number of non-improving iterations > stopIt 
    ### and the total gurobi solve time exceeds 5 minutes (used for small instances)
    ### stops running LNS 
    if quitRunning and cumulaTime > 300:
        print('quitRunning')
        break

    print(spacer)
    currTime = time.time()-heurStart
solveTime = round(time.time() - heurStart,2)

    
## sets gap to 100% bc we do not have a provable gap
gap = 1
## same for setting the lower bound to 0
lwrBd = 0
## last objective value
objVal = round(mmtc_odtMip.objVal,0) 
## total heuristic running time
runTime = round(solveTime)

print(' ')
print(spacer)
print(f'final objective: {objVal}')

binVars = mmtc_odtMip.numBinVars
intVars = mmtc_odtMip.numIntVars - binVars
contVars = mmtc_odtMip.numVars - mmtc_odtMip.numIntVars
numConstr = mmtc_odtMip.numConstrs


file1 = open(str(args.resultsFolder)+'/'+str(instNm)+"_solveData.csv","w")

for i in range(len(timeList)):
    file1.write(f'{iterList[i]},{timeList[i]},{objList[i]},{lbList[i]},{gapList[i]}\n')
    
file1.close()



#instance stats
numDmds = len(heurData.dfD)
numVnds = len(heurData.dfD[heurData.dfD['originID'].str.contains('V')]['originID'].unique())
numFcs = len(heurData.dfD[heurData.dfD['originID'].str.contains('F')]['originID'].unique())
numLmds = len(heurData.dfD['destID'].unique())
numPaths = len(data.dfP)
numArcMode = len(data.dfA)
numArc = len(data.dfA['arcID'].unique())
numDirArcMode = len(data.dfA[(data.dfA['originID'].str.contains('V'))&(data.dfA['destID'].str.contains('L'))])
numDirArc = len(data.dfA[(data.dfA['originID'].str.contains('V'))&(data.dfA['destID'].str.contains('L'))]['arcID'].unique())
numConArcMode = numArcMode - numDirArcMode
numConArc = numArc - numDirArc
totVol = data.dfD['wgt'].sum()

file2 = open(f"{args.loc}/{model}HeurStats.csv","a")

file2.write(f'{args.inst},{binVars},{intVars},{contVars},{numConstr},{objVal},{lwrBd},{gap},{runTime},{args.timeLimit},{numVnds},{numFcs},{numLmds},{numDmds},{totVol},{numPaths},{numArcMode},{numArc},{numConArcMode},{numConArc},{numDirArcMode},{numDirArc},{percPaths}\n')

file2.close()
