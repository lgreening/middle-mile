# models/mmtc_odt_bin.py
"""
Binary-linearized multi-mode + ODT model builder. Uses z_{a,m,f} binary selectors
for frequencies and constructs time constraints based on z.
"""
import gurobipy as gp
from gurobipy import GRB
from optFunctions import defPMObj

def gen_zvals(data):
    """Generate list of (arc,mode,freq) triples used for binary-frequency z-vars."""
    zVals = []
    for a, m in data.arcMode:
        # bigMVals should exist for (a,m)
        zVals += [(a, m, f) for f in range(1, data.bigMVals[a, m] + 1)]
    return zVals

# needs the lanes (acr+mode, arcModesU) used, paths used, ODTs used, and weights of paths used in the non-sensitivity run 
## non-sensitivity is chgcurve = 1 
def build(data, odt_data, arcModesU, pathsU, odtsU, pathWgtsU, conserv = 0.8):
    model  = gp.Model(name = 'PM_Bin')
    vars = {}

    #creating the variables
    ##x_r are binary variables indicates whether or not route r is selected
    vars['x'] = model.addVars(data.paths, lb = 0, name='x', vtype = GRB.BINARY)
    ##z_{l,m,f} are binary variables indicating if lane l uses mode m with a frequency of f
    zVals = gen_zvals(data)
    vars['z'] = model.addVars(zVals, lb = 0, name = 'z', vtype = GRB.BINARY)
    ##w_{l,m} are continuous variables that capture the amount of weight flowing on leg l using mode m
    vars['v'] = model.addVars(data.arcMode, lb = 0, name = 'v') 
    #w_{k,t} are binary variables indicating the lead time t that commodity k chooses
    vars['w'] = model.addVars(list(odt_data.conRates.keys()), lb = 0, name = 'w', vtype = GRB.BINARY)
    #u_r are continuous variables that capture the volume flowing along route r
    vars['u'] = model.addVars(odt_data.conPaths, lb = 0, name = 'u')

    #Objective Function
    # Obtain economic parameters (expects you have defPMObj implemented)
    # defPMObj should return: ppCost, dirPathCost, sales, cogs, dirSales, dirCOGS
    ppCost, dirPathCost, sales, cogs, dirSales, dirCOGS = defPMObj(data, odt_data, conserv)

    x = vars.get('x')
    z = vars.get('z')
    v = vars.get('v')
    w = vars.get('w')
    u = vars.get('u')


    # --- costPath: non-conv via x, conv via u (ppCost), direct conv via dirPathCost (if applicable) ---
    # non-conversion paths use data.pathCost and x
    costPath_nonconv = gp.quicksum(data.pathCost[p] * x[p] for p in odt_data.nonConPaths)

    # consolidation path cost: two cases:
    # - if a conPath is not a direct path in dirPaths -> use ppCost * u[p]
    # - if it is a direct candidate (dirPaths maps demand->direct path), treat it as x-cost (direct path cost)
    costPath_conv_pp = gp.quicksum(ppCost[p] * u[p] for p in odt_data.conPaths if p not in odt_data.dirPaths.values())
    costPath_conv_dir = gp.quicksum(dirPathCost[p] * x[p] for p in odt_data.dirPaths.values())

    costPath_expr = costPath_nonconv + costPath_conv_pp + costPath_conv_dir

    # --- revenue ---
    # revenue from conversion demands (depends on chosen lead time via w, and direct sales if direct route chosen)
    # note: odt_data.conODTs is the demand->list of ODTs 
    #cost of the route - first summation in objective
    rev_conv = gp.quicksum(
        (sales[k] - cogs[k]) *
        gp.quicksum(odt_data.conRates[(k, t)] * w[(k, t)] for t in odt_data.conODTs[k])
        + (dirSales[k] - dirCOGS[k]) * x[odt_data.dirPaths[k]]
        for k in odt_data.conDmds
    )

    # revenue from non-conversion demands: fixed revenue (sales - cogs)
    rev_nonconv = gp.quicksum((sales[k] - cogs[k]) for k in odt_data.nonConDmds)

    rev_expr = rev_conv + rev_nonconv

    # --- lane costs ---
    costLanes_expr = gp.quicksum(
        data.arcCosts[a, m]['fixedCost'] * gp.quicksum(f*z[a, m, f] for f in range(1, data.bigMVals[a, m] + 1))+
        data.arcCosts[a, m]['varCost'] * v[a, m]
        for (a, m) in data.arcMode
    )

    # --- Combined objective expression (profit = revenue - costs) ---
    ObjExpr = rev_expr - costPath_expr - costLanes_expr

    # set objective 
    model.setObjective(ObjExpr, GRB.MAXIMIZE)
    model.update()


    #Constraints

    #each demand pair must be satisfied
    model.addConstrs(gp.quicksum(x[r] for r in data.dmdPaths[d]) == 1
                                        for d in list(data.dmdPaths))  

    ### DIFFERENT #determining weight flowing along conversion-demand route r
    model.addConstrs(u[r] <= data.pathParams[r]['wgt']*gp.quicksum(odt_data.conRates[odt_data.pathDmd[r],t]*w[odt_data.pathDmd[r],t] 
                                                          for t in odt_data.conODTs[odt_data.pathDmd[r]]) for r in odt_data.conPaths)
    
    #big M constraint that forces u_r to be 0 if route isn't used
    model.addConstrs(u[r] <= odt_data.conRMax[odt_data.pathDmd[r]]*data.pathParams[r]['wgt']*x[r] for r in odt_data.conPaths)

    #constraint for volume on legs
    model.addConstrs(gp.quicksum(v[l,m] for m in data.modeDict[l]) >= gp.quicksum(u[r] for r in data.arcPaths[l] if r in odt_data.conPaths)
                                + gp.quicksum(data.pathParams[r]['wgt']*x[r] for r in data.arcPaths[l] if r not in odt_data.conPaths)
                                    for l in data.arcs)

    ### CHANGE TO MINIMUM(PLANNED, ACTUAL)
    model.addConstrs(v[l,arcModesU[l][0]] >= min(arcModesU[l][2], 
                                                 sum(data.pathParams[r]['wgt']*odt_data.conRates[odt_data.pathDmd[r],odtsU[odt_data.pathDmd[r]]] for r in data.arcPaths[l] if r in pathsU and r in odt_data.conPaths)
                                                +sum(data.pathParams[r]['wgt'] for r in data.arcPaths[l] if r in pathsU and odt_data.pathDmd[r] not in odt_data.conPaths))
                                      for l in arcModesU if l in data.arcs and arcModesU[l][0] != 0)
    ### adding a constr for u_vars with LTL in the route
    model.addConstrs(u[r] >= min(pathWgtsU[r], odt_data.conRates[odt_data.pathDmd[r],odtsU[odt_data.pathDmd[r]]]*data.pathParams[r]['wgt']) for r in odt_data.conPaths if r in pathsU)

    #calculating frequency of trips based on maximum weight - constraint (1i)
    model.addConstrs(v[l,m] <= data.arcParams[(l,m)]['maxWgt']*(gp.quicksum(f*z[l, m, f] for f in range(1, data.bigMVals[l, m] + 1)))
                    for l,m in data.arcMode)

    #calculating frequency of trips based on minimum weight - constraint (1j)
    model.addConstrs(v[l,m] >=  data.arcParams[(l,m)]['minWgt']*(gp.quicksum(f*z[l, m, f] for f in range(1, data.bigMVals[l, m] + 1)))
                    for l,m in data.arcMode if data.arcParams[(l,m)]['minWgt'] > 0)

    #only one mode for each leg allowed - constraint (1e)
    model.addConstrs(gp.quicksum(gp.quicksum(z[l, m, f] for f in range(1, data.bigMVals[l, m] + 1)) for m in data.modeDict[l]) <= 1 for l in data.arcs)


    #Lead Time-related Constraints

    #only one lead time selected for each commodity
    model.addConstrs(gp.quicksum(w[k,t] for t in odt_data.conODTs[k]) == 1 - x[odt_data.dirPaths[k]]
                            for k in odt_data.conDmds)

    #lead time constraints
    model.addConstrs(1/7*gp.quicksum(1/odt_data.cVals[r,t]*(t-data.pathParams[r]['fixedTT'])*w[odt_data.pathDmd[r],t] for t in odt_data.conODTs[odt_data.pathDmd[r]] if (r,t) in odt_data.cVals)
                + (1-x[r])*data.pathParams[r]['arcCt']  
                >= gp.quicksum(
                        gp.quicksum(
                            gp.quicksum(
                                (1 / min(f, data.maxF[m])) * z[a, m, f]
                                for f in range(1, data.bigMVals[a, m] + 1)
                            ) for m in data.modeDict[a]
                        ) for a in data.arcsConstrPaths[r]
                    )
                            for r in odt_data.conPaths)

    model.addConstrs(x[r]*(1/((odt_data.cValsFC[r])*7))*data.pathParams[r]['w_hat'] + (1-x[r])*data.pathParams[r]['arcCt']  
                >= gp.quicksum(
                        gp.quicksum(
                            gp.quicksum(
                                (1 / min(f, data.maxF[m])) * z[a, m, f]
                                for f in range(1, data.bigMVals[a, m] + 1)
                            ) for m in data.modeDict[a]
                        ) for a in data.arcsConstrPaths[r]
                    ) 
                            for r in data.conPaths if r not in odt_data.conPaths and r in odt_data.cValsFC)


        
    ##restricting modes on legs for certain routes
    ### truckload
    model.addConstrs(x[r] <= gp.quicksum(z[data.arcTL[r], 0, f] for f in range(1, data.bigMVals[data.arcTL[r], 0] + 1)) for r in data.arcTL)
    ### LTL
    model.addConstrs(x[r] <= gp.quicksum(gp.quicksum(z[data.arcLTL[r], m, f] for f in range(1, data.bigMVals[data.arcLTL[r], m] + 1))
                                          for m in data.modeDict[data.arcLTL[r]] if m!=0) for r in data.arcLTL)
    

    for (k,t) in list(odt_data.conRates.keys()):
        if t != odtsU[k]:
            w[k,t].ub = 0
    
    for r in pathsU:
        x[r].lb = 1

    ## fixing modes 
    for l,m in data.arcMode:
        if l in arcModesU:
            ## prevent all modes and frequencies not equal to original
            if m != arcModesU[l][0]: 
                for f in range(1, data.bigMVals[l, m] + 1):
                    z[l,m,f].ub = 0 
            # prevent frequencies not used for mode used
            else:
                for f in range(1, data.bigMVals[l, m] + 1):
                    if f != arcModesU[l][1]:
                        z[l,m,f].ub = 0
                    else:
                        z[l,m,f].lb = 1
        else:
            for f in range(1, data.bigMVals[l, m] + 1):
                z[l,m,f].ub = 0

    model.update()
    return model, vars