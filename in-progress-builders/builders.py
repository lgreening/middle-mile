# models/builders.py
"""
Reusable builders for variables, objectives and shared constraints used by
the four models (sm, mmtc, mmtc_odt_pw, mmtc_odt_bin).

Functions mutate the model in-place. They return or update a `vars` dict to
collect variable references.
"""
from typing import Dict, Tuple, Iterable
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


def add_path_vars(model: gp.Model, data, vars: Dict):
    vars['x'] = model.addVars(data.paths, lb=0, vtype=GRB.BINARY, name='x')
    return vars


def add_u_vars_for_odt(model: gp.Model, odt_data, vars: Dict):
    # u variables exist only for consolidation (conversion) paths
    vars['u'] = model.addVars(odt_data.conPaths, lb=0, name='u')
    return vars


def add_arc_mode_vars_piecewise(model: gp.Model, data, vars: Dict):
    """Add the piecewise-style frequency vars: v (continuous), y (mode active binary),
    f (integer number of trips), and h (continuous inverse-frequency proxy per arc)."""
    vars['v'] = model.addVars(data.arcMode, lb=0, name='v')           # volume on (a,m)
    vars['y'] = model.addVars(data.arcMode, lb=0, vtype=GRB.BINARY, name='y')  # mode activation
    vars['f'] = model.addVars(data.arcMode, lb=0, vtype=GRB.INTEGER, name='f') # frequency per (a,m)
    vars['h'] = model.addVars(data.arcs, name='h')  # h per arc
    return vars


def add_arc_mode_vars_binary(model: gp.Model, data, vars: Dict):
    """
    Add z binary vars for (a,m,f) frequencies + v.
    """
    vars['v'] = model.addVars(data.arcMode, lb=0, name='v')  # volume on (a,m)
    zVals = gen_zvals(data)
    vars['z'] = model.addVars(zVals, lb=0, vtype=GRB.BINARY, name='z')
    # do NOT create h here for binary formulation
    return vars


def add_one_path_per_demand(model: gp.Model, data, vars: Dict):
    """Each demand must have exactly one path selected (direct or consolidation)."""
    x = vars['x']
    model.addConstrs(
        (gp.quicksum(x[p] for p in data.dmdPaths[k]) == 1
         for k in data.dmdPaths),
        name='one_path_per_demand'
    )


def add_odt_flow_constraints(model: gp.Model, data, odt_data, vars: Dict):
    """Flow constraints linking u_vars (conversion path flow) with w_vars (ODT selection)
    and x_vars (path selection). Adds lower bound (demand realized) and big-M upper bound.
    Requires vars['u'] and vars['x'] and presence of odt_data."""
    x = vars['x']; u = vars['u']; w = vars['w']
    # lower bound: u_p >= path_wgt * sum(conRate * w_var) - (1-x_p)*conRMax*path_wgt
    model.addConstrs(
        (u[p] >= data.pathParams[p]['wgt'] * gp.quicksum(
            odt_data.conRates[(odt_data.pathDmd[p], t)] * w[(odt_data.pathDmd[p], t)]
            for t in odt_data.conODTs[odt_data.pathDmd[p]]
        ) - (1 - x[p]) * odt_data.conRMax[odt_data.pathDmd[p]] * data.pathParams[p]['wgt']
        for p in odt_data.conPaths),
        name='odt_flow_lower'
    )
    # upper bound big-M: u_p <= conRMax * path_wgt * x_p
    model.addConstrs(
        (u[p] <= odt_data.conRMax[odt_data.pathDmd[p]] * data.pathParams[p]['wgt'] * x[p]
        for p in odt_data.conPaths),
        name='odt_flow_upper'
    )


def add_frequency_activation_piecewise(model: gp.Model, data, vars: Dict):
    """f <= bigM * y activation for piecewise (f,y)."""
    f = vars['f']; y = vars['y']
    model.addConstrs(
        (f[a, m] <= data.bigMVals[a, m] * y[a, m]
        for (a, m) in data.arcMode),
        name='freq_activation_piecewise'
    )


def add_frequency_activation_binary(model: gp.Model, data, vars: Dict):
    """Single z selection per arc constraint for binary formulation."""
    z = vars['z']
    model.addConstrs(
        (gp.quicksum(gp.quicksum(z[a, m, f] for f in range(1, data.bigMVals[a, m] + 1))
                    for m in data.modeDict[a]) <= 1
        for a in data.arcs),
        name='freq_activation_binary'
    )


def add_volume_balance(model: gp.Model, data, vars: Dict, odt_data=None):
    """Volume on each arc must cover flows from consolidation paths OR selected path weights.

    If odt_data is provided, `odt_data.conPaths` are considered conversion paths;
    their flows are tracked by u_vars. Other paths are handled via x_vars*path_wgt.
    """
    v = vars['v']; x = vars['x']
    u = vars.get('u', None)
    if odt_data:
        model.addConstrs(
            (gp.quicksum(v[a, m] for m in data.modeDict[a]) ==
            gp.quicksum(u[p] for p in data.arcPaths[a] if p in odt_data.conPaths)
            + gp.quicksum(data.pathParams[p]['wgt'] * x[p] for p in data.arcPaths[a] if p in odt_data.nonConPaths)
            for a in data.arcs),
            name='volume_balance_odt'
        )
    else:
        model.addConstrs(
            (gp.quicksum(v[a, m] for m in data.modeDict[a]) ==
            gp.quicksum(data.pathParams[p]['wgt'] * x[p] for p in data.arcPaths[a])
            for a in data.arcs),
            name='volume_balance'
        )


def add_maxmin_bounds_piecewise(model: gp.Model, data, vars: Dict):
    """v <= maxWgt * f  and  v >= minWgt * f (if minWgt != 0), for piecewise (f) formulation."""
    v = vars['v']; f = vars['f']
    model.addConstrs(
        (v[a, m] <= data.arcParams[a, m]['maxWgt'] * f[a, m]
        for (a, m) in data.arcMode),
        name='max_bound_piecewise'
    )
    model.addConstrs(
        (v[a, m] >= data.arcParams[a, m]['minWgt'] * f[a, m]
        for (a, m) in data.arcMode
        if data.arcParams[a, m]['minWgt'] != 0),
        name='min_bound_piecewise'
    )


def add_maxmin_bounds_binary(model: gp.Model, data, vars: Dict):
    """v <= maxWgt * sum(f * z) and v >= minWgt * sum(f * z) for binary z formulation.
    Here f is index in the z triple, so 1/min(...) factors and sums are used where needed."""
    v = vars['v']; z = vars['z']
    model.addConstrs(
        (v[a, m] <= data.arcParams[a, m]['maxWgt'] *
        gp.quicksum(f * z[a, m, f] for f in range(1, data.bigMVals[a, m] + 1))
        for (a, m) in data.arcMode),
        name='max_bound_binary'
    )
    model.addConstrs(
        (v[a, m] >= data.arcParams[a, m]['minWgt'] *
        gp.quicksum(f * z[a, m, f] for f in range(1, data.bigMVals[a, m] + 1))
        for (a, m) in data.arcMode
        if data.arcParams[a, m]['minWgt'] != 0),
        name='min_bound_binary'
    )


def add_one_mode_per_arc_piecewise(model: gp.Model, data, vars: Dict):
    """Only one mode active per arc (piecewise with y)."""
    y = vars['y']
    model.addConstrs(
        (gp.quicksum(y[a, m] for m in data.modeDict[a]) <= 1
        for a in data.arcs),
        name='one_mode_per_arc_piecewise'
    )


def add_one_lead_time_per_conversion(model: gp.Model, odt_data, vars: Dict):
    """Each conversion demand chooses exactly one lead time (unless direct route used)."""
    w = vars['w']; x = vars['x']
    model.addConstrs(
        (gp.quicksum(w[(k, t)] for t in odt_data.conODTs[k]) == 1 - x[odt_data.dirPaths[k]]
        for k in odt_data.conDmds),
        name='one_lead_time_per_conversion'
    )


def add_time_constraints_piecewise(model: gp.Model, data, odt_data, vars: Dict):
    """
    Add time constraints for consolidation paths using piecewise (f,y) variables if present,
    otherwise falls back to requiring z-vars (binary) and uses an expression based on z.

    LHS = (1/7) * sum_t (1/cVal(p,t) * (t - fixedTT) * w[pathDmd, t]) + (1-x_p)*arcCt
    RHS = sum over arcs in path of (something depending on f or z)
    Note: if exact min(f, maxF[m]) terms are needed, the z-binary formulation is exact.
    """
    x = vars['x']; w = vars['w']; h = vars.get('h', None)
    # Prefer z-based RHS if z is present (binary linearization)
    if 'z' in vars:
        z = vars['z']
        # RHS: sum_{a in arcsConstrPaths[p]} sum_{m} sum_{f} (1/min(f, maxF[m])) * z[a,m,f]
        model.addConstrs(
            ((1 / 7) *
            gp.quicksum(
                (1 / odt_data.cVals[(p, t)]) *
                (t - data.pathParams[p]['fixedTT']) *
                w[(odt_data.pathDmd[p], t)]
                for t in odt_data.conODTs[odt_data.pathDmd[p]]
                if (p, t) in odt_data.cVals
            )
            + (1 - x[p]) * data.pathParams[p]['arcCt']
            >= gp.quicksum(
                gp.quicksum(
                    gp.quicksum((1 / min(f, data.maxF[m])) * z[a, m, f]
                                for f in range(1, data.bigMVals[a, m] + 1))
                    for m in data.modeDict[a])
                for a in data.arcsConstrPaths[p]
            )
            for p in odt_data.conPaths),
            name='time_cons_z'
        )
    elif ('f' in vars) and ('y' in vars):
        # fallback: use numeric approximation with f_vars and data.maxF for min(...) replacement:
        f = vars['f']
        model.addConstrs(
            ((1 / 7) *
            gp.quicksum(
                (1 / odt_data.cVals[(p, t)]) *
                (t - data.pathParams[p]['fixedTT']) *
                w[(odt_data.pathDmd[p], t)]
                for t in odt_data.conODTs[odt_data.pathDmd[p]]
                if (p, t) in odt_data.cVals
            )
            + (1 - x[p]) * data.pathParams[p]['arcCt']
            >= gp.quicksum(
                gp.quicksum((1 / data.maxF[m]) * f[a, m] for m in data.modeDict[a])
                for a in data.arcsConstrPaths[p]
            )
            for p in odt_data.conPaths),
            name='time_cons_f'
        )
    else:
        raise RuntimeError("No frequency variables present for time constraint (need 'z' or 'f' in vars).")


def add_fc_time_constraints(model: gp.Model, data, odt_data, vars: Dict):
    """Special FC constraints that use cValsFC dict (for FC final-arc constraints)."""
    x = vars['x']; h = vars.get('h', None)
    model.addConstrs(
        (x[p] * (1 / (odt_data.cValsFC[p] * 7)) * data.pathParams[p]['w_hat']
        + (1 - x[p]) * data.pathParams[p]['arcCt']
        >= gp.quicksum(h[a] for a in data.arcsConstrPaths[p])
        for p in data.conPaths
        if (p not in odt_data.conPaths) and (p in odt_data.cValsFC)),
        name='fc_time_constraints'
    )


def add_h_bounds_piecewise(model: gp.Model, data, vars: Dict, bigMTime=7, maxLTLShip=5):
    """h >= wCoefs[...] * f + ... + 1.5*y  and  h bounds w.r.t y for piecewise formulation."""
    wCoefs = {}
    for m in data.dfA['mode'].unique():
        if m == 0:
            wCoefs[m] = [(-1/(n*(n+1)),(2*n+1)/(n*(n+1))-3/2) for n in range(1,bigMTime)]
        else:
            wCoefs[m] = [(-1/(n*(n+1)),(2*n+1)/(n*(n+1))-3/2) for n in range(1,maxLTLShip+1)]

    h = vars['h']; f = vars['f']; y = vars['y']
    for (a, m) in data.arcMode:
        for n in range(len(wCoefs[m])):
            model.addConstr(
                h[a] >= wCoefs[m][n][0] * f[a, m] + wCoefs[m][n][1] + 1.5 * y[a, m]
            )
    # relative bounds
    model.addConstrs(
        (h[a] >= (1 / bigMTime) * gp.quicksum(y[a, m] for m in data.modeDict[a])
        for a in data.arcs),
        name='h_lower_bound_piecewise'
    )
    model.addConstrs(
       ( h[a] <= gp.quicksum(y[a, m] for m in data.modeDict[a])
        for a in data.arcs),
        name='h_upper_bound_piecewise'
    )


def add_time_constraints_binary(model: gp.Model, data, odt_data, vars: Dict):
    """
    Lead-time / time constraints using z-variables (binary linearization).
    RHS uses sum_{a in arcsConstrPaths[p]} sum_{m} sum_{f}
    (1 / min(f, data.maxF[m])) * z[a,m,f].
    Also implements the non-conversion (FC) constraints when cValsFC present.
    """
    x = vars['x']
    w = vars['w']
    z = vars['z']

    # conversion consolidation paths
    model.addConstrs(
        ((1 / 7) *
        gp.quicksum(
            (1 / odt_data.cVals[(p, t)]) * (t - data.pathParams[p]['fixedTT']) *
            w[(odt_data.pathDmd[p], t)]
            for t in odt_data.conODTs[odt_data.pathDmd[p]]
            if (p, t) in odt_data.cVals
        )
        + (1 - x[p]) * data.pathParams[p]['arcCt']
        >=
        gp.quicksum(
            gp.quicksum(
                gp.quicksum(
                    (1 / min(f, data.maxF[m])) * z[a, m, f]
                    for f in range(1, data.bigMVals[a, m] + 1)
                ) for m in data.modeDict[a]
            ) for a in data.arcsConstrPaths[p]
        )
        for p in odt_data.conPaths),
        name='time_cons_z_conv'
    )

    # non-conversion (FC) constraints, if present in odt_data as cValsFC and W_hat
    # Only add for those paths p where a special FC cVal exists.
    if hasattr(odt_data, 'cValsFC') and hasattr(odt_data, 'W_hat'):
        model.addConstrs(
            (x[p] * (1 / (odt_data.cValsFC[p] * 7)) * odt_data.W_hat[p]
            + (1 - x[p]) * data.pathParams[p]['arcCt']
            >=
            gp.quicksum(
                gp.quicksum(
                    gp.quicksum(
                        (1 / min(f, data.maxF[m])) * z[a, m, f]
                        for f in range(1, data.bigMVals[a, m] + 1)
                    ) for m in data.modeDict[a]
                ) for a in data.arcsConstrPaths[p]
            )
            for p in data.conPaths
            if (p not in odt_data.conPaths) and (p in odt_data.cValsFC)),
            name='time_cons_z_hse'
        )


def add_mode_restrictions(model: gp.Model, data, vars: Dict):
    """Restrict final-arc modes for certain consolidation paths (truckload/LTL restrictions)."""
    x = vars['x']
    if 'y' in vars:
        y = vars['y']
        model.addConstrs((x[p] <= y[(data.arcTL[p], 0)] for p in data.arcTL), name='mode_restriction_TL')
        model.addConstrs(
            (x[p] <= gp.quicksum(y[(data.arcLTL[p], m)] for m in data.modeDict[data.arcLTL[p]] if m != 0)
            for p in data.arcLTL),
            name='mode_restriction_LTL'
        )
    elif 'z' in vars:
        z = vars['z']
        model.addConstrs(
            (x[p] <= gp.quicksum(z[data.arcTL[p], 0, f] for f in range(1, data.bigMVals[data.arcTL[p], 0] + 1))
            for p in data.arcTL),
            name='mode_restriction_TL'
        )
        model.addConstrs(
            (x[p] <= gp.quicksum(
                gp.quicksum(z[data.arcLTL[p], m, f]
                            for f in range(1, data.bigMVals[data.arcLTL[p], m] + 1))
                for m in data.modeDict[data.arcLTL[p]] if m != 0)
            for p in data.arcLTL),
            name='mode_restriction_LTL'
        )

def build_revenue_and_costs(model, data, odt_data, vars,
                            minProb=None,
                            piecewise=True):
    """
    Build revenue, path-cost, and lane-cost expressions for ODT models.

    Parameters
    ----------
    model : gurobipy.Model
        Gurobi model instance.
    data : optData
        Network/path data container.
    odt_data : odt_data
        ODT/conversion data container.
    vars : dict
        Dict of Gurobi variables. Required keys:
          - 'x' : path-selection binaries
          - 'u' : consolidation path continuous flows (only for odt models)
          - 'w' : ODT lead-time binaries
        If piecewise:
          - 'f' and 'v' must exist
        If binary:
          - 'z' and 'v' must exist
    minProb : float or None
        Passed to defPMObj if you use it there (keeps signature compatible).
    piecewise : bool
        If True, build lane cost using f and v (piecewise model).
        If False, build lane cost using z and v (binary linearization).

    Returns
    -------
    updated model
    """

    # Obtain economic parameters (expects you have defPMObj implemented)
    # defPMObj should return: ppCost, dirPathCost, sales, cogs, dirSales, dirCOGS
    ppCost, dirPathCost, sales, cogs, dirSales, dirCOGS = defPMObj(data, odt_data, minProb)

    x = vars.get('x')
    u = vars.get('u')
    w = vars.get('w')

    if x is None or w is None:
        raise ValueError("vars must contain at least 'x' and 'w'")

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
    if piecewise:
        # Expect 'f' and 'v' in vars
        f = vars.get('f')
        v = vars.get('v')
        if f is None or v is None:
            raise ValueError("piecewise=True requires 'f' and 'v' in vars")

        costLanes_expr = gp.quicksum(
            data.arcCosts[a, m]['fixedCost'] * f[a, m] +
            data.arcCosts[a, m]['varCost'] * v[a, m]
            for (a, m) in data.arcMode
        )
    else:
        # binary linearization: uses z and v (no h)
        z = vars.get('z')
        v = vars.get('v')
        if z is None or v is None:
            raise ValueError("piecewise=False requires 'z' and 'v' in vars")

        costLanes_expr = gp.quicksum(
            data.arcCosts[a, m]['fixedCost'] *
            gp.quicksum(f * z[a, m, f] for f in range(1, data.bigMVals[a, m] + 1))
            + data.arcCosts[a, m]['varCost'] * v[a, m]
            for (a, m) in data.arcMode
        )

    # --- Combined objective expression (profit = revenue - costs) ---
    ObjExpr = rev_expr - costPath_expr - costLanes_expr

    # set objective 
    model.setObjective(ObjExpr, GRB.MAXIMIZE)
    model.update()

    return model