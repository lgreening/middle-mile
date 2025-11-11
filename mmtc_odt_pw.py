# models/mmtc_odt_pw.py
"""
Piecewise (y/f) multi-mode ODT-capable MIP builder.
This builder uses the piecewise activation variables y (binary) and f (integer).
It expects odt_data to exist and to provide conRates, conLTs, conPaths, path_to_dmd, etc.
"""
import gurobipy as gp
from gurobipy import GRB
from .builders import (
    add_path_vars, add_u_vars_for_odt, add_arc_mode_vars_piecewise,
    add_one_path_per_demand, add_odt_flow_constraints, add_frequency_activation_piecewise,
    add_volume_balance, add_maxmin_bounds_piecewise, add_one_mode_per_arc_piecewise,
    add_one_lead_time_per_conversion, add_time_constraints_piecewise,
    add_fc_time_constraints, add_h_bounds_piecewise, add_mode_restrictions, build_revenue_and_costs
)


def build(data, odt_data, minProb = 0.8):
    model = gp.Model('mmtc_odt_piecewise')
    vars = {}

    # vars
    vars = add_path_vars(model, data, vars)
    # add odt u variables
    vars = add_u_vars_for_odt(model, odt_data, vars)
    # arc-mode piecewise vars: v,y,f,h
    vars = add_arc_mode_vars_piecewise(model, data, vars)
    # w-vars for ODT lead times
    vars['w'] = model.addVars(list(odt_data.conRates.keys()), lb=0, vtype=GRB.BINARY, name='w')

    # Objective: path costs (nonConv via x, conv via u) + lane costs via z & v, revenue via odt_data
    model = build_revenue_and_costs(
    model, data, odt_data, vars, minProb=minProb, piecewise=True, set_objective=True, maximize=True
    )

    # Constraints
    add_one_path_per_demand(model, data, vars)
    add_odt_flow_constraints(model, data, odt_data, vars)
    add_frequency_activation_piecewise(model, data, vars)
    add_volume_balance(model, data, vars, odt_data=odt_data)
    add_maxmin_bounds_piecewise(model, data, vars)
    add_one_mode_per_arc_piecewise(model, data, vars)
    add_one_lead_time_per_conversion(model, odt_data, vars)
    add_time_constraints_piecewise(model, data, odt_data, vars)
    add_fc_time_constraints(model, data, odt_data, vars)
    add_h_bounds_piecewise(model, data, odt_data, vars)
    add_mode_restrictions(model, data, vars)

    model.update()
    return model, vars
