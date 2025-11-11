# models/mmtc_odt_bin.py
"""
Binary-linearized multi-mode + ODT model builder. Uses z_{a,m,f} binary selectors
for frequencies and constructs time constraints based on z.
"""
import gurobipy as gp
from gurobipy import GRB
from .builders import (
    add_path_vars, add_u_vars_for_odt, add_arc_mode_vars_binary,
    add_one_path_per_demand, add_odt_flow_constraints, add_frequency_activation_binary,
    add_volume_balance, add_maxmin_bounds_binary, add_one_lead_time_per_conversion,
    add_time_constraints_binary, add_mode_restrictions, build_revenue_and_costs
)



def build(data, odt_data, minProb = 0.8):
    model = gp.Model('mmtc_odt_bin')
    vars = {}

    # vars
    vars = add_path_vars(model, data, vars)           # x
    vars = add_u_vars_for_odt(model, odt_data, vars)  # u for consolidation paths
    vars = add_arc_mode_vars_binary(model, data, vars)  # v and z (no h)
    # w-vars for ODT lead times (binary (k,t))
    vars['w'] = model.addVars(list(odt_data.conRates.keys()), lb=0, vtype=GRB.BINARY, name='w')

    # Objective: path costs (nonConv via x, conv via u) + lane costs via z & v, revenue via odt_data
    model = build_revenue_and_costs(
    model, data, odt_data, vars, minProb=minProb, piecewise=False, set_objective=True, maximize=True
    )

    # constraints
    add_one_path_per_demand(model, data, vars)
    add_odt_flow_constraints(model, data, odt_data, vars)
    add_frequency_activation_binary(model, data, vars)
    add_volume_balance(model, data, vars, odt_data=odt_data)
    add_maxmin_bounds_binary(model, data, vars)
    add_one_lead_time_per_conversion(model, odt_data, vars)

    # Time constraints implemented using z (exact binary-linearization)
    add_time_constraints_binary(model, data, odt_data, vars)

    # Mode restrictions on final arcs (truckload / LTL) using z
    add_mode_restrictions(model, data, vars)

    model.update()
    return model, vars