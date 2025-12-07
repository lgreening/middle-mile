"""
LOGIC OVERVIEW:
-------------------------------------------------------------------------
This function adjusts a supply chain solution by converting "excess" 
Truckload (TL) volume into Less-than-Truckload (LTL) shipments where 
appropriate.

1.  INPUTS: 
    - Takes two solutions: "Original" (baseline) and "Conversion" (current).
    - Takes cost data (laneCosts) and physical constraints (Q/Max Weight).
    - Data frame columns in ALL CAPS are not in the original data, they are 
      calculated from the results.

2.  IDENTIFY CANDIDATES (The Loop):
    - Iterate through every lane in the "Conversion" solution.
    - Filter for lanes using Truckload (Mode 0).
    - Check if the frequency of trucks has INCREASED compared to the "Original".
      (e.g., Original used 2 trucks, Conversion uses 3).

3.  CALCULATE EXCESS:
    - If frequency increased, calculate the "excess weight" that pushed 
      the solution over the limit.
      Formula: Excess = Total Weight - (Original Frequency * Max Truck Weight).
    - Calculate the "Old Cost" (the Fixed Cost of that extra truck we are removing).

4.  CALCULATE NEW COST:
    - Determine the cost to ship that "excess weight" via LTL instead.
    - This requires an external function `LTLcost` that returns the LTL 
      price, mode, and frequency based on distance and weight.

5.  CALCULATE ADJUSTMENT:
    - Objective Adjustment = (New LTL Cost) - (Old Fixed TL Cost).

6.  UPDATE GLOBAL STATS (dfSC):
    - Adjust Total Objective Value, Lane Costs, and Log Costs by the adjustment amount.
    - Recalculate Profit Margins.

7.  UPDATE DETAILED LANE DATA (freqsUsedC):
    - EXISTING TL ROWS: 
        - Reduce Frequency by 1 (remove the extra truck).
        - Reduce Weight by the excess amount.
        - Reduce Cost proportionally.
    - NEW LTL ROWS:
        - Create new rows for the LTL shipments.
        - Assign the calculated LTL cost, mode, and frequency.
    - Concatenate these new rows to the dataframe.
-------------------------------------------------------------------------
"""

import numpy as np

#LTL cost for 3-bucket LTL (maxLTL is max wgt for any ltl mode; midLTL is the upper bound for middle bucket)
def LTLcost(dist, wgt, midLTL = 2000, maxLTL = 4000):
    fc = 0.05*(750 + 1.27*dist)
    v2 = 0.1875 + 0.0003175*dist
    v1 = 1.25*v2
    midBrk = (fc + midLTL*v1)/v2
    modeCost = {}
    modeCost[1] = (fc*np.ceil(wgt/midLTL) + v1*wgt, np.ceil(wgt/midLTL))
    modeCost[2] = (fc*np.ceil(wgt/midBrk) + v1*midLTL, np.ceil(wgt/midBrk))
    modeCost[3] = (v2*wgt, np.ceil(wgt/maxLTL))
    minCost = 100000
    for m in modeCost:
        if modeCost[m][0] < minCost:
            minCost = modeCost[m][0]
            mode = m
            freq = modeCost[m][1]
    return minCost, mode, int(freq)



import pandas as pd
import math
from typing import Dict, Tuple, Any

def calculate_lane_adjustments(
    dfSC: pd.DataFrame,
    freqsUsedC: pd.DataFrame,
    lanesO: Dict[Any, Tuple[int, int, float]],
    laneCosts: Dict[Tuple[Any, int], Dict[str, Any]],
    Q: float = 12000.0
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:

    # Create a quick lookup dict for the "Conversion" lanes
    # Assumes freqsUsedC has: arcID, mode, maxWgt, LANE_FREQ, distance, LANE_WGT
    ## LANE_FREQ (lane frequency) and LANE_WGT are calculated from solution (not in original lanes_data)
    lanesC = (freqsUsedC[['arcID', 'mode', 'maxWgt', 'LANE_FREQ', 'distance',
                          'LANE_WGT']].set_index(['arcID']).to_dict('index'))

    exLn = {}
    oldCost = 0

    # --- STEP 1: Identify lanes to adjust ---
    for l in lanesC:
        # Check only for Truckload (Mode 0)
        if lanesC[l]['mode'] == 0:
            # Get original frequency (default to 0 if new lane)
            original_freq = lanesO.get(l, (None, 0, 0))[1]
            freq_diff = lanesC[l]['LANE_FREQ'] - original_freq
            
            # If we are using strictly MORE trucks than before
            if freq_diff >= 1:
                # Calculate the weight that is "spilling over" into the extra truck
                excess_wgt = lanesC[l]['LANE_WGT'] - original_freq * lanesC[l]['maxWgt']
                
                # Store the original frequency and the excess weight for later
                exLn[l] = (original_freq, excess_wgt)
                
                # Calculate what we are currently paying for that extra truck (Fixed Cost)
                # Note: This assumes we are removing exactly (freq_diff) trucks, 
                # but usually we just remove the 1 marginal truck.
                fixed_cost = laneCosts.get((l, 0), {}).get('fixedCost', 0)
                oldCost += fixed_cost * freq_diff

    # --- STEP 2: Calculate new LTL costs for the excess weight ---
    newLn = {}
    newCost = 0
    for l, (original_freq, excess_wgt) in exLn.items():
        # Get LTL details
        minCost, md, freq = LTLcost(lanesC[l]['distance'], excess_wgt)
        
        # Store: (Mode, Cost, Freq)
        newLn[l] = (md, minCost, freq)
        newCost += minCost

    # --- STEP 3: Calculate the financial impact ---
    objAdj = newCost - oldCost

    # --- STEP 4: Update Global Statistics (dfSC) ---
    # Adjust costs and objective
    dfSC['OBJVAL ($)'] -= objAdj
    dfSC['LANE_COSTS'] += objAdj
    
    # Recalculate margins
    # Ensure 'sales' and 'cogs' exist in dfSC or this will error
    if 'sales' in dfSC.columns and 'cogs' in dfSC.columns:
        dfSC['profMarg'] = dfSC.apply(
            lambda x: (x['sales'] - x['cogs'] - x['logCost']) / x['sales'] 
            if x['sales'] != 0 else 0, axis=1
        )

    # --- STEP 5: Update Lane Detail DataFrame (freqsUsedC) ---
    
    # A. Adjust the existing Truckload rows (remove the excess weight/freq)
    for l in exLn:
        # Filter for the specific lane
        mask = freqsUsedC['arcID'] == l
        
        # Current values
        current_freq = freqsUsedC.loc[mask, 'LANE_FREQ']
        
        # 1. Adjust Cost proportionally (remove cost of 1 truck)
        # Avoid division by zero
        freqsUsedC.loc[mask, 'LANE_COST'] *= (current_freq - 1) / current_freq.replace(0, 1)
        
        # 2. Remove the excess weight
        freqsUsedC.loc[mask, 'LANE_WGT'] -= exLn[l][1]
        
        # 3. Reduce frequency by 1
        freqsUsedC.loc[mask, 'LANE_FREQ'] -= 1
        
    # B. Create new LTL rows
    # Extract the rows that need LTL components
    newLTL_rows = freqsUsedC[freqsUsedC['arcID'].isin(exLn)].copy()
    
    # Overwrite columns with LTL specific data
    newLTL_rows['LANE_FREQ'] = newLTL_rows['arcID'].apply(lambda x: newLn[x][2])
    newLTL_rows['LANE_COST'] = newLTL_rows['arcID'].apply(lambda x: newLn[x][1])
    newLTL_rows['mode'] = newLTL_rows['arcID'].apply(lambda x: newLn[x][0])
    newLTL_rows['LANE_WGT'] = newLTL_rows['arcID'].apply(lambda x: exLn[x][1])
    
    # Append the new LTL rows to the main dataframe
    freqsUsedC = pd.concat([freqsUsedC, newLTL_rows], ignore_index=True)

    return dfSC, freqsUsedC, objAdj