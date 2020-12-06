from typing import Dict
from dataclasses import dataclass
from scipy.optimize import minimize

import os
import json
import itertools
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None)

water_characteristics_file = os.path.dirname(os.path.abspath(__file__)) + '\water_characteristics.json'
with open(water_characteristics_file) as json_file:
    raw_water_data = json.load(json_file)

all_available_water_characteristics = pd.DataFrame.from_dict(raw_water_data, orient='index')

"""
Omer Ytzhaki

The X Axis is Alkalinity in CaCO3 which is equivalent to Bicarbonate (HCO3) *  0.8202.
The Volvic spec is 71ppm -> 58.2ppm of CaCO3.

For Total Hardness, you would need to calculate as follows:
Total Hardness (as CaCO3) = Calcium (Ca2+) * 2.5 + Magnesium (Mg2+) * 4.1
In the Volvic case: 11.5*2.5 + 8*4.1 = 61.5
"""

all_available_water_characteristics['Total Alkalinity'] = 0.8202 * all_available_water_characteristics['Bicarbonate (HCO3)']
all_available_water_characteristics['Total Hardness (as CaCO3)'] = 2.5 * all_available_water_characteristics['Calcium (Ca)'] + 4.1 * all_available_water_characteristics['Magnesium (Mg)']
print(all_available_water_characteristics)


@dataclass(eq=True, frozen=True)
class Target:
    variable: str
    target: float
    min_range: float
    max_range: float
    weight: float
    out_of_range_penalty: float


""" 
https://scanews.coffee/2013/07/08/dissecting-scaas-water-quality-standard/

Total Dissolved Solvents (TDS): between 75-250 mg/L TDS, with a target of 150
Calcium Hardness: 17-85 mg/L, with a target of 51-68 mg/L
Total Alkalinity: At or near 40 mg/L
pH: 6.5-7.5, with a target of 7
Sodium: Less than 30 ml/L
"""

ideal_water = [
    Target('TDS', 150.0, 75.0, 250.0, 1.0, 2.0),
    Target('Total Hardness (as CaCO3)', 59.5, 17.0, 85.0, 1.0, 2.0),
    Target('Total Alkalinity', 40.0, 35.0, 45.0, 1.0, 2.0),
    Target('pH', 7.0, 6.5, 7.5, 1.0, 2.0),
    Target('Sodium', 10.0, 0.0, 30.0, 0.0, 1.0),
]


def mix_characteristics(mix_ratio, available_water_characteristics) -> Dict[str, float]:
    return {
        'TDS': float(np.sum(mix_ratio * available_water_characteristics['TDS'])),
        'Total Hardness (as CaCO3)': float(np.sum(mix_ratio * available_water_characteristics['Total Hardness (as CaCO3)'])),
        'Total Alkalinity': float(np.sum(mix_ratio * available_water_characteristics['Total Alkalinity'])),
        'pH': float(np.sum(mix_ratio * available_water_characteristics['pH'])),
        'Sodium': float(np.sum(mix_ratio * available_water_characteristics['Sodium'])),
        'Bicarbonate (HCO3)': float(np.sum(mix_ratio * available_water_characteristics['Bicarbonate (HCO3)'])),
        'Calcium (Ca)': float(np.sum(mix_ratio * available_water_characteristics['Calcium (Ca)'])),
        'Magnesium (Mg)': float(np.sum(mix_ratio * available_water_characteristics['Magnesium (Mg)'])),
    }


def deviation_from_target(x, available_water_characteristics) -> float:
    mix = mix_characteristics(np.append(x, 1 - sum(x)), available_water_characteristics)

    def target_deviation(target):
        deviation = target.weight * ((mix[target.variable] - target.target) / target.target) ** 2.0

        if mix[target.variable] < target.min_range or mix[target.variable] > target.max_range:
            return target.out_of_range_penalty * deviation
        else:
            return deviation

    return sum([target_deviation(target) for target in ideal_water])


water_mix_characteristics = pd.DataFrame(columns=[water + ' (%)' for water in all_available_water_characteristics.index.to_list()] + ['Target Deviation'])

for waters_to_mix in list(range(1, 3)) + [all_available_water_characteristics.shape[0]]:
    combination_water_mix_characteristics = pd.DataFrame(columns=[water + ' (%)' for water in all_available_water_characteristics.index.to_list()] + ['Target Deviation'])
    for combination in itertools.combinations(all_available_water_characteristics.index.to_list(), waters_to_mix):
        available_water_characteristics_subset = all_available_water_characteristics.loc[combination, :]

        if len(combination) == 1:
            x = np.zeros(available_water_characteristics_subset.shape[0])
            x[available_water_characteristics_subset.index.to_list().index(combination[0])] = 1.0
            x = x[:-1]
            results = {
                'x': x,
                'fun': deviation_from_target(x, available_water_characteristics_subset)
            }
        else:
            results = minimize(deviation_from_target,
                               x0=(available_water_characteristics_subset.shape[0] - 1) * [0.0],
                               bounds=(available_water_characteristics_subset.shape[0] - 1) * [(0.0, 1.0)],
                               constraints={'type': 'ineq', 'fun': lambda x: 1.0 - sum(x)},
                               args=(available_water_characteristics_subset),
                               options={'disp': False}
                               )

        optimal_mix_ratio = np.append(results['x'], max(0.0, 1.0 - sum(results['x'])))
        optimal_mix_characteristics = mix_characteristics(optimal_mix_ratio, available_water_characteristics_subset)
        optimal_mix = {**{variable: round(value, 2) for variable, value in optimal_mix_characteristics.items()},
                       'Target Deviation': results['fun'],
                       **{available_water_characteristics_subset.index[i] + ' (%)': round(100.0 * w, 1) for i, w in enumerate(optimal_mix_ratio)}}
        combination_water_mix_characteristics = combination_water_mix_characteristics.append(pd.DataFrame(optimal_mix, index=[0]))

    if waters_to_mix == 1:
        water_mix_characteristics = water_mix_characteristics.append(combination_water_mix_characteristics)
    else:
        best = min(water_mix_characteristics['Target Deviation'])
        combination_water_mix_characteristics.sort_values('Target Deviation', inplace=True)
        if combination_water_mix_characteristics.iloc[0]['Target Deviation'] < best:
            water_mix_characteristics = water_mix_characteristics.append(combination_water_mix_characteristics.iloc[0])

water_mix_characteristics.sort_values('Target Deviation', inplace=True)
water_mix_characteristics.reset_index(drop=True, inplace=True)
water_mix_characteristics.fillna(0, inplace=True)

print(water_mix_characteristics.to_string(index=False))
pass
