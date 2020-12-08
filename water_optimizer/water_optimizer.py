from typing import Dict
from dataclasses import dataclass
from scipy.optimize import minimize

import os
import json
import itertools
import numpy as np
import pandas as pd


pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None)

water_characteristics_file = os.path.dirname(os.path.abspath(__file__)) + '/water_characteristics.json'
with open(water_characteristics_file) as json_file:
    raw_water_data = json.load(json_file)

all_available_water_characteristics = pd.DataFrame.from_dict(raw_water_data, orient='index')

"""
Omer Ytzhaki

Alkalinity in CaCO3 which is equivalent to Bicarbonate (HCO3) *  0.8202.
Total Hardness (as CaCO3) = Calcium (Ca2+) * 2.5 + Magnesium (Mg2+) * 4.1
"""

all_available_water_characteristics['Total Alkalinity'] = 0.8202 * all_available_water_characteristics['Bicarbonate (HCO3)']
all_available_water_characteristics['Total Hardness'] = 2.5 * all_available_water_characteristics['Calcium (Ca)'] + 4.1 * all_available_water_characteristics['Magnesium (Mg)']

# print(all_available_water_characteristics)


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

water_targets = [
    Target('TDS', 150.0, 75.0, 250.0, 1.0, 2.0),
    Target('Total Hardness', 59.5, 17.0, 85.0, 1.0, 2.0),
    Target('Total Alkalinity', 40.0, 35.0, 45.0, 1.0, 2.0),
    Target('pH', 7.0, 6.5, 7.5, 1.0, 2.0),
    Target('Sodium (Na)', 10.0, 0.0, 30.0, 0.0, 1.0),
]

min_content = 20.0/100.0  # Discard any optimal solution that has a ratio of less than 5% for one of the waters.
ratio_rounding = 1  # Number of decimal places to round each water ratio

# Number of decimal places to round each water characteristic
characteristics_rounding = {
    'TDS': 1,
    'Total Hardness': 2,
    'Total Alkalinity': 2,
    'pH': 1,
    'Sodium (Na)': 1,
    'Bicarbonate (HCO3)': 1,
    'Calcium (Ca)': 1,
    'Magnesium (Mg)': 1
}


def mix_characteristics(mix_ratio, available_water_characteristics) -> Dict[str, float]:
    return {
        'TDS': float(np.sum(mix_ratio * available_water_characteristics['TDS'])),
        'Total Hardness': float(np.sum(mix_ratio * available_water_characteristics['Total Hardness'])),
        'Total Alkalinity': float(np.sum(mix_ratio * available_water_characteristics['Total Alkalinity'])),
        'pH': float(np.sum(mix_ratio * available_water_characteristics['pH'])),
        'Sodium (Na)': float(np.sum(mix_ratio * available_water_characteristics['Sodium (Na)'])),
        'Bicarbonate (HCO3)': float(np.sum(mix_ratio * available_water_characteristics['Bicarbonate (HCO3)'])),
        'Calcium (Ca)': float(np.sum(mix_ratio * available_water_characteristics['Calcium (Ca)'])),
        'Magnesium (Mg)': float(np.sum(mix_ratio * available_water_characteristics['Magnesium (Mg)'])),
    }


def deviation_from_target(mix_ratio, available_water_characteristics) -> float:
    mix = mix_characteristics(np.append(mix_ratio, 1 - sum(mix_ratio)), available_water_characteristics)

    def target_deviation(target):
        deviation = target.weight * ((mix[target.variable] - target.target) / target.target) ** 2.0

        if mix[target.variable] < target.min_range or mix[target.variable] > target.max_range:
            return target.out_of_range_penalty * deviation
        else:
            return deviation

    return sum([target_deviation(target) for target in water_targets])


water_mix_columns = [water + ' %' for water in all_available_water_characteristics.index.to_list()]
water_mix_characteristics = pd.DataFrame(columns=water_mix_columns + ['Target Deviation', 'Relative Deviation'])

for waters_to_mix in range(1, all_available_water_characteristics.shape[0]):
    combination_water_mix_characteristics = pd.DataFrame(columns=water_mix_columns + ['Target Deviation'])
    for combination in itertools.combinations(all_available_water_characteristics.index.to_list(), waters_to_mix):
        available_water_characteristics_subset = all_available_water_characteristics.loc[combination, :]

        if available_water_characteristics_subset.shape[0] * min_content > 1.0:
            continue

        if len(combination) == 1:
            x = np.zeros(available_water_characteristics_subset.shape[0])
            x[available_water_characteristics_subset.index.to_list().index(combination[0])] = 1.0
            x = x[:-1]
            results = {
                'x': x,
                'fun': deviation_from_target(x, available_water_characteristics_subset),
                'success': True
            }
        else:
            results = minimize(deviation_from_target,
                               x0=(available_water_characteristics_subset.shape[0] - 1) * [0.0],
                               bounds=(available_water_characteristics_subset.shape[0] - 1) * [(min_content, 1.0)],
                               constraints={'type': 'ineq', 'fun': lambda x: 1.0 - sum(x)},
                               args=(available_water_characteristics_subset),
                               options={'disp': False}
                               )

        if results['success']:
            rounded_x = np.round(100.0*results['x'], ratio_rounding)/100.0
            optimal_mix_ratio = np.append(rounded_x, max(0.0, 1.0 - sum(rounded_x)))
            optimal_mix_characteristics = mix_characteristics(optimal_mix_ratio, available_water_characteristics_subset)
            optimal_mix = {**{variable: round(value, characteristics_rounding[variable]) for variable, value in optimal_mix_characteristics.items()},
                           'Target Deviation': deviation_from_target(rounded_x, available_water_characteristics_subset),
                           **{available_water_characteristics_subset.index[i] + ' %': 100.0*w for i, w in enumerate(optimal_mix_ratio)}}
            combination_water_mix_characteristics = combination_water_mix_characteristics.append(pd.DataFrame(optimal_mix, index=[0]))

    if waters_to_mix == 1:
        water_mix_characteristics = water_mix_characteristics.append(combination_water_mix_characteristics)
    else:
        best = min(water_mix_characteristics['Target Deviation'])
        combination_water_mix_characteristics.sort_values('Target Deviation', inplace=True)
        if combination_water_mix_characteristics.shape[0] > 0 and combination_water_mix_characteristics.iloc[0]['Target Deviation'] < best:
            water_mix_characteristics = water_mix_characteristics.append(combination_water_mix_characteristics.iloc[0])

water_mix_characteristics.sort_values('Target Deviation', inplace=True)
water_mix_characteristics.reset_index(drop=True, inplace=True)
water_mix_characteristics['Relative Deviation'] = round(water_mix_characteristics['Target Deviation'] / water_mix_characteristics['Target Deviation'][0], 1)
water_mix_characteristics.fillna(0, inplace=True)

print(water_mix_characteristics.to_string(index=False))
pass
