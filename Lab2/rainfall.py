"""
Małgorzata Cichowlas s16512
Prediction of rain [%] based on parameteres: temperature, temperature of ground and humidity. 

Used for this code: Program 02_fuzzyLogicExample.py
To run program install
pip install scikit-fuzzy
pip install matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Inputs:
temperature = ctrl.Antecedent(np.arange(0, 40, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'humidity')
groundTemperature = ctrl.Antecedent(np.arange(0, 40, 1), 'groundTemperature')

# Output:
rainfall = ctrl.Consequent(np.arange(0, 100, 1), 'rainfall')

# Membership functions:
rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 50])
rainfall['medium'] = fuzz.trimf(rainfall.universe, [0, 50, 100])
rainfall['high'] = fuzz.trimf(rainfall.universe, [50, 100, 100])

temperature.automf(3)
humidity.automf(3)
groundTemperature.automf(3)

# Visualition of these functions:
"""
temperature.view()

humidity.view()

pressure.view()

rainfall.view()
"""

# Rules - moderate and veryfing prediction of rainfall
rule1 = ctrl.Rule(temperature['poor'] | humidity['poor'] | (groundTemperature['poor']) , rainfall['low'])
rule2 = ctrl.Rule(temperature['poor'] | humidity['average'] | (groundTemperature['poor']) , rainfall['low'])
rule3 = ctrl.Rule(temperature['poor'] | humidity['poor'] | (groundTemperature['average']) , rainfall['low'])
rule3 = ctrl.Rule(temperature['poor'] | humidity['good'] | (groundTemperature['poor']) , rainfall['medium'])
rule4 = ctrl.Rule(humidity['average'], rainfall['medium'])
rule5 = ctrl.Rule(temperature['average'] | groundTemperature['average'] | humidity['good'], rainfall['high'])
rule6 = ctrl.Rule(temperature['good'] | humidity['good'], rainfall['high'])


pre_rainfall_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])

# Simulation with inputs:
pre_rainfall = ctrl.ControlSystemSimulation(pre_rainfall_ctrl)
pre_rainfall.input['temperature'] = 5
pre_rainfall.input['humidity'] = 30
pre_rainfall.input['groundTemperature'] = 2

pre_rainfall.compute()

# Output [%]:
print (pre_rainfall.output['rainfall'])
rainfall.view(sim=pre_rainfall)
plt.show() 