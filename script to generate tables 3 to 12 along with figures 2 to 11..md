# 1\. FOR M1(G)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

    "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

    "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

    "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

    "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

    "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

    "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

    "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

    "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

    "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

    "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



    df\_subset = df\[\["M1", prop]].dropna()

    X = df\_subset\[\["M1"]]

    y = df\_subset\[prop]



    # Fit OLS model (for CI \& p-values)

    X\_sm = sm.add\_constant(X)

    ols\_model = sm.OLS(y, X\_sm).fit()



    intercept, slope = ols\_model.params

    se\_intercept, se\_slope = ols\_model.bse



    # 95% Confidence Intervals

    ci = ols\_model.conf\_int(alpha=0.05)

    ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

    ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



    # Predictions for R²

    y\_pred = ols\_model.predict(X\_sm)



    # Statistics

    r = np.corrcoef(X\["M1"], y)\[0, 1]

    r2 = r2\_score(y, y\_pred)

    se = np.sqrt(np.mean((y - y\_pred)\*\*2))

    f\_stat = ols\_model.fvalue

    p\_value = ols\_model.pvalues\[1]



    # Store results

    results.append(\[

        f"{intercept:.4f} + {slope:.4f}·M1",

        r,

        r2,

        se,

        f\_stat,

        p\_value,

        ci\_slope

    ])





\# Create DataFrame for results

table = pd.DataFrame(

    results,

    columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

    index=properties

)







\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["M1"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["M1"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("M1")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs M1")



plt.tight\_layout()

fig.savefig("reg M1.pdf", bbox\_inches='tight')

plt.show()





# 2\. FOR M2(G)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["M2", prop]].dropna()

&nbsp;   X = df\_subset\[\["M2"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["M2"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·M2",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["M2"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["M2"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("M2")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs M2")



plt.tight\_layout()

fig.savefig("reg M2.pdf", bbox\_inches='tight')

plt.show()



# 3\. FOR H(G)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["H", prop]].dropna()

&nbsp;   X = df\_subset\[\["H"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["H"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·H",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["H"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["H"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("H")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs H")



plt.tight\_layout()

fig.savefig("reg H.pdf", bbox\_inches='tight')

plt.show()





# 4\. FOR HM(G)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["HM", prop]].dropna()

&nbsp;   X = df\_subset\[\["HM"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["HM"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·HM",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["HM"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["HM"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("HM")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs HM")



plt.tight\_layout()

fig.savefig("reg HM.pdf", bbox\_inches='tight')

plt.show()



# 5\. FOR F(G)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["F", prop]].dropna()

&nbsp;   X = df\_subset\[\["F"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["F"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·F",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["F"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["F"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("F")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs F")



plt.tight\_layout()

fig.savefig("reg F.pdf", bbox\_inches='tight')

plt.show()



# 6\. FOR R(G)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["R", prop]].dropna()

&nbsp;   X = df\_subset\[\["R"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["R"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·R",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["R"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["R"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("R")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs R")



plt.tight\_layout()

fig.savefig("reg R.pdf", bbox\_inches='tight')

plt.show()



# 7\. FOR RR(G)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["RR", prop]].dropna()

&nbsp;   X = df\_subset\[\["RR"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["RR"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·RR",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["RR"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["RR"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("RR")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs RR")



plt.tight\_layout()

fig.savefig("reg RR.pdf", bbox\_inches='tight')

plt.show()



# 

# 8\. FOR SCI(G)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["SCI", prop]].dropna()

&nbsp;   X = df\_subset\[\["SCI"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["SCI"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·SCI",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)



\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["SCI"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["SCI"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("SCI")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs SCI")



plt.tight\_layout()

fig.savefig("reg SCI.pdf", bbox\_inches='tight')

plt.show()

# 

# &nbsp;9. FOR GA(G)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5, 35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["GA", prop]].dropna()

&nbsp;   X = df\_subset\[\["GA"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["GA"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·GA",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)





\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["GA"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["GA"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("GA")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs GA")



plt.tight\_layout()

fig.savefig("reg GA.pdf", bbox\_inches='tight')

plt.show()



# 10\. FOR ABC(G)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score



\# Data (Replace with actual values if different)

data = {

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;   "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8]

}



\# Convert to DataFrame

df = pd.DataFrame(data)



\# List of dependent variables

properties = \["BP", "EV", "FP", "IR", "MR", "PSA", "P", "ST", "MV"]



\# Prepare table storage

results = \[]



\# Linear Regression Model with Confidence Intervals

for prop in properties:



&nbsp;   df\_subset = df\[\["ABC", prop]].dropna()

&nbsp;   X = df\_subset\[\["ABC"]]

&nbsp;   y = df\_subset\[prop]



&nbsp;   # Fit OLS model (for CI \& p-values)

&nbsp;   X\_sm = sm.add\_constant(X)

&nbsp;   ols\_model = sm.OLS(y, X\_sm).fit()



&nbsp;   intercept, slope = ols\_model.params

&nbsp;   se\_intercept, se\_slope = ols\_model.bse



&nbsp;   # 95% Confidence Intervals

&nbsp;   ci = ols\_model.conf\_int(alpha=0.05)

&nbsp;   ci\_intercept = f"\[{ci.iloc\[0,0]:.4f}, {ci.iloc\[0,1]:.4f}]"

&nbsp;   ci\_slope = f"\[{ci.iloc\[1,0]:.4f}, {ci.iloc\[1,1]:.4f}]"



&nbsp;   # Predictions for R²

&nbsp;   y\_pred = ols\_model.predict(X\_sm)



&nbsp;   # Statistics

&nbsp;   r = np.corrcoef(X\["ABC"], y)\[0, 1]

&nbsp;   r2 = r2\_score(y, y\_pred)

&nbsp;   se = np.sqrt(np.mean((y - y\_pred)\*\*2))

&nbsp;   f\_stat = ols\_model.fvalue

&nbsp;   p\_value = ols\_model.pvalues\[1]



&nbsp;   # Store results

&nbsp;   results.append(\[

&nbsp;       f"{intercept:.4f} + {slope:.4f}·ABC",

&nbsp;       r,

&nbsp;       r2,

&nbsp;       se,

&nbsp;       f\_stat,

&nbsp;       p\_value,

&nbsp;       ci\_slope

&nbsp;   ])





\# Create DataFrame for results

table = pd.DataFrame(

&nbsp;   results,

&nbsp;   columns=\["Model", "R", "R²", "SE", "F", "p-value", "95% CI (Slope)"],

&nbsp;   index=properties

)



\# Display table

print(table)



\# Create a single figure with two subplots

fig, axes = plt.subplots(1, 2, figsize=(14, 6))



\# --- Plot 1: R-squared values ---

sns.barplot(x=table.index, y=table\["R²"], palette="Set2", ax=axes\[0])

axes\[0].set\_xlabel("Chemical Property")

axes\[0].set\_ylabel("R-squared Value")

axes\[0].set\_title("R-squared Values for Each Property")



\# --- Plot 2: Best Model Regression Plot ---

best\_property = table\["R²"].idxmax()  # Get the property with the highest R²

X = df\[\["ABC"]]

y = df\[best\_property]

model = LinearRegression()

model.fit(X, y)

y\_pred = model.predict(X)



sns.regplot(x=df\["ABC"], y=df\[best\_property], ci=95, scatter\_kws={"s": 50}, line\_kws={"color": "red"}, ax=axes\[1])

axes\[1].set\_xlabel("ABC")

axes\[1].set\_ylabel(best\_property)

axes\[1].set\_title(f"Best Model: {best\_property} vs ABC")



plt.tight\_layout()

fig.savefig("reg ABC.pdf", bbox\_inches='tight')

plt.show()





