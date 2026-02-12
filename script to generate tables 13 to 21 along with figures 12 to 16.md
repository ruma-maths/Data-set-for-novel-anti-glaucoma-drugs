1. # Prediction of BP

import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;   "Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "BP": \[395.5,	448,	629.8,	518.6,	762.4,	290,	372.1,	583.8,	658.3,	464.4,	682,	494.9,	458.7,	711.9,	709.9,	431.8,	497.2,	552.9,	487.2,	691.2,	584.8,	562.9],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_BP\_M1(m1): return 250.5465 + 2.4147 \* m1

def calc\_BP\_M2(m2): return 273.2967 + 1.9364 \* m2

def calc\_BP\_H(h):   return 255.2305 + 26.0114 \* h

def calc\_BP\_HM(hm):   return 274.9828 + 0.4526 \* hm

def calc\_BP\_F(f): return 277.5787 + 0.8463 \* f

def calc\_BP\_R(r): return  250.6678 + 25.3152 \* r

def calc\_BP\_RR(rr): return 250.6270 + 4.9978 \* rr

def calc\_BP\_SCI(sci): return  255.2305 + 52.0228 \* sci

def calc\_BP\_GA(ga):  return 249.5538 + 11.9332 \* ga

def calc\_BP\_ABC(abc):  return 250.3594 + 15.9572 \* abc



\# Apply the functions to DataFrame

df\['BP\_M1'] = df\['M1'].apply(calc\_BP\_M1)

df\['BP\_M2'] = df\['M2'].apply(calc\_BP\_M2)

df\['BP\_H']  = df\['H'].apply(calc\_BP\_H)

df\['BP\_HM']  = df\['HM'].apply(calc\_BP\_HM)

df\['BP\_F'] = df\['F'].apply(calc\_BP\_F)

df\['BP\_R'] = df\['R'].apply(calc\_BP\_R)

df\['BP\_RR'] = df\['RR'].apply(calc\_BP\_RR)

df\['BP\_SCI'] = df\['SCI'].apply(calc\_BP\_SCI)

df\['BP\_GA'] = df\['GA'].apply(calc\_BP\_GA)

df\['BP\_ABC'] = df\['ABC'].apply(calc\_BP\_ABC)



print(df\[\['Drug', 'BP', 'BP\_M1', 'BP\_M2', 'BP\_H', 'BP\_HM', 'BP\_F', 'BP\_R', 'BP\_RR', 'BP\_SCI', 'BP\_GA', 'BP\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['BP'], label='BP', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['BP\_M1'], label='M1')

plt.plot(df\['Drug'], df\['BP\_M2'], label='M2')

plt.plot(df\['Drug'], df\['BP\_H'], label='H')

plt.plot(df\['Drug'], df\['BP\_HM'], label='HM')

plt.plot(df\['Drug'], df\['BP\_F'], label='F')

plt.plot(df\['Drug'], df\['BP\_R'], label='R')

plt.plot(df\['Drug'], df\['BP\_RR'], label='RR')

plt.plot(df\['Drug'], df\['BP\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['BP\_GA'], label='GA')

plt.plot(df\['Drug'], df\['BP\_ABC'], label='ABC')



plt.xlabel('Drug Index')

plt.ylabel('Boiling Point (BP)')

plt.title('Actual and Predicted values of Boiling Point (BP)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 200, 400, 600, 800, 1000])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("boiling\_point\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 2\. Prediction of EV

import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;   "Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "EV": \[64.6,	74.4,	97.9,	83.3,	126.6,	61.4,	71.6,	91.8,	101.8,	76.5,	105.1,	87.8,	75.8,	104.1,	103.8,	68.8,	76.5,	87.7,	79.3,	106.4,	91.9,	97.2],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_EV\_M1(m1): return    52.6303 + 0.2895 \* m1

def calc\_EV\_M2(m2): return  54.3060 + 0.2397 \* m2

def calc\_EV\_H(h):   return  54.7138 + 2.9818 \* h

def calc\_EV\_HM(hm):   return  54.4426 + 0.0561 \* hm

def calc\_EV\_F(f): return   54.6990 + 0.1052 \* f

def calc\_EV\_R(r): return   54.0070 + 2.9178 \* r

def calc\_EV\_RR(rr): return   52.7229 + 0.5978 \* rr

def calc\_EV\_SCI(sci):  return   54.7138 + 5.9636 \* sci

def calc\_EV\_GA(ga):  return   53.5643 + 1.3881 \* ga

def calc\_EV\_ABC(abc):  return   54.0389 + 1.8356 \* abc



\# Apply the functions to DataFrame

df\['EV\_M1'] = df\['M1'].apply(calc\_EV\_M1)

df\['EV\_M2'] = df\['M2'].apply(calc\_EV\_M2)

df\['EV\_H']  = df\['H'].apply(calc\_EV\_H)

df\['EV\_HM']  = df\['HM'].apply(calc\_EV\_HM)

df\['EV\_F'] = df\['F'].apply(calc\_EV\_F)

df\['EV\_R'] = df\['R'].apply(calc\_EV\_R)

df\['EV\_RR'] = df\['RR'].apply(calc\_EV\_RR)

df\['EV\_SCI'] = df\['SCI'].apply(calc\_EV\_SCI)

df\['EV\_GA'] = df\['GA'].apply(calc\_EV\_GA)

df\['EV\_ABC'] = df\['ABC'].apply(calc\_EV\_ABC)



print(df\[\['Drug', 'EV', 'EV\_M1', 'EV\_M2', 'EV\_H', 'EV\_HM', 'EV\_F', 'EV\_R', 'EV\_RR', 'EV\_SCI', 'EV\_GA', 'EV\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['EV'], label='EV', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['EV\_M1'], label='M1')

plt.plot(df\['Drug'], df\['EV\_M2'], label='M2')

plt.plot(df\['Drug'], df\['EV\_H'], label='H')

plt.plot(df\['Drug'], df\['EV\_HM'], label='HM')

plt.plot(df\['Drug'], df\['EV\_F'], label='F')

plt.plot(df\['Drug'], df\['EV\_R'], label='R')

plt.plot(df\['Drug'], df\['EV\_RR'], label='RR')

plt.plot(df\['Drug'], df\['EV\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['EV\_GA'], label='GA')

plt.plot(df\['Drug'], df\['EV\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('enthalpy of vaporization (EV)')

plt.title('Actual and Predicted values of enthalpy of vaporization (EV)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 50, 100, 150, 200])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("enthalpy of vaporization\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 3\. Prediction of FP



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;   "Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "FP": \[193, 224.7,	334.7,	267.4,	274.3,	160,	178.8,	188.3,	351.9,	234.7,	366.3,	292.5,	231.2,	384.3,	383.1,	215,	254.5,	288.2,	248.5,	371.8,	307.5, 308.3],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;  "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_FP\_M1(m1): return   148.3255 + 1.0424 \* m1

def calc\_FP\_M2(m2): return 166.2082 + 0.7785 \* m2

def calc\_FP\_H(h):   return  134.1791 + 12.6810 \*h

def calc\_FP\_HM(hm):   return  168.1915 + 0.1798 \* hm

def calc\_FP\_F(f): return 170.4084 + 0.3324 \*  f

def calc\_FP\_R(r): return  133.6725 + 12.1938 \* r

def calc\_FP\_RR(rr): return 147.3185 + 2.1753 \* rr

def calc\_FP\_SCI(sci):  return   134.1791 + 25.3620 \* sci

def calc\_FP\_GA(ga):  return    137.7898 + 5.5600 \* ga

def calc\_FP\_ABC(abc):  return   136.2323 + 7.5396 \* abc



\# Apply the functions to DataFrame

df\['FP\_M1'] = df\['M1'].apply(calc\_FP\_M1)

df\['FP\_M2'] = df\['M2'].apply(calc\_FP\_M2)

df\['FP\_H']  = df\['H'].apply(calc\_FP\_H)

df\['FP\_HM']  = df\['HM'].apply(calc\_FP\_HM)

df\['FP\_F'] = df\['F'].apply(calc\_FP\_F)

df\['FP\_R'] = df\['R'].apply(calc\_FP\_R)

df\['FP\_RR'] = df\['RR'].apply(calc\_FP\_RR)

df\['FP\_SCI'] = df\['SCI'].apply(calc\_FP\_SCI)

df\['FP\_GA'] = df\['GA'].apply(calc\_FP\_GA)

df\['FP\_ABC'] = df\['ABC'].apply(calc\_FP\_ABC)



print(df\[\['Drug', 'FP', 'FP\_M1', 'FP\_M2', 'FP\_H', 'FP\_HM', 'FP\_F', 'FP\_R', 'FP\_RR', 'FP\_SCI', 'FP\_GA', 'FP\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['FP'], label='FP', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['FP\_M1'], label='M1')

plt.plot(df\['Drug'], df\['FP\_M2'], label='M2')

plt.plot(df\['Drug'], df\['FP\_H'], label='H')

plt.plot(df\['Drug'], df\['FP\_HM'], label='HM')

plt.plot(df\['Drug'], df\['FP\_F'], label='F')

plt.plot(df\['Drug'], df\['FP\_R'], label='R')

plt.plot(df\['Drug'], df\['FP\_RR'], label='RR')

plt.plot(df\['Drug'], df\['FP\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['FP\_GA'], label='GA')

plt.plot(df\['Drug'], df\['FP\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('Flash Point (FP)')

plt.title('Actual and Predicted values of Flash Point (FP)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 100, 200, 300, 400])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Flash\_point\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 4\. Prediction of IR



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;   "Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "IR": \[1.719,	1.53,	1.591,	1.542,	1.651,	1.49,	1.562,	1.538,	1.544,	1.543,	1.681,	1.597,	1.513,	1.667,	1.641,	1.585,	1.589,	1.549,	1.549,	1.816,	1.547,	1.509],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;  "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_IR\_M1(m1): return 1.5244 + 0.0005 \* m1

def calc\_IR\_M2(m2): return 1.5180 + 0.0005 \* m2

def calc\_IR\_H(h):   return  1.5442 + 0.0040 \* h

def calc\_IR\_HM(hm): return 1.5217 + 0.0001 \* hm

def calc\_IR\_F(f): return  1.5254 + 0.0002 \* f

def calc\_IR\_R(r): return  1.5457 + 0.0037 \* r

def calc\_IR\_RR(rr): return 1.5224 + 0.0011 \* rr

def calc\_IR\_SCI(sci): return  1.5442 + 0.0080 \*  sci

def calc\_IR\_GA(ga):  return 1.5317 + 0.0023 \* ga

def calc\_IR\_ABC(abc):  return  1.5384 + 0.0027 \* abc





\# Apply the functions to DataFrame

df\['IR\_M1'] = df\['M1'].apply(calc\_IR\_M1)

df\['IR\_M2'] = df\['M2'].apply(calc\_IR\_M2)

df\['IR\_H']  = df\['H'].apply(calc\_IR\_H)

df\['IR\_HM']  = df\['HM'].apply(calc\_IR\_HM)

df\['IR\_F'] = df\['F'].apply(calc\_IR\_F)

df\['IR\_R'] = df\['R'].apply(calc\_IR\_R)

df\['IR\_RR'] = df\['RR'].apply(calc\_IR\_RR)

df\['IR\_SCI'] = df\['SCI'].apply(calc\_IR\_SCI)

df\['IR\_GA'] = df\['GA'].apply(calc\_IR\_GA)

df\['IR\_ABC'] = df\['ABC'].apply(calc\_IR\_ABC)



print(df\[\['Drug', 'IR', 'IR\_M1', 'IR\_M2', 'IR\_H', 'IR\_HM', 'IR\_F', 'IR\_R', 'IR\_RR', 'IR\_SCI', 'IR\_GA', 'IR\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['IR'], label='IR', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['IR\_M1'], label='M1')

plt.plot(df\['Drug'], df\['IR\_M2'], label='M2')

plt.plot(df\['Drug'], df\['IR\_H'], label='H')

plt.plot(df\['Drug'], df\['IR\_HM'], label='HM')

plt.plot(df\['Drug'], df\['IR\_F'], label='F')

plt.plot(df\['Drug'], df\['IR\_R'], label='R')

plt.plot(df\['Drug'], df\['IR\_RR'], label='RR')

plt.plot(df\['Drug'], df\['IR\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['IR\_GA'], label='GA')

plt.plot(df\['Drug'], df\['IR\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('Index of refraction (IR)')

plt.title('Actual and Predicted values of Index of refraction (IR)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 0.5, 1, 1.5,2])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Index of refraction\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 5\. Prediction of MR



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;   "Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "MR": \[59.2,	88.9,	122.7,	81.4,	94.2,	20.5,	32.1,	123.6,	136.1,	82.7,	118.2,	38.9,	87,	135,	143.9,	57,	84.3,	121.3,	82.2,	87.8,	127.5,	106.8],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;  "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_MR\_M1(m1): return     11.0302 + 0.6669 \* m1

def calc\_MR\_M2(m2): return  25.1537 + 0.4789 \* m2

def calc\_MR\_H(h):   return  -0.4428 + 8.3302 \* h

def calc\_MR\_HM(hm):  return  25.2680 + 0.1124 \* hm

def calc\_MR\_F(f): return    25.6379 + 0.2111 \* f

def calc\_MR\_R(r): return  -1.5567 + 8.0774 \* r

def calc\_MR\_RR(rr): return  11.0741 + 1.3799 \* rr

def calc\_MR\_SCI(sci):  return  -0.4428 + 16.6604 \* sci

def calc\_MR\_GA(ga):  return  3.4654 + 3.5903 \* ga

def calc\_MR\_ABC(abc): return 0.5330 + 4.9730 \* abc



\# Apply the functions to DataFrame

df\['MR\_M1'] = df\['M1'].apply(calc\_MR\_M1)

df\['MR\_M2'] = df\['M2'].apply(calc\_MR\_M2)

df\['MR\_H']  = df\['H'].apply(calc\_MR\_H)

df\['MR\_HM']  = df\['HM'].apply(calc\_MR\_HM)

df\['MR\_F'] = df\['F'].apply(calc\_MR\_F)

df\['MR\_R'] = df\['R'].apply(calc\_MR\_R)

df\['MR\_RR'] = df\['RR'].apply(calc\_MR\_RR)

df\['MR\_SCI'] = df\['SCI'].apply(calc\_MR\_SCI)

df\['MR\_GA'] = df\['GA'].apply(calc\_MR\_GA)

df\['MR\_ABC'] = df\['ABC'].apply(calc\_MR\_ABC)



print(df\[\['Drug', 'MR', 'MR\_M1', 'MR\_M2', 'MR\_H', 'MR\_HM', 'MR\_F', 'MR\_R', 'MR\_RR', 'MR\_SCI', 'MR\_GA', 'MR\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['MR'], label='MR', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['MR\_M1'], label='M1')

plt.plot(df\['Drug'], df\['MR\_M2'], label='M2')

plt.plot(df\['Drug'], df\['MR\_H'], label='H')

plt.plot(df\['Drug'], df\['MR\_HM'], label='HM')

plt.plot(df\['Drug'], df\['MR\_F'], label='F')

plt.plot(df\['Drug'], df\['MR\_R'], label='R')

plt.plot(df\['Drug'], df\['MR\_RR'], label='RR')

plt.plot(df\['Drug'], df\['MR\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['MR\_GA'], label='GA')

plt.plot(df\['Drug'], df\['MR\_ABC'], label='ABC')



plt.xlabel('Drug Index')

plt.ylabel('Molar refractivity (MR)')

plt.title('Actual and Predicted values of Molar refractivity (MR)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 50, 100, 150, 200])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Molar refractivity\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 6\. Prediction of PSA



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;"Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "PSA": \[62,	51,	90,	71,	149,	61,	59,	87,	142,	59,	93,	121,	87,	94,	128,	44,	71,	76,	108,	160,	96,	95],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_PSA\_M1(m1): return    23.6797 + 0.5194 \* m1

def calc\_PSA\_M2(m2): return   25.6662 + 0.4265 \* m2

def calc\_PSA\_H(h):   return    21.6449 + 6.1173 \* h

def calc\_PSA\_HM(hm):   return    25.9531 + 0.1005 \* hm

def calc\_PSA\_F(f): return   26.3869 + 0.1897 \* f

def calc\_PSA\_R(r): return   21.2650 + 5.8949 \* r

def calc\_PSA\_RR(rr): return  23.6699 + 1.0726 \* rr

def calc\_PSA\_SCI(sci):  return   21.6748 + 12.2296 \* sci

def calc\_PSA\_GA(ga):  return  44.9970 + 1.8622 \* ga

def calc\_PSA\_ABC(abc):  return   22.1490 + 3.6044 \* abc



\# Apply the functions to DataFrame

df\['PSA\_M1'] = df\['M1'].apply(calc\_PSA\_M1)

df\['PSA\_M2'] = df\['M2'].apply(calc\_PSA\_M2)

df\['PSA\_H']  = df\['H'].apply(calc\_PSA\_H)

df\['PSA\_HM']  = df\['HM'].apply(calc\_PSA\_HM)

df\['PSA\_F'] = df\['F'].apply(calc\_PSA\_F)

df\['PSA\_R'] = df\['R'].apply(calc\_PSA\_R)

df\['PSA\_RR'] = df\['RR'].apply(calc\_PSA\_RR)

df\['PSA\_SCI'] = df\['SCI'].apply(calc\_PSA\_SCI)

df\['PSA\_GA'] = df\['GA'].apply(calc\_PSA\_GA)

df\['PSA\_ABC'] = df\['ABC'].apply(calc\_PSA\_ABC)



print(df\[\['Drug', 'PSA', 'PSA\_M1', 'PSA\_M2', 'PSA\_H', 'PSA\_HM', 'PSA\_F', 'PSA\_R', 'PSA\_RR', 'PSA\_SCI', 'PSA\_GA', 'PSA\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['PSA'], label='PSA', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['PSA\_M1'], label='M1')

plt.plot(df\['Drug'], df\['PSA\_M2'], label='M2')

plt.plot(df\['Drug'], df\['PSA\_H'], label='H')

plt.plot(df\['Drug'], df\['PSA\_HM'], label='HM')

plt.plot(df\['Drug'], df\['PSA\_F'], label='F')

plt.plot(df\['Drug'], df\['PSA\_R'], label='R')

plt.plot(df\['Drug'], df\['PSA\_RR'], label='RR')

plt.plot(df\['Drug'], df\['PSA\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['PSA\_GA'], label='GA')

plt.plot(df\['Drug'], df\['PSA\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('Polar Surface Area (PSA)')

plt.title('Actual and Predicted values of Polar Surface Area (PSA)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 50, 100, 150, 200])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("polar surface area.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 7\. Prediction of P

import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;"Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "P": \[23.5,	35.3,	48.6,	32.3,	37.3,	8.1,	12.7,	49,	53.9,	32.8,	46.9,	15.4,	34.5,	53.5,	57.1,	22.6,	33.4,	48.1,	32.6,	34.8,	50.6,	42.3],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_P\_M1(m1): return 5.8026 + 0.2585 \* m1

def calc\_P\_M2(m2): return 11.4920 + 0.1841 \* m2

def calc\_P\_H(h):   return 0.9365 + 3.2667 \* h

def calc\_P\_HM(hm): return 11.5768 + 0.0432 \* hm

def calc\_P\_F(f): return 11.7558 + 0.0809 \* f

def calc\_P\_R(r): return 0.5628 + 3.1621 \* r

def calc\_P\_RR(rr): return 5.7841 + 0.5355 \* rr

def calc\_P\_SCI(sci):  return 0.9365 + 6.5334 \* sci

def calc\_P\_GA(ga):  return 1.3657 + 1.4237 \* ga

def calc\_P\_ABC(abc):return 1.4634 + 1.9423 \* abc



\# Apply the functions to DataFrame

df\['P\_M1'] = df\['M1'].apply(calc\_P\_M1)

df\['P\_M2'] = df\['M2'].apply(calc\_P\_M2)

df\['P\_H']  = df\['H'].apply(calc\_P\_H)

df\['P\_HM']  = df\['HM'].apply(calc\_P\_HM)

df\['P\_F'] = df\['F'].apply(calc\_P\_F)

df\['P\_R'] = df\['R'].apply(calc\_P\_R)

df\['P\_RR'] = df\['RR'].apply(calc\_P\_RR)

df\['P\_SCI'] = df\['SCI'].apply(calc\_P\_SCI)

df\['P\_GA'] = df\['GA'].apply(calc\_P\_GA)

df\['P\_ABC'] = df\['ABC'].apply(calc\_P\_ABC)



print(df\[\['Drug', 'P', 'P\_M1', 'P\_M2', 'P\_H', 'P\_HM', 'P\_F', 'P\_R', 'P\_RR', 'P\_SCI', 'P\_GA', 'P\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['P'], label='P', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['P\_M1'], label='M1')

plt.plot(df\['Drug'], df\['P\_M2'], label='M2')

plt.plot(df\['Drug'], df\['P\_H'], label='H')

plt.plot(df\['Drug'], df\['P\_HM'], label='HM')

plt.plot(df\['Drug'], df\['P\_F'], label='F')

plt.plot(df\['Drug'], df\['P\_R'], label='R')

plt.plot(df\['Drug'], df\['P\_RR'], label='RR')

plt.plot(df\['Drug'], df\['P\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['P\_GA'], label='GA')

plt.plot(df\['Drug'], df\['P\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('Polarizability (P)')

plt.title('Actual and Predicted values of Polarizability (P)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 20, 40, 60, 80])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Polarizability\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 8\. Prediction of ST



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;"Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;   "ST": \[59.4,	41.5,	48,	43.6,	79,	62,	56.7,	43.4,	48.3,	44,	53.3,	99.9,	34.5,	58,	53,	42,	46.9,	40.8,	52.5,	89.4,	41.7,	42.4],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_ST\_M1(m1): return    58.0743 + -0.0363 \* m1

def calc\_ST\_M2(m2): return 53.7273 + -0.0006 \* m2

def calc\_ST\_H(h):   return     64.0690 + -0.9355 \* h

def calc\_ST\_HM(hm):   return  54.0159 + -0.0006 \*  hm

def calc\_ST\_F(f): return 54.2750 + -0.0020 \* f

def calc\_ST\_R(r): return  64.1009 + -0.8991 \* r

def calc\_ST\_RR(rr): return  57.9911 + -0.0737 \* rr

def calc\_ST\_SCI(sci):  return  64.0690 + -1.8710 \* sci

def calc\_ST\_GA(ga):  return  61.5665 + -0.3198 \* ga

def calc\_ST\_ABC(abc):  return  63.2576 + -0.5205 \* abc



\# Apply the functions to DataFrame

df\['ST\_M1'] = df\['M1'].apply(calc\_ST\_M1)

df\['ST\_M2'] = df\['M2'].apply(calc\_ST\_M2)

df\['ST\_H']  = df\['H'].apply(calc\_ST\_H)

df\['ST\_HM']  = df\['HM'].apply(calc\_ST\_HM)

df\['ST\_F'] = df\['F'].apply(calc\_ST\_F)

df\['ST\_R'] = df\['R'].apply(calc\_ST\_R)

df\['ST\_RR'] = df\['RR'].apply(calc\_ST\_RR)

df\['ST\_SCI'] = df\['SCI'].apply(calc\_ST\_SCI)

df\['ST\_GA'] = df\['GA'].apply(calc\_ST\_GA)

df\['ST\_ABC'] = df\['ABC'].apply(calc\_ST\_ABC)



print(df\[\['Drug', 'ST', 'ST\_M1', 'ST\_M2', 'ST\_H', 'ST\_HM', 'ST\_F', 'ST\_R', 'ST\_RR', 'ST\_SCI', 'ST\_GA', 'ST\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['ST'], label='ST', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['ST\_M1'], label='M1')

plt.plot(df\['Drug'], df\['ST\_M2'], label='M2')

plt.plot(df\['Drug'], df\['ST\_H'], label='H')

plt.plot(df\['Drug'], df\['ST\_HM'], label='HM')

plt.plot(df\['Drug'], df\['ST\_F'], label='F')

plt.plot(df\['Drug'], df\['ST\_R'], label='R')

plt.plot(df\['Drug'], df\['ST\_RR'], label='RR')

plt.plot(df\['Drug'], df\['ST\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['ST\_GA'], label='GA')

plt.plot(df\['Drug'], df\['ST\_ABC'], label='ABC')



plt.xlabel('Drug Index')

plt.ylabel('Surface tension (ST)')

plt.title('Actual and Predicted values of Surface tension (ST)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 20, 40, 60, 80,100])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Surface tension\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()



# 9\. Prediction of MV



import pandas as pd

import matplotlib.pyplot as plt



\# Input data from the table

data = {

&nbsp;"Drug": \["Apraclonidine", "Betaxolol", "Bimatoprost", "Carteolol", "Ginkgolide B", "Glycerol", "Isosorbide", "Latanoprost ", "Latanoprostene bunod ", "Levobunolol", "Losartan", "Mannitol", "Metipranolol", "Netarsudil", "Omidenepag isopropyl", "Pilocarpine", "Ripasudil", "Tafluprost", "Timolol", "Trabodenoson", "Travoprost", "Unoprostone"],

&nbsp;    "MV": \[150,	288,	362.8,	258.6,	258,	71,	99,	395.5,	430.7,	262.6,	312.5,	114.1,	289.5,	362.6,	399.2,	170.2,	250.1,	381.5,	258.5,	202.5,	402,	357.8],

&nbsp;  "M1": \[76,	104,	140,	106,	200,	20,	54,	146,	166,	106,	156, 50,	104,	176,	190,	76,	118,	154,	104,	146,	170,	120],

&nbsp;   "M2": \[87,	113,	155,	118,	279,	19,	65,	161,	181,	119,	185,	55,	116,	205,	218,	89,	142,	170,	115,	175,	189,	130],

&nbsp;   "H": \[6.866667,	10.3,	14.06667,	9.3,	12.85476,	2.633333,	4.6,	14.3,	16.8,	9.333333,	14.43333,	5.133333,	9.533333,	15.93333,	17.20476,	6.966667,	10.10476,	14.46667,	9.5,	12.63333,	15.58571,	12.46667],

&nbsp; "HM": \[368,	476,	648,	520,	1198,	84,	272,	680,	760,	522,	760,	240,	504,	856,	924,	372,	602,	736,	502,	728,	822,	548],

&nbsp;  "F": \[194,	250,	338,	284,	640,	46,	142,	358,	398,	284,	390,	130,	272,	446,	488,	194,	318,	396,	272,	378,	444,	288],

&nbsp;   "R": \[7.164704,	10.63103,	14.50724,	9.849155,	13.83997,	2.80806,	4.787694,	14.86308,	17.36308,	9.865992,	14.70704,	5.540111,	10.21812,	16.43989,	17.8077,	7.219545,	10.51487,	15.17634,	9.955308,	13.04171,	16.46825,	12.9516],

&nbsp;   "RR": \[36.79207,	50.59003,	68.2868,	50.33794,	95.44305,	9.459457,	26.26206,	70.78618,	80.78618,	50.43896,	76.85333,	23.65561,	49.45232,	85.88272,	92.11415,	37.02474,	57.08924,	73.81201,	49.70691,	71.32208,	81.14926,	58.2209],

&nbsp;   "SCI": \[3.433333,	5.15,	7.033333,	4.65,	6.427381,	1.316667,	2.3,	7.15,	8.4,	4.666667,	7.216667,	2.566667,	4.766667,	7.966667,	8.602381,	3.483333,	5.052381,	7.233333,	4.75,	6.316667,	7.792857,	6.233333],

&nbsp;   "GA": \[15.4364,	22.35563,	30.18467,	20.89302,	33.06481,	4.711235,	10.65123,	30.9537,	35.9537,

&nbsp;          20.93343,	32.46879,	10.30931,	20.76657,	36.02303,	38.77348,	15.53343,

&nbsp;          23.19031,	31.6137,	21.06741,	29.22165,	34.30304,	26.13151],

&nbsp;   "ABC": \[11.561,	16.59163,	22.23655,	16.171,	23.24362,	3.644924,	7.875634,	23.16244,	26.69797,	16.13056,	23.20127,	8.094413,	16.26971,	26.39831,	28.76662,	11.41117,	17.2222,	24.07799,	16.06161,	21.36768,	26.40601,	19.51751],

}

\# Convert to DataFrame

df = pd.DataFrame(data)

\# Define functions for each equation

def calc\_MV\_M1(m1): return    51.8952 + 1.8400 \* m1

def calc\_MV\_M2(m2): return   97.4506 + 1.2744 \* m2

def calc\_MV\_H(h):   return  9.7395 + 23.9267 \* h

def calc\_MV\_HM(hm):   return  96.4060 + 0.3015 \* hm

def calc\_MV\_F(f): return   96.1782 + 0.5699 \* f

def calc\_MV\_R(r): return   5.6626 + 23.2759 \* r

def calc\_MV\_RR(rr): return 52.7987 + 3.7941 \* rr

def calc\_MV\_SCI(sci):  return  9.7395 + 47.8533 \* sci

def calc\_MV\_GA(ga):  return 26.2066 + 10.1005 \* ga

def calc\_MV\_ABC(abc):  return   14.5313 + 14.1760 \* abc



\# Apply the functions to DataFrame

df\['MV\_M1'] = df\['M1'].apply(calc\_MV\_M1)

df\['MV\_M2'] = df\['M2'].apply(calc\_MV\_M2)

df\['MV\_H']  = df\['H'].apply(calc\_MV\_H)

df\['MV\_HM']  = df\['HM'].apply(calc\_MV\_HM)

df\['MV\_F'] = df\['F'].apply(calc\_MV\_F)

df\['MV\_R'] = df\['R'].apply(calc\_MV\_R)

df\['MV\_RR'] = df\['RR'].apply(calc\_MV\_RR)

df\['MV\_SCI'] = df\['SCI'].apply(calc\_MV\_SCI)

df\['MV\_GA'] = df\['GA'].apply(calc\_MV\_GA)

df\['MV\_ABC'] = df\['ABC'].apply(calc\_MV\_ABC)



print(df\[\['Drug', 'MV', 'MV\_M1', 'MV\_M2', 'MV\_H', 'MV\_HM', 'MV\_F', 'MV\_R', 'MV\_RR', 'MV\_SCI', 'MV\_GA', 'MV\_ABC']])



\# Plotting

plt.figure(figsize=(6, 4))



\# Actual Boiling Point

plt.plot(df\['Drug'], df\['MV'], label='MV', color='blue')

\# Predicted

plt.plot(df\['Drug'], df\['MV\_M1'], label='M1')

plt.plot(df\['Drug'], df\['MV\_M2'], label='M2')

plt.plot(df\['Drug'], df\['MV\_H'], label='H')

plt.plot(df\['Drug'], df\['MV\_HM'], label='HM')

plt.plot(df\['Drug'], df\['MV\_F'], label='F')

plt.plot(df\['Drug'], df\['MV\_R'], label='R')

plt.plot(df\['Drug'], df\['MV\_RR'], label='RR')

plt.plot(df\['Drug'], df\['MV\_SCI'], label='SCI')

plt.plot(df\['Drug'], df\['MV\_GA'], label='GA')

plt.plot(df\['Drug'], df\['MV\_ABC'], label='ABC')





plt.xlabel('Drug Index')

plt.ylabel('Molar volume (MV)')

plt.title('Actual and Predicted values of Molar volume (MV)')



\# Remove x-axis labels (drug names)

plt.xticks(\[], \[])

plt.yticks(\[0, 150, 300, 450])



\# Move legend to bottom

plt.legend(loc='upper center', bbox\_to\_anchor=(0.5, -0.15), ncol=6, frameon=False)



plt.tight\_layout()

plt.subplots\_adjust(bottom=0.25)  # Ensure space for the legend

plt.grid(True)



plt.savefig("Molar volume\_plot.pdf", format='pdf', bbox\_inches='tight')  # Save as PDF

plt.show()





