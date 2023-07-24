"""
script to measure fractionalization
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


data_df = pd.read_excel("GDP.xls", sheet_name="Data")
data_df = data_df[["country", "gdp_2021"]].set_index("country")

frac_df = pd.read_excel("GDP.xls", sheet_name="Fractional")
frac_df = frac_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
frac_df = frac_df.set_index("country")

data_df = pd.merge(data_df, frac_df, left_index=True, right_index=True, how='inner')
data_df = data_df.sort_values(ascending=False, by="gdp_2021")
data_df = data_df[:43]

# drop_nations = ["Monaco", "Luxembourg", "Bermuda", "Faroe Islands", "Andorra", "Guam", "Malta", 
#                 "Aruba", "Cyprus", "Barbados", 'Trinidad and Tobago', 'Antigua and Barbuda',
#                 'American Samoa', "Seychelles", "Nauru", "Palau", "Grenada", "Tuvalu", "Marshall Islands"]
drop_nations = ["Monaco", "Luxembourg", "Bermuda", "Faroe Islands", "Andorra", "Guam", "Malta", 
                "Aruba", "Cyprus"]
data_df = data_df.drop(drop_nations, axis=0)

# m, b = np.polyfit(data_df['ethnic_frac'], data_df['2021'], 1)
# create a dataframe with example data
# fit a linear regression model
model = smf.ols(formula='gdp_2021 ~ ethnic_frac', data=data_df).fit()

# get the slope and intercept of the regression line
m = model.params['ethnic_frac']
b = model.params['Intercept']


# create a scatter plot with the x and y values based on column names
pdb.set_trace()
fig, ax = plt.subplots()
ax.scatter(data_df['ethnic_frac'], data_df['gdp_2021'])
# Add labels to the dots using the 'label' column of the dataframe
for i, label in enumerate(data_df.index):
    ax.text(data_df['ethnic_frac'][i], data_df['gdp_2021'][i], label)

# plot the trend line using plot()
ax.plot(data_df['ethnic_frac'], m*data_df['ethnic_frac'] + b, color='red')

# set the plot title and labels
plt.title('Ethnic Frac')
plt.xlabel('Ethnic_frac')
plt.ylabel('GDP')
plt.savefig("ethnic_frac.png")
plt.close()
