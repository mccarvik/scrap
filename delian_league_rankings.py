import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = {
    'Name': ['Brigi', 'Keith', 'Pine', 'Lask', 'Yoff', 'Roth', 'Eric', 'Slevi'],
    '2007': [0, 0, 0, 3.939876521, 1.96993826, 1.313292174, 1.313292174, 0],
    '2008': [0, 1.397119334, 0, 0, 4.191358001, 2.095679, 0, 0],
    '2009': [0, 3.715742908, 1.486297163, 0, 2.972594327, 1.486297163, 0, 0],
    '2010': [0, 1.581167195, 0, 1.581167195, 3.952917988, 0, 3.16233439, 0],
    '2011': [0, 3.364185521, 0, 0, 1.682092761, 0, 1.682092761, 4.205231902],
    '2012': [0, 0, 0, 0, 3.578920767, 4.473650959, 1.789460384, 1.789460384],
    '2013': [0, 2.855521889, 0, 1.903681259, 0, 0, 2.855521889, 4.759203148],
    '2014': [0, 0, 3.037789243, 0, 0, 2.025192829, 3.037789243, 5.062982072],
    '2015': [0, 5.386151141, 3.231690685, 0, 2.154460456, 3.231690685, 0, 0],
    '2016': [0, 5.729948022, 0, 0, 4.583958418, 2.291979209, 0, 2.291979209],
    '2017': [2.438275754, 0, 4.876551508, 0, 0, 6.095689385, 2.438275754, 0],
    '2018': [5.187820754, 6.484775942, 0, 0, 0, 0, 2.593910377, 2.593910377],
    '2019': [0, 0, 0, 0, 4.139218686, 2.759479124, 4.139218686, 6.898697811],
    '2020': [2.93561609, 0, 0, 0, 0, 2.93561609, 7.339040224, 5.871232179],
    '2021': [3.12299584, 4.68449376, 0, 0, 0, 4.68449376, 7.8074896, 0],
    '2022': [3.322336, 4.983504, 9.967008, 3.322336, 0, 0, 0, 0],
    '2023': [3.5344, 5.3016, 0, 5.3016, 8.836, 0, 0, 0],
    '2024': [0, 0, 0, 3.76, 11.28, 0, 3.76, 5.64],
    '2025': [0, 4, 0, 6, 4, 0, 0, 12]
}

df = pd.DataFrame(data)
df.set_index('Name', inplace=True)

# Calculate cumulative scores
df_cumulative = df.cumsum(axis=1)

# Normalize cumulative scores for each year based on the top scorer for that year
df_normalized = df_cumulative.copy()
for year in df_cumulative.columns:
    max_cumulative_score = df_cumulative[year].max()
    if max_cumulative_score > 0:  # Avoid division by zero
        df_normalized[year] = df_cumulative[year] / max_cumulative_score




# Plotting
plt.figure(figsize=(12, 8))
for person in df_normalized.index:
    plt.plot(df_normalized.columns, df_normalized.loc[person], marker='o', label=person)

# Add vertical lines with captions
plt.axvline(x='2009', color='gray', linestyle='--', alpha=0.5)
plt.text('2009', 1.05, 'Brigi/Slevi join', rotation=90, verticalalignment='bottom', horizontalalignment='right')
plt.axvline(x='2021', color='gray', linestyle='--', alpha=0.5)
plt.text('2021', 1.05, 'Kev joins Lask', rotation=90, verticalalignment='bottom', horizontalalignment='right')

plt.title('Cumulative Scores Normalized by Top Scorer Each Year')
plt.xlabel('Year')
plt.ylabel('Normalized Cumulative Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("delian_league_normalized.png", dpi=300)
plt.show()

# Plotting raw cumulative scores
plt.figure(figsize=(12, 8))
for person in df_cumulative.index:
    plt.plot(df_cumulative.columns, df_cumulative.loc[person], marker='o', label=person)

# Add vertical lines with captions
plt.axvline(x='2009', color='gray', linestyle='--', alpha=0.5)
plt.text('2009', 1.05, 'Brigi/Slevi join', rotation=90, verticalalignment='bottom', horizontalalignment='right')
plt.axvline(x='2021', color='gray', linestyle='--', alpha=0.5)
plt.text('2021', 1.05, 'Kev joins Lask', rotation=90, verticalalignment='bottom', horizontalalignment='right')

plt.title('Raw Cumulative Scores')
plt.xlabel('Year')
plt.ylabel('Cumulative Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("delian_league_raw.png", dpi=300)
plt.show()