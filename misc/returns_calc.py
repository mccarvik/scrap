

age = 22
years = 38
initial_investment = 120000
annual_addition = 20000
annual_interest = 0.105
inflation = 0.03
tax_rate = 0.20

future_value = initial_investment
year = 2024

# Calculate future value
for year in range(years+1):
    print("year: {}  age: {}   value: {:.2f}".format(year, age, future_value))
    future_value = future_value * (1 + (annual_interest-inflation)*(1-tax_rate))
    # future_value = future_value * (1 + (annual_interest-inflation))
    future_value = future_value + annual_addition
    age+=1
print("Future value: {:.2f}".format(future_value))

