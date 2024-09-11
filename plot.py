
import matplotlib.pyplot as plt

import pandas as pd
csv_name = "individual_prompt3"
df = pd.read_csv(f"{csv_name}.csv")

y_prd = df['Prediction'].values.tolist()
y_true = df['Ground Truth'].values.tolist()

# Plotting the lines
plt.figure(figsize=(10, 5))
plt.plot(y_prd , label='Prediction', linestyle='-', marker='o')
plt.plot(y_true, label='Besttrack', linestyle='-', marker='x')

# Adding titles and labels
plt.title('Comparison of Predicted and True Values')
plt.xlabel('Step')
plt.ylabel('M.S wind speed (m/s)')

# Adding a grid
plt.grid(True, linestyle=':')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()
plt.savefig(f"{csv_name}.png")
