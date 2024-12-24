import pandas as pd
import matplotlib.pyplot as plt



# Load the CSV files
cpu_data = pd.read_csv('out_files/mesures/cpu_1973633.csv')  # Replace with your actual file name
gpu_data = pd.read_csv('out_files/mesures/gpu_1973633.csv')  # Replace with your actual file name

# Merge the two DataFrames
merged_data = pd.concat([cpu_data, gpu_data])

# Optionally, sort the merged data by time
merged_data['time'] = pd.to_datetime(merged_data['time'])  # Ensure 'time' column is datetime
merged_data = merged_data.sort_values(by='time')

# Save the merged data to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)

print(merged_data.head())  # Display the first few rows

# Replace GPU measure by the cumulative sum of GPU measures
merged_data.loc[merged_data['type'] == 'GPU', 'measure'] = merged_data.loc[
    merged_data['type'] == 'GPU', 'measure'
].cumsum()


# Plot the data
plt.figure(figsize=(12, 6))
for data_type, group in merged_data.groupby('type'):
    plt.plot(group['time'], group['measure'], label=data_type)


# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Measure')
plt.title('Measure vs Time')
plt.legend(title='Type')
plt.grid(True)

plt.savefig('measure_vs_time.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution

