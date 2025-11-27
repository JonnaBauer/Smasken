import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_data_ht2025.csv")

# Create binary: 1 if high real demand (i.e. "low_bike_demand" = bikes are being used)
df['high_demand'] = (df['increase_stock'] == 'low_bike_demand').astype(int)

# Group by holiday and calculate percentage of high demand hours
holiday_stats = df.groupby('holiday')['high_demand'].mean() * 100

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(
    [0, 1],
    holiday_stats.values,
    color=['#3498db', '#e74c3c'],
    width=0.6
)

# Add percentage labels on top
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

# Clean looks
plt.xticks([0, 1], ['Regular Day', 'Holiday'], fontsize=12)
plt.ylabel('High Bike Demand (%)', fontsize=13)
plt.title('Do People Ride More Bikes on Holidays?', fontsize=16, pad=20)
plt.ylim(0, 100)
plt.box(False)  # remove top/right borders

# Save
plt.tight_layout()
plt.savefig("holiday_bike_demand.png", dpi=300, transparent=False)
plt.show()

print("penis")