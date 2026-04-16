import xarray as xr
import pandas as pd
import numpy as np

# 1. Load the dataset
all_ds = xr.open_dataset('../processed/2026-04-15/results.nc')

# 2. Convert to a flat DataFrame
# We include the necessary data variables for diameter and intensity
df = all_ds.to_dataframe().reset_index()

# 3. Step 1: Aggregate per individual cell
# This calculates the mean diameter and intensity for each specific cell
cell_level = df.groupby(['image', 'cell_label']).agg(
    spot_count=('feret_diameter_max', 'size'),
    cell_avg_diameter=('feret_diameter_max', 'mean'),
    cell_avg_intensity=('intensity_mean', 'mean')
).reset_index()

# 4. Step 2: Aggregate per image
# This calculates the final summary statistics across all cells in each image
summary = cell_level.groupby('image').agg(
    number_of_cells=('cell_label', 'nunique'),
    avg_spots_per_cell=('spot_count', 'mean'),
    avg_spot_diameter=('cell_avg_diameter', 'mean'),
    avg_spot_intensity=('cell_avg_intensity', 'mean')
).reset_index()

# 5. Export to CSV
summary.to_csv('image_summary.csv', index=False)

# --- DISPLAY RESULTS ---
print("FINAL IMAGE SUMMARY:")
print(summary)

exit()

# 1. Load the dataset
all_ds = xr.open_dataset('../processed/2026-04-15/results.nc')

# 2. Convert to a flat DataFrame for easier summary statistics
# This moves 'image' and 'cell_label' into accessible columns
df = all_ds.coords.to_dataset().to_dataframe().reset_index()

# 3. Calculate spots per individual cell (Intermediate step)
# This creates a table where each row is one cell
spots_per_cell = df.groupby(['image', 'cell_label']).size().reset_index(name='spot_count')

# 4. Create the final summary per image
# We aggregate the intermediate table to get:
# - Unique cell count
# - Average of the spot counts
summary = spots_per_cell.groupby('image').agg(
    number_of_cells=('cell_label', 'nunique'),
    avg_spots_per_cell=('spot_count', 'mean')
).reset_index()

# 5. Export to CSV
summary.to_csv('image_summary.csv', index=False)

# --- DISPLAY RESULTS ---
print("FINAL IMAGE SUMMARY:")
print(summary)

exit()


# 1. Load the dataset
all_ds = xr.open_dataset('../processed/2026-04-15/results.nc')

# 2. Set the MultiIndex
# We drop any existing index on 'id' to avoid the "indexes involved" error
if 'id' in all_ds.dims:
    all_ds = all_ds.drop_indexes("id", errors="ignore")
all_ds = all_ds.set_index(id=["image", "cell_label"])

# 3. COUNT: Number of Unique Cells per Image
# We convert coords to a DataFrame to use the efficient .nunique() method
cells_per_image = (all_ds.coords.to_dataset()
                  .to_dataframe()
                  .reset_index()
                  .groupby("image")["cell_label"]
                  .nunique())

# 4. COUNT: Number of Spots per Cell
# Grouping by the MultiIndex 'id' (image + cell_label) counts the rows/spots
# We use one of your data variables (intensity_mean) to hold the count
spots_per_cell = all_ds.groupby("id").count().intensity_mean

# --- DISPLAY SUMMARY ---
print("SUMMARY: CELLS PER IMAGE")
print(cells_per_image)

print("\nSUMMARY: SPOTS PER CELL (Top 10)")
# Converting to DataFrame makes the MultiIndex output much easier to read
print(spots_per_cell.to_dataframe(name="spot_count").head(10))


exit()


all_ds = xr.open_dataset('../processed/2026-04-15/results.nc')

#all_ds = all_ds.set_index(instance=["image", "cell_label"])
#cell_data = all_ds.sel(instance=("96wellplate_63x_03042026_processed-Scene-001-P1-B03", 6))

# 1. Clear any existing indexes on the 'id' dimension
all_ds = all_ds.drop_indexes("id", errors="ignore")

# 2. Explicitly create the MultiIndex
all_ds = all_ds.set_index(id=["image", "cell_label"])

# 3. Select using the new MultiIndex dimension name
cell_data = all_ds.sel(id=("96wellplate_63x_03042026_processed-Scene-001-P1-B03", 6))


print(cell_data.intensity_mean.size)

print(cell_data.intensity_mean.values)

cell_counts = (all_ds.coords.to_dataset()
               .to_dataframe()
               .reset_index() # This moves 'cell_label' from the index to a column
               .groupby("image")["cell_label"]
               .nunique())

print(cell_counts)


exit()


df = all_ds.to_dataframe()

df = df.rename(columns={
    'feret_diameter_max': 'spot_diameter'
})

print("All data")
print("--------------")
print(all_ds)
print(f"\n")

# print(df)
# Calculate summary statistics by image
cell_count = df.groupby('image')['cell_label'].nunique()
cell_count_xr = cell_count.to_xarray().rename("cell_count")

# Group the data
grouped_data = all_ds.groupby(["image", "cell_label"]).count()

print("Grouped data")
print("--------------")
print(grouped_data)
print(f"\n")

# Calculate summary statistics - change to number of spots per cell, number of cells etc.
spots_per_cell = grouped_data["intensity_mean"].groupby("image").mean()

print("Spots Per Cell")
print("--------------")
print(spots_per_cell)
print(f"\n")

mean_spots_per_cell = spots_per_cell.groupby("image").mean()


print("Mean Spots Per Cell")
print("--------------")
print(mean_spots_per_cell)
print(f"\n")

spot_intensity_per_cell = all_ds.intensity_mean.groupby(["image", "cell_label"]).mean().rename("avg_intensity")
mean_spot_intensity_per_cell = spot_intensity_per_cell.groupby("image").mean()

summary_ds = xr.merge([cell_count_xr, mean_spots_per_cell, mean_spot_intensity_per_cell])
summary_df = summary_ds.to_dataframe()