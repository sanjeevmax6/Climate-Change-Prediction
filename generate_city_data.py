import pandas as pd

# Read the dataset into a DataFrame
df = pd.read_csv('./data/GlobalLandTemperaturesByMajorCity.csv')

unique_cities = df["City"].unique()

print(unique_cities)

# Iterate over cities
for city in unique_cities:
    # Filter the DataFrame for the current city
    city_df = df[df['City'] == city]

    # Generate the output file name with city name included
    output_file = f"{city.replace(' ', '_')}_data.csv"

    # Save the filtered DataFrame to a CSV file
    city_df.to_csv(output_file, index=False)

    print(f"Saved data for {city} to {output_file}")

