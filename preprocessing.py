import pandas as pd
import numpy as np

def datetime_to_spike_train(datetime_values, time_interval, temperature_values):

    has_na = temperature_values.isna().any()

    if has_na:
        print("The dataset contains {} missing values, hence proceeding with backward fill".format(temperature_values.isna().sum()))
        temperature_values = temperature_values.bfill()
        print("Backward fill complete")
    else:
        print("No missing entries found!")

    min_datetime = min(datetime_values)
    max_datetime = max(datetime_values)

    datetime_difference = max_datetime - min_datetime

    num_time_steps = int(np.ceil(datetime_difference.total_seconds() / time_interval)) + 1

    temperature_values_minmax_normalized = (temperature_values - temperature_values.min()) / (temperature_values.max() - temperature_values.min())
    temperature_values = temperature_values_minmax_normalized


    # Initialize spike trains
    spike_trains = []
    # Iterate through each datetime value
    for dt, temp in zip(datetime_values, temperature_values):
        # Calculate the time step for the current datetime value
        time_step = int((dt - min_datetime).total_seconds() / time_interval)
        
        # Create a spike train with zeros
        spike_train = np.zeros(num_time_steps)
        
        # Set the spike at the corresponding time step
        spike_train[time_step] = temp  # Using temperature to influence spike train
        
        # Append the spike train to the list
        spike_trains.append(spike_train)

    return spike_trains