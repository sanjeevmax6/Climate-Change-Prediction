activate python venv (source spike_env/bin/activate)

1) Run SNN_basline.py to generate results on the model provided - this will be trained on the global average temperatures dataset
2) Run SNN_indivudal_city.py to train the model on individual cities generated from global average temperatures by major cities dataset 
3) Run SNN_modified.py to generate the actual results depicted in the paper (induces randomness and the atan smoothing function)
4) generated plots in root folder
5) the data folder contains all the datasets and datasets for each city generated
6) generate_city() generates datasets for major cities and creates individual cities