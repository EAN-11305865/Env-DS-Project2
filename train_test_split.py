import pandas as pd
from data_preprocess import preprocess_data, load_and_merge_datasets

def split_train_test(data, year_threshold=2050):
    train = data[data['time'].dt.year < year_threshold]
    test = data[data['time'].dt.year >= year_threshold]
    return train, test

if __name__ == "__main__":
    file_names = [
        '003_2006_2080_352_360.nc', '004_2006_2080_352_360.nc', '005_2006_2080_352_360.nc',
        '006_2006_2080_352_360.nc', '007_2006_2080_352_360.nc', '008_2006_2080_352_360.nc'
    ]
    combined_dataset = load_and_merge_datasets(file_names)
    manchester_lat, manchester_lon = 53.246075, 357.5
    data = preprocess_data(combined_dataset, manchester_lat, manchester_lon)
    
    train_data, test_data = split_train_test(data)
    
    print("Train Data Sample:")
    print(train_data.head())
    print("\nTest Data Sample:")
    print(test_data.head())
