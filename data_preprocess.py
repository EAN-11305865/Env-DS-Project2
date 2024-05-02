import xarray as xr
import pandas as pd

def load_and_merge_datasets(file_names):
    datasets = [xr.open_dataset(f).expand_dims("source").assign_coords(source=[f]) for f in file_names]
    combined_dataset = xr.concat(datasets, dim="source")
    combined_dataset.to_netcdf('combined_dataset.nc')
    return combined_dataset

def preprocess_data(dataset, lat, lon):
    ds = dataset.sel(lat=lat, lon=lon, method='nearest')
    df = ds.to_dataframe().reset_index()
    df = df.drop(['lat', 'lon'], axis=1)
    df['time'] = pd.to_datetime(df['time'].astype(str))
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['dayofyear'] = df['time'].dt.dayofyear
    df['source'] = df['source'].str.split('_').str[0].astype(int)
    return df

if __name__ == "__main__":
    file_names = [
        '003_2006_2080_352_360.nc', '004_2006_2080_352_360.nc', '005_2006_2080_352_360.nc',
        '006_2006_2080_352_360.nc', '007_2006_2080_352_360.nc', '008_2006_2080_352_360.nc'
    ]
    combined_dataset = load_and_merge_datasets(file_names)
    manchester_lat, manchester_lon = 53.246075, 357.5
    manchester_data = preprocess_data(combined_dataset, manchester_lat, manchester_lon)
    print(manchester_data.head())
