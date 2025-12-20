# IMPORTING ALL DEPENDANCIES I NEED FOR THIS PROJECT


```python
import numpy as np
import pandas as pd
import missingno as msno 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.graphics.correlation as sgc
from statsmodels.graphics.gofplots import qqplot
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```

# CONNECTION TO MY DATABASE ON POSTGRES

This is the connection link to my database on postgreSQL, the actual connection function is on the file **db_connect.py**


```python
# Import necessary packages
import pandas as pd
import sys
sys.path.append('..')  # Go up one level to the root folder
from db_connect import connect_to_db

# Step 1: Connect to the database
conn = connect_to_db()

# Step 2: Create a cursor and run a query
cursor = conn.cursor()
query = "SELECT * FROM airbnbs_nairobi.listing_data_yearly;"
cursor.execute(query)

# Step 3: Fetch results and convert to a DataFrame
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

# Step 4: Display the data
print("Connection successful! Previewing data:")
display(df.head())
```

    Database connection successful!
    Connection successful! Previewing data:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>listing_name</th>
      <th>listing_type</th>
      <th>room_type</th>
      <th>cover_photo_url</th>
      <th>photos_count</th>
      <th>minimum_nights</th>
      <th>cancellation_policy</th>
      <th>professional_management</th>
      <th>registration</th>
      <th>...</th>
      <th>rating_communication</th>
      <th>rating_location</th>
      <th>rating_value</th>
      <th>revenue_per_year</th>
      <th>avg_rate_per_year</th>
      <th>annual_occupancy(%)</th>
      <th>revenue_per_night_yearly</th>
      <th>reserved_days_in_year</th>
      <th>blocked_days_in_year</th>
      <th>available_days_in_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75683</td>
      <td>Kiloranhouse Apt Prime Bedroom</td>
      <td>Private room in home</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/5499026/ef...</td>
      <td>13</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.8</td>
      <td>4.7</td>
      <td>141049.0</td>
      <td>6474.0</td>
      <td>6.6</td>
      <td>386.4</td>
      <td>24</td>
      <td>0</td>
      <td>341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>471581</td>
      <td>Located In a Serene Environment</td>
      <td>Entire cottage</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/6434524/bc...</td>
      <td>37</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.8</td>
      <td>4.8</td>
      <td>804490.0</td>
      <td>5791.6</td>
      <td>54.8</td>
      <td>3058.9</td>
      <td>144</td>
      <td>102</td>
      <td>221</td>
    </tr>
    <tr>
      <th>2</th>
      <td>906958</td>
      <td>Makena's Place Karen - Flamingo Room</td>
      <td>Private room in cottage</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/68ecc57f-d...</td>
      <td>29</td>
      <td>1</td>
      <td>Firm</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>594869.0</td>
      <td>6772.2</td>
      <td>24.4</td>
      <td>1629.8</td>
      <td>89</td>
      <td>0</td>
      <td>276</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1023556</td>
      <td>Guesthouse Near Nairobi National Park &amp; Airport</td>
      <td>Entire guesthouse</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/ddd8badc-1...</td>
      <td>20</td>
      <td>1</td>
      <td>Flexible</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.7</td>
      <td>4.8</td>
      <td>29004.0</td>
      <td>3631.3</td>
      <td>3.0</td>
      <td>79.5</td>
      <td>11</td>
      <td>0</td>
      <td>354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1237886</td>
      <td>Hob House</td>
      <td>Room in bed and breakfast</td>
      <td>hotel_room</td>
      <td>https://a0.muscache.com/im/pictures/cbdab7e1-f...</td>
      <td>8</td>
      <td>1</td>
      <td>Flexible</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>4.6</td>
      <td>4.7</td>
      <td>4.8</td>
      <td>168583.0</td>
      <td>15401.5</td>
      <td>3.0</td>
      <td>461.9</td>
      <td>11</td>
      <td>0</td>
      <td>354</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>listing_name</th>
      <th>listing_type</th>
      <th>room_type</th>
      <th>cover_photo_url</th>
      <th>photos_count</th>
      <th>minimum_nights</th>
      <th>cancellation_policy</th>
      <th>professional_management</th>
      <th>registration</th>
      <th>...</th>
      <th>rating_communication</th>
      <th>rating_location</th>
      <th>rating_value</th>
      <th>revenue_per_year</th>
      <th>avg_rate_per_year</th>
      <th>annual_occupancy(%)</th>
      <th>revenue_per_night_yearly</th>
      <th>reserved_days_in_year</th>
      <th>blocked_days_in_year</th>
      <th>available_days_in_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75683</td>
      <td>Kiloranhouse Apt Prime Bedroom</td>
      <td>Private room in home</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/5499026/ef...</td>
      <td>13</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.8</td>
      <td>4.7</td>
      <td>141049.0</td>
      <td>6474.0</td>
      <td>6.6</td>
      <td>386.4</td>
      <td>24</td>
      <td>0</td>
      <td>341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>471581</td>
      <td>Located In a Serene Environment</td>
      <td>Entire cottage</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/6434524/bc...</td>
      <td>37</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.8</td>
      <td>4.8</td>
      <td>804490.0</td>
      <td>5791.6</td>
      <td>54.8</td>
      <td>3058.9</td>
      <td>144</td>
      <td>102</td>
      <td>221</td>
    </tr>
    <tr>
      <th>2</th>
      <td>906958</td>
      <td>Makena's Place Karen - Flamingo Room</td>
      <td>Private room in cottage</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/68ecc57f-d...</td>
      <td>29</td>
      <td>1</td>
      <td>Firm</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>594869.0</td>
      <td>6772.2</td>
      <td>24.4</td>
      <td>1629.8</td>
      <td>89</td>
      <td>0</td>
      <td>276</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1023556</td>
      <td>Guesthouse Near Nairobi National Park &amp; Airport</td>
      <td>Entire guesthouse</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/ddd8badc-1...</td>
      <td>20</td>
      <td>1</td>
      <td>Flexible</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.7</td>
      <td>4.8</td>
      <td>29004.0</td>
      <td>3631.3</td>
      <td>3.0</td>
      <td>79.5</td>
      <td>11</td>
      <td>0</td>
      <td>354</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1237886</td>
      <td>Hob House</td>
      <td>Room in bed and breakfast</td>
      <td>hotel_room</td>
      <td>https://a0.muscache.com/im/pictures/cbdab7e1-f...</td>
      <td>8</td>
      <td>1</td>
      <td>Flexible</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>4.6</td>
      <td>4.7</td>
      <td>4.8</td>
      <td>168583.0</td>
      <td>15401.5</td>
      <td>3.0</td>
      <td>461.9</td>
      <td>11</td>
      <td>0</td>
      <td>354</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>42123446</td>
      <td>Mvuli Luxury Suites</td>
      <td>Entire rental unit</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/238557fd-c...</td>
      <td>24</td>
      <td>1</td>
      <td>Firm</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.5</td>
      <td>4.3</td>
      <td>4.4</td>
      <td>59105.0</td>
      <td>3710.9</td>
      <td>4.4</td>
      <td>161.9</td>
      <td>16</td>
      <td>0</td>
      <td>349</td>
    </tr>
    <tr>
      <th>296</th>
      <td>42139551</td>
      <td>Modern 1BR | King Bed | Fast Wi-Fi | Near CBD</td>
      <td>Entire condo</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/a10f889a-4...</td>
      <td>27</td>
      <td>1</td>
      <td>Flexible</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>4.8</td>
      <td>228931.0</td>
      <td>6572.8</td>
      <td>9.0</td>
      <td>663.6</td>
      <td>31</td>
      <td>20</td>
      <td>334</td>
    </tr>
    <tr>
      <th>297</th>
      <td>42187559</td>
      <td>Cosy &amp; Airy Studio with Balcony WI-FI and Netflix</td>
      <td>Tiny home</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/miso/Hosti...</td>
      <td>43</td>
      <td>1</td>
      <td>Moderate</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.7</td>
      <td>5.0</td>
      <td>25317.0</td>
      <td>2041.4</td>
      <td>3.3</td>
      <td>69.4</td>
      <td>12</td>
      <td>0</td>
      <td>353</td>
    </tr>
    <tr>
      <th>298</th>
      <td>42207619</td>
      <td>Kazuri Ivy Serene &amp; Spacious Nairobi Apartment</td>
      <td>Entire rental unit</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/hosting/Ho...</td>
      <td>19</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.9</td>
      <td>4.4</td>
      <td>4.8</td>
      <td>422289.0</td>
      <td>4969.1</td>
      <td>25.6</td>
      <td>1272.0</td>
      <td>85</td>
      <td>33</td>
      <td>280</td>
    </tr>
    <tr>
      <th>299</th>
      <td>42223689</td>
      <td>Airy &amp; Light-Filled: Indoor-Outdoor Home-Stay</td>
      <td>Private room in rental unit</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/fd146f8e-5...</td>
      <td>17</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>4.8</td>
      <td>4.9</td>
      <td>4.9</td>
      <td>190766.0</td>
      <td>2930.6</td>
      <td>22.4</td>
      <td>657.8</td>
      <td>65</td>
      <td>75</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 42 columns</p>
</div>



# Data Exploration an attempt at understanding my data


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 300 entries, 0 to 299
    Data columns (total 42 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   listing_id                300 non-null    int64  
     1   listing_name              300 non-null    object 
     2   listing_type              300 non-null    object 
     3   room_type                 300 non-null    object 
     4   cover_photo_url           300 non-null    object 
     5   photos_count              300 non-null    int64  
     6   minimum_nights            300 non-null    int64  
     7   cancellation_policy       300 non-null    object 
     8   professional_management   300 non-null    bool   
     9   registration              300 non-null    bool   
     10  instant_book              300 non-null    bool   
     11  amenities                 300 non-null    object 
     12  host_id                   300 non-null    int64  
     13  host_name                 300 non-null    object 
     14  cohost_ids                300 non-null    object 
     15  cohost_names              300 non-null    object 
     16  owned_by                  300 non-null    object 
     17  owners                    300 non-null    int64  
     18  superhost                 300 non-null    bool   
     19  latitude                  300 non-null    float64
     20  longitude                 300 non-null    float64
     21  guests_allowed            300 non-null    int64  
     22  bedrooms                  300 non-null    int64  
     23  beds                      300 non-null    int64  
     24  baths                     300 non-null    float64
     25  cleaning_fee              300 non-null    float64
     26  extra_guest_fee           300 non-null    float64
     27  num_reviews               300 non-null    int64  
     28  rating_overall            300 non-null    float64
     29  rating_accuracy           300 non-null    float64
     30  rating_checkin            300 non-null    float64
     31  rating_cleanliness        300 non-null    float64
     32  rating_communication      300 non-null    float64
     33  rating_location           300 non-null    float64
     34  rating_value              300 non-null    float64
     35  revenue_per_year          300 non-null    float64
     36  avg_rate_per_year         300 non-null    float64
     37  annual_occupancy(%)       300 non-null    float64
     38  revenue_per_night_yearly  300 non-null    float64
     39  reserved_days_in_year     300 non-null    int64  
     40  blocked_days_in_year      300 non-null    int64  
     41  available_days_in_year    300 non-null    int64  
    dtypes: bool(4), float64(16), int64(12), object(10)
    memory usage: 90.4+ KB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>photos_count</th>
      <th>minimum_nights</th>
      <th>host_id</th>
      <th>owners</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>guests_allowed</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>...</th>
      <th>rating_communication</th>
      <th>rating_location</th>
      <th>rating_value</th>
      <th>revenue_per_year</th>
      <th>avg_rate_per_year</th>
      <th>annual_occupancy(%)</th>
      <th>revenue_per_night_yearly</th>
      <th>reserved_days_in_year</th>
      <th>blocked_days_in_year</th>
      <th>available_days_in_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000e+02</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>3.000000e+02</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>...</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>3.000000e+02</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.818099e+07</td>
      <td>30.100000</td>
      <td>2.176667</td>
      <td>1.294209e+08</td>
      <td>1.580000</td>
      <td>-1.282688</td>
      <td>36.793314</td>
      <td>5.360000</td>
      <td>1.736667</td>
      <td>2.176667</td>
      <td>...</td>
      <td>4.869000</td>
      <td>4.825333</td>
      <td>4.775000</td>
      <td>5.557893e+05</td>
      <td>7504.654667</td>
      <td>23.385333</td>
      <td>1733.102333</td>
      <td>73.513333</td>
      <td>39.670000</td>
      <td>291.486667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.191095e+07</td>
      <td>16.297916</td>
      <td>3.445533</td>
      <td>1.062699e+08</td>
      <td>0.890455</td>
      <td>0.035017</td>
      <td>0.039478</td>
      <td>4.856854</td>
      <td>1.571269</td>
      <td>2.031225</td>
      <td>...</td>
      <td>0.157549</td>
      <td>0.174527</td>
      <td>0.194769</td>
      <td>8.759067e+05</td>
      <td>6837.522117</td>
      <td>20.727560</td>
      <td>2581.110728</td>
      <td>66.100738</td>
      <td>52.670057</td>
      <td>66.100738</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.568300e+04</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.699700e+04</td>
      <td>1.000000</td>
      <td>-1.379600</td>
      <td>36.669400</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>3.700000</td>
      <td>3.600000</td>
      <td>3.000000</td>
      <td>1.826400e+04</td>
      <td>1366.000000</td>
      <td>2.700000</td>
      <td>50.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000117e+07</td>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>3.594571e+07</td>
      <td>1.000000</td>
      <td>-1.297350</td>
      <td>36.779075</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.800000</td>
      <td>4.800000</td>
      <td>4.700000</td>
      <td>1.338565e+05</td>
      <td>4382.800000</td>
      <td>6.675000</td>
      <td>399.450000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>261.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.198200e+07</td>
      <td>28.000000</td>
      <td>1.000000</td>
      <td>1.008633e+08</td>
      <td>1.000000</td>
      <td>-1.284950</td>
      <td>36.791200</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>4.900000</td>
      <td>4.900000</td>
      <td>4.800000</td>
      <td>3.001395e+05</td>
      <td>6198.150000</td>
      <td>15.700000</td>
      <td>950.500000</td>
      <td>50.000000</td>
      <td>20.000000</td>
      <td>315.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.868569e+07</td>
      <td>37.000000</td>
      <td>2.000000</td>
      <td>2.237596e+08</td>
      <td>2.000000</td>
      <td>-1.264500</td>
      <td>36.806925</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>4.900000</td>
      <td>4.900000</td>
      <td>6.591980e+05</td>
      <td>8770.775000</td>
      <td>35.100000</td>
      <td>2171.350000</td>
      <td>104.000000</td>
      <td>61.250000</td>
      <td>343.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.222369e+07</td>
      <td>122.000000</td>
      <td>30.000000</td>
      <td>4.292661e+08</td>
      <td>6.000000</td>
      <td>-1.187200</td>
      <td>36.909600</td>
      <td>16.000000</td>
      <td>15.000000</td>
      <td>19.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1.070000e+07</td>
      <td>88421.700000</td>
      <td>90.700000</td>
      <td>32344.600000</td>
      <td>331.000000</td>
      <td>258.000000</td>
      <td>355.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>



# Data Cleaning


```python
import re

def to_snake_case(name):
    # Convert to lowercase
    name = name.lower()
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove special characters like parentheses
    name = re.sub(r'[(%\)]+', '', name)
    # Replace multiple underscores with single underscore
    name = re.sub(r'_+', '_', name)
    return name

# Apply to all columns
df.columns = [to_snake_case(col) for col in df.columns]

print(df.columns)
```

    Index(['listing_id', 'listing_name', 'listing_type', 'room_type',
           'cover_photo_url', 'photos_count', 'minimum_nights',
           'cancellation_policy', 'professional_management', 'registration',
           'instant_book', 'amenities', 'host_id', 'host_name', 'cohost_ids',
           'cohost_names', 'owned_by', 'owners', 'superhost', 'latitude',
           'longitude', 'guests_allowed', 'bedrooms', 'beds', 'baths',
           'cleaning_fee_', 'extra_guest_fee', 'num_reviews', 'rating_overall',
           'rating_accuracy', 'rating_checkin', 'rating_cleanliness',
           'rating_communication', 'rating_location', 'rating_value',
           'revenue_per_year', 'avg_rate_per_year', 'annual_occupancy',
           'revenue_per_night_yearly', 'reserved_days_in_year',
           'blocked_days_in_year', 'available_days_in_year'],
          dtype='object')



```python
msno.matrix(df)
```




    <Axes: >




    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_11_1.png)
    


# Pricing & Revenue Analysis

## What's the correlation between listing type and average nightly rate?
For this analysis, room type is used as the primary classification variable, as it provides a more general and consistent categorization of listings. In contrast, listing type contains a large number of highly specific categories, which can introduce unnecessary complexity and reduce comparability across observations.


```python
listing_type = df['listing_type'].unique()
len(listing_type)
```




    29




```python
room_type = df['room_type'].unique()
len(room_type)
room_type
```




    array(['private_room', 'entire_home', 'hotel_room'], dtype=object)



While **listing types** consist of **29 unique categories** across the dataset, **room types** are limited to **three broad and representative categories** that better capture the nature of the accommodation. These include **Private Room**, **Entire Home/Apartment**, and **Hotel Room**, the latter of which appears only once in the dataset.

#### Distribution of Room Type across the entire dataset


```python
room_type_count = df['room_type'].value_counts()
plt.figure(figsize=(10,8))
plt.pie(room_type_count, labels=room_type_count.index, colors=("#0FB9B9", "#E29015", "#F3013E"), autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
plt.title("Room Type Distribution Across the Dataset", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_18_0.png)
    



```python
listing_type_vs_avg_rate_per_night = df.groupby('room_type')['avg_rate_per_year'].mean().sort_values(ascending=False).round(2)
plt.figure(figsize=(10, 6))
listing_type_vs_avg_rate_per_night.plot(kind='bar', color=["#1AF64D", "#10EFCD", "#F60F0FF3"])
plt.title('Listing Type correlation to Average Rate Per Night', fontsize=14, fontweight='bold')
plt.ylabel('Average rate per Night', fontsize=14)
plt.xlabel('Type of Listing', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# display the findings
listing_type_vs_avg_rate_per_night
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_19_0.png)
    





    room_type
    hotel_room      15401.50
    entire_home      7488.58
    private_room     7424.20
    Name: avg_rate_per_year, dtype: float64



Based on the analysis, **hotel rooms** command the highest average nightly rate at **KSh 15,401**. In contrast, **entire homes** and **private rooms** are priced at relatively similar levels, with average nightly rates of **KSh 7,488** and **KSh 7,424**, respectively.

## Which room types generate the most revenue despite having lower average rates?


```python
avg_rate_vs_revenue_per_room_type = df.groupby('room_type')['revenue_per_year'].mean().sort_values(ascending=False).round(2)
avg_rate_vs_revenue_per_room_type
```




    room_type
    entire_home     619762.21
    private_room    229331.23
    hotel_room      168583.00
    Name: revenue_per_year, dtype: float64




```python
avg_rate_vs_revenue_per_room_type = (
    df.groupby('room_type')
      .agg(
          avg_rate_per_night=('avg_rate_per_year', 'mean'),
          revenue_per_year=('revenue_per_year', 'mean')
      )
).sort_values(by='revenue_per_year', ascending=False).round(2)

avg_rate_vs_revenue_per_room_type
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_rate_per_night</th>
      <th>revenue_per_year</th>
    </tr>
    <tr>
      <th>room_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>entire_home</th>
      <td>7488.58</td>
      <td>619762.21</td>
    </tr>
    <tr>
      <th>private_room</th>
      <td>7424.20</td>
      <td>229331.23</td>
    </tr>
    <tr>
      <th>hotel_room</th>
      <td>15401.50</td>
      <td>168583.00</td>
    </tr>
  </tbody>
</table>
</div>



The listing type generating the most revenue is **Entire Home, Ksh 619762** despite being the second highest rate per night, followed by **Private Room** generating **Ksh 229331** per year and lastly **Hotel Rooms** generating **Ksh 168583** despite being the highest rate per night

# Host Strategy & Management

## Are professionally managed listings priced higher than individually managed ones?

A listing is considered **professionally managed** when it is:

*   Operated by a **property management company** or hospitality firm
    
*   Managed by hosts who handle **multiple properties** as a business rather than a single personal residence
    

This typically includes:

*   Standardized check-in/check-out procedures
    
*   Dedicated cleaning and maintenance teams
    
*   Consistent pricing and availability management
    
*   Formal guest communication and support systems
    

How this differs from individual hosts

*   **Individually managed listings** are usually run by:
    
    *   A single host
        
    *   Often the property owner
        
    *   With more personalized and less standardized operations
        
*   **Professionally managed listings** tend to:
    
    *   Have **stricter policies** (e.g., cancellation, minimum nights)
        
    *   Operate more like hotels or serviced apartments
        
    *   Prioritize occupancy optimization and operational efficiency


```python
# Map boolean values to labels
management_labels = df['professional_management'].map({
    True: 'Professionally Managed',
    False: 'Individually Managed'
})

professional_management_pie = management_labels.value_counts()

colors = ("#5da718", "#79059c")

plt.figure(figsize=(10, 8))
plt.pie(
    professional_management_pie,
    labels=professional_management_pie.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11}
)
plt.title("Professionally Managed vs Individually Managed Listings")
plt.tight_layout()
plt.show()

# Display counts
print(professional_management_pie)

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_28_0.png)
    


    professional_management
    Individually Managed      201
    Professionally Managed     99
    Name: count, dtype: int64


so 99 Listings making up 33% of all the listings are professionally managed while 201 Listings making uo 67% of all the listings are Individually managed


```python
management_labels = df['professional_management'].map({
    True: 'Professionally Managed',
    False: 'Individually Managed'
})

pricing_vs_avg_rate_per_year = df.groupby(management_labels)['avg_rate_per_year'].mean().round(2)
pricing_vs_avg_rate_per_year
```




    professional_management
    Individually Managed      7218.56
    Professionally Managed    8085.51
    Name: avg_rate_per_year, dtype: float64




```python
plt.figure(figsize=(10,6))
pricing_vs_avg_rate_per_year.plot(kind='bar', color=["#05258D","#067538"])
plt.title('Professional Management Correlation To Average Price', fontsize=12, fontweight='bold')
plt.xlabel('Professional Management', fontsize=12)
plt.xticks(rotation=0)
plt.ylabel('Average Rate Per Year', fontsize=12)
plt.tight_layout()
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_31_0.png)
    


based on our chart above, the professionally managed listings tend to be priced higher compared to individually run listings, With professionally managed listings charging an average a rate of **8086 per Night** and Individually Managed Listings charging an average rate of **7219 per Night**

## Do professionally managed listings have better reviews/ratings?


```python
professionally_managed_vs_reviews = df.groupby(management_labels)['rating_overall'].mean().sort_values(ascending=True).round(2)
professionally_managed_vs_reviews
```




    professional_management
    Professionally Managed    4.77
    Individually Managed      4.79
    Name: rating_overall, dtype: float64



**Individually Managed** Listings are slightly better rated at **4.79** compared to **Professionally Managed** listings at **4.77**

## Is there a relationship between professional management and cancellation strictness?

To assess whether host management strategy influences booking flexibility, we examined the distribution of cancellation policies across professionally and individually managed listings.


```python
professional_management_vs_cancellation_strictness = df.groupby('cancellation_policy')['professional_management'].value_counts
professional_management_vs_cancellation_strictness
```




    <bound method SeriesGroupBy.value_counts of <pandas.core.groupby.generic.SeriesGroupBy object at 0x79cb866339e0>>




```python
pm_vs_policy = (
    pd.crosstab(
        df['cancellation_policy'],
        df['professional_management'].map({
            True: 'Professionally Managed',
            False: 'Individually Managed'
        })
    )
)

```


```python
pm_vs_policy.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6)
)

plt.title('Professional Management vs Cancellation Policy')
plt.xlabel('Cancellation Policy')
plt.ylabel('Number of Listings')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_40_0.png)
    


### Key Insights: Professional Management vs Cancellation Policy

1.  **Flexible policies are dominated by individually managed listings**The majority of listings with _Flexible_ cancellation policies are individually managed, suggesting that solo hosts prioritize flexibility to attract bookings and remain competitive.
    
2.  **Professionally managed listings are more concentrated in stricter policies**Under _Strict_ and _Moderate_ cancellation policies, professionally managed listings make up a noticeably larger share. This indicates that professional operators are more comfortable enforcing stricter terms, likely due to better demand forecasting, operational buffers, and portfolio diversification.
    
3.  **Moderate policies represent a middle ground for both host types**Both individual and professional hosts are strongly represented under _Moderate_ policies, reinforcing the idea that this policy balances guest appeal with revenue protection.
    
4.  **Firm and Limited policies are relatively rare**Very few listings adopt _Firm_ or _Limited_ cancellation policies, suggesting low market preference—possibly due to reduced guest willingness to book under highly restrictive conditions.
    

### Strategic Interpretation

Overall, **professional management is associated with stricter cancellation strategies**, while **individual hosts rely more on flexibility to drive demand**. This highlights differing risk tolerance and pricing strategies between professional operators and independent hosts in Nairobi’s Airbnb market.

If you want, I can now help you **connect this insight directly to pricing or occupancy outcomes** for a stronger narrative in your report.

## What price range attracts the most bookings (occupancy)?



```python
# Create price bins
df['price_range'] = pd.cut(df['avg_rate_per_year'], 
                            bins=[0, 5000, 10000, 15000, 20000],
                            labels=['Budget (0-5k)', 'Mid-range (5-10k)', 'Premium (10-15k)', 'Luxury (15k+)'])

# Group by price range
best_price_metric = df.groupby('price_range')['annual_occupancy'].agg(['mean', 'count']).sort_values('mean', ascending=False)

plt.figure(figsize=(10, 6))
best_price_metric['mean'].plot(kind='bar', color='steelblue')
plt.title('Average Occupancy by Price Range', fontsize=16, fontweight='bold')
plt.xlabel('Price Range', fontsize=14)
plt.ylabel('Average Occupancy (%)', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(best_price_metric)
```

    /tmp/ipykernel_54075/1302887327.py:7: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      best_price_metric = df.groupby('price_range')['annual_occupancy'].agg(['mean', 'count']).sort_values('mean', ascending=False)



    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_43_1.png)
    


                            mean  count
    price_range                        
    Mid-range (5-10k)  24.607586    145
    Luxury (15k+)      23.842857     14
    Budget (0-5k)      22.109615    104
    Premium (10-15k)   22.013333     30


### Insights: Average Occupancy by Price Range


1.  **Mid-range listings (KSh 5k–10k) attract the most customers**With the highest average occupancy, mid-range listings appear to offer the best balance between price and perceived value. This price segment is likely the most attractive to the majority of guests in Nairobi.
    
2.  **Luxury listings (KSh 15k+) maintain strong demand**Despite higher prices, luxury listings show relatively high occupancy, suggesting a consistent market for premium accommodation—possibly driven by business travelers, expatriates, or high-end tourists.
    
3.  **Budget listings (Below KSh 5k) do not achieve the highest occupancy**While budget listings are cheaper, their lower occupancy compared to mid-range listings suggests that **price alone is not the primary driver of demand**. Factors such as location, amenities, and perceived quality likely play a significant role.
    
4.  **Premium listings (KSh 10k–15k) show moderate occupancy**Premium listings fall between mid-range and budget listings in terms of occupancy, indicating that demand tapers slightly as prices rise beyond the mid-range threshold.
    

### Key Takeaway


> **Mid-range Airbnb listings (KSh 5k–10k) attract the highest customer demand, highlighting a value-for-money sweet spot in Nairobi’s short-term rental market.**

## How does cancellation policy affect pricing strategy?

In this segment, we check the various types of cancellation policies present within our dataset


```python
plt.figure(figsize=(10, 8))
cancellation_policy_count = df['cancellation_policy'].value_counts()
colors = ['#FF6B6B', "#6F0DD0", '#45B7D1', "#EDC914", "#4EED14"]
plt.pie(cancellation_policy_count, labels=cancellation_policy_count.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 11})
plt.title("Cancellation Policy Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Display the counts
print(cancellation_policy_count)
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_47_0.png)
    


    cancellation_policy
    Flexible    131
    Moderate    110
    Firm         34
    Strict       24
    Limited       1
    Name: count, dtype: int64


### Cancellation Policy Distribution

The dataset is dominated by **Flexible** and **Moderate** cancellation policies, with **43.7%** and **36.7%** listings respectively. This suggests that most hosts prefer policies that allow guests greater freedom to cancel reservations with minimal penalties.

**Firm** and **Strict** policies are less common, appearing in **11.3%** and **8.0%** listings respectively, indicating a smaller segment of hosts who prioritize booking certainty over flexibility.

Only **one** listing follows a **Limited** cancellation policy, making it negligible in the overall distribution.

### Key Insight

Overall, the distribution reflects a **guest-friendly marketplace**, where flexible cancellation options are more prevalent than restrictive policies, likely aimed at attracting short-term and undecided travelers.


```python
valid_policies = ['Flexible', 'Moderate', 'Firm', 'Strict']
cancellation_policy_vs_pricing = df[df['cancellation_policy'].isin(valid_policies)].groupby('cancellation_policy')['avg_rate_per_year'].mean().sort_values(ascending=False).round(2)
cancellation_policy_vs_pricing
```




    cancellation_policy
    Moderate    8850.24
    Strict      7568.66
    Firm        6986.59
    Flexible    6418.48
    Name: avg_rate_per_year, dtype: float64



so i got rid of **Limited** since it's only one entry, i don't think it a good representation of the insights i'd like to derive from the dataset


```python
plt.figure(figsize=(10,6))
cancellation_policy_vs_pricing.plot(kind='bar')
plt.title('Cancellation Policy and it\'s effect on Pricing of the Listing',fontsize=14,fontweight='bold' )
plt.ylabel('Average Rate per Night', fontsize=12)
plt.xlabel('Cancellation Policy', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_51_0.png)
    


### Effect of Cancellation Policy on Listing Pricing

The chart shows a **clear relationship between cancellation strictness and average nightly price**.

*   **Moderate cancellation policies** have the **highest average nightly rates**, suggesting that hosts with mid-level flexibility are able to charge a premium—possibly balancing guest trust with revenue protection.
    
*   **Strict policies** follow closely, indicating that listings enforcing tighter cancellation rules still command relatively high prices, likely due to stronger demand, desirable locations, or higher-quality listings.
    
*   **Firm policies** sit in the middle range, reflecting a moderate pricing strategy.
    
*   **Flexible cancellation policies** are associated with the **lowest average nightly rates**, suggesting that hosts may lower prices to compensate for the higher risk of last-minute cancellations.
    

###  Interpretation

Overall, **less flexible cancellation policies tend to correlate with higher pricing**, implying that hosts who restrict cancellations may do so confidently when their listings have strong market appeal. Conversely, greater flexibility appears to be used as a competitive pricing lever to attract more bookings.

#  Occupancy & Booking Patterns

## What's the average occupancy rate across the dataset?
The average occupancy rate across all active Airbnb listings in Nairobi provides a baseline measure of market demand. This metric helps contextualize how different pricing strategies, room types, and cancellation policies perform relative to the overall market.


```python
average_occupancy_rate = df['annual_occupancy'].mean().round(2)
average_occupancy_rate
```




    np.float64(23.39)



Average occupancy rate across the dataset is **23.39%**

## Do flexible cancellation policies lead to higher occupancy?


```python
valid_policies = ['Flexible', 'Moderate', 'Firm', 'Strict']
cancellation_policy_vs_occupancy = df[df['cancellation_policy'].isin(valid_policies)].groupby('cancellation_policy')['annual_occupancy'].mean().sort_values(ascending=False).round(2)
cancellation_policy_vs_occupancy
```




    cancellation_policy
    Strict      26.35
    Moderate    23.61
    Firm        23.29
    Flexible    22.42
    Name: annual_occupancy, dtype: float64



From our Findings,Listings with  **Strict** cancellation policy holds the largest percentage of annual occupancy rate at **26.35%**, Listings with **Moderate** cancellation policy come in second with **23.61%**, Firm Cancellation policy listings hold a **23.29%** annual occupancy rate and listings with **Flexible** cancellation Policy hold the least annual occupancy percentage at **22.42%**. So to answer the question, **No** 
#### Flexible Cancellation policies do not attract higher annual occupancy

## How does minimum night requirement affect booking frequency?


```python
minimum_nights_vs_booking_frequency = df.groupby('minimum_nights')['annual_occupancy'].mean().sort_values(ascending=False)
minimum_nights_vs_booking_frequency
```




    minimum_nights
    30    81.600000
    28    43.850000
    7     30.800000
    3     29.932258
    2     26.201031
    1     19.886275
    4     13.850000
    14     4.200000
    5      3.600000
    Name: annual_occupancy, dtype: float64




```python
plt.figure(figsize=(10,6))
minimum_nights_vs_booking_frequency.plot(kind='barh')
plt.title('Minimum Nights vs Booking Frequency', fontsize=14, fontweight='bold')
plt.ylabel('Minimum Nights', fontsize=12)
plt.xlabel('Annual Occupancy Percentage', fontsize=12)
plt.tight_layout()
plt.show()
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_62_0.png)
    


**1\. Long-Term Stays Dominate Annual Occupancy**

*   **30-Day Stays are King:** The most significant finding is that bookings with a minimum stay of **30 nights** account for the highest annual occupancy percentage by a wide margin, reaching approximately **82%**.
    
*   **Strong Performance of 28-Day Stays:** The second-highest occupancy is for **28-night** minimum stays, at around **44%**.
    
*   **Conclusion:** This suggests a very strong market for monthly or near-monthly rentals, which could be driven by corporate housing, digital nomads, or temporary relocations. These long stays are the primary drivers of occupancy in this dataset.
    

**2\. Weekly and Long-Weekend Stays are Important Secondary Drivers**

*   **The "Sweet Spot" for Shorter Stays:** Minimum stays of **7 nights** and **3 nights** have very similar and significant occupancy percentages, both hovering around the **30-31%** mark.
    
*   **Conclusion:** There is substantial demand for weekly vacations and extended weekend trips. For hosts not targeting the monthly market, these two durations appear to be the most effective for maintaining occupancy.
    

**3\. Very Short Stays Contribute Less to Overall Occupancy**

*   **1-2 Night Stays:** While extremely common in the short-term rental market, minimum stays of **2 nights** (~26%) and **1 night** (~20%) contribute less to the total annual occupancy than the 3, 7, 28, and 30-night options.
    
*   **Implication:** This could indicate that while these bookings may be frequent, the higher turnover results in more unbooked "gap days," leading to a lower overall occupancy percentage over the course of a year compared to longer, more continuous stays.
    

**4\. Specific Durations are Less Effective**

*   **Low Occupancy for 4, 5, and 14 Nights:** Minimum stays of **4 nights** (~14%), **14 nights** (~4%), and **5 nights** (~3%) show the lowest annual occupancy percentages.
    
*   **Conclusion:** These specific booking windows seem to be less popular with guests or less successfully utilized by hosts compared to the standard 1-night, weekend (2-3 nights), weekly (7 nights), or monthly models.
    

**Strategic Implication for Hosts:** To maximize annual occupancy, the data suggests targeting the **30+ day market** is the most effective strategy. If that is not feasible, focusing on **7-night (weekly)** or **3-night (long weekend)** minimums would be better than setting minimums of 1, 2, 4, 5, or 14 nights.

**Key Insight** The data suggests that decreasing booking frequency (by raising minimum nights to 28+) actually increases your overall business performance (occupancy).

## Which listing types have the highest occupancy rates?


```python
listing_types_vs_occupancy_rate = df.groupby('room_type')['annual_occupancy'].mean().sort_values(ascending=False).round(2)
listing_types_vs_occupancy_rate
```




    room_type
    entire_home     25.05
    private_room    15.08
    hotel_room       3.00
    Name: annual_occupancy, dtype: float64



**Entire Homes** are the most common listing type to be booked at a rate of **25.05%**, followed by **Private Room** at a rate of **15.08%** and lastly**Hotel Rooms** making up **3.0%** of the total occupancy rate

## How many listings per host on average?


```python
no_of_hosts = df['host_id'].unique()
len(no_of_hosts)
```




    205



Our dataset has 205 unique hosts all who own either one or a bunch of listings within Nairobi


```python
no_of_listing_per_host = (
    df.groupby('host_id')
      .agg(
          host_name=('host_name', 'first'),
          number_of_listings=('listing_id', 'count')
      ).sort_values(by='number_of_listings',ascending=False)
)

no_of_listing_per_host
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_name</th>
      <th>number_of_listings</th>
    </tr>
    <tr>
      <th>host_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35945714</th>
      <td>Samra Apartments</td>
      <td>19</td>
    </tr>
    <tr>
      <th>308523342</th>
      <td>Damaris</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8042369</th>
      <td>Sherry</td>
      <td>7</td>
    </tr>
    <tr>
      <th>145631743</th>
      <td>Diana</td>
      <td>6</td>
    </tr>
    <tr>
      <th>43851715</th>
      <td>Duncan</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>322215915</th>
      <td>David</td>
      <td>1</td>
    </tr>
    <tr>
      <th>326653326</th>
      <td>Alain</td>
      <td>1</td>
    </tr>
    <tr>
      <th>327031637</th>
      <td>Lillian</td>
      <td>1</td>
    </tr>
    <tr>
      <th>363302880</th>
      <td>Nyangaga</td>
      <td>1</td>
    </tr>
    <tr>
      <th>429266147</th>
      <td>Anata</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 2 columns</p>
</div>




```python
top_host = no_of_listing_per_host.iloc[0]
print(f"Top host: {top_host['host_name']} with {top_host['number_of_listings']} listings")
```

    Top host: Samra Apartments with 19 listings



```python
print(f"Total hosts: {len(no_of_listing_per_host)}")
print(f"Average listings per host: {no_of_listing_per_host['number_of_listings'].mean():.2f}")
print(f"Max listings by one host: {no_of_listing_per_host['number_of_listings'].max()}")
print(f"Min listings by one host: {no_of_listing_per_host['number_of_listings'].min()}")
```

    Total hosts: 205
    Average listings per host: 1.46
    Max listings by one host: 19
    Min listings by one host: 1


The dataset contains **205 unique hosts**, with an **average of 1.46 listings per host**, indicating that the market is largely composed of small-scale hosts. Most hosts operate **a single listing**, as reflected by the minimum of one listing per host. However, there is evidence of **professional or portfolio-style hosting**, with the largest host, **Samra Apartments**, managing **19 listings**. This highlights a market structure dominated by individual hosts alongside a small number of high-volume operators.

# Geographic/Neighborhood Patterns


```python
plt.figure(figsize=(8, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['avg_rate_per_year'], alpha=0.6)
plt.colorbar(label='Average Nightly Rate')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Pricing Distribution Across Nairobi')
plt.show()
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_75_0.png)
    



```python
NAIROBI_BOUNDS = {
    "lat_min": -1.5,
    "lat_max": -1.1,
    "lon_min": 36.6,
    "lon_max": 37.1
}

```


```python
df_nairobi = df[
    (df['latitude'] >= NAIROBI_BOUNDS['lat_min']) &
    (df['latitude'] <= NAIROBI_BOUNDS['lat_max']) &
    (df['longitude'] >= NAIROBI_BOUNDS['lon_min']) &
    (df['longitude'] <= NAIROBI_BOUNDS['lon_max'])
].copy()

print(f"Listings before: {len(df)}")
print(f"Listings in Nairobi: {len(df_nairobi)}")

```

    Listings before: 300
    Listings in Nairobi: 300



```python
import folium
from IPython.display import display

m = folium.Map(
    location=[-1.286389, 36.817223],  # Nairobi CBD
    zoom_start=12,
    min_zoom=11,
    max_zoom=16,
    max_bounds=True,
    bounds=NAIROBI_BOUNDS,
    tiles="OpenStreetMap"
)

```


```python
#m.fit_bounds(NAIROBI_BOUNDS)
```


```python
for row in df_nairobi.itertuples():
    if row.avg_rate_per_year < 5000:
        color = '#2EC4B6'   # Budget
    elif row.avg_rate_per_year < 10000:
        color = '#1F77B4'   # Mid-range
    elif row.avg_rate_per_year < 15000:
        color = '#FF9F1C'   # Premium
    else:
        color = '#E63946'   # Luxury

    # Create popup with more details
    popup_text = f"""
    <b style='font-size: 14px'>{row.listing_name}</b><br>
    <b>Type:</b> {row.listing_type}<br>
    <b>Room:</b> {row.room_type}<br>
    <b>Rating:</b> {row.rating_overall}/5.0<br>
    <b>Price:</b> KSh {row.avg_rate_per_year:.0f}/night
    """
    
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(m)

display(m)
```


<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_9f59ba5306fc0ae6e73b99c769527a34 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

            &lt;style&gt;html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }
            &lt;/style&gt;

            &lt;style&gt;#map {
                position:absolute;
                top:0;
                bottom:0;
                right:0;
                left:0;
                }
            &lt;/style&gt;

            &lt;script&gt;
                L_NO_TOUCH = false;
                L_DISABLE_3D = false;
            &lt;/script&gt;


&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_9f59ba5306fc0ae6e73b99c769527a34&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_9f59ba5306fc0ae6e73b99c769527a34 = L.map(
                &quot;map_9f59ba5306fc0ae6e73b99c769527a34&quot;,
                {
                    center: [-1.286389, 36.817223],
                    crs: L.CRS.EPSG3857,
                    ...{
  &quot;maxBounds&quot;: [
[
-90,
-180,
],
[
90,
180,
],
],
  &quot;zoom&quot;: 12,
  &quot;zoomControl&quot;: true,
  &quot;preferCanvas&quot;: false,
  &quot;bounds&quot;: {
  &quot;latMin&quot;: -1.5,
  &quot;latMax&quot;: -1.1,
  &quot;lonMin&quot;: 36.6,
  &quot;lonMax&quot;: 37.1,
},
}

                }
            );





            var tile_layer_5630b5d654622c2c7c4202f906a7a0eb = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {
  &quot;minZoom&quot;: 11,
  &quot;maxZoom&quot;: 16,
  &quot;maxNativeZoom&quot;: 16,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 1,
}

            );


            tile_layer_5630b5d654622c2c7c4202f906a7a0eb.addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


            var circle_marker_1b78f7b52e41415cc00f912ed2dfcbe4 = L.circleMarker(
                [-1.2848, 36.7924],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5f7d197a1216da8ffa037c10cf20a0cf = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dffda6cde16c3d76e7a3ec5c949148e8 = $(`&lt;div id=&quot;html_dffda6cde16c3d76e7a3ec5c949148e8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Kiloranhouse Apt Prime Bedroom&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6474/night     &lt;/div&gt;`)[0];
                popup_5f7d197a1216da8ffa037c10cf20a0cf.setContent(html_dffda6cde16c3d76e7a3ec5c949148e8);



        circle_marker_1b78f7b52e41415cc00f912ed2dfcbe4.bindPopup(popup_5f7d197a1216da8ffa037c10cf20a0cf)
        ;




            var circle_marker_7d7ea87af2d0d524b3cfca0d4d972812 = L.circleMarker(
                [-1.2268, 36.8577],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1d0304bf8b62a152f0a89f013039d4b9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_fb2ef1b3069d27eedd3c70180551d89d = $(`&lt;div id=&quot;html_fb2ef1b3069d27eedd3c70180551d89d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Located In a Serene Environment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5792/night     &lt;/div&gt;`)[0];
                popup_1d0304bf8b62a152f0a89f013039d4b9.setContent(html_fb2ef1b3069d27eedd3c70180551d89d);



        circle_marker_7d7ea87af2d0d524b3cfca0d4d972812.bindPopup(popup_1d0304bf8b62a152f0a89f013039d4b9)
        ;




            var circle_marker_6a35003309605f8083ec463a1f7aa63f = L.circleMarker(
                [-1.324, 36.7053],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fb22251a7011e959d6f49ef1d12eca01 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b1443cf6b414d05ba53187f8b6c71a0c = $(`&lt;div id=&quot;html_b1443cf6b414d05ba53187f8b6c71a0c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Makena&#x27;s Place Karen - Flamingo Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6772/night     &lt;/div&gt;`)[0];
                popup_fb22251a7011e959d6f49ef1d12eca01.setContent(html_b1443cf6b414d05ba53187f8b6c71a0c);



        circle_marker_6a35003309605f8083ec463a1f7aa63f.bindPopup(popup_fb22251a7011e959d6f49ef1d12eca01)
        ;




            var circle_marker_04eec9fe56541b3cf988739402ab6a65 = L.circleMarker(
                [-1.3222, 36.7852],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_24da93b8fdcb450e95d637ffd542aa40 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_548c49278388d3bd9744f27a32455ddc = $(`&lt;div id=&quot;html_548c49278388d3bd9744f27a32455ddc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Guesthouse Near Nairobi National Park &amp; Airport&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3631/night     &lt;/div&gt;`)[0];
                popup_24da93b8fdcb450e95d637ffd542aa40.setContent(html_548c49278388d3bd9744f27a32455ddc);



        circle_marker_04eec9fe56541b3cf988739402ab6a65.bindPopup(popup_24da93b8fdcb450e95d637ffd542aa40)
        ;




            var circle_marker_101e578639d0b98249e2a8440c8cb44a = L.circleMarker(
                [-1.2258, 36.7679],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1d26838ca5d518cb04587085793e6a8f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_548141f1f7936e2455947a00b904b914 = $(`&lt;div id=&quot;html_548141f1f7936e2455947a00b904b914&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Hob House&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; hotel_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 15402/night     &lt;/div&gt;`)[0];
                popup_1d26838ca5d518cb04587085793e6a8f.setContent(html_548141f1f7936e2455947a00b904b914);



        circle_marker_101e578639d0b98249e2a8440c8cb44a.bindPopup(popup_1d26838ca5d518cb04587085793e6a8f)
        ;




            var circle_marker_6256d2a85fb1a80f8963280d2748c1fb = L.circleMarker(
                [-1.3237, 36.7059],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2371a47c4e272f965adda95b43082976 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ef1d396ca5bbe047914e67f1fa265413 = $(`&lt;div id=&quot;html_ef1d396ca5bbe047914e67f1fa265413&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Makena&#x27;s Place Karen - All Rooms&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17025/night     &lt;/div&gt;`)[0];
                popup_2371a47c4e272f965adda95b43082976.setContent(html_ef1d396ca5bbe047914e67f1fa265413);



        circle_marker_6256d2a85fb1a80f8963280d2748c1fb.bindPopup(popup_2371a47c4e272f965adda95b43082976)
        ;




            var circle_marker_17fae8196a9f9753a7a2f3c900b72da2 = L.circleMarker(
                [-1.3233, 36.7077],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_319444889256577f71d03fdca88b981d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_32befb0284905df7788dc58b7d12fbb1 = $(`&lt;div id=&quot;html_32befb0284905df7788dc58b7d12fbb1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Makena&#x27;s Place Karen - K Girl &amp; Gecko Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4668/night     &lt;/div&gt;`)[0];
                popup_319444889256577f71d03fdca88b981d.setContent(html_32befb0284905df7788dc58b7d12fbb1);



        circle_marker_17fae8196a9f9753a7a2f3c900b72da2.bindPopup(popup_319444889256577f71d03fdca88b981d)
        ;




            var circle_marker_6bedf99ebd586545f9c5a73c4a040f80 = L.circleMarker(
                [-1.3734, 36.7403],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9d647d6cb2746fac1137818555d292a1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1cd8c0b240b6691408b1803187587951 = $(`&lt;div id=&quot;html_1cd8c0b240b6691408b1803187587951&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stunning guest cottage in tranquil setting&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.98/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 23465/night     &lt;/div&gt;`)[0];
                popup_9d647d6cb2746fac1137818555d292a1.setContent(html_1cd8c0b240b6691408b1803187587951);



        circle_marker_6bedf99ebd586545f9c5a73c4a040f80.bindPopup(popup_9d647d6cb2746fac1137818555d292a1)
        ;




            var circle_marker_7fd4826bb26be659a5a2975fb15f0e64 = L.circleMarker(
                [-1.2743, 36.7746],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6477bfbf8bb106530a72608311c771a3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0e09b6f01e1ee182ad4d459711991a41 = $(`&lt;div id=&quot;html_0e09b6f01e1ee182ad4d459711991a41&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy 2 Bedroom Guest House in Lavington Green&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8391/night     &lt;/div&gt;`)[0];
                popup_6477bfbf8bb106530a72608311c771a3.setContent(html_0e09b6f01e1ee182ad4d459711991a41);



        circle_marker_7fd4826bb26be659a5a2975fb15f0e64.bindPopup(popup_6477bfbf8bb106530a72608311c771a3)
        ;




            var circle_marker_b43735c74db384bf541a73777b2c0aeb = L.circleMarker(
                [-1.2298, 36.7976],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8aae770a038456385c8ace8c4678dd36 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_716bdaf12c0f8c160628e1c82cb545c0 = $(`&lt;div id=&quot;html_716bdaf12c0f8c160628e1c82cb545c0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy Garden Cottage / House&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4454/night     &lt;/div&gt;`)[0];
                popup_8aae770a038456385c8ace8c4678dd36.setContent(html_716bdaf12c0f8c160628e1c82cb545c0);



        circle_marker_b43735c74db384bf541a73777b2c0aeb.bindPopup(popup_8aae770a038456385c8ace8c4678dd36)
        ;




            var circle_marker_d672534f3a6be9ed2a73556e3ddf25d6 = L.circleMarker(
                [-1.2964, 36.7636],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f907af627b774eb80166540d0cf97b47 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_40e9ebfab4894a141899b2ffd6199d47 = $(`&lt;div id=&quot;html_40e9ebfab4894a141899b2ffd6199d47&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Charming &amp; Cozy 2BD penthouse, Riara Road, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6046/night     &lt;/div&gt;`)[0];
                popup_f907af627b774eb80166540d0cf97b47.setContent(html_40e9ebfab4894a141899b2ffd6199d47);



        circle_marker_d672534f3a6be9ed2a73556e3ddf25d6.bindPopup(popup_f907af627b774eb80166540d0cf97b47)
        ;




            var circle_marker_8e501aeab26fa2d36f1c53faf52fef74 = L.circleMarker(
                [-1.2977, 36.7982],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_619ca5be7c7bdc762d18994eb47fd39f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cfb571d0083ee4b5bc38195cab1c0cd6 = $(`&lt;div id=&quot;html_cfb571d0083ee4b5bc38195cab1c0cd6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Kilimani Century Gardens B602&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.62/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4776/night     &lt;/div&gt;`)[0];
                popup_619ca5be7c7bdc762d18994eb47fd39f.setContent(html_cfb571d0083ee4b5bc38195cab1c0cd6);



        circle_marker_8e501aeab26fa2d36f1c53faf52fef74.bindPopup(popup_619ca5be7c7bdc762d18994eb47fd39f)
        ;




            var circle_marker_56eb2e0d2304210daaeb533252d46de8 = L.circleMarker(
                [-1.2968, 36.796],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_06930ea0e16ac682bc54b01fb251fdf7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7cb35c5795c615ddf94afb20f81a8d74 = $(`&lt;div id=&quot;html_7cb35c5795c615ddf94afb20f81a8d74&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Kilimani Century Gardens B601&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4793/night     &lt;/div&gt;`)[0];
                popup_06930ea0e16ac682bc54b01fb251fdf7.setContent(html_7cb35c5795c615ddf94afb20f81a8d74);



        circle_marker_56eb2e0d2304210daaeb533252d46de8.bindPopup(popup_06930ea0e16ac682bc54b01fb251fdf7)
        ;




            var circle_marker_5ce2ecfc6aa14714f282d94e99c0c907 = L.circleMarker(
                [-1.239, 36.8447],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8ac8702f34e707761c91c3d6bd4be9bc = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6d8553dc2760c0a2159e19bbc51a6760 = $(`&lt;div id=&quot;html_6d8553dc2760c0a2159e19bbc51a6760&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene Melrose Gardens 3 br, 3 bath Guest-wing.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8058/night     &lt;/div&gt;`)[0];
                popup_8ac8702f34e707761c91c3d6bd4be9bc.setContent(html_6d8553dc2760c0a2159e19bbc51a6760);



        circle_marker_5ce2ecfc6aa14714f282d94e99c0c907.bindPopup(popup_8ac8702f34e707761c91c3d6bd4be9bc)
        ;




            var circle_marker_a757cb36c215f29a9fac5d98a8816740 = L.circleMarker(
                [-1.2948, 36.7836],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e38db1afaf7937caf78fc4553651552e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0f6a09d05129c0ebf6a973114ef476ef = $(`&lt;div id=&quot;html_0f6a09d05129c0ebf6a973114ef476ef&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;2 Bedrooms Executive furnished apartmt Yaya Brooks&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.47/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7948/night     &lt;/div&gt;`)[0];
                popup_e38db1afaf7937caf78fc4553651552e.setContent(html_0f6a09d05129c0ebf6a973114ef476ef);



        circle_marker_a757cb36c215f29a9fac5d98a8816740.bindPopup(popup_e38db1afaf7937caf78fc4553651552e)
        ;




            var circle_marker_0368ea3da3a1bf30cb7b9a11c29fa3f6 = L.circleMarker(
                [-1.2946, 36.791],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_15bf5da8849e443d09c623bd253f2c6f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_146843551bfd37234ce889de90a8e0ba = $(`&lt;div id=&quot;html_146843551bfd37234ce889de90a8e0ba&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;GeoMara Marcus Garden 511&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7853/night     &lt;/div&gt;`)[0];
                popup_15bf5da8849e443d09c623bd253f2c6f.setContent(html_146843551bfd37234ce889de90a8e0ba);



        circle_marker_0368ea3da3a1bf30cb7b9a11c29fa3f6.bindPopup(popup_15bf5da8849e443d09c623bd253f2c6f)
        ;




            var circle_marker_2aecd36c4b12dacb1db7151782703c5a = L.circleMarker(
                [-1.374, 36.732],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9f6429fa892282c8af247fb6ec9d096e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_850b7110bb163b21757073bf80374358 = $(`&lt;div id=&quot;html_850b7110bb163b21757073bf80374358&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bamboo Cottage, Langata&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12436/night     &lt;/div&gt;`)[0];
                popup_9f6429fa892282c8af247fb6ec9d096e.setContent(html_850b7110bb163b21757073bf80374358);



        circle_marker_2aecd36c4b12dacb1db7151782703c5a.bindPopup(popup_9f6429fa892282c8af247fb6ec9d096e)
        ;




            var circle_marker_1e6e5306579c2205d940c524c273a2f3 = L.circleMarker(
                [-1.2289, 36.8617],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_49fb8aa5b75f99efef2351f93387ad41 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7d6dc6fbfd7bd8fe1a637628ea2e1bf4 = $(`&lt;div id=&quot;html_7d6dc6fbfd7bd8fe1a637628ea2e1bf4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Home In a Serene Environment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3960/night     &lt;/div&gt;`)[0];
                popup_49fb8aa5b75f99efef2351f93387ad41.setContent(html_7d6dc6fbfd7bd8fe1a637628ea2e1bf4);



        circle_marker_1e6e5306579c2205d940c524c273a2f3.bindPopup(popup_49fb8aa5b75f99efef2351f93387ad41)
        ;




            var circle_marker_654154c05771b06313131a59e84c1693 = L.circleMarker(
                [-1.2598, 36.7987],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b42ab50b806807d12aad71bdbd62494d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1bb4d315c76a05a3899f30edb6c71a18 = $(`&lt;div id=&quot;html_1bb4d315c76a05a3899f30edb6c71a18&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Entire 1 bdrm in Westlands proximity to Sarit Cent&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.34/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5673/night     &lt;/div&gt;`)[0];
                popup_b42ab50b806807d12aad71bdbd62494d.setContent(html_1bb4d315c76a05a3899f30edb6c71a18);



        circle_marker_654154c05771b06313131a59e84c1693.bindPopup(popup_b42ab50b806807d12aad71bdbd62494d)
        ;




            var circle_marker_06e106db8b03cce6f6981d1972139ed5 = L.circleMarker(
                [-1.2892, 36.7594],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1a5b6b7917ca68706f4888372ec36011 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bd4f862ea2f45507e1a99a60ed72e42b = $(`&lt;div id=&quot;html_bd4f862ea2f45507e1a99a60ed72e42b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;LAVINGTON LOVELY APARTMENT&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4337/night     &lt;/div&gt;`)[0];
                popup_1a5b6b7917ca68706f4888372ec36011.setContent(html_bd4f862ea2f45507e1a99a60ed72e42b);



        circle_marker_06e106db8b03cce6f6981d1972139ed5.bindPopup(popup_1a5b6b7917ca68706f4888372ec36011)
        ;




            var circle_marker_891cbeacfffc1329a9fd4ed0e38d8eb5 = L.circleMarker(
                [-1.2809, 36.8271],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_334bc065f13322b8b64adf472d9bae68 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0ffc43dd7fb462b1aa2dbab0c2747e0a = $(`&lt;div id=&quot;html_0ffc43dd7fb462b1aa2dbab0c2747e0a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy Furnished Studio Apartment in Nairobi&#x27;s CBD&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.51/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2895/night     &lt;/div&gt;`)[0];
                popup_334bc065f13322b8b64adf472d9bae68.setContent(html_0ffc43dd7fb462b1aa2dbab0c2747e0a);



        circle_marker_891cbeacfffc1329a9fd4ed0e38d8eb5.bindPopup(popup_334bc065f13322b8b64adf472d9bae68)
        ;




            var circle_marker_bb79e23ecac808fbae555129d399c96a = L.circleMarker(
                [-1.2258, 36.7678],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_90434f70a542c1164e6d93f8c25083ad = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0b4f98f0ef24b1680f48d72fb5334d96 = $(`&lt;div id=&quot;html_0b4f98f0ef24b1680f48d72fb5334d96&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Hob House Habibi Suite&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 18832/night     &lt;/div&gt;`)[0];
                popup_90434f70a542c1164e6d93f8c25083ad.setContent(html_0b4f98f0ef24b1680f48d72fb5334d96);



        circle_marker_bb79e23ecac808fbae555129d399c96a.bindPopup(popup_90434f70a542c1164e6d93f8c25083ad)
        ;




            var circle_marker_2bf90e0fa7202da0d244cef99924f918 = L.circleMarker(
                [-1.2894, 36.7887],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_100a384429259acef995014a69ef96fd = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_caf52bb194f843d47836e76d408bb0b4 = $(`&lt;div id=&quot;html_caf52bb194f843d47836e76d408bb0b4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;3 Bed Lux Spacious Apartment to Let - Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.57/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9890/night     &lt;/div&gt;`)[0];
                popup_100a384429259acef995014a69ef96fd.setContent(html_caf52bb194f843d47836e76d408bb0b4);



        circle_marker_2bf90e0fa7202da0d244cef99924f918.bindPopup(popup_100a384429259acef995014a69ef96fd)
        ;




            var circle_marker_a0cfc2d2c41315b52a96e3deb916a266 = L.circleMarker(
                [-1.2808, 36.8239],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e03c4831227e295cd7a9b133a68da230 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ce48d31680b9fa144ccdd7dcfe0b14e1 = $(`&lt;div id=&quot;html_ce48d31680b9fa144ccdd7dcfe0b14e1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Elegant studio apartment in the City Centre&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.77/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2675/night     &lt;/div&gt;`)[0];
                popup_e03c4831227e295cd7a9b133a68da230.setContent(html_ce48d31680b9fa144ccdd7dcfe0b14e1);



        circle_marker_a0cfc2d2c41315b52a96e3deb916a266.bindPopup(popup_e03c4831227e295cd7a9b133a68da230)
        ;




            var circle_marker_50f3c92ae8179901b5f7cab2287117b6 = L.circleMarker(
                [-1.3137, 36.8354],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c2a548ad8906d8198f27cdf2af24470f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_99475fd201e5d85dff67b1733a15fb57 = $(`&lt;div id=&quot;html_99475fd201e5d85dff67b1733a15fb57&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Boushel Place - For best times&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2546/night     &lt;/div&gt;`)[0];
                popup_c2a548ad8906d8198f27cdf2af24470f.setContent(html_99475fd201e5d85dff67b1733a15fb57);



        circle_marker_50f3c92ae8179901b5f7cab2287117b6.bindPopup(popup_c2a548ad8906d8198f27cdf2af24470f)
        ;




            var circle_marker_0fd55d379e4a63bf8dd6f011b6dfab97 = L.circleMarker(
                [-1.2942, 36.7965],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_968e7222ec338d06878cffe7d3fc8610 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_46fbf8c7a1e5642b1ef73ddf677de795 = $(`&lt;div id=&quot;html_46fbf8c7a1e5642b1ef73ddf677de795&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;C5 Samra 1  bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4670/night     &lt;/div&gt;`)[0];
                popup_968e7222ec338d06878cffe7d3fc8610.setContent(html_46fbf8c7a1e5642b1ef73ddf677de795);



        circle_marker_0fd55d379e4a63bf8dd6f011b6dfab97.bindPopup(popup_968e7222ec338d06878cffe7d3fc8610)
        ;




            var circle_marker_857f3acb76bdee11512081d83a86c43b = L.circleMarker(
                [-1.2975, 36.7937],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9b489b3e5e27193c073bf7ab3a897e0d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8cc4e381b3a966561d71aa965d312b70 = $(`&lt;div id=&quot;html_8cc4e381b3a966561d71aa965d312b70&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Jue&#x27;s Cosy Family House with Garden in Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6343/night     &lt;/div&gt;`)[0];
                popup_9b489b3e5e27193c073bf7ab3a897e0d.setContent(html_8cc4e381b3a966561d71aa965d312b70);



        circle_marker_857f3acb76bdee11512081d83a86c43b.bindPopup(popup_9b489b3e5e27193c073bf7ab3a897e0d)
        ;




            var circle_marker_c6eed8b6a68d8f027272e514370a519b = L.circleMarker(
                [-1.2961, 36.7748],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b14ffd8342f0941cc09c734cc390f452 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_4c204ea9a78f56a2fb0f4f233ce0b737 = $(`&lt;div id=&quot;html_4c204ea9a78f56a2fb0f4f233ce0b737&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;A Breath of Fresh Air: Indoor-Outdoor Home-Stay&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.82/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2207/night     &lt;/div&gt;`)[0];
                popup_b14ffd8342f0941cc09c734cc390f452.setContent(html_4c204ea9a78f56a2fb0f4f233ce0b737);



        circle_marker_c6eed8b6a68d8f027272e514370a519b.bindPopup(popup_b14ffd8342f0941cc09c734cc390f452)
        ;




            var circle_marker_daf7f1b19f4fba8336260770b8e0c33c = L.circleMarker(
                [-1.3149, 36.8062],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_364dbaf8658c67417ec3e2c50e0c6a7e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ee278f8afc2cf2444ab3540d3d176529 = $(`&lt;div id=&quot;html_ee278f8afc2cf2444ab3540d3d176529&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;WiFied flatshare nr T-Mall btw town &amp; JKIA/SGR&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.53/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1428/night     &lt;/div&gt;`)[0];
                popup_364dbaf8658c67417ec3e2c50e0c6a7e.setContent(html_ee278f8afc2cf2444ab3540d3d176529);



        circle_marker_daf7f1b19f4fba8336260770b8e0c33c.bindPopup(popup_364dbaf8658c67417ec3e2c50e0c6a7e)
        ;




            var circle_marker_9b6327fd51ca1cafdfedead235c8f4d0 = L.circleMarker(
                [-1.2796, 36.8249],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a553cd11be0a5ae27424dec9030dd9d0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9b6e58ad2900e659e144fdc7864bb404 = $(`&lt;div id=&quot;html_9b6e58ad2900e659e144fdc7864bb404&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Affordable furnished apartment in the City&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2202/night     &lt;/div&gt;`)[0];
                popup_a553cd11be0a5ae27424dec9030dd9d0.setContent(html_9b6e58ad2900e659e144fdc7864bb404);



        circle_marker_9b6327fd51ca1cafdfedead235c8f4d0.bindPopup(popup_a553cd11be0a5ae27424dec9030dd9d0)
        ;




            var circle_marker_615d7ffd13b20cc073ff2d0a173e82d1 = L.circleMarker(
                [-1.2288, 36.8165],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e48e46fc4767bf03e21d8f7d8bb8cbbe = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_903adc9b1e6ed370ce86fb0277308a07 = $(`&lt;div id=&quot;html_903adc9b1e6ed370ce86fb0277308a07&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cozy safe room near UNEP HQ, Karura Forest &amp; mall.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5214/night     &lt;/div&gt;`)[0];
                popup_e48e46fc4767bf03e21d8f7d8bb8cbbe.setContent(html_903adc9b1e6ed370ce86fb0277308a07);



        circle_marker_615d7ffd13b20cc073ff2d0a173e82d1.bindPopup(popup_e48e46fc4767bf03e21d8f7d8bb8cbbe)
        ;




            var circle_marker_f7796b2badb336b3e414b8fe863031b8 = L.circleMarker(
                [-1.229, 36.8612],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1d7f8c1e6cafb29be9ee25db447693dc = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_be64f01d1c21f874ab843242c88ce83f = $(`&lt;div id=&quot;html_be64f01d1c21f874ab843242c88ce83f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Loft in a Serene Location&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Tiny home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.94/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3869/night     &lt;/div&gt;`)[0];
                popup_1d7f8c1e6cafb29be9ee25db447693dc.setContent(html_be64f01d1c21f874ab843242c88ce83f);



        circle_marker_f7796b2badb336b3e414b8fe863031b8.bindPopup(popup_1d7f8c1e6cafb29be9ee25db447693dc)
        ;




            var circle_marker_1cf97e4bc5e4b5b88eb559891bc049d7 = L.circleMarker(
                [-1.3535, 36.7324],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_77580d10cb64e438b8647687911278fa = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9f8f01edc5dd5c1c765b66cd0eb20e22 = $(`&lt;div id=&quot;html_9f8f01edc5dd5c1c765b66cd0eb20e22&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Starehe Base&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4393/night     &lt;/div&gt;`)[0];
                popup_77580d10cb64e438b8647687911278fa.setContent(html_9f8f01edc5dd5c1c765b66cd0eb20e22);



        circle_marker_1cf97e4bc5e4b5b88eb559891bc049d7.bindPopup(popup_77580d10cb64e438b8647687911278fa)
        ;




            var circle_marker_12f091f4a8afcec96d77d84cace1f8b5 = L.circleMarker(
                [-1.2499, 36.7584],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_84579c7c6dbbd4dba8413bac4f02106b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b64d3f53d0ea955a3756155d28a93be9 = $(`&lt;div id=&quot;html_b64d3f53d0ea955a3756155d28a93be9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene, Safe &amp; Peaceful Guesthouse&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4566/night     &lt;/div&gt;`)[0];
                popup_84579c7c6dbbd4dba8413bac4f02106b.setContent(html_b64d3f53d0ea955a3756155d28a93be9);



        circle_marker_12f091f4a8afcec96d77d84cace1f8b5.bindPopup(popup_84579c7c6dbbd4dba8413bac4f02106b)
        ;




            var circle_marker_088a1475a278dc0c7582fe00e2ed9457 = L.circleMarker(
                [-1.2868, 36.7924],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_604b879422efc631b0b5371de4dbd01c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8605ab30c7e119dc2e953b2b6e6367da = $(`&lt;div id=&quot;html_8605ab30c7e119dc2e953b2b6e6367da&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Kilimani Nairobi Woodmere Studio Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.47/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4458/night     &lt;/div&gt;`)[0];
                popup_604b879422efc631b0b5371de4dbd01c.setContent(html_8605ab30c7e119dc2e953b2b6e6367da);



        circle_marker_088a1475a278dc0c7582fe00e2ed9457.bindPopup(popup_604b879422efc631b0b5371de4dbd01c)
        ;




            var circle_marker_744f285c6e4104b8c1a282f429ea1040 = L.circleMarker(
                [-1.3148, 36.6825],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_89b23c4d45e2343a792e312aa9f96536 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c034f421f4ee6fb4a953287f1ffbfa63 = $(`&lt;div id=&quot;html_c034f421f4ee6fb4a953287f1ffbfa63&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;An Irish welcome in Karen - Hill Cottage&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.76/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7185/night     &lt;/div&gt;`)[0];
                popup_89b23c4d45e2343a792e312aa9f96536.setContent(html_c034f421f4ee6fb4a953287f1ffbfa63);



        circle_marker_744f285c6e4104b8c1a282f429ea1040.bindPopup(popup_89b23c4d45e2343a792e312aa9f96536)
        ;




            var circle_marker_c2f6129b46a0d5996e8a405ace77cc25 = L.circleMarker(
                [-1.2908, 36.7581],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9dbead61c419e3a49072386dc0eddc4d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_393318fc0817dc3c9ce72153edba93ec = $(`&lt;div id=&quot;html_393318fc0817dc3c9ce72153edba93ec&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Splendid 3 bed modern apart in Lavington&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5999/night     &lt;/div&gt;`)[0];
                popup_9dbead61c419e3a49072386dc0eddc4d.setContent(html_393318fc0817dc3c9ce72153edba93ec);



        circle_marker_c2f6129b46a0d5996e8a405ace77cc25.bindPopup(popup_9dbead61c419e3a49072386dc0eddc4d)
        ;




            var circle_marker_1a6a8a987ad34538e59bd6f8c7f1e90c = L.circleMarker(
                [-1.2994, 36.7752],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e7f4f621422875c1db171a49b3f0ac67 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f2a60f8260ba63baa2d486a3a3787c1b = $(`&lt;div id=&quot;html_f2a60f8260ba63baa2d486a3a3787c1b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Rafiki-private Room - naturally inviting&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in townhouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3821/night     &lt;/div&gt;`)[0];
                popup_e7f4f621422875c1db171a49b3f0ac67.setContent(html_f2a60f8260ba63baa2d486a3a3787c1b);



        circle_marker_1a6a8a987ad34538e59bd6f8c7f1e90c.bindPopup(popup_e7f4f621422875c1db171a49b3f0ac67)
        ;




            var circle_marker_2ecc3869de65dc0895ebad11c198c822 = L.circleMarker(
                [-1.2985, 36.7931],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fd897cd9f17e059b58e571e24022ff96 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_21e9ffd50d6d395d77d5cdee2ce8a0f9 = $(`&lt;div id=&quot;html_21e9ffd50d6d395d77d5cdee2ce8a0f9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;lovely Monroe 3bedroom furnished apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11621/night     &lt;/div&gt;`)[0];
                popup_fd897cd9f17e059b58e571e24022ff96.setContent(html_21e9ffd50d6d395d77d5cdee2ce8a0f9);



        circle_marker_2ecc3869de65dc0895ebad11c198c822.bindPopup(popup_fd897cd9f17e059b58e571e24022ff96)
        ;




            var circle_marker_45aeee48afa57572c3bdeb8a1515e952 = L.circleMarker(
                [-1.2783, 36.7849],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e49ae36149ddd1a66d8dcf6be483dd09 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e0c4d92db0cd809a54be6a3d96f2e586 = $(`&lt;div id=&quot;html_e0c4d92db0cd809a54be6a3d96f2e586&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ensuite room1 w/pool &amp;gym in Kileleshwa(Flatshare)&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1787/night     &lt;/div&gt;`)[0];
                popup_e49ae36149ddd1a66d8dcf6be483dd09.setContent(html_e0c4d92db0cd809a54be6a3d96f2e586);



        circle_marker_45aeee48afa57572c3bdeb8a1515e952.bindPopup(popup_e49ae36149ddd1a66d8dcf6be483dd09)
        ;




            var circle_marker_533cb21a0b9cdf86b5838a22b56d7847 = L.circleMarker(
                [-1.2954, 36.7591],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ae9791a9590a79e0cc6b5593365d03db = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c82726f5ebdf3283b8a01f8acb5bea78 = $(`&lt;div id=&quot;html_c82726f5ebdf3283b8a01f8acb5bea78&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Trendy Apartment, 90m to Junction Mall, Sleeps 8!&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6132/night     &lt;/div&gt;`)[0];
                popup_ae9791a9590a79e0cc6b5593365d03db.setContent(html_c82726f5ebdf3283b8a01f8acb5bea78);



        circle_marker_533cb21a0b9cdf86b5838a22b56d7847.bindPopup(popup_ae9791a9590a79e0cc6b5593365d03db)
        ;




            var circle_marker_a4561d091da1f06a2daa8ccc8511ff67 = L.circleMarker(
                [-1.2783, 36.785],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ccd3f7456f4c08c29284a6c4ed145b0e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_051fc4f69b7c47d4cdf8ee247f188415 = $(`&lt;div id=&quot;html_051fc4f69b7c47d4cdf8ee247f188415&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ensuite room2 w/pool &amp;gym in Kileleshwa(Flatshare)&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.73/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1715/night     &lt;/div&gt;`)[0];
                popup_ccd3f7456f4c08c29284a6c4ed145b0e.setContent(html_051fc4f69b7c47d4cdf8ee247f188415);



        circle_marker_a4561d091da1f06a2daa8ccc8511ff67.bindPopup(popup_ccd3f7456f4c08c29284a6c4ed145b0e)
        ;




            var circle_marker_2b5484da79f358c887ed659b2a128f8b = L.circleMarker(
                [-1.3028, 36.8117],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4163ac37b900bc110ac1c7a659e5fee6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5b696fa37b8d193b088c49dce15b5114 = $(`&lt;div id=&quot;html_5b696fa37b8d193b088c49dce15b5114&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Green House @ Upper Hill Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6518/night     &lt;/div&gt;`)[0];
                popup_4163ac37b900bc110ac1c7a659e5fee6.setContent(html_5b696fa37b8d193b088c49dce15b5114);



        circle_marker_2b5484da79f358c887ed659b2a128f8b.bindPopup(popup_4163ac37b900bc110ac1c7a659e5fee6)
        ;




            var circle_marker_d535190645df89b9c80da1e44cc112c5 = L.circleMarker(
                [-1.3272, 36.7332],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f136e85d0a3003e46fd3cc183f64af7f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_943b20d6b01c2621051acbf7ab37d3d4 = $(`&lt;div id=&quot;html_943b20d6b01c2621051acbf7ab37d3d4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Maison Mitwaba - Room 1&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8080/night     &lt;/div&gt;`)[0];
                popup_f136e85d0a3003e46fd3cc183f64af7f.setContent(html_943b20d6b01c2621051acbf7ab37d3d4);



        circle_marker_d535190645df89b9c80da1e44cc112c5.bindPopup(popup_f136e85d0a3003e46fd3cc183f64af7f)
        ;




            var circle_marker_d40ae0d7231230ec139a65ddfa096ca4 = L.circleMarker(
                [-1.2789, 36.7932],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7f66b65a9deec1ba4eb35a6c2edf3f59 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_783997269463c9bc98aeb202d61bb5bd = $(`&lt;div id=&quot;html_783997269463c9bc98aeb202d61bb5bd&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Symphony of Style 3BR-Kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.58/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10522/night     &lt;/div&gt;`)[0];
                popup_7f66b65a9deec1ba4eb35a6c2edf3f59.setContent(html_783997269463c9bc98aeb202d61bb5bd);



        circle_marker_d40ae0d7231230ec139a65ddfa096ca4.bindPopup(popup_7f66b65a9deec1ba4eb35a6c2edf3f59)
        ;




            var circle_marker_46cdfd3bb54c94ee5bca9cce602e1307 = L.circleMarker(
                [-1.2542, 36.7751],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_222b10ef9f710d9218d944fda7789b42 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_624c68ac61b609d95f7f33ee59645925 = $(`&lt;div id=&quot;html_624c68ac61b609d95f7f33ee59645925&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;SHERRY’s Qwetu Studio Cottage: SAFE GET-AWAY&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.76/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3788/night     &lt;/div&gt;`)[0];
                popup_222b10ef9f710d9218d944fda7789b42.setContent(html_624c68ac61b609d95f7f33ee59645925);



        circle_marker_46cdfd3bb54c94ee5bca9cce602e1307.bindPopup(popup_222b10ef9f710d9218d944fda7789b42)
        ;




            var circle_marker_1e90161401e2a590024ceea694b8b3b8 = L.circleMarker(
                [-1.359, 36.7186],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_eb88ca99477504d0be58d817ad5d1d60 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_69c10b58d024c437f9693b43753d5e82 = $(`&lt;div id=&quot;html_69c10b58d024c437f9693b43753d5e82&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;★The Brandy Bus, Glamping In a Quiet Paradise&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Bus&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17858/night     &lt;/div&gt;`)[0];
                popup_eb88ca99477504d0be58d817ad5d1d60.setContent(html_69c10b58d024c437f9693b43753d5e82);



        circle_marker_1e90161401e2a590024ceea694b8b3b8.bindPopup(popup_eb88ca99477504d0be58d817ad5d1d60)
        ;




            var circle_marker_d8f32fcd8f82b206a1bd42b8af4191a7 = L.circleMarker(
                [-1.2906, 36.7791],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0871c3825944e891440224bf7c742b8e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_2fd734695ee95107277edaeb50f455ad = $(`&lt;div id=&quot;html_2fd734695ee95107277edaeb50f455ad&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Brookview Apartments - 1 Bedroom - Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.64/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3692/night     &lt;/div&gt;`)[0];
                popup_0871c3825944e891440224bf7c742b8e.setContent(html_2fd734695ee95107277edaeb50f455ad);



        circle_marker_d8f32fcd8f82b206a1bd42b8af4191a7.bindPopup(popup_0871c3825944e891440224bf7c742b8e)
        ;




            var circle_marker_02845a5674b57cded8671141b6bbf418 = L.circleMarker(
                [-1.2315, 36.8048],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_11f74e17ff05b2ee6c5b8621e55413a7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3799f0bc26ee268fb9f6ea92329ad3a1 = $(`&lt;div id=&quot;html_3799f0bc26ee268fb9f6ea92329ad3a1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Orchid Homes - Deluxe King Room with Jacuzzi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 29061/night     &lt;/div&gt;`)[0];
                popup_11f74e17ff05b2ee6c5b8621e55413a7.setContent(html_3799f0bc26ee268fb9f6ea92329ad3a1);



        circle_marker_02845a5674b57cded8671141b6bbf418.bindPopup(popup_11f74e17ff05b2ee6c5b8621e55413a7)
        ;




            var circle_marker_399b1a9192ed30a873746f5fe65ba9e4 = L.circleMarker(
                [-1.2293, 36.8155],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_948f0405aeaf2754592e618c1f6af86d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f49af662fb28a286f41d9962f9bc8663 = $(`&lt;div id=&quot;html_f49af662fb28a286f41d9962f9bc8663&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Elegant suite, Facing UNEP/ Karura Forest.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7925/night     &lt;/div&gt;`)[0];
                popup_948f0405aeaf2754592e618c1f6af86d.setContent(html_f49af662fb28a286f41d9962f9bc8663);



        circle_marker_399b1a9192ed30a873746f5fe65ba9e4.bindPopup(popup_948f0405aeaf2754592e618c1f6af86d)
        ;




            var circle_marker_600b12c01598f9a01a53397e69ad2a1e = L.circleMarker(
                [-1.2315, 36.8064],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_531c80fe58fdc15f55762415f677f095 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a2ba2257bb7e7e84fd0316b915627c9f = $(`&lt;div id=&quot;html_a2ba2257bb7e7e84fd0316b915627c9f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bonsai Villa - Standard King Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9642/night     &lt;/div&gt;`)[0];
                popup_531c80fe58fdc15f55762415f677f095.setContent(html_a2ba2257bb7e7e84fd0316b915627c9f);



        circle_marker_600b12c01598f9a01a53397e69ad2a1e.bindPopup(popup_531c80fe58fdc15f55762415f677f095)
        ;




            var circle_marker_6e12b87f93da42d0d9eac84b56b33138 = L.circleMarker(
                [-1.2584, 36.7974],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_419be21632b08560c66c6696c9beb615 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_68e88cb86346c1c0aae9bc0585328461 = $(`&lt;div id=&quot;html_68e88cb86346c1c0aae9bc0585328461&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene serviced apartment-2BR GRACIOUS APARTMENTS&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9770/night     &lt;/div&gt;`)[0];
                popup_419be21632b08560c66c6696c9beb615.setContent(html_68e88cb86346c1c0aae9bc0585328461);



        circle_marker_6e12b87f93da42d0d9eac84b56b33138.bindPopup(popup_419be21632b08560c66c6696c9beb615)
        ;




            var circle_marker_1fccc7e52d31bebe4e0f2fed0bcd85d9 = L.circleMarker(
                [-1.2782, 36.785],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6cccaac8769ea43f0cb1fb0d19923cee = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1f360bc828772498dfcaade52f65d594 = $(`&lt;div id=&quot;html_1f360bc828772498dfcaade52f65d594&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;SQ in Kileleshwa flatshare w/ pool &amp; gym&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1392/night     &lt;/div&gt;`)[0];
                popup_6cccaac8769ea43f0cb1fb0d19923cee.setContent(html_1f360bc828772498dfcaade52f65d594);



        circle_marker_1fccc7e52d31bebe4e0f2fed0bcd85d9.bindPopup(popup_6cccaac8769ea43f0cb1fb0d19923cee)
        ;




            var circle_marker_28c9917d5244a4b5592278701dd18bad = L.circleMarker(
                [-1.2837, 36.784],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_744248cbd4b85e18c06b48f0c4f060ec = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_2e979a3938e0471801fe503d6776bb3c = $(`&lt;div id=&quot;html_2e979a3938e0471801fe503d6776bb3c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Urban elegance with balcony views&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4759/night     &lt;/div&gt;`)[0];
                popup_744248cbd4b85e18c06b48f0c4f060ec.setContent(html_2e979a3938e0471801fe503d6776bb3c);



        circle_marker_28c9917d5244a4b5592278701dd18bad.bindPopup(popup_744248cbd4b85e18c06b48f0c4f060ec)
        ;




            var circle_marker_e9f99a7eb765573a0f57fdbc3b13bbdf = L.circleMarker(
                [-1.2678, 36.8054],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_dd92bd93f1983d56ab10c7bfdb95b626 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_26f723754a20fc9cc04cf3a2ea19be19 = $(`&lt;div id=&quot;html_26f723754a20fc9cc04cf3a2ea19be19&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;SHERRY’s ZANA STUDIO  gym &amp; central location&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.43/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5265/night     &lt;/div&gt;`)[0];
                popup_dd92bd93f1983d56ab10c7bfdb95b626.setContent(html_26f723754a20fc9cc04cf3a2ea19be19);



        circle_marker_e9f99a7eb765573a0f57fdbc3b13bbdf.bindPopup(popup_dd92bd93f1983d56ab10c7bfdb95b626)
        ;




            var circle_marker_54551aa1e5f2cb4f27d591990045b54c = L.circleMarker(
                [-1.297, 36.7505],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d98d3048e52e45aae8bf4bb5102c17a6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bf0fe07db101ba961f1f7a859168bb00 = $(`&lt;div id=&quot;html_bf0fe07db101ba961f1f7a859168bb00&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Skywalk Katerina, Exquisite 1 BR Loft With Pool&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire loft&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3284/night     &lt;/div&gt;`)[0];
                popup_d98d3048e52e45aae8bf4bb5102c17a6.setContent(html_bf0fe07db101ba961f1f7a859168bb00);



        circle_marker_54551aa1e5f2cb4f27d591990045b54c.bindPopup(popup_d98d3048e52e45aae8bf4bb5102c17a6)
        ;




            var circle_marker_b11ad2b77f8770a3060130ddb1d842ed = L.circleMarker(
                [-1.2952, 36.7863],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_75ec8a11ebdbaa26105881ced7c1c287 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_025a97ddad7c4933d76af35ab575c424 = $(`&lt;div id=&quot;html_025a97ddad7c4933d76af35ab575c424&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cozy with amazing city skyline and infinity pool&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.27/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7553/night     &lt;/div&gt;`)[0];
                popup_75ec8a11ebdbaa26105881ced7c1c287.setContent(html_025a97ddad7c4933d76af35ab575c424);



        circle_marker_b11ad2b77f8770a3060130ddb1d842ed.bindPopup(popup_75ec8a11ebdbaa26105881ced7c1c287)
        ;




            var circle_marker_4b005bb90ed833de096a1adf19dd68e8 = L.circleMarker(
                [-1.215, 36.7887],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_86a3a29d9a168cef5f2a4c7c1496709b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8bd5c7c8f57add7ea896fe5c8c0c3c16 = $(`&lt;div id=&quot;html_8bd5c7c8f57add7ea896fe5c8c0c3c16&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Rosslyn NGH2  1 or 2 bedrooms&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.61/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9636/night     &lt;/div&gt;`)[0];
                popup_86a3a29d9a168cef5f2a4c7c1496709b.setContent(html_8bd5c7c8f57add7ea896fe5c8c0c3c16);



        circle_marker_4b005bb90ed833de096a1adf19dd68e8.bindPopup(popup_86a3a29d9a168cef5f2a4c7c1496709b)
        ;




            var circle_marker_37114755e7561fd41c511535c5a5c340 = L.circleMarker(
                [-1.3711, 36.7559],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1bb792737b733ba127c6e4e57a3c9e76 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_79e160b9de5deceb2ec495f3e186d82c = $(`&lt;div id=&quot;html_79e160b9de5deceb2ec495f3e186d82c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Acacia Rustic Cottage&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7370/night     &lt;/div&gt;`)[0];
                popup_1bb792737b733ba127c6e4e57a3c9e76.setContent(html_79e160b9de5deceb2ec495f3e186d82c);



        circle_marker_37114755e7561fd41c511535c5a5c340.bindPopup(popup_1bb792737b733ba127c6e4e57a3c9e76)
        ;




            var circle_marker_be6b700ba4d8eb656040b404c6779745 = L.circleMarker(
                [-1.2153, 36.7893],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6055102447b1a2f0018b4f61dad577b5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3ae06bad34e07b3fc7126f3d28ce07b6 = $(`&lt;div id=&quot;html_3ae06bad34e07b3fc7126f3d28ce07b6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Colobus Cottage on Rosslyn Lone Tree, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17958/night     &lt;/div&gt;`)[0];
                popup_6055102447b1a2f0018b4f61dad577b5.setContent(html_3ae06bad34e07b3fc7126f3d28ce07b6);



        circle_marker_be6b700ba4d8eb656040b404c6779745.bindPopup(popup_6055102447b1a2f0018b4f61dad577b5)
        ;




            var circle_marker_de1f11367d5dcf72b2b187860be19660 = L.circleMarker(
                [-1.2121, 36.7989],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7306871649327f70c66371041890c3fd = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_48e79b365c5b8796029666241abac11c = $(`&lt;div id=&quot;html_48e79b365c5b8796029666241abac11c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;lofts...great living and views...&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire loft&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4845/night     &lt;/div&gt;`)[0];
                popup_7306871649327f70c66371041890c3fd.setContent(html_48e79b365c5b8796029666241abac11c);



        circle_marker_de1f11367d5dcf72b2b187860be19660.bindPopup(popup_7306871649327f70c66371041890c3fd)
        ;




            var circle_marker_e42e73491cfa4faba9308f9ee8c31e31 = L.circleMarker(
                [-1.2297, 36.8767],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ee0d1b0304935df2db83f1b2c878b031 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8fb765b50d7f54d998404957d1caad50 = $(`&lt;div id=&quot;html_8fb765b50d7f54d998404957d1caad50&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Penthouse Studio-Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2627/night     &lt;/div&gt;`)[0];
                popup_ee0d1b0304935df2db83f1b2c878b031.setContent(html_8fb765b50d7f54d998404957d1caad50);



        circle_marker_e42e73491cfa4faba9308f9ee8c31e31.bindPopup(popup_ee0d1b0304935df2db83f1b2c878b031)
        ;




            var circle_marker_85f2552607fa619a95ef6a4287690fda = L.circleMarker(
                [-1.2314, 36.8784],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6f5194b347e5270ce073435475ad2ffb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f6994017a6a1af1d7a02a7a42767c262 = $(`&lt;div id=&quot;html_f6994017a6a1af1d7a02a7a42767c262&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stay in a Stylish 2BR Apartment at Garden City&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6401/night     &lt;/div&gt;`)[0];
                popup_6f5194b347e5270ce073435475ad2ffb.setContent(html_f6994017a6a1af1d7a02a7a42767c262);



        circle_marker_85f2552607fa619a95ef6a4287690fda.bindPopup(popup_6f5194b347e5270ce073435475ad2ffb)
        ;




            var circle_marker_eca8b55946987aaba6335af3dc0fa21f = L.circleMarker(
                [-1.2696, 36.7388],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_59c1da053d2d24aa33c3d452522de650 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d37d11cfb168129a7cb028604db67875 = $(`&lt;div id=&quot;html_d37d11cfb168129a7cb028604db67875&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Private Guest House In Lush Nairobi Suburb&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2704/night     &lt;/div&gt;`)[0];
                popup_59c1da053d2d24aa33c3d452522de650.setContent(html_d37d11cfb168129a7cb028604db67875);



        circle_marker_eca8b55946987aaba6335af3dc0fa21f.bindPopup(popup_59c1da053d2d24aa33c3d452522de650)
        ;




            var circle_marker_df2b2ff49816299fc0087a1b68b4350f = L.circleMarker(
                [-1.2871, 36.7984],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_82fd10b643d55fa65c3145d1b2a68abf = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5112b697fe00e51f8dcadfe45ae17c4a = $(`&lt;div id=&quot;html_5112b697fe00e51f8dcadfe45ae17c4a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Homely &amp; Elegant&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9338/night     &lt;/div&gt;`)[0];
                popup_82fd10b643d55fa65c3145d1b2a68abf.setContent(html_5112b697fe00e51f8dcadfe45ae17c4a);



        circle_marker_df2b2ff49816299fc0087a1b68b4350f.bindPopup(popup_82fd10b643d55fa65c3145d1b2a68abf)
        ;




            var circle_marker_fd0f257a3e34c37c58ffed420db86c30 = L.circleMarker(
                [-1.2867, 36.9019],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_41b63955963cd7b8a65d5b6c8486154c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_842ff1b78b9271e9232572748a8949aa = $(`&lt;div id=&quot;html_842ff1b78b9271e9232572748a8949aa&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Comfort Homes&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.57/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 37485/night     &lt;/div&gt;`)[0];
                popup_41b63955963cd7b8a65d5b6c8486154c.setContent(html_842ff1b78b9271e9232572748a8949aa);



        circle_marker_fd0f257a3e34c37c58ffed420db86c30.bindPopup(popup_41b63955963cd7b8a65d5b6c8486154c)
        ;




            var circle_marker_a5817eb5dfe6394f95a43248627feacc = L.circleMarker(
                [-1.364, 36.7419],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b68661a6576dcb791355ad77299118b6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bc0e75e2abfcebcf029f556166bff0d5 = $(`&lt;div id=&quot;html_bc0e75e2abfcebcf029f556166bff0d5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;215 Karen Garden - 3 Bedroom Bungalow&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire bungalow&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.5/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9774/night     &lt;/div&gt;`)[0];
                popup_b68661a6576dcb791355ad77299118b6.setContent(html_bc0e75e2abfcebcf029f556166bff0d5);



        circle_marker_a5817eb5dfe6394f95a43248627feacc.bindPopup(popup_b68661a6576dcb791355ad77299118b6)
        ;




            var circle_marker_8db14ce30200301321271e2d4cecee8e = L.circleMarker(
                [-1.3164, 36.6825],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_95613b489c1ad129bc908fa31f568590 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_90ffa9532b5584c9378a76c861317778 = $(`&lt;div id=&quot;html_90ffa9532b5584c9378a76c861317778&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;An Irish Welcome in Karen - River Cottage Two&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire chalet&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7957/night     &lt;/div&gt;`)[0];
                popup_95613b489c1ad129bc908fa31f568590.setContent(html_90ffa9532b5584c9378a76c861317778);



        circle_marker_8db14ce30200301321271e2d4cecee8e.bindPopup(popup_95613b489c1ad129bc908fa31f568590)
        ;




            var circle_marker_51003170c977a720bf94645f59503ba7 = L.circleMarker(
                [-1.3738, 36.754],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bb8ddad62f91ea5c45e23ad3e82c7e9b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ed299d2a89353fb3b2ee88521ed78d93 = $(`&lt;div id=&quot;html_ed299d2a89353fb3b2ee88521ed78d93&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Casa La Ndoto (House of good dreams)- 2 Guestrooms&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 44787/night     &lt;/div&gt;`)[0];
                popup_bb8ddad62f91ea5c45e23ad3e82c7e9b.setContent(html_ed299d2a89353fb3b2ee88521ed78d93);



        circle_marker_51003170c977a720bf94645f59503ba7.bindPopup(popup_bb8ddad62f91ea5c45e23ad3e82c7e9b)
        ;




            var circle_marker_55662800688e32c51441e5f0817d8e26 = L.circleMarker(
                [-1.3699, 36.7501],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fedbc3d4a947172d9822ecb5b5686ff5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_485412bceba457235c9a5e4b3c556348 = $(`&lt;div id=&quot;html_485412bceba457235c9a5e4b3c556348&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Treehouse Nr3 at NgongHouse  on 4ha of nature.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Treehouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 13364/night     &lt;/div&gt;`)[0];
                popup_fedbc3d4a947172d9822ecb5b5686ff5.setContent(html_485412bceba457235c9a5e4b3c556348);



        circle_marker_55662800688e32c51441e5f0817d8e26.bindPopup(popup_fedbc3d4a947172d9822ecb5b5686ff5)
        ;




            var circle_marker_66f3f184a9cd68a8d9f91828be61db49 = L.circleMarker(
                [-1.2907, 36.7805],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6f3acc15ed4ea4e5750cc7baff370285 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d0f5d3893e6821ef50d3624a16a0a4e0 = $(`&lt;div id=&quot;html_d0f5d3893e6821ef50d3624a16a0a4e0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stylish Apartment Near Yaya Mall&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6710/night     &lt;/div&gt;`)[0];
                popup_6f3acc15ed4ea4e5750cc7baff370285.setContent(html_d0f5d3893e6821ef50d3624a16a0a4e0);



        circle_marker_66f3f184a9cd68a8d9f91828be61db49.bindPopup(popup_6f3acc15ed4ea4e5750cc7baff370285)
        ;




            var circle_marker_11364722d977b2034eb2e2065719f5ed = L.circleMarker(
                [-1.2353, 36.875],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4c455a55567a19d222326f44cb59c030 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bc532c7600a3679476c6eb9d82cbea79 = $(`&lt;div id=&quot;html_bc532c7600a3679476c6eb9d82cbea79&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Affordable Garden City 7th Flr Duplex Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10974/night     &lt;/div&gt;`)[0];
                popup_4c455a55567a19d222326f44cb59c030.setContent(html_bc532c7600a3679476c6eb9d82cbea79);



        circle_marker_11364722d977b2034eb2e2065719f5ed.bindPopup(popup_4c455a55567a19d222326f44cb59c030)
        ;




            var circle_marker_ce2a444d932e1f9ccdebc00dd78cfae5 = L.circleMarker(
                [-1.3533, 36.7114],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6c12d914105c6075abad9404c15b97b1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_4c8b48d3069a00f51216791e8114d7e7 = $(`&lt;div id=&quot;html_4c8b48d3069a00f51216791e8114d7e7&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Justin&#x27;s cottage in a beautiful garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.96/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7712/night     &lt;/div&gt;`)[0];
                popup_6c12d914105c6075abad9404c15b97b1.setContent(html_4c8b48d3069a00f51216791e8114d7e7);



        circle_marker_ce2a444d932e1f9ccdebc00dd78cfae5.bindPopup(popup_6c12d914105c6075abad9404c15b97b1)
        ;




            var circle_marker_3aabce92fdf524aef3e1c85469ca0564 = L.circleMarker(
                [-1.2565, 36.7514],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_91e7b801b240ad93633f35f56ac4887f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bdc39f122436c5e1b30fd15e461284d8 = $(`&lt;div id=&quot;html_bdc39f122436c5e1b30fd15e461284d8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;★★Two Floor Hideout - Best of Both Worlds★★&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.64/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5536/night     &lt;/div&gt;`)[0];
                popup_91e7b801b240ad93633f35f56ac4887f.setContent(html_bdc39f122436c5e1b30fd15e461284d8);



        circle_marker_3aabce92fdf524aef3e1c85469ca0564.bindPopup(popup_91e7b801b240ad93633f35f56ac4887f)
        ;




            var circle_marker_ecf630db276be44ca9e4bb3215a932ef = L.circleMarker(
                [-1.2671, 36.8065],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4e465aefca98a4c11a656c9b6ded2f8d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f3407b10a639acf8bf918fe36e1c862b = $(`&lt;div id=&quot;html_f3407b10a639acf8bf918fe36e1c862b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;SHERRY’s ZANA II STUDIO&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.47/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4606/night     &lt;/div&gt;`)[0];
                popup_4e465aefca98a4c11a656c9b6ded2f8d.setContent(html_f3407b10a639acf8bf918fe36e1c862b);



        circle_marker_ecf630db276be44ca9e4bb3215a932ef.bindPopup(popup_4e465aefca98a4c11a656c9b6ded2f8d)
        ;




            var circle_marker_7588059b37639591c335faddea882085 = L.circleMarker(
                [-1.3299, 36.7485],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_376df9eb509c017202bb48944853c398 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_054d7e5708d93e022ce271eda97fb735 = $(`&lt;div id=&quot;html_054d7e5708d93e022ce271eda97fb735&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy Cottage in Karen&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.96/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9335/night     &lt;/div&gt;`)[0];
                popup_376df9eb509c017202bb48944853c398.setContent(html_054d7e5708d93e022ce271eda97fb735);



        circle_marker_7588059b37639591c335faddea882085.bindPopup(popup_376df9eb509c017202bb48944853c398)
        ;




            var circle_marker_5ca0b37936d13d2b18e3229f2e25e3e7 = L.circleMarker(
                [-1.2836, 36.8143],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_37d89a921f5dfd2a08b9c99f607b9665 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_458d5ee3f3da0a041d2f4dfd59c47b78 = $(`&lt;div id=&quot;html_458d5ee3f3da0a041d2f4dfd59c47b78&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ywca Furnished  Budget Studio Rooms Near Town&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.26/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3358/night     &lt;/div&gt;`)[0];
                popup_37d89a921f5dfd2a08b9c99f607b9665.setContent(html_458d5ee3f3da0a041d2f4dfd59c47b78);



        circle_marker_5ca0b37936d13d2b18e3229f2e25e3e7.bindPopup(popup_37d89a921f5dfd2a08b9c99f607b9665)
        ;




            var circle_marker_a5594026641912fe4f6ea157102348e3 = L.circleMarker(
                [-1.2812, 36.7937],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_db04601ee177c8b941981f5f5037fd50 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6eaed2dc478a9d38314e47dc5cc6e5c3 = $(`&lt;div id=&quot;html_6eaed2dc478a9d38314e47dc5cc6e5c3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cottage Apartment, 1 BR,  with Private Garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4488/night     &lt;/div&gt;`)[0];
                popup_db04601ee177c8b941981f5f5037fd50.setContent(html_6eaed2dc478a9d38314e47dc5cc6e5c3);



        circle_marker_a5594026641912fe4f6ea157102348e3.bindPopup(popup_db04601ee177c8b941981f5f5037fd50)
        ;




            var circle_marker_1fde35d27c88b4ef5a36fcaa5f4099e7 = L.circleMarker(
                [-1.2304, 36.8067],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7b79e0e983ba8910bbca22ec7573af76 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ee18d19f7e14a3e367cdfcb28d134172 = $(`&lt;div id=&quot;html_ee18d19f7e14a3e367cdfcb28d134172&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bonsai Villa - Deluxe Double/Triple Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12155/night     &lt;/div&gt;`)[0];
                popup_7b79e0e983ba8910bbca22ec7573af76.setContent(html_ee18d19f7e14a3e367cdfcb28d134172);



        circle_marker_1fde35d27c88b4ef5a36fcaa5f4099e7.bindPopup(popup_7b79e0e983ba8910bbca22ec7573af76)
        ;




            var circle_marker_53b5ded3bcda0c70f3703eff72c5d9dc = L.circleMarker(
                [-1.2151, 36.7888],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3fc8f09273ad37dbb2184b8ddc46dc72 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8f46fa416d06a8c4f3d0b8dd5c30e1d3 = $(`&lt;div id=&quot;html_8f46fa416d06a8c4f3d0b8dd5c30e1d3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;1 bedroom cottage - Rosslyn Lone Tree Estate&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.83/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8899/night     &lt;/div&gt;`)[0];
                popup_3fc8f09273ad37dbb2184b8ddc46dc72.setContent(html_8f46fa416d06a8c4f3d0b8dd5c30e1d3);



        circle_marker_53b5ded3bcda0c70f3703eff72c5d9dc.bindPopup(popup_3fc8f09273ad37dbb2184b8ddc46dc72)
        ;




            var circle_marker_403a230f9fb5be95f5f67ee3c3039f61 = L.circleMarker(
                [-1.2932, 36.8106],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_79774df22f5879174f88843ff1880dc7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cf9ae6e3171802bf067b093692e88869 = $(`&lt;div id=&quot;html_cf9ae6e3171802bf067b093692e88869&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nairobi Hill Elegance- Upper Hill 2 bedrooms&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7382/night     &lt;/div&gt;`)[0];
                popup_79774df22f5879174f88843ff1880dc7.setContent(html_cf9ae6e3171802bf067b093692e88869);



        circle_marker_403a230f9fb5be95f5f67ee3c3039f61.bindPopup(popup_79774df22f5879174f88843ff1880dc7)
        ;




            var circle_marker_7022a9f77b92ac7e69cc486d641acd5e = L.circleMarker(
                [-1.2985, 36.7629],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0432ad019935aab8ff91d19f709006a5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9091c16eec609ec2e3b47ea692094c22 = $(`&lt;div id=&quot;html_9091c16eec609ec2e3b47ea692094c22&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;You always have a home here!&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.58/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2184/night     &lt;/div&gt;`)[0];
                popup_0432ad019935aab8ff91d19f709006a5.setContent(html_9091c16eec609ec2e3b47ea692094c22);



        circle_marker_7022a9f77b92ac7e69cc486d641acd5e.bindPopup(popup_0432ad019935aab8ff91d19f709006a5)
        ;




            var circle_marker_39b6f750175b99e5e5a836dcaa2ab6ec = L.circleMarker(
                [-1.2806, 36.7933],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b2892a27c16e81e5db2bd932adfff1c0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3f2b11c062e4e04632210e9cec5e30d4 = $(`&lt;div id=&quot;html_3f2b11c062e4e04632210e9cec5e30d4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cozy Studio Cottage Apartment with Private Garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4465/night     &lt;/div&gt;`)[0];
                popup_b2892a27c16e81e5db2bd932adfff1c0.setContent(html_3f2b11c062e4e04632210e9cec5e30d4);



        circle_marker_39b6f750175b99e5e5a836dcaa2ab6ec.bindPopup(popup_b2892a27c16e81e5db2bd932adfff1c0)
        ;




            var circle_marker_fab7b40cfb09dda472876c16ab5f7fdb = L.circleMarker(
                [-1.2806, 36.7932],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_67054aba6b3f1db8c34de37770c3d444 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a0ac78855ec72242427cd242e04d7e6d = $(`&lt;div id=&quot;html_a0ac78855ec72242427cd242e04d7e6d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene 1-BR Cottage Apartment with private yard&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3479/night     &lt;/div&gt;`)[0];
                popup_67054aba6b3f1db8c34de37770c3d444.setContent(html_a0ac78855ec72242427cd242e04d7e6d);



        circle_marker_fab7b40cfb09dda472876c16ab5f7fdb.bindPopup(popup_67054aba6b3f1db8c34de37770c3d444)
        ;




            var circle_marker_1237cf4d5c2d37606363918a0106598c = L.circleMarker(
                [-1.2795, 36.7812],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8a0493709a582b1ce4a05a4db6fc48c7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e95c85d7f202b855ef39318927f07e7c = $(`&lt;div id=&quot;html_e95c85d7f202b855ef39318927f07e7c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Crescent Apartments;  3 Bed Immaculate Condo&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.96/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 15587/night     &lt;/div&gt;`)[0];
                popup_8a0493709a582b1ce4a05a4db6fc48c7.setContent(html_e95c85d7f202b855ef39318927f07e7c);



        circle_marker_1237cf4d5c2d37606363918a0106598c.bindPopup(popup_8a0493709a582b1ce4a05a4db6fc48c7)
        ;




            var circle_marker_25719edd800cc135c05e548ea99c7145 = L.circleMarker(
                [-1.2911, 36.779],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d6866b9298ae337b1f2eb0aacd415cca = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c4a033629acbdbd9514a3f4a91cffcc9 = $(`&lt;div id=&quot;html_c4a033629acbdbd9514a3f4a91cffcc9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Brookview Serviced Apartments Kilimani - 2 Bedroom&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5369/night     &lt;/div&gt;`)[0];
                popup_d6866b9298ae337b1f2eb0aacd415cca.setContent(html_c4a033629acbdbd9514a3f4a91cffcc9);



        circle_marker_25719edd800cc135c05e548ea99c7145.bindPopup(popup_d6866b9298ae337b1f2eb0aacd415cca)
        ;




            var circle_marker_c6223f45d9ffe1b00ceedba2593a252f = L.circleMarker(
                [-1.2111, 36.8401],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9529f7a0d2464f3f9c2384be381e8e87 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_edcd3851506ba6a13704f1ba49e33c99 = $(`&lt;div id=&quot;html_edcd3851506ba6a13704f1ba49e33c99&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Beautiful cozy 1-bed in a lush green suburb&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.74/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5501/night     &lt;/div&gt;`)[0];
                popup_9529f7a0d2464f3f9c2384be381e8e87.setContent(html_edcd3851506ba6a13704f1ba49e33c99);



        circle_marker_c6223f45d9ffe1b00ceedba2593a252f.bindPopup(popup_9529f7a0d2464f3f9c2384be381e8e87)
        ;




            var circle_marker_a24f1f6ef9a00055f200ef4bd85eaa4c = L.circleMarker(
                [-1.2805, 36.8276],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0533c5fc5768f99281df0b0630f3a677 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_969b2e2b8665363cd7b0a295e2c5a4e1 = $(`&lt;div id=&quot;html_969b2e2b8665363cd7b0a295e2c5a4e1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern, Stylish Apartment with Netflix&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2591/night     &lt;/div&gt;`)[0];
                popup_0533c5fc5768f99281df0b0630f3a677.setContent(html_969b2e2b8665363cd7b0a295e2c5a4e1);



        circle_marker_a24f1f6ef9a00055f200ef4bd85eaa4c.bindPopup(popup_0533c5fc5768f99281df0b0630f3a677)
        ;




            var circle_marker_ba23849071bdddc372ea5cbdd13f28fc = L.circleMarker(
                [-1.2763, 36.8082],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f9f6afadc25597604a97b9be88973ef4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_23f0b7aed2bbc9962be914cfb059a30e = $(`&lt;div id=&quot;html_23f0b7aed2bbc9962be914cfb059a30e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Charming &amp; serene home in the tree tops.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8060/night     &lt;/div&gt;`)[0];
                popup_f9f6afadc25597604a97b9be88973ef4.setContent(html_23f0b7aed2bbc9962be914cfb059a30e);



        circle_marker_ba23849071bdddc372ea5cbdd13f28fc.bindPopup(popup_f9f6afadc25597604a97b9be88973ef4)
        ;




            var circle_marker_3069554ea47897b2322e3fa9f2779ea2 = L.circleMarker(
                [-1.2744, 36.795],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7f491abc22284385341c635b16d9ec29 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7826e3d16e85c81b3e47efe77fd31d1a = $(`&lt;div id=&quot;html_7826e3d16e85c81b3e47efe77fd31d1a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ramis Suite Deluxe, Kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9569/night     &lt;/div&gt;`)[0];
                popup_7f491abc22284385341c635b16d9ec29.setContent(html_7826e3d16e85c81b3e47efe77fd31d1a);



        circle_marker_3069554ea47897b2322e3fa9f2779ea2.bindPopup(popup_7f491abc22284385341c635b16d9ec29)
        ;




            var circle_marker_58e7bbdb0abbaeafba2d6263a27967fc = L.circleMarker(
                [-1.2192, 36.8746],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_31bbb8101f43e2ac374706da8f91a433 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_653955ec49956b1bc15c683427cccc17 = $(`&lt;div id=&quot;html_653955ec49956b1bc15c683427cccc17&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Great experience homestay with serene surroundings&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6051/night     &lt;/div&gt;`)[0];
                popup_31bbb8101f43e2ac374706da8f91a433.setContent(html_653955ec49956b1bc15c683427cccc17);



        circle_marker_58e7bbdb0abbaeafba2d6263a27967fc.bindPopup(popup_31bbb8101f43e2ac374706da8f91a433)
        ;




            var circle_marker_30bf4c3a1ce51af2f76c6f8908f8015b = L.circleMarker(
                [-1.3149, 36.6826],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d97b55407f8a6e3582e225b1d0105e9a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_14faf118f6f0e1ac45ef746fa63c4c4d = $(`&lt;div id=&quot;html_14faf118f6f0e1ac45ef746fa63c4c4d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;An Irish Welcome in Karen - River Cottage One&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire chalet&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.76/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9428/night     &lt;/div&gt;`)[0];
                popup_d97b55407f8a6e3582e225b1d0105e9a.setContent(html_14faf118f6f0e1ac45ef746fa63c4c4d);



        circle_marker_30bf4c3a1ce51af2f76c6f8908f8015b.bindPopup(popup_d97b55407f8a6e3582e225b1d0105e9a)
        ;




            var circle_marker_77201ec9f5cf9f2201e2b6f5062fc156 = L.circleMarker(
                [-1.3276, 36.7325],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5f49d9497dd4e5cae3f0d38366a0504b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8acfd8d407dcbff0d83ef5065b0d301d = $(`&lt;div id=&quot;html_8acfd8d407dcbff0d83ef5065b0d301d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Maison Mitwaba-room 2&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8728/night     &lt;/div&gt;`)[0];
                popup_5f49d9497dd4e5cae3f0d38366a0504b.setContent(html_8acfd8d407dcbff0d83ef5065b0d301d);



        circle_marker_77201ec9f5cf9f2201e2b6f5062fc156.bindPopup(popup_5f49d9497dd4e5cae3f0d38366a0504b)
        ;




            var circle_marker_f3a578470c5d2f8c7eaae3aeb6b357ef = L.circleMarker(
                [-1.3273, 36.7331],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6d9285cdb104aecea0305b28a26446e3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bdd86ee60a41734b744b327d57862906 = $(`&lt;div id=&quot;html_bdd86ee60a41734b744b327d57862906&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Huge mini apartment, Maison Mitwaba-Room 4&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.94/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9453/night     &lt;/div&gt;`)[0];
                popup_6d9285cdb104aecea0305b28a26446e3.setContent(html_bdd86ee60a41734b744b327d57862906);



        circle_marker_f3a578470c5d2f8c7eaae3aeb6b357ef.bindPopup(popup_6d9285cdb104aecea0305b28a26446e3)
        ;




            var circle_marker_ffd87eb6c4ff8b86e73f403db1761d98 = L.circleMarker(
                [-1.2872, 36.7902],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_63f6da56965333cd0a61d604a60d91e5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5f0726b611de624405182e75bebbb9cc = $(`&lt;div id=&quot;html_5f0726b611de624405182e75bebbb9cc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury 3 Bed Apartment - Kilimani, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.58/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7446/night     &lt;/div&gt;`)[0];
                popup_63f6da56965333cd0a61d604a60d91e5.setContent(html_5f0726b611de624405182e75bebbb9cc);



        circle_marker_ffd87eb6c4ff8b86e73f403db1761d98.bindPopup(popup_63f6da56965333cd0a61d604a60d91e5)
        ;




            var circle_marker_a42d9dfb0e8315934ab2b495b1110d70 = L.circleMarker(
                [-1.3274, 36.7331],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e3dd45adc1e9880e6262d28dd39ed4a1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5f4be6e1df3c582d7200135056b64a5a = $(`&lt;div id=&quot;html_5f4be6e1df3c582d7200135056b64a5a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Private garden veranda, Maison Mitwaba-Room 5&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8137/night     &lt;/div&gt;`)[0];
                popup_e3dd45adc1e9880e6262d28dd39ed4a1.setContent(html_5f4be6e1df3c582d7200135056b64a5a);



        circle_marker_a42d9dfb0e8315934ab2b495b1110d70.bindPopup(popup_e3dd45adc1e9880e6262d28dd39ed4a1)
        ;




            var circle_marker_32d7ce57bf42e7fa7ba68d2c99811b15 = L.circleMarker(
                [-1.3137, 36.6694],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_21cc41a9e2ccb067f67acd7024ec331c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6ef01306445ad5021f318326656636ce = $(`&lt;div id=&quot;html_6ef01306445ad5021f318326656636ce&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy Cottage with Garden in Karen&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7155/night     &lt;/div&gt;`)[0];
                popup_21cc41a9e2ccb067f67acd7024ec331c.setContent(html_6ef01306445ad5021f318326656636ce);



        circle_marker_32d7ce57bf42e7fa7ba68d2c99811b15.bindPopup(popup_21cc41a9e2ccb067f67acd7024ec331c)
        ;




            var circle_marker_ba1d9a036a15ea00c89feb89fc11d8ec = L.circleMarker(
                [-1.284, 36.8715],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_22f14201b5eb7f8c6b1aa12703840d3d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_860e27b7e9c7802ea189f3edef1b08b6 = $(`&lt;div id=&quot;html_860e27b7e9c7802ea189f3edef1b08b6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Secure and quiet 2 bedroom Bungalow in BuruBuru.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire bungalow&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4461/night     &lt;/div&gt;`)[0];
                popup_22f14201b5eb7f8c6b1aa12703840d3d.setContent(html_860e27b7e9c7802ea189f3edef1b08b6);



        circle_marker_ba1d9a036a15ea00c89feb89fc11d8ec.bindPopup(popup_22f14201b5eb7f8c6b1aa12703840d3d)
        ;




            var circle_marker_9b78c7ef3b8f71433c397b1da03bdc93 = L.circleMarker(
                [-1.2882, 36.7885],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1b45fc2d9c9e415791187d92a19dc328 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d7a001335a308ccb4bc6193d5d504385 = $(`&lt;div id=&quot;html_d7a001335a308ccb4bc6193d5d504385&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Comfortable new 3 bedroom apartment in Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5993/night     &lt;/div&gt;`)[0];
                popup_1b45fc2d9c9e415791187d92a19dc328.setContent(html_d7a001335a308ccb4bc6193d5d504385);



        circle_marker_9b78c7ef3b8f71433c397b1da03bdc93.bindPopup(popup_1b45fc2d9c9e415791187d92a19dc328)
        ;




            var circle_marker_e012036a956c3dfb9d040a905a64a41d = L.circleMarker(
                [-1.298, 36.7968],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9ec7cba8ddb9cb1251075a2c36414e35 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_905327160ed1fae1c2c42d5d163fe021 = $(`&lt;div id=&quot;html_905327160ed1fae1c2c42d5d163fe021&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;2 Bedroom Fully Furnished Apartment, Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6800/night     &lt;/div&gt;`)[0];
                popup_9ec7cba8ddb9cb1251075a2c36414e35.setContent(html_905327160ed1fae1c2c42d5d163fe021);



        circle_marker_e012036a956c3dfb9d040a905a64a41d.bindPopup(popup_9ec7cba8ddb9cb1251075a2c36414e35)
        ;




            var circle_marker_a96606d542b24281a5f06e9cd78d7e70 = L.circleMarker(
                [-1.3243, 36.8435],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2354287ba1df6e640041b2d1816f4324 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5b7f241c028bb1a75889a9e8c81bbc94 = $(`&lt;div id=&quot;html_5b7f241c028bb1a75889a9e8c81bbc94&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nextgen-Ruby Apt 1 Bedrm Msa Rd next to Eka Hotel&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5178/night     &lt;/div&gt;`)[0];
                popup_2354287ba1df6e640041b2d1816f4324.setContent(html_5b7f241c028bb1a75889a9e8c81bbc94);



        circle_marker_a96606d542b24281a5f06e9cd78d7e70.bindPopup(popup_2354287ba1df6e640041b2d1816f4324)
        ;




            var circle_marker_b00929217870f798a3e86055870c2fd2 = L.circleMarker(
                [-1.3765, 36.7409],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f810dc5d5887bebab90d9b3460b7ee1a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c115f6e6d6ad5b22340623a5c6741ca2 = $(`&lt;div id=&quot;html_c115f6e6d6ad5b22340623a5c6741ca2&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Airy studio getaway on stunning grounds in Karen&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8688/night     &lt;/div&gt;`)[0];
                popup_f810dc5d5887bebab90d9b3460b7ee1a.setContent(html_c115f6e6d6ad5b22340623a5c6741ca2);



        circle_marker_b00929217870f798a3e86055870c2fd2.bindPopup(popup_f810dc5d5887bebab90d9b3460b7ee1a)
        ;




            var circle_marker_0e2bd6c9b6919d03bce779469004f45d = L.circleMarker(
                [-1.268, 36.7978],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_34c5f68da4b503626fdef9828dddeed8 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d360d58a5ba061e6b35598c82ffc31a9 = $(`&lt;div id=&quot;html_d360d58a5ba061e6b35598c82ffc31a9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Zebra Pad - Spacious 1BR Apt in Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.92/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5472/night     &lt;/div&gt;`)[0];
                popup_34c5f68da4b503626fdef9828dddeed8.setContent(html_d360d58a5ba061e6b35598c82ffc31a9);



        circle_marker_0e2bd6c9b6919d03bce779469004f45d.bindPopup(popup_34c5f68da4b503626fdef9828dddeed8)
        ;




            var circle_marker_4f164a9a69231f47bc963bd19977e19d = L.circleMarker(
                [-1.2867, 36.7829],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fb52b8143ab306f2099bf32850a0faf5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dd1fa825d9d6f884e3d5b2f49e12c933 = $(`&lt;div id=&quot;html_dd1fa825d9d6f884e3d5b2f49e12c933&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Leah’s Quiet Peaceful Space&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3439/night     &lt;/div&gt;`)[0];
                popup_fb52b8143ab306f2099bf32850a0faf5.setContent(html_dd1fa825d9d6f884e3d5b2f49e12c933);



        circle_marker_4f164a9a69231f47bc963bd19977e19d.bindPopup(popup_fb52b8143ab306f2099bf32850a0faf5)
        ;




            var circle_marker_f2cabb5a4fcb67192546226b34b292d8 = L.circleMarker(
                [-1.2805, 36.7933],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1beaa8526da4d89fe6beb465f1b33b9d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9a9f8d5e3db432c264ce32f5ff02d37e = $(`&lt;div id=&quot;html_9a9f8d5e3db432c264ce32f5ff02d37e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cozy Serviced Cottage Apartment with Private yard&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3137/night     &lt;/div&gt;`)[0];
                popup_1beaa8526da4d89fe6beb465f1b33b9d.setContent(html_9a9f8d5e3db432c264ce32f5ff02d37e);



        circle_marker_f2cabb5a4fcb67192546226b34b292d8.bindPopup(popup_1beaa8526da4d89fe6beb465f1b33b9d)
        ;




            var circle_marker_02cd65bd81a6c526c810bb948cedb5b1 = L.circleMarker(
                [-1.2953, 36.798],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d96db00181e35fb21e7f84f5398de0cd = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_83bafaf34ab601106cd98cdb74d3ec3b = $(`&lt;div id=&quot;html_83bafaf34ab601106cd98cdb74d3ec3b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;F Samra 2 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5134/night     &lt;/div&gt;`)[0];
                popup_d96db00181e35fb21e7f84f5398de0cd.setContent(html_83bafaf34ab601106cd98cdb74d3ec3b);



        circle_marker_02cd65bd81a6c526c810bb948cedb5b1.bindPopup(popup_d96db00181e35fb21e7f84f5398de0cd)
        ;




            var circle_marker_5c3d089f747b9bc8d97cdfe63651ca93 = L.circleMarker(
                [-1.2741, 36.7909],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bd3e8d66599afd0285e358e9ab1d441c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_85b6a68001f792c18efcf32673dad668 = $(`&lt;div id=&quot;html_85b6a68001f792c18efcf32673dad668&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cozy Ground Flr Studio with Balcony In Kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.62/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3179/night     &lt;/div&gt;`)[0];
                popup_bd3e8d66599afd0285e358e9ab1d441c.setContent(html_85b6a68001f792c18efcf32673dad668);



        circle_marker_5c3d089f747b9bc8d97cdfe63651ca93.bindPopup(popup_bd3e8d66599afd0285e358e9ab1d441c)
        ;




            var circle_marker_5f2e35607d054c96ff0fc0826a52c74a = L.circleMarker(
                [-1.2738, 36.7902],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0f51230f102062f4153334388295e8c7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bb2a24698c2f2aa1e44344d0288d2792 = $(`&lt;div id=&quot;html_bb2a24698c2f2aa1e44344d0288d2792&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Superior 3BR on Grd Flr- Pool &amp;Playground&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10012/night     &lt;/div&gt;`)[0];
                popup_0f51230f102062f4153334388295e8c7.setContent(html_bb2a24698c2f2aa1e44344d0288d2792);



        circle_marker_5f2e35607d054c96ff0fc0826a52c74a.bindPopup(popup_0f51230f102062f4153334388295e8c7)
        ;




            var circle_marker_c973be44017c80569e5448272085c8f3 = L.circleMarker(
                [-1.2585, 36.7997],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e302463aa91f21932951806c75dcccfa = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0645f5213ab697eb937f439e5eaa8d25 = $(`&lt;div id=&quot;html_0645f5213ab697eb937f439e5eaa8d25&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Charming Sundowner 2BDR - Westgate Mall, Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6114/night     &lt;/div&gt;`)[0];
                popup_e302463aa91f21932951806c75dcccfa.setContent(html_0645f5213ab697eb937f439e5eaa8d25);



        circle_marker_c973be44017c80569e5448272085c8f3.bindPopup(popup_e302463aa91f21932951806c75dcccfa)
        ;




            var circle_marker_f9e0c29e37551f313f5e36f362506cba = L.circleMarker(
                [-1.2851, 36.7782],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5cb5421e1c69b3e967df8d4c57912205 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_929fce6d12d61cb6f4ff57527d0c4c65 = $(`&lt;div id=&quot;html_929fce6d12d61cb6f4ff57527d0c4c65&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;KILELESHWA&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.82/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12716/night     &lt;/div&gt;`)[0];
                popup_5cb5421e1c69b3e967df8d4c57912205.setContent(html_929fce6d12d61cb6f4ff57527d0c4c65);



        circle_marker_f9e0c29e37551f313f5e36f362506cba.bindPopup(popup_5cb5421e1c69b3e967df8d4c57912205)
        ;




            var circle_marker_75e8d517140d37383e9392ba328586e9 = L.circleMarker(
                [-1.2744, 36.7931],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bed22551fdaa7479a346f55e5c6b9e88 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8d8690091eaaf638e50fbe5c39ec56ba = $(`&lt;div id=&quot;html_8d8690091eaaf638e50fbe5c39ec56ba&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Tiny house nestled in Kileleshwa, Nairobi, Kenya&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Tiny home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.77/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4118/night     &lt;/div&gt;`)[0];
                popup_bed22551fdaa7479a346f55e5c6b9e88.setContent(html_8d8690091eaaf638e50fbe5c39ec56ba);



        circle_marker_75e8d517140d37383e9392ba328586e9.bindPopup(popup_bed22551fdaa7479a346f55e5c6b9e88)
        ;




            var circle_marker_c232009a479c8d87957291f75a4d24b3 = L.circleMarker(
                [-1.2578, 36.7959],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_98a1719f7031223493588a414dd3174b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e9de44e5a40bdb0a763324d7343b2996 = $(`&lt;div id=&quot;html_e9de44e5a40bdb0a763324d7343b2996&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Urban Retro In A Leafy Serene Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5639/night     &lt;/div&gt;`)[0];
                popup_98a1719f7031223493588a414dd3174b.setContent(html_e9de44e5a40bdb0a763324d7343b2996);



        circle_marker_c232009a479c8d87957291f75a4d24b3.bindPopup(popup_98a1719f7031223493588a414dd3174b)
        ;




            var circle_marker_29288a69f384c549b252f1667670ac82 = L.circleMarker(
                [-1.2802, 36.7941],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ce84dd580ca8baa4abd69fa3a5200d21 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a5cb31f61c707a82a936b6c55ff7a2cf = $(`&lt;div id=&quot;html_a5cb31f61c707a82a936b6c55ff7a2cf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Charming Cottage Apartment with Private Garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4760/night     &lt;/div&gt;`)[0];
                popup_ce84dd580ca8baa4abd69fa3a5200d21.setContent(html_a5cb31f61c707a82a936b6c55ff7a2cf);



        circle_marker_29288a69f384c549b252f1667670ac82.bindPopup(popup_ce84dd580ca8baa4abd69fa3a5200d21)
        ;




            var circle_marker_36adbe377b5fae197ef841729d4e4e7f = L.circleMarker(
                [-1.2158, 36.8995],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_813d48e5bcd7dd70c3baeea8d6230298 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1188acc947d1815de44b354e5c2eafd3 = $(`&lt;div id=&quot;html_1188acc947d1815de44b354e5c2eafd3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene studio with adequate parking space&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.5/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1430/night     &lt;/div&gt;`)[0];
                popup_813d48e5bcd7dd70c3baeea8d6230298.setContent(html_1188acc947d1815de44b354e5c2eafd3);



        circle_marker_36adbe377b5fae197ef841729d4e4e7f.bindPopup(popup_813d48e5bcd7dd70c3baeea8d6230298)
        ;




            var circle_marker_9607e2558ad502c86ce6ef25224bdde1 = L.circleMarker(
                [-1.2816, 36.7935],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_acd1aab73fd7ea2972d624fd2876635e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_38403d602c58228b5db4c16990bba27b = $(`&lt;div id=&quot;html_38403d602c58228b5db4c16990bba27b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cottage Apartment, 1 Bedroom, with Private Garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5053/night     &lt;/div&gt;`)[0];
                popup_acd1aab73fd7ea2972d624fd2876635e.setContent(html_38403d602c58228b5db4c16990bba27b);



        circle_marker_9607e2558ad502c86ce6ef25224bdde1.bindPopup(popup_acd1aab73fd7ea2972d624fd2876635e)
        ;




            var circle_marker_ae05971ce7c3a41636b41c39351a91ec = L.circleMarker(
                [-1.2798, 36.7826],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_345fd3bee212ffd91b1bc25c2e153729 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_11ca4d9bd9f1b2ce94d081fa25565350 = $(`&lt;div id=&quot;html_11ca4d9bd9f1b2ce94d081fa25565350&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Crescent Apartments; 1 Bed Immaculate Condo&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11066/night     &lt;/div&gt;`)[0];
                popup_345fd3bee212ffd91b1bc25c2e153729.setContent(html_11ca4d9bd9f1b2ce94d081fa25565350);



        circle_marker_ae05971ce7c3a41636b41c39351a91ec.bindPopup(popup_345fd3bee212ffd91b1bc25c2e153729)
        ;




            var circle_marker_5b900fa00f6a8b649d554463aa695014 = L.circleMarker(
                [-1.2235, 36.8213],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2c331462cf11e3957b2f0e8614863678 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_167617d983ca11496c757c0030b4e71e = $(`&lt;div id=&quot;html_167617d983ca11496c757c0030b4e71e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lotus House - Annex&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5693/night     &lt;/div&gt;`)[0];
                popup_2c331462cf11e3957b2f0e8614863678.setContent(html_167617d983ca11496c757c0030b4e71e);



        circle_marker_5b900fa00f6a8b649d554463aa695014.bindPopup(popup_2c331462cf11e3957b2f0e8614863678)
        ;




            var circle_marker_286bd28f4202c684b22b9ab48e3c8ba3 = L.circleMarker(
                [-1.2642, 36.7999],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c662bfe0cafdc0e7630b96655b85b1e7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f3da0a1c898fbb899bf1994d9aae9b26 = $(`&lt;div id=&quot;html_f3da0a1c898fbb899bf1994d9aae9b26&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Westlands Place , Elegant 1BR  with Gym &amp;Lounge&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.76/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5738/night     &lt;/div&gt;`)[0];
                popup_c662bfe0cafdc0e7630b96655b85b1e7.setContent(html_f3da0a1c898fbb899bf1994d9aae9b26);



        circle_marker_286bd28f4202c684b22b9ab48e3c8ba3.bindPopup(popup_c662bfe0cafdc0e7630b96655b85b1e7)
        ;




            var circle_marker_eda6df07be6dab838e092248c2e50f65 = L.circleMarker(
                [-1.2664, 36.7998],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c374c3f883c549452ec78661a04092fd = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0df3c200b65456cc7e4223b22e45ae84 = $(`&lt;div id=&quot;html_0df3c200b65456cc7e4223b22e45ae84&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nairobi Savannah Pad - 3 bedroom Apt in Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.82/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10842/night     &lt;/div&gt;`)[0];
                popup_c374c3f883c549452ec78661a04092fd.setContent(html_0df3c200b65456cc7e4223b22e45ae84);



        circle_marker_eda6df07be6dab838e092248c2e50f65.bindPopup(popup_c374c3f883c549452ec78661a04092fd)
        ;




            var circle_marker_fc0030194a4543438b025276c1598839 = L.circleMarker(
                [-1.2904, 36.7686],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_710612f6dca6b0b4779429fe0a0fc419 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d6cd88acd4c5cb2936b467d6a4efb3ed = $(`&lt;div id=&quot;html_d6cd88acd4c5cb2936b467d6a4efb3ed&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern charming cosy retreat&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire bungalow&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.57/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4467/night     &lt;/div&gt;`)[0];
                popup_710612f6dca6b0b4779429fe0a0fc419.setContent(html_d6cd88acd4c5cb2936b467d6a4efb3ed);



        circle_marker_fc0030194a4543438b025276c1598839.bindPopup(popup_710612f6dca6b0b4779429fe0a0fc419)
        ;




            var circle_marker_43d0fc0405ae97d975e4fdb196ddb1be = L.circleMarker(
                [-1.3055, 36.8181],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_320ec467fff45010a77ce44b89593ca4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_76d0938f0ad452ac7a856a9589cd27b1 = $(`&lt;div id=&quot;html_76d0938f0ad452ac7a856a9589cd27b1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Madaraka Furnished Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.25/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4888/night     &lt;/div&gt;`)[0];
                popup_320ec467fff45010a77ce44b89593ca4.setContent(html_76d0938f0ad452ac7a856a9589cd27b1);



        circle_marker_43d0fc0405ae97d975e4fdb196ddb1be.bindPopup(popup_320ec467fff45010a77ce44b89593ca4)
        ;




            var circle_marker_fbd0d1c97d62ccd390fea36b2d52f0b8 = L.circleMarker(
                [-1.2149, 36.8419],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7c74caffcd663dc062e45152f2067b30 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7fc053be74275de3f2c28d18777bed0f = $(`&lt;div id=&quot;html_7fc053be74275de3f2c28d18777bed0f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Fourways Junction VIP Suite - Near UN HQ&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7058/night     &lt;/div&gt;`)[0];
                popup_7c74caffcd663dc062e45152f2067b30.setContent(html_7fc053be74275de3f2c28d18777bed0f);



        circle_marker_fbd0d1c97d62ccd390fea36b2d52f0b8.bindPopup(popup_7c74caffcd663dc062e45152f2067b30)
        ;




            var circle_marker_c170af1fd5c10798e1ac385bb507871a = L.circleMarker(
                [-1.2591, 36.7979],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d6985d1788fe2a46d66e28573563e9aa = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_01bcb3d6f990041dc59fe2ca5446fa75 = $(`&lt;div id=&quot;html_01bcb3d6f990041dc59fe2ca5446fa75&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;SHERRY’s Tulia 1 Bedroom  -Near Westgate Mall&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.56/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5353/night     &lt;/div&gt;`)[0];
                popup_d6985d1788fe2a46d66e28573563e9aa.setContent(html_01bcb3d6f990041dc59fe2ca5446fa75);



        circle_marker_c170af1fd5c10798e1ac385bb507871a.bindPopup(popup_d6985d1788fe2a46d66e28573563e9aa)
        ;




            var circle_marker_e12c5b12ec3b75e501edf805534eb4fc = L.circleMarker(
                [-1.2893, 36.7811],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_194daad81e89375059e5a06f720645e0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_905ea382fed836a48ffee02579191e60 = $(`&lt;div id=&quot;html_905ea382fed836a48ffee02579191e60&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Upscale King Bed in Kilimani.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7080/night     &lt;/div&gt;`)[0];
                popup_194daad81e89375059e5a06f720645e0.setContent(html_905ea382fed836a48ffee02579191e60);



        circle_marker_e12c5b12ec3b75e501edf805534eb4fc.bindPopup(popup_194daad81e89375059e5a06f720645e0)
        ;




            var circle_marker_b7c9440fd84e6e6c34ad6e67a607f903 = L.circleMarker(
                [-1.2935, 36.7868],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_585b2c4876006e4f169098e85f0c7891 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cdca909004c690d609803405ab4d867e = $(`&lt;div id=&quot;html_cdca909004c690d609803405ab4d867e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury Solos Lavington 3-Bedroom Dream  Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8080/night     &lt;/div&gt;`)[0];
                popup_585b2c4876006e4f169098e85f0c7891.setContent(html_cdca909004c690d609803405ab4d867e);



        circle_marker_b7c9440fd84e6e6c34ad6e67a607f903.bindPopup(popup_585b2c4876006e4f169098e85f0c7891)
        ;




            var circle_marker_624a916fe2a01ee79f8473a8f934d30a = L.circleMarker(
                [-1.2858, 36.7892],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ca973db8346e75239b1a77dc7c01a8ac = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_742caf8e2b3a752844eccc6bd56eb83f = $(`&lt;div id=&quot;html_742caf8e2b3a752844eccc6bd56eb83f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy 1 bedroom recently renovated in Kilimani Yaya&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4715/night     &lt;/div&gt;`)[0];
                popup_ca973db8346e75239b1a77dc7c01a8ac.setContent(html_742caf8e2b3a752844eccc6bd56eb83f);



        circle_marker_624a916fe2a01ee79f8473a8f934d30a.bindPopup(popup_ca973db8346e75239b1a77dc7c01a8ac)
        ;




            var circle_marker_11eec59c1e8bfec0c35f6ba9030c8cd5 = L.circleMarker(
                [-1.2687, 36.8079],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5fb534e7c0f9082d9928434715782c96 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e427b87e96a446dc8a021fefe2429b4d = $(`&lt;div id=&quot;html_e427b87e96a446dc8a021fefe2429b4d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;King&#x27;s Place&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.56/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2094/night     &lt;/div&gt;`)[0];
                popup_5fb534e7c0f9082d9928434715782c96.setContent(html_e427b87e96a446dc8a021fefe2429b4d);



        circle_marker_11eec59c1e8bfec0c35f6ba9030c8cd5.bindPopup(popup_5fb534e7c0f9082d9928434715782c96)
        ;




            var circle_marker_4361fa5387f260119b3da4d80852606f = L.circleMarker(
                [-1.223, 36.825],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9771cb633832d97f138bdbab2ceb69bb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_29581773b3b84d4306b33df3f695e307 = $(`&lt;div id=&quot;html_29581773b3b84d4306b33df3f695e307&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Villa Rockstop Falls, Homestay, Bed &amp; Breakfast.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.6/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6921/night     &lt;/div&gt;`)[0];
                popup_9771cb633832d97f138bdbab2ceb69bb.setContent(html_29581773b3b84d4306b33df3f695e307);



        circle_marker_4361fa5387f260119b3da4d80852606f.bindPopup(popup_9771cb633832d97f138bdbab2ceb69bb)
        ;




            var circle_marker_c065e188f11b932255b0abb71f5dc428 = L.circleMarker(
                [-1.2911, 36.781],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_24230391d880fb01842a2ee1f4bcd4ad = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_47f7d48d51c90ea8e550d477df3d8fa9 = $(`&lt;div id=&quot;html_47f7d48d51c90ea8e550d477df3d8fa9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spacious 2 bed, sleeps 4,1 bath near Yaya Center&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.65/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5089/night     &lt;/div&gt;`)[0];
                popup_24230391d880fb01842a2ee1f4bcd4ad.setContent(html_47f7d48d51c90ea8e550d477df3d8fa9);



        circle_marker_c065e188f11b932255b0abb71f5dc428.bindPopup(popup_24230391d880fb01842a2ee1f4bcd4ad)
        ;




            var circle_marker_02e61882458dc6d85247b355d64f4063 = L.circleMarker(
                [-1.2973, 36.7723],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_834fc8fa00bf4f4984d2f17f6cc0d766 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_400e32aec9da1a5ee5f0fa6149c80889 = $(`&lt;div id=&quot;html_400e32aec9da1a5ee5f0fa6149c80889&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;9th Floor View&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.49/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3672/night     &lt;/div&gt;`)[0];
                popup_834fc8fa00bf4f4984d2f17f6cc0d766.setContent(html_400e32aec9da1a5ee5f0fa6149c80889);



        circle_marker_02e61882458dc6d85247b355d64f4063.bindPopup(popup_834fc8fa00bf4f4984d2f17f6cc0d766)
        ;




            var circle_marker_f3c0267f04d1c3d61e3499692b713bc9 = L.circleMarker(
                [-1.2572, 36.7985],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_eeb7c731db0f52eba487cfc59aeaf42e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6e2a3c88ada55b94e021485d4b6d3f47 = $(`&lt;div id=&quot;html_6e2a3c88ada55b94e021485d4b6d3f47&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;7 min walk-Sarit Mall, Light noise, Great Value!&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5485/night     &lt;/div&gt;`)[0];
                popup_eeb7c731db0f52eba487cfc59aeaf42e.setContent(html_6e2a3c88ada55b94e021485d4b6d3f47);



        circle_marker_f3c0267f04d1c3d61e3499692b713bc9.bindPopup(popup_eeb7c731db0f52eba487cfc59aeaf42e)
        ;




            var circle_marker_77772615f93921d48e219e274c0bcef9 = L.circleMarker(
                [-1.2856, 36.7577],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6496b850c878998105c20899d3461c91 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6eb3b089567800af308c6731fecf6f41 = $(`&lt;div id=&quot;html_6eb3b089567800af308c6731fecf6f41&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Serene Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9439/night     &lt;/div&gt;`)[0];
                popup_6496b850c878998105c20899d3461c91.setContent(html_6eb3b089567800af308c6731fecf6f41);



        circle_marker_77772615f93921d48e219e274c0bcef9.bindPopup(popup_6496b850c878998105c20899d3461c91)
        ;




            var circle_marker_a9fac1b068d537953774fd66f5381f56 = L.circleMarker(
                [-1.2647, 36.7926],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a2097a3e17577ca4a56141d58323f182 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c5fb792609bd46c14c693624f96e6f18 = $(`&lt;div id=&quot;html_c5fb792609bd46c14c693624f96e6f18&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Westlands, Rafiki Residence 2, East Church Road.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3035/night     &lt;/div&gt;`)[0];
                popup_a2097a3e17577ca4a56141d58323f182.setContent(html_c5fb792609bd46c14c693624f96e6f18);



        circle_marker_a9fac1b068d537953774fd66f5381f56.bindPopup(popup_a2097a3e17577ca4a56141d58323f182)
        ;




            var circle_marker_ca74d7710bf22c16c95d6ac10968bdbd = L.circleMarker(
                [-1.2318, 36.8103],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e9714ab422b506e2275ebe7229bee749 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1e8ff18383b7eb20a473a62cddf8472e = $(`&lt;div id=&quot;html_1e8ff18383b7eb20a473a62cddf8472e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bonsai Villa - Diplomatic Suite/Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 14460/night     &lt;/div&gt;`)[0];
                popup_e9714ab422b506e2275ebe7229bee749.setContent(html_1e8ff18383b7eb20a473a62cddf8472e);



        circle_marker_ca74d7710bf22c16c95d6ac10968bdbd.bindPopup(popup_e9714ab422b506e2275ebe7229bee749)
        ;




            var circle_marker_9ee44da71dd8b92e63b950c3747b0806 = L.circleMarker(
                [-1.2929, 36.7881],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_9ba9b07095c5e14a89b46f20ce4ba26a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a4308a0a9ad5c6a3a33496b78da18420 = $(`&lt;div id=&quot;html_a4308a0a9ad5c6a3a33496b78da18420&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;MONROE FURNISHED APARTMENT- KILLIMANI .YAYA CENTER&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.74/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10015/night     &lt;/div&gt;`)[0];
                popup_9ba9b07095c5e14a89b46f20ce4ba26a.setContent(html_a4308a0a9ad5c6a3a33496b78da18420);



        circle_marker_9ee44da71dd8b92e63b950c3747b0806.bindPopup(popup_9ba9b07095c5e14a89b46f20ce4ba26a)
        ;




            var circle_marker_89a1a50e21185450993002e126467336 = L.circleMarker(
                [-1.2866, 36.7871],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_395795d760e2f0aa4ed9e37c2c8c867d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dee855f34f9f70298721bce3abb389bb = $(`&lt;div id=&quot;html_dee855f34f9f70298721bce3abb389bb&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Crestpark Penthouse Spectacular Views&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.73/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11239/night     &lt;/div&gt;`)[0];
                popup_395795d760e2f0aa4ed9e37c2c8c867d.setContent(html_dee855f34f9f70298721bce3abb389bb);



        circle_marker_89a1a50e21185450993002e126467336.bindPopup(popup_395795d760e2f0aa4ed9e37c2c8c867d)
        ;




            var circle_marker_ec7d22d593036b655f641a9b7f6c1969 = L.circleMarker(
                [-1.2986, 36.7677],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_22340b349cc01e3947a567341966dd06 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8d99ee225f271fc91d92fb77bf71b84f = $(`&lt;div id=&quot;html_8d99ee225f271fc91d92fb77bf71b84f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;4BR Elegant spacious secure,  kilimani great view&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.39/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9298/night     &lt;/div&gt;`)[0];
                popup_22340b349cc01e3947a567341966dd06.setContent(html_8d99ee225f271fc91d92fb77bf71b84f);



        circle_marker_ec7d22d593036b655f641a9b7f6c1969.bindPopup(popup_22340b349cc01e3947a567341966dd06)
        ;




            var circle_marker_d6c4aa7859781036780752fe98587751 = L.circleMarker(
                [-1.3176, 36.8434],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e0897a405463e4eef106cf58d38080f2 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_596a7a0292b97abd632f6dd7136762b0 = $(`&lt;div id=&quot;html_596a7a0292b97abd632f6dd7136762b0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nancy&#x27;s LOFT 07 then 25 then 843310&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2601/night     &lt;/div&gt;`)[0];
                popup_e0897a405463e4eef106cf58d38080f2.setContent(html_596a7a0292b97abd632f6dd7136762b0);



        circle_marker_d6c4aa7859781036780752fe98587751.bindPopup(popup_e0897a405463e4eef106cf58d38080f2)
        ;




            var circle_marker_4636312f398d89490e21e3ea6de5c67b = L.circleMarker(
                [-1.3264, 36.8874],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4a4984416f1db888bad4b3b19568b4de = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_51dd3f1de638dde4e0eb924adccfedf5 = $(`&lt;div id=&quot;html_51dd3f1de638dde4e0eb924adccfedf5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Imara Daima Furnished Apartment near Airport &amp; SGR&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2773/night     &lt;/div&gt;`)[0];
                popup_4a4984416f1db888bad4b3b19568b4de.setContent(html_51dd3f1de638dde4e0eb924adccfedf5);



        circle_marker_4636312f398d89490e21e3ea6de5c67b.bindPopup(popup_4a4984416f1db888bad4b3b19568b4de)
        ;




            var circle_marker_ffdbede04e24ef2e0eb44b3745b43d7d = L.circleMarker(
                [-1.2296, 36.8051],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b9522c380fef4a06acede7ab2f852835 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1f2b9721cd07f0d6392b898f0c5ea019 = $(`&lt;div id=&quot;html_1f2b9721cd07f0d6392b898f0c5ea019&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bonsai Villa - Standard King Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9561/night     &lt;/div&gt;`)[0];
                popup_b9522c380fef4a06acede7ab2f852835.setContent(html_1f2b9721cd07f0d6392b898f0c5ea019);



        circle_marker_ffdbede04e24ef2e0eb44b3745b43d7d.bindPopup(popup_b9522c380fef4a06acede7ab2f852835)
        ;




            var circle_marker_fbcc1c6475786f446d75afb01ce519cf = L.circleMarker(
                [-1.2285, 36.8109],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a630406c3a90800044465ff5e8951b44 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6d2990afccf96ac7c27a9d9a7df6b608 = $(`&lt;div id=&quot;html_6d2990afccf96ac7c27a9d9a7df6b608&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Orchid Homes -Standard Double Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 16873/night     &lt;/div&gt;`)[0];
                popup_a630406c3a90800044465ff5e8951b44.setContent(html_6d2990afccf96ac7c27a9d9a7df6b608);



        circle_marker_fbcc1c6475786f446d75afb01ce519cf.bindPopup(popup_a630406c3a90800044465ff5e8951b44)
        ;




            var circle_marker_60e08dffeb80170993f53ef175bcebb5 = L.circleMarker(
                [-1.2315, 36.8801],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1368b09efb9fe433d3269cfe511d25f8 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3d638498c561d3e7eeb74a0a348c15b4 = $(`&lt;div id=&quot;html_3d638498c561d3e7eeb74a0a348c15b4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern 2-bed/2.5bath duplex apt at Garden City&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11938/night     &lt;/div&gt;`)[0];
                popup_1368b09efb9fe433d3269cfe511d25f8.setContent(html_3d638498c561d3e7eeb74a0a348c15b4);



        circle_marker_60e08dffeb80170993f53ef175bcebb5.bindPopup(popup_1368b09efb9fe433d3269cfe511d25f8)
        ;




            var circle_marker_1c7514ec953058aa7c88a59fe3153cfb = L.circleMarker(
                [-1.29, 36.775],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ec274401bbb613fca09e472ce34d4c14 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6de54cd988164a2e72365b187b65be8b = $(`&lt;div id=&quot;html_6de54cd988164a2e72365b187b65be8b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury condo in lavington&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4334/night     &lt;/div&gt;`)[0];
                popup_ec274401bbb613fca09e472ce34d4c14.setContent(html_6de54cd988164a2e72365b187b65be8b);



        circle_marker_1c7514ec953058aa7c88a59fe3153cfb.bindPopup(popup_ec274401bbb613fca09e472ce34d4c14)
        ;




            var circle_marker_f3dfad40ccee35372836f6b28924f2d4 = L.circleMarker(
                [-1.283, 36.7851],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c870f38477b091f250287dff2735b9ae = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6d267d5a33593a1cb5d2fa3775f9c8ca = $(`&lt;div id=&quot;html_6d267d5a33593a1cb5d2fa3775f9c8ca&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spacious En-suite private room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2904/night     &lt;/div&gt;`)[0];
                popup_c870f38477b091f250287dff2735b9ae.setContent(html_6d267d5a33593a1cb5d2fa3775f9c8ca);



        circle_marker_f3dfad40ccee35372836f6b28924f2d4.bindPopup(popup_c870f38477b091f250287dff2735b9ae)
        ;




            var circle_marker_7f415fa7d62328283e7c08c7bbef9e55 = L.circleMarker(
                [-1.2944, 36.7985],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7e6bd9cdd5d75b6d48dd7eaa4abb9992 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_db0073ea259b656552315a1f502dfbb3 = $(`&lt;div id=&quot;html_db0073ea259b656552315a1f502dfbb3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;H Samra  2 Bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4607/night     &lt;/div&gt;`)[0];
                popup_7e6bd9cdd5d75b6d48dd7eaa4abb9992.setContent(html_db0073ea259b656552315a1f502dfbb3);



        circle_marker_7f415fa7d62328283e7c08c7bbef9e55.bindPopup(popup_7e6bd9cdd5d75b6d48dd7eaa4abb9992)
        ;




            var circle_marker_cadde7653f8997a3c95eb09fe23902f5 = L.circleMarker(
                [-1.2666, 36.8092],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_30f1ef57a54c1e029d2e5157fd0feea5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_eca2458edc6e27d85a08d1a80073ebd1 = $(`&lt;div id=&quot;html_eca2458edc6e27d85a08d1a80073ebd1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Humble Abode&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1366/night     &lt;/div&gt;`)[0];
                popup_30f1ef57a54c1e029d2e5157fd0feea5.setContent(html_eca2458edc6e27d85a08d1a80073ebd1);



        circle_marker_cadde7653f8997a3c95eb09fe23902f5.bindPopup(popup_30f1ef57a54c1e029d2e5157fd0feea5)
        ;




            var circle_marker_877406190b40c2599873a8ae7c7f295d = L.circleMarker(
                [-1.2317, 36.8063],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5d11c86068a78576337cb1acf83b7245 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f09df7ecde9563e5dbd66d7388cf01cf = $(`&lt;div id=&quot;html_f09df7ecde9563e5dbd66d7388cf01cf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bonsai Villa Penthouse Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 16197/night     &lt;/div&gt;`)[0];
                popup_5d11c86068a78576337cb1acf83b7245.setContent(html_f09df7ecde9563e5dbd66d7388cf01cf);



        circle_marker_877406190b40c2599873a8ae7c7f295d.bindPopup(popup_5d11c86068a78576337cb1acf83b7245)
        ;




            var circle_marker_853eed3baeb68ae0919b8cc6afc3862a = L.circleMarker(
                [-1.2164, 36.8362],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6db40c46e2f94d6719dc39deacc231d8 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d7e96009db65ef5383785ec21db52f6d = $(`&lt;div id=&quot;html_d7e96009db65ef5383785ec21db52f6d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Executive Suite-N003 @ Lymack Apartments&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.62/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7109/night     &lt;/div&gt;`)[0];
                popup_6db40c46e2f94d6719dc39deacc231d8.setContent(html_d7e96009db65ef5383785ec21db52f6d);



        circle_marker_853eed3baeb68ae0919b8cc6afc3862a.bindPopup(popup_6db40c46e2f94d6719dc39deacc231d8)
        ;




            var circle_marker_7bcf613aa53a7645cbf7d2a5bfad56a0 = L.circleMarker(
                [-1.2124, 36.8759],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_eaf534e7d2183fbbae1c1dbc4beeb56a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6bd8f3cdd277ada8e009cceaeb52130e = $(`&lt;div id=&quot;html_6bd8f3cdd277ada8e009cceaeb52130e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;257 Place House 2&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11068/night     &lt;/div&gt;`)[0];
                popup_eaf534e7d2183fbbae1c1dbc4beeb56a.setContent(html_6bd8f3cdd277ada8e009cceaeb52130e);



        circle_marker_7bcf613aa53a7645cbf7d2a5bfad56a0.bindPopup(popup_eaf534e7d2183fbbae1c1dbc4beeb56a)
        ;




            var circle_marker_0ba35310eaefd6c7ee6795e605b89af6 = L.circleMarker(
                [-1.2481, 36.7652],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_062bef42c29414ed6d694b9444314564 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e574cb8c572b008398a423640e19c4f8 = $(`&lt;div id=&quot;html_e574cb8c572b008398a423640e19c4f8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nairobi Treehouse with a View&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Treehouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 14371/night     &lt;/div&gt;`)[0];
                popup_062bef42c29414ed6d694b9444314564.setContent(html_e574cb8c572b008398a423640e19c4f8);



        circle_marker_0ba35310eaefd6c7ee6795e605b89af6.bindPopup(popup_062bef42c29414ed6d694b9444314564)
        ;




            var circle_marker_2b5264685038ea316f32b34ffccdf094 = L.circleMarker(
                [-1.202, 36.7786],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_affc1ddb7072d7f9312b1f73b6753f84 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_fb7f83329887b4e0c73d33cd918957c5 = $(`&lt;div id=&quot;html_fb7f83329887b4e0c73d33cd918957c5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Perfect Haven: Near UN, USA Embassy, Village Mkt&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.98/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4211/night     &lt;/div&gt;`)[0];
                popup_affc1ddb7072d7f9312b1f73b6753f84.setContent(html_fb7f83329887b4e0c73d33cd918957c5);



        circle_marker_2b5264685038ea316f32b34ffccdf094.bindPopup(popup_affc1ddb7072d7f9312b1f73b6753f84)
        ;




            var circle_marker_376eb7a5a72accc92b9e957d7fde6fe8 = L.circleMarker(
                [-1.2926, 36.7888],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7aad27236496e765d60764361a7c92c5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6513ebc2e8062f35fa0a976c882a0c6f = $(`&lt;div id=&quot;html_6513ebc2e8062f35fa0a976c882a0c6f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern 2 bedroom apartment with a study room.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.17/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6342/night     &lt;/div&gt;`)[0];
                popup_7aad27236496e765d60764361a7c92c5.setContent(html_6513ebc2e8062f35fa0a976c882a0c6f);



        circle_marker_376eb7a5a72accc92b9e957d7fde6fe8.bindPopup(popup_7aad27236496e765d60764361a7c92c5)
        ;




            var circle_marker_7d540bf4bf83463305ba268f2fee673d = L.circleMarker(
                [-1.2614, 36.7893],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0e1a6cb3a90966ff2c901892d548115a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0c6a6176a35af6db23a87616bd9011c6 = $(`&lt;div id=&quot;html_0c6a6176a35af6db23a87616bd9011c6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Skyline View Two Bedroom Apt @ Le&#x27;Mac Church Road&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.5/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11247/night     &lt;/div&gt;`)[0];
                popup_0e1a6cb3a90966ff2c901892d548115a.setContent(html_0c6a6176a35af6db23a87616bd9011c6);



        circle_marker_7d540bf4bf83463305ba268f2fee673d.bindPopup(popup_0e1a6cb3a90966ff2c901892d548115a)
        ;




            var circle_marker_0e3b05843e666181e63dc16e434b7ee3 = L.circleMarker(
                [-1.2544, 36.8306],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_638ee4aeaee7ebe52f267f5b2145a44f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3bde72fa402a93c75b8f45ad83569c88 = $(`&lt;div id=&quot;html_3bde72fa402a93c75b8f45ad83569c88&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Beautiful Historic House by YourHost, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 15342/night     &lt;/div&gt;`)[0];
                popup_638ee4aeaee7ebe52f267f5b2145a44f.setContent(html_3bde72fa402a93c75b8f45ad83569c88);



        circle_marker_0e3b05843e666181e63dc16e434b7ee3.bindPopup(popup_638ee4aeaee7ebe52f267f5b2145a44f)
        ;




            var circle_marker_f47da95bcf135fa87f1a1115d78bdd21 = L.circleMarker(
                [-1.3742, 36.7544],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_237f57163c8c21b565e45a4858f66471 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b16b6f9728ff403647c1b903770e0993 = $(`&lt;div id=&quot;html_b16b6f9728ff403647c1b903770e0993&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ol Losowan Main House with Pool in Karen Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire chalet&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 88422/night     &lt;/div&gt;`)[0];
                popup_237f57163c8c21b565e45a4858f66471.setContent(html_b16b6f9728ff403647c1b903770e0993);



        circle_marker_f47da95bcf135fa87f1a1115d78bdd21.bindPopup(popup_237f57163c8c21b565e45a4858f66471)
        ;




            var circle_marker_e9887b18997317453794eff066a1378a = L.circleMarker(
                [-1.3783, 36.7576],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_be510362fd7372c65d8c9c6d3519ba35 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e91636a170690ec9456852f0a75fa159 = $(`&lt;div id=&quot;html_e91636a170690ec9456852f0a75fa159&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Jua Cottage with Pool in Karen Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cabin&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17197/night     &lt;/div&gt;`)[0];
                popup_be510362fd7372c65d8c9c6d3519ba35.setContent(html_e91636a170690ec9456852f0a75fa159);



        circle_marker_e9887b18997317453794eff066a1378a.bindPopup(popup_be510362fd7372c65d8c9c6d3519ba35)
        ;




            var circle_marker_f6a8e4d02efb93ceed979e4024ec74ba = L.circleMarker(
                [-1.2606, 36.7887],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8aaf1a0b874fedc7b311cf0a50f73a27 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bb9af57a5809bd66569caa8de4352c0e = $(`&lt;div id=&quot;html_bb9af57a5809bd66569caa8de4352c0e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lemac Furnished air conditioned Prime apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.94/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7383/night     &lt;/div&gt;`)[0];
                popup_8aaf1a0b874fedc7b311cf0a50f73a27.setContent(html_bb9af57a5809bd66569caa8de4352c0e);



        circle_marker_f6a8e4d02efb93ceed979e4024ec74ba.bindPopup(popup_8aaf1a0b874fedc7b311cf0a50f73a27)
        ;




            var circle_marker_e94b888c37b8fc0f491dcdd87ac156c8 = L.circleMarker(
                [-1.3272, 36.8403],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8022326275d862cf8cf9873d388c423c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_93819a5dedb9e662e6304bb99cc8a43e = $(`&lt;div id=&quot;html_93819a5dedb9e662e6304bb99cc8a43e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;South C Private Bed Room Close to SGR, JKIA &amp; CBD&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1717/night     &lt;/div&gt;`)[0];
                popup_8022326275d862cf8cf9873d388c423c.setContent(html_93819a5dedb9e662e6304bb99cc8a43e);



        circle_marker_e94b888c37b8fc0f491dcdd87ac156c8.bindPopup(popup_8022326275d862cf8cf9873d388c423c)
        ;




            var circle_marker_56644bc0911edcb6745a762004ad38d6 = L.circleMarker(
                [-1.261, 36.7872],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_38927d5584e7305152f3e5ce7e7c6288 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cd3db0f549d90c95386d417627e3c14d = $(`&lt;div id=&quot;html_cd3db0f549d90c95386d417627e3c14d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene and Luxury Living.  Self check in apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7092/night     &lt;/div&gt;`)[0];
                popup_38927d5584e7305152f3e5ce7e7c6288.setContent(html_cd3db0f549d90c95386d417627e3c14d);



        circle_marker_56644bc0911edcb6745a762004ad38d6.bindPopup(popup_38927d5584e7305152f3e5ce7e7c6288)
        ;




            var circle_marker_b84c90e5ce68ae643bef493da36365b7 = L.circleMarker(
                [-1.3363, 36.7159],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b32507683635bfbff2575b9879e111bb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c0c947baf0d02b51c48534f4dd5d62e3 = $(`&lt;div id=&quot;html_c0c947baf0d02b51c48534f4dd5d62e3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Enchanting One Bedroom Garden Suite, Karen, Kenya&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5962/night     &lt;/div&gt;`)[0];
                popup_b32507683635bfbff2575b9879e111bb.setContent(html_c0c947baf0d02b51c48534f4dd5d62e3);



        circle_marker_b84c90e5ce68ae643bef493da36365b7.bindPopup(popup_b32507683635bfbff2575b9879e111bb)
        ;




            var circle_marker_bbd48528a68106dfb45faee0963507d1 = L.circleMarker(
                [-1.2951, 36.7968],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5dda597a8a12a5f0e0012f0201028a6b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ee76bf4f90ca54bf19290e993e403b56 = $(`&lt;div id=&quot;html_ee76bf4f90ca54bf19290e993e403b56&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Q Samra 2 bedroom  fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6772/night     &lt;/div&gt;`)[0];
                popup_5dda597a8a12a5f0e0012f0201028a6b.setContent(html_ee76bf4f90ca54bf19290e993e403b56);



        circle_marker_bbd48528a68106dfb45faee0963507d1.bindPopup(popup_5dda597a8a12a5f0e0012f0201028a6b)
        ;




            var circle_marker_60c07afbc3006bea38ba70c3fa3572a3 = L.circleMarker(
                [-1.293, 36.7966],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_535ef80cf13564272bc782c14ed06526 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e5ff41ab6f70f7eb6578abda20cb96b1 = $(`&lt;div id=&quot;html_e5ff41ab6f70f7eb6578abda20cb96b1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;P Samra 2 bedroom Apt fully furnished &amp; Serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6651/night     &lt;/div&gt;`)[0];
                popup_535ef80cf13564272bc782c14ed06526.setContent(html_e5ff41ab6f70f7eb6578abda20cb96b1);



        circle_marker_60c07afbc3006bea38ba70c3fa3572a3.bindPopup(popup_535ef80cf13564272bc782c14ed06526)
        ;




            var circle_marker_963a8f9e5e15ad3a11e596d6b008270d = L.circleMarker(
                [-1.2612, 36.7889],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_26451f188d94ad95b86e0f56953ad8e4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cf34b2e22d26c5e089cfed3cb2a0129e = $(`&lt;div id=&quot;html_cf34b2e22d26c5e089cfed3cb2a0129e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxurious Corner Two Bedroom @Prime Living&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12387/night     &lt;/div&gt;`)[0];
                popup_26451f188d94ad95b86e0f56953ad8e4.setContent(html_cf34b2e22d26c5e089cfed3cb2a0129e);



        circle_marker_963a8f9e5e15ad3a11e596d6b008270d.bindPopup(popup_26451f188d94ad95b86e0f56953ad8e4)
        ;




            var circle_marker_e709abf623d9ea3c3961aed04df5419a = L.circleMarker(
                [-1.2949, 36.7954],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6573d13f97d0055a24f3d7533dfc7ac7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_fc44ebb5ee67539c3a3a6f5b37b55f65 = $(`&lt;div id=&quot;html_fc44ebb5ee67539c3a3a6f5b37b55f65&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;S Samra 2 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5914/night     &lt;/div&gt;`)[0];
                popup_6573d13f97d0055a24f3d7533dfc7ac7.setContent(html_fc44ebb5ee67539c3a3a6f5b37b55f65);



        circle_marker_e709abf623d9ea3c3961aed04df5419a.bindPopup(popup_6573d13f97d0055a24f3d7533dfc7ac7)
        ;




            var circle_marker_b8ff6264d22fd53c79f6601cdc975b6c = L.circleMarker(
                [-1.2945, 36.7965],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4aca525913e1563c7c80a4e000be16d9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1505189e05b2d3f829d7966da3a777d0 = $(`&lt;div id=&quot;html_1505189e05b2d3f829d7966da3a777d0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;C6 Samra 1 bedroom Apt fully furnished serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.74/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4441/night     &lt;/div&gt;`)[0];
                popup_4aca525913e1563c7c80a4e000be16d9.setContent(html_1505189e05b2d3f829d7966da3a777d0);



        circle_marker_b8ff6264d22fd53c79f6601cdc975b6c.bindPopup(popup_4aca525913e1563c7c80a4e000be16d9)
        ;




            var circle_marker_edfad2463b8c1c93fd406ad7756bc66f = L.circleMarker(
                [-1.2945, 36.795],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0f253a9a4c34ffb5d83bf0a476cb2cf9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_71ba783427fa8ef263840fbd1036558e = $(`&lt;div id=&quot;html_71ba783427fa8ef263840fbd1036558e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Y Samra 1 bedroom apartment  furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4680/night     &lt;/div&gt;`)[0];
                popup_0f253a9a4c34ffb5d83bf0a476cb2cf9.setContent(html_71ba783427fa8ef263840fbd1036558e);



        circle_marker_edfad2463b8c1c93fd406ad7756bc66f.bindPopup(popup_0f253a9a4c34ffb5d83bf0a476cb2cf9)
        ;




            var circle_marker_6140c084336a9f6b5a7e3c568859ab31 = L.circleMarker(
                [-1.293, 36.7953],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f80c3bf71a7154246f01954b2f8c4e03 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d1306e5a8d7de7faff2a1e813753fbf6 = $(`&lt;div id=&quot;html_d1306e5a8d7de7faff2a1e813753fbf6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;A1 Samra 2 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.5/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6524/night     &lt;/div&gt;`)[0];
                popup_f80c3bf71a7154246f01954b2f8c4e03.setContent(html_d1306e5a8d7de7faff2a1e813753fbf6);



        circle_marker_6140c084336a9f6b5a7e3c568859ab31.bindPopup(popup_f80c3bf71a7154246f01954b2f8c4e03)
        ;




            var circle_marker_c424ae8e2f3a62bec952d2513309684a = L.circleMarker(
                [-1.295, 36.7967],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4aa2abbf5e522033f5a0dfc8be77b03d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_949791b379d59e8b1e5d05169ec6b979 = $(`&lt;div id=&quot;html_949791b379d59e8b1e5d05169ec6b979&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;A3 Samra 2 bedroom Apt fully furnished serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.7/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6035/night     &lt;/div&gt;`)[0];
                popup_4aa2abbf5e522033f5a0dfc8be77b03d.setContent(html_949791b379d59e8b1e5d05169ec6b979);



        circle_marker_c424ae8e2f3a62bec952d2513309684a.bindPopup(popup_4aa2abbf5e522033f5a0dfc8be77b03d)
        ;




            var circle_marker_a9d1ec25cb6d5f95c6d03ae013f18aaf = L.circleMarker(
                [-1.295, 36.7954],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fe1c7b24fea87ba58e9d310d8ee91e3b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_80b70f499e1aece27c36e24b10ce033d = $(`&lt;div id=&quot;html_80b70f499e1aece27c36e24b10ce033d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;G Samra 3 bedroom Apt fully furnished serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6245/night     &lt;/div&gt;`)[0];
                popup_fe1c7b24fea87ba58e9d310d8ee91e3b.setContent(html_80b70f499e1aece27c36e24b10ce033d);



        circle_marker_a9d1ec25cb6d5f95c6d03ae013f18aaf.bindPopup(popup_fe1c7b24fea87ba58e9d310d8ee91e3b)
        ;




            var circle_marker_9a22569af90d20db156de4493f7536c5 = L.circleMarker(
                [-1.295, 36.7953],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_74926276d70a9a4035a90034698cfe89 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_46477d0c4d9ef3b5b2e8519ebf0c71ac = $(`&lt;div id=&quot;html_46477d0c4d9ef3b5b2e8519ebf0c71ac&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;I  Samra 3 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.52/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5676/night     &lt;/div&gt;`)[0];
                popup_74926276d70a9a4035a90034698cfe89.setContent(html_46477d0c4d9ef3b5b2e8519ebf0c71ac);



        circle_marker_9a22569af90d20db156de4493f7536c5.bindPopup(popup_74926276d70a9a4035a90034698cfe89)
        ;




            var circle_marker_f2e6b5c587b24c55ded0300a6661e88f = L.circleMarker(
                [-1.2931, 36.7971],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0e774b555674e03fa6517b25d6426566 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_32fe63f0fe7f600af800aaa0d45a1f82 = $(`&lt;div id=&quot;html_32fe63f0fe7f600af800aaa0d45a1f82&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;T Samra 2 bedroom Apt fully furnished serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6220/night     &lt;/div&gt;`)[0];
                popup_0e774b555674e03fa6517b25d6426566.setContent(html_32fe63f0fe7f600af800aaa0d45a1f82);



        circle_marker_f2e6b5c587b24c55ded0300a6661e88f.bindPopup(popup_0e774b555674e03fa6517b25d6426566)
        ;




            var circle_marker_cdb9e9ea1ddb4b2094148d5dde57237a = L.circleMarker(
                [-1.2944, 36.7971],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_99f87127ac65de62f962d080b944dff5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_71b57da9ea46cc3e74adf74f1e2e4873 = $(`&lt;div id=&quot;html_71b57da9ea46cc3e74adf74f1e2e4873&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;C Samra 3 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6081/night     &lt;/div&gt;`)[0];
                popup_99f87127ac65de62f962d080b944dff5.setContent(html_71b57da9ea46cc3e74adf74f1e2e4873);



        circle_marker_cdb9e9ea1ddb4b2094148d5dde57237a.bindPopup(popup_99f87127ac65de62f962d080b944dff5)
        ;




            var circle_marker_6e51914bb6abbad65832074cd78c1167 = L.circleMarker(
                [-1.2945, 36.7949],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_af1295172ce1575a727f2c969d6c786b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_753970ae3e06f4447d32af93e28d9e54 = $(`&lt;div id=&quot;html_753970ae3e06f4447d32af93e28d9e54&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;B1  1 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.65/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4461/night     &lt;/div&gt;`)[0];
                popup_af1295172ce1575a727f2c969d6c786b.setContent(html_753970ae3e06f4447d32af93e28d9e54);



        circle_marker_6e51914bb6abbad65832074cd78c1167.bindPopup(popup_af1295172ce1575a727f2c969d6c786b)
        ;




            var circle_marker_08ed05a287e5359304d6062a6729f07e = L.circleMarker(
                [-1.2936, 36.7972],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_606217ebb000fe34d32f95743b583131 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6f6d4435966cbbea02824c30cc6bd7ad = $(`&lt;div id=&quot;html_6f6d4435966cbbea02824c30cc6bd7ad&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;A4 Samra 2 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5991/night     &lt;/div&gt;`)[0];
                popup_606217ebb000fe34d32f95743b583131.setContent(html_6f6d4435966cbbea02824c30cc6bd7ad);



        circle_marker_08ed05a287e5359304d6062a6729f07e.bindPopup(popup_606217ebb000fe34d32f95743b583131)
        ;




            var circle_marker_ba3c1493112ee8398697ac611e70ae4d = L.circleMarker(
                [-1.2948, 36.7964],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3faed1b53b05b336eb244d1c658acd21 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d33e6770394b042e914913cc11dbbe35 = $(`&lt;div id=&quot;html_d33e6770394b042e914913cc11dbbe35&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;R Samra 2 bedroom Apt fully furnished &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.68/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6026/night     &lt;/div&gt;`)[0];
                popup_3faed1b53b05b336eb244d1c658acd21.setContent(html_d33e6770394b042e914913cc11dbbe35);



        circle_marker_ba3c1493112ee8398697ac611e70ae4d.bindPopup(popup_3faed1b53b05b336eb244d1c658acd21)
        ;




            var circle_marker_38297abb623d23807409f222411c14ba = L.circleMarker(
                [-1.2936, 36.7971],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b7d96918b9ad4fe73829e8fc3716c128 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f5b29712b22587ac0638b416f524c303 = $(`&lt;div id=&quot;html_f5b29712b22587ac0638b416f524c303&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;B Samra 2 bedroom Apt fully furnish &amp; serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4526/night     &lt;/div&gt;`)[0];
                popup_b7d96918b9ad4fe73829e8fc3716c128.setContent(html_f5b29712b22587ac0638b416f524c303);



        circle_marker_38297abb623d23807409f222411c14ba.bindPopup(popup_b7d96918b9ad4fe73829e8fc3716c128)
        ;




            var circle_marker_cb1934a3473f5996604feaa28cad536e = L.circleMarker(
                [-1.295, 36.7959],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_99eac05c73c391238083f9f3e498bdb1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c5bcc888952cc42f7e184e9578b59a9c = $(`&lt;div id=&quot;html_c5bcc888952cc42f7e184e9578b59a9c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;K Samra 3 bedroom Apt Fully furnished &amp; Serviced&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.68/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6400/night     &lt;/div&gt;`)[0];
                popup_99eac05c73c391238083f9f3e498bdb1.setContent(html_c5bcc888952cc42f7e184e9578b59a9c);



        circle_marker_cb1934a3473f5996604feaa28cad536e.bindPopup(popup_99eac05c73c391238083f9f3e498bdb1)
        ;




            var circle_marker_22fd6c296dbcb923cded24decfb48d6a = L.circleMarker(
                [-1.2673, 36.8065],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4c475cc74fe596cbe756ec44675b0d7e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b9d988046677f22d1d2a18d5a420e215 = $(`&lt;div id=&quot;html_b9d988046677f22d1d2a18d5a420e215&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;W Place Close Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6069/night     &lt;/div&gt;`)[0];
                popup_4c475cc74fe596cbe756ec44675b0d7e.setContent(html_b9d988046677f22d1d2a18d5a420e215);



        circle_marker_22fd6c296dbcb923cded24decfb48d6a.bindPopup(popup_4c475cc74fe596cbe756ec44675b0d7e)
        ;




            var circle_marker_f2c721b28416995e52e22c57972a5c80 = L.circleMarker(
                [-1.3419, 36.6952],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a6cee013beb791b54abaa55aeeb33921 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a62d9ca600be7be1cce3a35918472258 = $(`&lt;div id=&quot;html_a62d9ca600be7be1cce3a35918472258&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Nest in Karen&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10986/night     &lt;/div&gt;`)[0];
                popup_a6cee013beb791b54abaa55aeeb33921.setContent(html_a62d9ca600be7be1cce3a35918472258);



        circle_marker_f2c721b28416995e52e22c57972a5c80.bindPopup(popup_a6cee013beb791b54abaa55aeeb33921)
        ;




            var circle_marker_837c8a5f92c4b24ba6fa97cca6220b3b = L.circleMarker(
                [-1.263, 36.7885],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ac11576d8b8cc049b54432578ee0343e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_40018de8ebf1766b8832d7862c9c455f = $(`&lt;div id=&quot;html_40018de8ebf1766b8832d7862c9c455f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Prime Living Two Bedroom @Church Road LeMac&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12031/night     &lt;/div&gt;`)[0];
                popup_ac11576d8b8cc049b54432578ee0343e.setContent(html_40018de8ebf1766b8832d7862c9c455f);



        circle_marker_837c8a5f92c4b24ba6fa97cca6220b3b.bindPopup(popup_ac11576d8b8cc049b54432578ee0343e)
        ;




            var circle_marker_ff6a658bb2eafca329896c70a2deba96 = L.circleMarker(
                [-1.2706, 36.8103],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_aee63a0901cbe7924d71d4613cabba3e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a32a7506f03a4355cf681c49c8a2e2b0 = $(`&lt;div id=&quot;html_a32a7506f03a4355cf681c49c8a2e2b0&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stylish Living BNB Furnished Apt Westlands Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4795/night     &lt;/div&gt;`)[0];
                popup_aee63a0901cbe7924d71d4613cabba3e.setContent(html_a32a7506f03a4355cf681c49c8a2e2b0);



        circle_marker_ff6a658bb2eafca329896c70a2deba96.bindPopup(popup_aee63a0901cbe7924d71d4613cabba3e)
        ;




            var circle_marker_7b31c14a2b9fdf50ce7ac3b2ae058048 = L.circleMarker(
                [-1.2341, 36.8598],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ea2962d72a4f1bc3ecb42ec835edc725 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3f413dd4e7851203cffee517a69819d8 = $(`&lt;div id=&quot;html_3f413dd4e7851203cffee517a69819d8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Haven: Serene, Secure, Garden access 1br guesthous&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3479/night     &lt;/div&gt;`)[0];
                popup_ea2962d72a4f1bc3ecb42ec835edc725.setContent(html_3f413dd4e7851203cffee517a69819d8);



        circle_marker_7b31c14a2b9fdf50ce7ac3b2ae058048.bindPopup(popup_ea2962d72a4f1bc3ecb42ec835edc725)
        ;




            var circle_marker_2c521545ae26255ef4b36b7a86c3ecb4 = L.circleMarker(
                [-1.2912, 36.7791],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4e0c82b82b591067c238d6c971a799c7 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6e9fd4cdfa3e4038973aa7021995a0aa = $(`&lt;div id=&quot;html_6e9fd4cdfa3e4038973aa7021995a0aa&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;1 Bedroom - BROOKVIEW APARTMENTS - Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3743/night     &lt;/div&gt;`)[0];
                popup_4e0c82b82b591067c238d6c971a799c7.setContent(html_6e9fd4cdfa3e4038973aa7021995a0aa);



        circle_marker_2c521545ae26255ef4b36b7a86c3ecb4.bindPopup(popup_4e0c82b82b591067c238d6c971a799c7)
        ;




            var circle_marker_c3dea8b94e790bc8e266cb98a794d658 = L.circleMarker(
                [-1.2341, 36.8609],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7d3b5e199998a8074a01f4502fb4306b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_add0fda55d8eda93b7ea1ef6bee397fa = $(`&lt;div id=&quot;html_add0fda55d8eda93b7ea1ef6bee397fa&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Top Haven: Cozy, Modern, 1bedroom +  Garden access&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3518/night     &lt;/div&gt;`)[0];
                popup_7d3b5e199998a8074a01f4502fb4306b.setContent(html_add0fda55d8eda93b7ea1ef6bee397fa);



        circle_marker_c3dea8b94e790bc8e266cb98a794d658.bindPopup(popup_7d3b5e199998a8074a01f4502fb4306b)
        ;




            var circle_marker_4e0b36d0bd871b9499cad4d198ffcba0 = L.circleMarker(
                [-1.2318, 36.7892],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_26f4742093418c459c836684a94fa6eb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e3577ed3579de84fdad3d6c408e89f44 = $(`&lt;div id=&quot;html_e3577ed3579de84fdad3d6c408e89f44&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Sinatra homes&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.92/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12699/night     &lt;/div&gt;`)[0];
                popup_26f4742093418c459c836684a94fa6eb.setContent(html_e3577ed3579de84fdad3d6c408e89f44);



        circle_marker_4e0b36d0bd871b9499cad4d198ffcba0.bindPopup(popup_26f4742093418c459c836684a94fa6eb)
        ;




            var circle_marker_71559403827045466bdcf1bf476cbebf = L.circleMarker(
                [-1.2786, 36.8296],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_53d91c6bb1a4ce84743d3dad5f4ee2ab = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3c6383b577e33c1e94311ebdefdf75a3 = $(`&lt;div id=&quot;html_3c6383b577e33c1e94311ebdefdf75a3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Just Good Vibes&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2171/night     &lt;/div&gt;`)[0];
                popup_53d91c6bb1a4ce84743d3dad5f4ee2ab.setContent(html_3c6383b577e33c1e94311ebdefdf75a3);



        circle_marker_71559403827045466bdcf1bf476cbebf.bindPopup(popup_53d91c6bb1a4ce84743d3dad5f4ee2ab)
        ;




            var circle_marker_b352075be79cd2fb120f2bf14a8889e8 = L.circleMarker(
                [-1.3796, 36.7562],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ddf782d200ca33380f8336e1685ecd1f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_163d3333438f6f26e3637c3797278b4c = $(`&lt;div id=&quot;html_163d3333438f6f26e3637c3797278b4c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Ol Losowan- Bahari Cottage with Pool Karen Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cabin&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.94/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17493/night     &lt;/div&gt;`)[0];
                popup_ddf782d200ca33380f8336e1685ecd1f.setContent(html_163d3333438f6f26e3637c3797278b4c);



        circle_marker_b352075be79cd2fb120f2bf14a8889e8.bindPopup(popup_ddf782d200ca33380f8336e1685ecd1f)
        ;




            var circle_marker_7d56ae68df3d79f56b397c3fbcf905d3 = L.circleMarker(
                [-1.2923, 36.7785],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_49ba0b472fa5c5dd9ff25c7258168cf3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6c72a574c91fc4b216d0de6eee035bea = $(`&lt;div id=&quot;html_6c72a574c91fc4b216d0de6eee035bea&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Fabulous 5 BR Afro Chic Duplex Near Yaya Mall&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9096/night     &lt;/div&gt;`)[0];
                popup_49ba0b472fa5c5dd9ff25c7258168cf3.setContent(html_6c72a574c91fc4b216d0de6eee035bea);



        circle_marker_7d56ae68df3d79f56b397c3fbcf905d3.bindPopup(popup_49ba0b472fa5c5dd9ff25c7258168cf3)
        ;




            var circle_marker_cc88c0dd7402942985b01f6dec763b1d = L.circleMarker(
                [-1.2335, 36.8039],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6ee47a6aa3ea965502136d5e82f6d62c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b60a10a67e5fc606344c28d6d2f562e1 = $(`&lt;div id=&quot;html_b60a10a67e5fc606344c28d6d2f562e1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Norah-Keushi Unique and Cozy, Home  away from home&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in villa&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9198/night     &lt;/div&gt;`)[0];
                popup_6ee47a6aa3ea965502136d5e82f6d62c.setContent(html_b60a10a67e5fc606344c28d6d2f562e1);



        circle_marker_cc88c0dd7402942985b01f6dec763b1d.bindPopup(popup_6ee47a6aa3ea965502136d5e82f6d62c)
        ;




            var circle_marker_66228a45ea71b41b81ba83d624c2ed8e = L.circleMarker(
                [-1.2975, 36.8118],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d5fe1074a77265fe603f53639437e2b6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_06908ab9d2fe482d4164eca36ef84c3c = $(`&lt;div id=&quot;html_06908ab9d2fe482d4164eca36ef84c3c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Emja Homes&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6444/night     &lt;/div&gt;`)[0];
                popup_d5fe1074a77265fe603f53639437e2b6.setContent(html_06908ab9d2fe482d4164eca36ef84c3c);



        circle_marker_66228a45ea71b41b81ba83d624c2ed8e.bindPopup(popup_d5fe1074a77265fe603f53639437e2b6)
        ;




            var circle_marker_f1b9f7d1c522e8aee5c4f41ded08455d = L.circleMarker(
                [-1.3161, 36.8426],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2510c9bb90a799bd1379bb4d63f4eb35 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ee9482f0c11abfeb6985ec1c204760b1 = $(`&lt;div id=&quot;html_ee9482f0c11abfeb6985ec1c204760b1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Homely stay, 15mins to JKIA, 10mins to Nrbi City&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.81/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2612/night     &lt;/div&gt;`)[0];
                popup_2510c9bb90a799bd1379bb4d63f4eb35.setContent(html_ee9482f0c11abfeb6985ec1c204760b1);



        circle_marker_f1b9f7d1c522e8aee5c4f41ded08455d.bindPopup(popup_2510c9bb90a799bd1379bb4d63f4eb35)
        ;




            var circle_marker_c34088d34d6deec4a704024fa173130a = L.circleMarker(
                [-1.2723, 36.9056],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8890f6df1a93fc1e72141f33a5082ea3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d7a0b7c1475fde8c4695ca80db347067 = $(`&lt;div id=&quot;html_d7a0b7c1475fde8c4695ca80db347067&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;K Heights cosy &amp; private Apartments&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.76/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4297/night     &lt;/div&gt;`)[0];
                popup_8890f6df1a93fc1e72141f33a5082ea3.setContent(html_d7a0b7c1475fde8c4695ca80db347067);



        circle_marker_c34088d34d6deec4a704024fa173130a.bindPopup(popup_8890f6df1a93fc1e72141f33a5082ea3)
        ;




            var circle_marker_aa8b42a4561cca41baa4c0f7d3acdc11 = L.circleMarker(
                [-1.2912, 36.779],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e326ac059ef3681e2072fb01748e0a12 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_268c971528178b82c49612678561144a = $(`&lt;div id=&quot;html_268c971528178b82c49612678561144a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;1 Bedroom Brookview Apartments in Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3557/night     &lt;/div&gt;`)[0];
                popup_e326ac059ef3681e2072fb01748e0a12.setContent(html_268c971528178b82c49612678561144a);



        circle_marker_aa8b42a4561cca41baa4c0f7d3acdc11.bindPopup(popup_e326ac059ef3681e2072fb01748e0a12)
        ;




            var circle_marker_eecb89760eb908a50a930d2ae2217f3d = L.circleMarker(
                [-1.263, 36.7892],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ead76082999164e1786f1b16cd6b451c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_82b89d4849c7d0a08ac9e42187f8a8a2 = $(`&lt;div id=&quot;html_82b89d4849c7d0a08ac9e42187f8a8a2&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stylish, modern 2-bd apt in the heart of Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.59/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12274/night     &lt;/div&gt;`)[0];
                popup_ead76082999164e1786f1b16cd6b451c.setContent(html_82b89d4849c7d0a08ac9e42187f8a8a2);



        circle_marker_eecb89760eb908a50a930d2ae2217f3d.bindPopup(popup_ead76082999164e1786f1b16cd6b451c)
        ;




            var circle_marker_30d04b400d9ac24c17336b37594f8223 = L.circleMarker(
                [-1.3125, 36.9096],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c6311afefb1a4766a6ec4561cc230d00 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1c442576bf81498e803a0db2c5fbd57d = $(`&lt;div id=&quot;html_1c442576bf81498e803a0db2c5fbd57d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Jane&#x27;s Nyayo Estate Embakasi Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7173/night     &lt;/div&gt;`)[0];
                popup_c6311afefb1a4766a6ec4561cc230d00.setContent(html_1c442576bf81498e803a0db2c5fbd57d);



        circle_marker_30d04b400d9ac24c17336b37594f8223.bindPopup(popup_c6311afefb1a4766a6ec4561cc230d00)
        ;




            var circle_marker_56594be3000aa99745d440b3dd2d4c0e = L.circleMarker(
                [-1.2676, 36.7356],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1aa60f725a3b02ba9474446d485d8201 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_850f8ea81f0b9d02fee663a1bd668db3 = $(`&lt;div id=&quot;html_850f8ea81f0b9d02fee663a1bd668db3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Explore Nairobi from Private Homely Bungalow&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire bungalow&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3690/night     &lt;/div&gt;`)[0];
                popup_1aa60f725a3b02ba9474446d485d8201.setContent(html_850f8ea81f0b9d02fee663a1bd668db3);



        circle_marker_56594be3000aa99745d440b3dd2d4c0e.bindPopup(popup_1aa60f725a3b02ba9474446d485d8201)
        ;




            var circle_marker_045f13a435f0c6141446a22f5e459428 = L.circleMarker(
                [-1.3046, 36.8133],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8f951aa34a8afcaca9591edde8a37aa2 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_730ff10c7e929c31b09a2c933d7b2c67 = $(`&lt;div id=&quot;html_730ff10c7e929c31b09a2c933d7b2c67&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Executive  Flat  : One Bedroom  with Home Office&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8610/night     &lt;/div&gt;`)[0];
                popup_8f951aa34a8afcaca9591edde8a37aa2.setContent(html_730ff10c7e929c31b09a2c933d7b2c67);



        circle_marker_045f13a435f0c6141446a22f5e459428.bindPopup(popup_8f951aa34a8afcaca9591edde8a37aa2)
        ;




            var circle_marker_1943427646e812a75526c44e87848db2 = L.circleMarker(
                [-1.2855, 36.7684],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_518b9111991d462e894c474acc8faa8e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6c7cc975c5d94ba63224194854c52445 = $(`&lt;div id=&quot;html_6c7cc975c5d94ba63224194854c52445&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Home Away from Home, Spacious 1 Bedroom, Braeside&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.61/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4598/night     &lt;/div&gt;`)[0];
                popup_518b9111991d462e894c474acc8faa8e.setContent(html_6c7cc975c5d94ba63224194854c52445);



        circle_marker_1943427646e812a75526c44e87848db2.bindPopup(popup_518b9111991d462e894c474acc8faa8e)
        ;




            var circle_marker_4dbcccd94a220c83634f2671d7ce2b7a = L.circleMarker(
                [-1.3274, 36.7169],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7f47efee0d612ab5f4265ea60ae10832 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e388193067ade639aa367955bf4d30ea = $(`&lt;div id=&quot;html_e388193067ade639aa367955bf4d30ea&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stylish and airy flat in the heart of Karen&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6383/night     &lt;/div&gt;`)[0];
                popup_7f47efee0d612ab5f4265ea60ae10832.setContent(html_e388193067ade639aa367955bf4d30ea);



        circle_marker_4dbcccd94a220c83634f2671d7ce2b7a.bindPopup(popup_7f47efee0d612ab5f4265ea60ae10832)
        ;




            var circle_marker_3b9cea2fd6f48e8c49cb2bfe61f5287e = L.circleMarker(
                [-1.3127, 36.6736],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_638bde9188e3aba76cbb2055899843a5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b110a3e9bec56846cc2433aed9e84414 = $(`&lt;div id=&quot;html_b110a3e9bec56846cc2433aed9e84414&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Poolside 2BR Apt @ the Jungle Oasis w/ heated pool&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 18555/night     &lt;/div&gt;`)[0];
                popup_638bde9188e3aba76cbb2055899843a5.setContent(html_b110a3e9bec56846cc2433aed9e84414);



        circle_marker_3b9cea2fd6f48e8c49cb2bfe61f5287e.bindPopup(popup_638bde9188e3aba76cbb2055899843a5)
        ;




            var circle_marker_7d6b632e41802fc45885e0a36119cbd7 = L.circleMarker(
                [-1.3273, 36.7332],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_cdf857387a3f3d4eac8ea8786120aff1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_331392e3b1671a0bfdedec2acff3f444 = $(`&lt;div id=&quot;html_331392e3b1671a0bfdedec2acff3f444&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Comfortable and stylish - Maison Mitwaba, Room 3&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in bed and breakfast&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8607/night     &lt;/div&gt;`)[0];
                popup_cdf857387a3f3d4eac8ea8786120aff1.setContent(html_331392e3b1671a0bfdedec2acff3f444);



        circle_marker_7d6b632e41802fc45885e0a36119cbd7.bindPopup(popup_cdf857387a3f3d4eac8ea8786120aff1)
        ;




            var circle_marker_7961e880df12864d9b8eb8ea58175601 = L.circleMarker(
                [-1.2944, 36.7853],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3b8e52fd8290be0a6b9404825801f54f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7e9c2d2cf4f9ec1414e873157adf5b8c = $(`&lt;div id=&quot;html_7e9c2d2cf4f9ec1414e873157adf5b8c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The ChatRoom , Wood Avenue , Yaya center, Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.77/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6609/night     &lt;/div&gt;`)[0];
                popup_3b8e52fd8290be0a6b9404825801f54f.setContent(html_7e9c2d2cf4f9ec1414e873157adf5b8c);



        circle_marker_7961e880df12864d9b8eb8ea58175601.bindPopup(popup_3b8e52fd8290be0a6b9404825801f54f)
        ;




            var circle_marker_4b510507c914552771838600d6f8c66c = L.circleMarker(
                [-1.2859, 36.7838],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_14d47b49838cb484d9acdc59f0a64e7a = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5c44020cd58774c4f0d35736a391efbb = $(`&lt;div id=&quot;html_5c44020cd58774c4f0d35736a391efbb&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Amazing 1BR nestled in quiet leafy Nairobi!&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.92/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6172/night     &lt;/div&gt;`)[0];
                popup_14d47b49838cb484d9acdc59f0a64e7a.setContent(html_5c44020cd58774c4f0d35736a391efbb);



        circle_marker_4b510507c914552771838600d6f8c66c.bindPopup(popup_14d47b49838cb484d9acdc59f0a64e7a)
        ;




            var circle_marker_6d053ed18b127b74a9665c38a7c95b78 = L.circleMarker(
                [-1.2967, 36.7906],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e6f432e2899b825cbb1b79628c572838 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c7d28400e62c0857eb0f43b6a3cc8c1b = $(`&lt;div id=&quot;html_c7d28400e62c0857eb0f43b6a3cc8c1b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Homely Apartment: Yaya Kilimani, close Yaya Center&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8253/night     &lt;/div&gt;`)[0];
                popup_e6f432e2899b825cbb1b79628c572838.setContent(html_c7d28400e62c0857eb0f43b6a3cc8c1b);



        circle_marker_6d053ed18b127b74a9665c38a7c95b78.bindPopup(popup_e6f432e2899b825cbb1b79628c572838)
        ;




            var circle_marker_5a36820cdd4b67f948cf8a956c2a780b = L.circleMarker(
                [-1.2935, 36.7637],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0801ecf04b4ecc907fdc8bd37eb10488 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_90c89a3c64d72687b93f86b2d05903af = $(`&lt;div id=&quot;html_90c89a3c64d72687b93f86b2d05903af&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Astoria Apartments 1 Br With a gym &amp;Pool Lavington&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.45/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7804/night     &lt;/div&gt;`)[0];
                popup_0801ecf04b4ecc907fdc8bd37eb10488.setContent(html_90c89a3c64d72687b93f86b2d05903af);



        circle_marker_5a36820cdd4b67f948cf8a956c2a780b.bindPopup(popup_0801ecf04b4ecc907fdc8bd37eb10488)
        ;




            var circle_marker_04f130e175658b452a61eb796edf1035 = L.circleMarker(
                [-1.2702, 36.8],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4712090c1d07316d3f9f055d31af45a4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1f213b0f10bd111831eabcb638a7e830 = $(`&lt;div id=&quot;html_1f213b0f10bd111831eabcb638a7e830&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Erica Residences - Riverside, 3 Bedroom Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.64/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11565/night     &lt;/div&gt;`)[0];
                popup_4712090c1d07316d3f9f055d31af45a4.setContent(html_1f213b0f10bd111831eabcb638a7e830);



        circle_marker_04f130e175658b452a61eb796edf1035.bindPopup(popup_4712090c1d07316d3f9f055d31af45a4)
        ;




            var circle_marker_67fd51cd4da866f582c29807763bbaf1 = L.circleMarker(
                [-1.2927, 36.8226],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_933b6911389feac0c2c8e5bfc941ff4b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_029e6600b9f26700ef9ac1415abac95d = $(`&lt;div id=&quot;html_029e6600b9f26700ef9ac1415abac95d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Waterfall side home stay UN Runda (Nairobi).&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in nature lodge&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7290/night     &lt;/div&gt;`)[0];
                popup_933b6911389feac0c2c8e5bfc941ff4b.setContent(html_029e6600b9f26700ef9ac1415abac95d);



        circle_marker_67fd51cd4da866f582c29807763bbaf1.bindPopup(popup_933b6911389feac0c2c8e5bfc941ff4b)
        ;




            var circle_marker_c762e9a22b42d9abff99b43d95ce36ba = L.circleMarker(
                [-1.2645, 36.7874],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4573a8fa0f7ad285f8ca4e3c6dd39b51 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ce70d19ceed1f9efd548b20ab354c678 = $(`&lt;div id=&quot;html_ce70d19ceed1f9efd548b20ab354c678&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Apartment near Artcaffe &amp; LeMac_ Westlands Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.94/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2877/night     &lt;/div&gt;`)[0];
                popup_4573a8fa0f7ad285f8ca4e3c6dd39b51.setContent(html_ce70d19ceed1f9efd548b20ab354c678);



        circle_marker_c762e9a22b42d9abff99b43d95ce36ba.bindPopup(popup_4573a8fa0f7ad285f8ca4e3c6dd39b51)
        ;




            var circle_marker_c8ad0a9195ec8b52da9e7d788eb8cb65 = L.circleMarker(
                [-1.2771, 36.8293],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6c1ddf4fd8788bb44380227502e2ea14 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_83bed44f82e23af9171febd1ce8e055a = $(`&lt;div id=&quot;html_83bed44f82e23af9171febd1ce8e055a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Yellow Pearl Luxury Gem Apartment in Nairobi Ngara&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.29/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1860/night     &lt;/div&gt;`)[0];
                popup_6c1ddf4fd8788bb44380227502e2ea14.setContent(html_83bed44f82e23af9171febd1ce8e055a);



        circle_marker_c8ad0a9195ec8b52da9e7d788eb8cb65.bindPopup(popup_6c1ddf4fd8788bb44380227502e2ea14)
        ;




            var circle_marker_c6eb049042e139bcb9c3d81ad610d696 = L.circleMarker(
                [-1.2626, 36.8076],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6b2383607eb58a0ac0baff980e3a89e5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c3ea433309cfe0bd26f40803542bdd82 = $(`&lt;div id=&quot;html_c3ea433309cfe0bd26f40803542bdd82&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lovely studio apartment with a nice  street view&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.69/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5145/night     &lt;/div&gt;`)[0];
                popup_6b2383607eb58a0ac0baff980e3a89e5.setContent(html_c3ea433309cfe0bd26f40803542bdd82);



        circle_marker_c6eb049042e139bcb9c3d81ad610d696.bindPopup(popup_6b2383607eb58a0ac0baff980e3a89e5)
        ;




            var circle_marker_73300dc07bc7cee91d07da7742d90c42 = L.circleMarker(
                [-1.3059, 36.8206],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_df90e892be6fc6cad740892e85f1bc3b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0890ed2714d9bd5e4a68e9854d466d4c = $(`&lt;div id=&quot;html_0890ed2714d9bd5e4a68e9854d466d4c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lotus suite&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2917/night     &lt;/div&gt;`)[0];
                popup_df90e892be6fc6cad740892e85f1bc3b.setContent(html_0890ed2714d9bd5e4a68e9854d466d4c);



        circle_marker_73300dc07bc7cee91d07da7742d90c42.bindPopup(popup_df90e892be6fc6cad740892e85f1bc3b)
        ;




            var circle_marker_dae859bd176267fc6fe02d5c18c59d68 = L.circleMarker(
                [-1.2969, 36.7877],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a0202230b810aa1480b00837c9a738d6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e037b5d7a6f6334c1cbbfb855627f750 = $(`&lt;div id=&quot;html_e037b5d7a6f6334c1cbbfb855627f750&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Elegant,Family friendly 3BR near Yaya Center&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.48/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9178/night     &lt;/div&gt;`)[0];
                popup_a0202230b810aa1480b00837c9a738d6.setContent(html_e037b5d7a6f6334c1cbbfb855627f750);



        circle_marker_dae859bd176267fc6fe02d5c18c59d68.bindPopup(popup_a0202230b810aa1480b00837c9a738d6)
        ;




            var circle_marker_3f293ec66e7a4c4c674a6a82f1ff5dcc = L.circleMarker(
                [-1.314, 36.8179],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0990a717b08043741907347b98d71bc1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_42e7bb4e1a28129a23f1e640a4dd13f6 = $(`&lt;div id=&quot;html_42e7bb4e1a28129a23f1e640a4dd13f6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Studio-Living Suite&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2868/night     &lt;/div&gt;`)[0];
                popup_0990a717b08043741907347b98d71bc1.setContent(html_42e7bb4e1a28129a23f1e640a4dd13f6);



        circle_marker_3f293ec66e7a4c4c674a6a82f1ff5dcc.bindPopup(popup_0990a717b08043741907347b98d71bc1)
        ;




            var circle_marker_5499ab0f67de5ac88f7b37b9315e5f35 = L.circleMarker(
                [-1.2743, 36.7904],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_eade4f31b3bc066085a3ec501000493c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0c5300892d445149ac79665ea079ff9d = $(`&lt;div id=&quot;html_0c5300892d445149ac79665ea079ff9d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Standard Ensuite Twin Room in a Condo -Kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4500/night     &lt;/div&gt;`)[0];
                popup_eade4f31b3bc066085a3ec501000493c.setContent(html_0c5300892d445149ac79665ea079ff9d);



        circle_marker_5499ab0f67de5ac88f7b37b9315e5f35.bindPopup(popup_eade4f31b3bc066085a3ec501000493c)
        ;




            var circle_marker_2d6ce6aa2f7413647eac987cdfe31bcc = L.circleMarker(
                [-1.2744, 36.7904],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bf62957dbaf98fdaf7e1f8a14d032f59 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_32a425e159a719858e4da215cf11386e = $(`&lt;div id=&quot;html_32a425e159a719858e4da215cf11386e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Deluxe 1BR with Tub, Pool and Balcony&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6322/night     &lt;/div&gt;`)[0];
                popup_bf62957dbaf98fdaf7e1f8a14d032f59.setContent(html_32a425e159a719858e4da215cf11386e);



        circle_marker_2d6ce6aa2f7413647eac987cdfe31bcc.bindPopup(popup_bf62957dbaf98fdaf7e1f8a14d032f59)
        ;




            var circle_marker_209461236cde337a94ac4503cdf4c3a8 = L.circleMarker(
                [-1.2745, 36.7888],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_10ee46a99b1f3e2c1f44b1fc221e1d48 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_459f37f165c6943a20717d9dbea56466 = $(`&lt;div id=&quot;html_459f37f165c6943a20717d9dbea56466&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Superior Ensuite Double in a Condo- Kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4131/night     &lt;/div&gt;`)[0];
                popup_10ee46a99b1f3e2c1f44b1fc221e1d48.setContent(html_459f37f165c6943a20717d9dbea56466);



        circle_marker_209461236cde337a94ac4503cdf4c3a8.bindPopup(popup_10ee46a99b1f3e2c1f44b1fc221e1d48)
        ;




            var circle_marker_7a8cfa8936dda765a4bef0c9f56988cf = L.circleMarker(
                [-1.2969, 36.7896],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_67b5027756ac2c229bfb0e4b93db2734 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_895c915a0d5c651358ede2426d8b606f = $(`&lt;div id=&quot;html_895c915a0d5c651358ede2426d8b606f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Family Friendly Loft near Yaya | Heated Pool + Gym&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9962/night     &lt;/div&gt;`)[0];
                popup_67b5027756ac2c229bfb0e4b93db2734.setContent(html_895c915a0d5c651358ede2426d8b606f);



        circle_marker_7a8cfa8936dda765a4bef0c9f56988cf.bindPopup(popup_67b5027756ac2c229bfb0e4b93db2734)
        ;




            var circle_marker_f0e81ae2811094db4c13ad8f9b3c643e = L.circleMarker(
                [-1.3163, 36.844],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_dd4a3bec7dd791ce4bff2d30cbcc6ee0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9cd48a21bb4f1134855f0d444ffae3ee = $(`&lt;div id=&quot;html_9cd48a21bb4f1134855f0d444ffae3ee&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;CONVENIENT apartment:S/B near JKIAIRPORT&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1832/night     &lt;/div&gt;`)[0];
                popup_dd4a3bec7dd791ce4bff2d30cbcc6ee0.setContent(html_9cd48a21bb4f1134855f0d444ffae3ee);



        circle_marker_f0e81ae2811094db4c13ad8f9b3c643e.bindPopup(popup_dd4a3bec7dd791ce4bff2d30cbcc6ee0)
        ;




            var circle_marker_300b6440be8464ede7079b8ccafd03d6 = L.circleMarker(
                [-1.249, 36.7914],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_80728f34319d234c815d201452caa090 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_13490e3e26ddf80ee8e2a304d3651199 = $(`&lt;div id=&quot;html_13490e3e26ddf80ee8e2a304d3651199&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lofty house with mature garden in Spring Valley&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5816/night     &lt;/div&gt;`)[0];
                popup_80728f34319d234c815d201452caa090.setContent(html_13490e3e26ddf80ee8e2a304d3651199);



        circle_marker_300b6440be8464ede7079b8ccafd03d6.bindPopup(popup_80728f34319d234c815d201452caa090)
        ;




            var circle_marker_585d09097fc8b863fd4628b9e67196bd = L.circleMarker(
                [-1.2634, 36.7823],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1992fabeaae6c8bd149fef7b99a40b3c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_05cbb6dec3cbd85ccb74fd36fc3d5d2f = $(`&lt;div id=&quot;html_05cbb6dec3cbd85ccb74fd36fc3d5d2f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene, cozy ,calm &amp; comfortable stay in westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.83/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8068/night     &lt;/div&gt;`)[0];
                popup_1992fabeaae6c8bd149fef7b99a40b3c.setContent(html_05cbb6dec3cbd85ccb74fd36fc3d5d2f);



        circle_marker_585d09097fc8b863fd4628b9e67196bd.bindPopup(popup_1992fabeaae6c8bd149fef7b99a40b3c)
        ;




            var circle_marker_700ac3290f542f88e3e6366384494b8a = L.circleMarker(
                [-1.2639, 36.7824],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3e045830cd4f74aa13dcec1a43350d18 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_741e1ba518b8ffccd154c1b875e24038 = $(`&lt;div id=&quot;html_741e1ba518b8ffccd154c1b875e24038&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;One bedroom Check in &amp; out at anytime 5% off&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7828/night     &lt;/div&gt;`)[0];
                popup_3e045830cd4f74aa13dcec1a43350d18.setContent(html_741e1ba518b8ffccd154c1b875e24038);



        circle_marker_700ac3290f542f88e3e6366384494b8a.bindPopup(popup_3e045830cd4f74aa13dcec1a43350d18)
        ;




            var circle_marker_36dd848814e36c040fa44ee2f9b3f516 = L.circleMarker(
                [-1.2853, 36.7814],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_01128d93b461778ff07a60978cf3987e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_4aed61be39486b3b0ca7561f9ca843d3 = $(`&lt;div id=&quot;html_4aed61be39486b3b0ca7561f9ca843d3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Stylish 1 bedroom in Kileleshwa W/unlimited WIFI&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6074/night     &lt;/div&gt;`)[0];
                popup_01128d93b461778ff07a60978cf3987e.setContent(html_4aed61be39486b3b0ca7561f9ca843d3);



        circle_marker_36dd848814e36c040fa44ee2f9b3f516.bindPopup(popup_01128d93b461778ff07a60978cf3987e)
        ;




            var circle_marker_e4c159a7319f1513486eddedaad2e0a8 = L.circleMarker(
                [-1.2914, 36.7711],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2fe9122e9555a41b5ddfcb1a19d3a629 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5afcc4eed21a78dbdfb73da735047b6e = $(`&lt;div id=&quot;html_5afcc4eed21a78dbdfb73da735047b6e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;A Home Away from Home; Comfortable and Welcoming&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4351/night     &lt;/div&gt;`)[0];
                popup_2fe9122e9555a41b5ddfcb1a19d3a629.setContent(html_5afcc4eed21a78dbdfb73da735047b6e);



        circle_marker_e4c159a7319f1513486eddedaad2e0a8.bindPopup(popup_2fe9122e9555a41b5ddfcb1a19d3a629)
        ;




            var circle_marker_7e6c99cca1cb4109252cd0ee2f178522 = L.circleMarker(
                [-1.2638, 36.7825],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f78612a02462be035bd85a7aa9b1c4b6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b85228d437757614f718d4fde29b92ab = $(`&lt;div id=&quot;html_b85228d437757614f718d4fde29b92ab&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;ONE-BED Offer on Rhapta road 5% OFF BOOK  Now&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7541/night     &lt;/div&gt;`)[0];
                popup_f78612a02462be035bd85a7aa9b1c4b6.setContent(html_b85228d437757614f718d4fde29b92ab);



        circle_marker_7e6c99cca1cb4109252cd0ee2f178522.bindPopup(popup_f78612a02462be035bd85a7aa9b1c4b6)
        ;




            var circle_marker_2b676e82da5cd9b8794f182a2b02ec90 = L.circleMarker(
                [-1.2634, 36.7826],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6f0e4dbf2c9f083ecb1ba1e3c28bc835 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_2d82dd08be2666a1c04c45d32aa89b74 = $(`&lt;div id=&quot;html_2d82dd08be2666a1c04c45d32aa89b74&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;December OFFER 5% off fast Wi-Fi &amp; Landry included&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8054/night     &lt;/div&gt;`)[0];
                popup_6f0e4dbf2c9f083ecb1ba1e3c28bc835.setContent(html_2d82dd08be2666a1c04c45d32aa89b74);



        circle_marker_2b676e82da5cd9b8794f182a2b02ec90.bindPopup(popup_6f0e4dbf2c9f083ecb1ba1e3c28bc835)
        ;




            var circle_marker_fffabaeb14b6b2addc7a9cedf943da3c = L.circleMarker(
                [-1.275, 36.7897],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b3981eed9c06a467e2caf436dbb24a88 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_781753c5b49b8c6fb1f641c09b973510 = $(`&lt;div id=&quot;html_781753c5b49b8c6fb1f641c09b973510&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bright Studio in Kileleshwa | Clean + Secure&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2776/night     &lt;/div&gt;`)[0];
                popup_b3981eed9c06a467e2caf436dbb24a88.setContent(html_781753c5b49b8c6fb1f641c09b973510);



        circle_marker_fffabaeb14b6b2addc7a9cedf943da3c.bindPopup(popup_b3981eed9c06a467e2caf436dbb24a88)
        ;




            var circle_marker_1d458e77fbc28d194a0f271f4d37d065 = L.circleMarker(
                [-1.2677, 36.8052],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_cc22fc0cc739babcbdedf84d14517a41 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f61dd2992bf24170b3092123bf101487 = $(`&lt;div id=&quot;html_f61dd2992bf24170b3092123bf101487&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;5 St*r U.N approved 1BR in the heart of Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7305/night     &lt;/div&gt;`)[0];
                popup_cc22fc0cc739babcbdedf84d14517a41.setContent(html_f61dd2992bf24170b3092123bf101487);



        circle_marker_1d458e77fbc28d194a0f271f4d37d065.bindPopup(popup_cc22fc0cc739babcbdedf84d14517a41)
        ;




            var circle_marker_5180cf6c4a43f0c96ac35f5367358555 = L.circleMarker(
                [-1.3062, 36.8227],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5780e7c2df99314b41da69c643a794eb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a38a369f9d7a6cfd489828258ed1c4af = $(`&lt;div id=&quot;html_a38a369f9d7a6cfd489828258ed1c4af&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Compact Home away from Home-Nairobi West Suite&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.66/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2904/night     &lt;/div&gt;`)[0];
                popup_5780e7c2df99314b41da69c643a794eb.setContent(html_a38a369f9d7a6cfd489828258ed1c4af);



        circle_marker_5180cf6c4a43f0c96ac35f5367358555.bindPopup(popup_5780e7c2df99314b41da69c643a794eb)
        ;




            var circle_marker_667cb1d224a1885ee1c734f76a28c77f = L.circleMarker(
                [-1.3117, 36.8389],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_81a95fdf5cceee67df94b2b6bcfee4b4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_425993fcf1a727ef61287dc1cf41eaed = $(`&lt;div id=&quot;html_425993fcf1a727ef61287dc1cf41eaed&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Homely Queen Bed Apartment Near Nairobi CBD&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2505/night     &lt;/div&gt;`)[0];
                popup_81a95fdf5cceee67df94b2b6bcfee4b4.setContent(html_425993fcf1a727ef61287dc1cf41eaed);



        circle_marker_667cb1d224a1885ee1c734f76a28c77f.bindPopup(popup_81a95fdf5cceee67df94b2b6bcfee4b4)
        ;




            var circle_marker_8a3d80f975a01f2c60ef7e0852137686 = L.circleMarker(
                [-1.3079, 36.8223],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_8858a2ef2c435394bc6303c851aadfd9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1dd5b76eeee6d39c3a8185af6b80b45a = $(`&lt;div id=&quot;html_1dd5b76eeee6d39c3a8185af6b80b45a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury living space  Nairobi west suits .&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3022/night     &lt;/div&gt;`)[0];
                popup_8858a2ef2c435394bc6303c851aadfd9.setContent(html_1dd5b76eeee6d39c3a8185af6b80b45a);



        circle_marker_8a3d80f975a01f2c60ef7e0852137686.bindPopup(popup_8858a2ef2c435394bc6303c851aadfd9)
        ;




            var circle_marker_796325f1e2112959dd5c9e8aa3c8bd5b = L.circleMarker(
                [-1.3688, 36.7492],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_69d9c7d2d979838f6292d85415d35eab = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_eaa37a7993df414cef5aa4637ac03306 = $(`&lt;div id=&quot;html_eaa37a7993df414cef5aa4637ac03306&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Tree-House Nr2 at Ngong House on 4ha in Karen.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Treehouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 12375/night     &lt;/div&gt;`)[0];
                popup_69d9c7d2d979838f6292d85415d35eab.setContent(html_eaa37a7993df414cef5aa4637ac03306);



        circle_marker_796325f1e2112959dd5c9e8aa3c8bd5b.bindPopup(popup_69d9c7d2d979838f6292d85415d35eab)
        ;




            var circle_marker_38c31a984d3453807233a360b460d7c7 = L.circleMarker(
                [-1.3267, 36.881],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_96655f9a0779636dc8980ecc5e2344bb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_a0a948327ef334fbe04d6f667208042c = $(`&lt;div id=&quot;html_a0a948327ef334fbe04d6f667208042c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Rustic Apt with airport view&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3780/night     &lt;/div&gt;`)[0];
                popup_96655f9a0779636dc8980ecc5e2344bb.setContent(html_a0a948327ef334fbe04d6f667208042c);



        circle_marker_38c31a984d3453807233a360b460d7c7.bindPopup(popup_96655f9a0779636dc8980ecc5e2344bb)
        ;




            var circle_marker_dc84be5d29175e025e4071810ed40ee4 = L.circleMarker(
                [-1.307, 36.8226],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_42d944ff3b761ec8285b615d29ac868b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b4fced05324ae5548285880b2cf3438a = $(`&lt;div id=&quot;html_b4fced05324ae5548285880b2cf3438a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lily suite&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2694/night     &lt;/div&gt;`)[0];
                popup_42d944ff3b761ec8285b615d29ac868b.setContent(html_b4fced05324ae5548285880b2cf3438a);



        circle_marker_dc84be5d29175e025e4071810ed40ee4.bindPopup(popup_42d944ff3b761ec8285b615d29ac868b)
        ;




            var circle_marker_2c16ee972243f7189b708501c18596d1 = L.circleMarker(
                [-1.2708, 36.8095],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bca682ca79adc6d66660e177cc11c7b1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_181e334bb1a1a1bf0b128e77745b8738 = $(`&lt;div id=&quot;html_181e334bb1a1a1bf0b128e77745b8738&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;KIPEPEO STUDIO APARTMENT IN WESTLANDS&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.63/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6375/night     &lt;/div&gt;`)[0];
                popup_bca682ca79adc6d66660e177cc11c7b1.setContent(html_181e334bb1a1a1bf0b128e77745b8738);



        circle_marker_2c16ee972243f7189b708501c18596d1.bindPopup(popup_bca682ca79adc6d66660e177cc11c7b1)
        ;




            var circle_marker_b3fad2cf35e41e541ff6b6a3c4cfbecb = L.circleMarker(
                [-1.2976, 36.7576],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_223da5ae035da3268577de3fe4d327ee = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_963562f9489b044fa3c4dda33fdd78db = $(`&lt;div id=&quot;html_963562f9489b044fa3c4dda33fdd78db&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Big house &amp; great garden near The Junction. A home&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.72/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 20094/night     &lt;/div&gt;`)[0];
                popup_223da5ae035da3268577de3fe4d327ee.setContent(html_963562f9489b044fa3c4dda33fdd78db);



        circle_marker_b3fad2cf35e41e541ff6b6a3c4cfbecb.bindPopup(popup_223da5ae035da3268577de3fe4d327ee)
        ;




            var circle_marker_13258a323e12df7fbe0fd5d8d1d8439c = L.circleMarker(
                [-1.2773, 36.7884],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_73c885832dd2ece5c01c9032719ef850 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8fac8b5180837d9d977cc19c408608de = $(`&lt;div id=&quot;html_8fac8b5180837d9d977cc19c408608de&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Zanna Guests, luxury apartment in quiet suburbs&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8213/night     &lt;/div&gt;`)[0];
                popup_73c885832dd2ece5c01c9032719ef850.setContent(html_8fac8b5180837d9d977cc19c408608de);



        circle_marker_13258a323e12df7fbe0fd5d8d1d8439c.bindPopup(popup_73c885832dd2ece5c01c9032719ef850)
        ;




            var circle_marker_8a4c0e4d81c6418fab410db6770a090f = L.circleMarker(
                [-1.3719, 36.7364],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_22496170dffc7c20165353a3b982137f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_560a4cdcbb9533b1bb61232f51c33a82 = $(`&lt;div id=&quot;html_560a4cdcbb9533b1bb61232f51c33a82&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Holiday Working Garden Home&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.83/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4893/night     &lt;/div&gt;`)[0];
                popup_22496170dffc7c20165353a3b982137f.setContent(html_560a4cdcbb9533b1bb61232f51c33a82);



        circle_marker_8a4c0e4d81c6418fab410db6770a090f.bindPopup(popup_22496170dffc7c20165353a3b982137f)
        ;




            var circle_marker_b74f3a527ae09fb6ac424831acb3dd0a = L.circleMarker(
                [-1.2638, 36.7881],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1e70bbfad0ae90f645202cf1d3e4bdae = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_79bf356e8deeb15135f619ea35f9f61a = $(`&lt;div id=&quot;html_79bf356e8deeb15135f619ea35f9f61a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Le Mac Executive Fully Furnished Apartment 1013&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7066/night     &lt;/div&gt;`)[0];
                popup_1e70bbfad0ae90f645202cf1d3e4bdae.setContent(html_79bf356e8deeb15135f619ea35f9f61a);



        circle_marker_b74f3a527ae09fb6ac424831acb3dd0a.bindPopup(popup_1e70bbfad0ae90f645202cf1d3e4bdae)
        ;




            var circle_marker_6d1c835d0e40e18e9569ee6e63cc9f0d = L.circleMarker(
                [-1.3151, 36.8919],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_73838389e224ab2e266a5d1cb6999416 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c7bfeb4ef97b4fb0523392c244c78332 = $(`&lt;div id=&quot;html_c7bfeb4ef97b4fb0523392c244c78332&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Quaint one bedroom guest house near main airport&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guesthouse&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.97/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1820/night     &lt;/div&gt;`)[0];
                popup_73838389e224ab2e266a5d1cb6999416.setContent(html_c7bfeb4ef97b4fb0523392c244c78332);



        circle_marker_6d1c835d0e40e18e9569ee6e63cc9f0d.bindPopup(popup_73838389e224ab2e266a5d1cb6999416)
        ;




            var circle_marker_a4d845f41371883e872c1d139ef397ba = L.circleMarker(
                [-1.2627, 36.7888],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6d6c5323820d84c2f538cc2fb79358a0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cbbe79a882bee17b158cdc361cd2a16f = $(`&lt;div id=&quot;html_cbbe79a882bee17b158cdc361cd2a16f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Le Mac Fully Furnished Luxury Apartment 1113&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7386/night     &lt;/div&gt;`)[0];
                popup_6d6c5323820d84c2f538cc2fb79358a0.setContent(html_cbbe79a882bee17b158cdc361cd2a16f);



        circle_marker_a4d845f41371883e872c1d139ef397ba.bindPopup(popup_6d6c5323820d84c2f538cc2fb79358a0)
        ;




            var circle_marker_e38ec14d2e7dead9a295d03eca927d98 = L.circleMarker(
                [-1.2609, 36.7831],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ea4dddfaee0d396ccfe2348e4b0bf979 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_8c42a7008841a4fcfb1eb58d4954bdfc = $(`&lt;div id=&quot;html_8c42a7008841a4fcfb1eb58d4954bdfc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury stay :  Looks like a Hotel, feels like home&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11317/night     &lt;/div&gt;`)[0];
                popup_ea4dddfaee0d396ccfe2348e4b0bf979.setContent(html_8c42a7008841a4fcfb1eb58d4954bdfc);



        circle_marker_e38ec14d2e7dead9a295d03eca927d98.bindPopup(popup_ea4dddfaee0d396ccfe2348e4b0bf979)
        ;




            var circle_marker_f27ba511a9bc0194a8d519f2ec104117 = L.circleMarker(
                [-1.2859, 36.7788],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d1792f09d6c571aa8b8034114073e8d9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_edd2d944f0409f8d2ff67501e85065b9 = $(`&lt;div id=&quot;html_edd2d944f0409f8d2ff67501e85065b9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Apartment, Serene &amp; Modern in Kileleshwa, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4784/night     &lt;/div&gt;`)[0];
                popup_d1792f09d6c571aa8b8034114073e8d9.setContent(html_edd2d944f0409f8d2ff67501e85065b9);



        circle_marker_f27ba511a9bc0194a8d519f2ec104117.bindPopup(popup_d1792f09d6c571aa8b8034114073e8d9)
        ;




            var circle_marker_e47f3be8ac74fa4d77734cb26cb4680a = L.circleMarker(
                [-1.2595, 36.783],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7d3093d7d0110ea896511276a28d0f98 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_86c6db68a7c953d6e399f0a3b7a859fe = $(`&lt;div id=&quot;html_86c6db68a7c953d6e399f0a3b7a859fe&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;✨ORAK✨Serene Minimalist ApartHotel/Pool/ Lavington&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.68/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8564/night     &lt;/div&gt;`)[0];
                popup_7d3093d7d0110ea896511276a28d0f98.setContent(html_86c6db68a7c953d6e399f0a3b7a859fe);



        circle_marker_e47f3be8ac74fa4d77734cb26cb4680a.bindPopup(popup_7d3093d7d0110ea896511276a28d0f98)
        ;




            var circle_marker_3354c25081460556fa40b03bb7272ca4 = L.circleMarker(
                [-1.2771, 36.7928],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6fc7c108e89d1809d8ac9480b4beee75 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f98504358a33c4f33d3458e1e27f42cd = $(`&lt;div id=&quot;html_f98504358a33c4f33d3458e1e27f42cd&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Fun &amp; airy suburban melting pot of cultures!&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.75/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3531/night     &lt;/div&gt;`)[0];
                popup_6fc7c108e89d1809d8ac9480b4beee75.setContent(html_f98504358a33c4f33d3458e1e27f42cd);



        circle_marker_3354c25081460556fa40b03bb7272ca4.bindPopup(popup_6fc7c108e89d1809d8ac9480b4beee75)
        ;




            var circle_marker_6d538807a8488d9fba2733c6d124cc8a = L.circleMarker(
                [-1.288, 36.78],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_13245df1737b250366310dc252642144 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b8c92dcb0ab5439b4b78df9a1828d65c = $(`&lt;div id=&quot;html_b8c92dcb0ab5439b4b78df9a1828d65c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lavish Private Room with A Private Balcony&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2740/night     &lt;/div&gt;`)[0];
                popup_13245df1737b250366310dc252642144.setContent(html_b8c92dcb0ab5439b4b78df9a1828d65c);



        circle_marker_6d538807a8488d9fba2733c6d124cc8a.bindPopup(popup_13245df1737b250366310dc252642144)
        ;




            var circle_marker_ec2687cd292b689d3cbef8ff86871afd = L.circleMarker(
                [-1.2216, 36.8347],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e6333086446d12662a9cf108fb170a53 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_625a8f7682f23392df0561e3be8b08bf = $(`&lt;div id=&quot;html_625a8f7682f23392df0561e3be8b08bf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Self Contained Modern cottage in Runda&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.96/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5437/night     &lt;/div&gt;`)[0];
                popup_e6333086446d12662a9cf108fb170a53.setContent(html_625a8f7682f23392df0561e3be8b08bf);



        circle_marker_ec2687cd292b689d3cbef8ff86871afd.bindPopup(popup_e6333086446d12662a9cf108fb170a53)
        ;




            var circle_marker_815de356180d6d09784a6c60a034950b = L.circleMarker(
                [-1.2774, 36.7836],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d15abddcedfefe99a74f44ceff283d28 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_9c82ca0cd905c16f2583c63207e34bb5 = $(`&lt;div id=&quot;html_9c82ca0cd905c16f2583c63207e34bb5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Sochati Casa Resort&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Room in boutique hotel&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.29/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3173/night     &lt;/div&gt;`)[0];
                popup_d15abddcedfefe99a74f44ceff283d28.setContent(html_9c82ca0cd905c16f2583c63207e34bb5);



        circle_marker_815de356180d6d09784a6c60a034950b.bindPopup(popup_d15abddcedfefe99a74f44ceff283d28)
        ;




            var circle_marker_ce56dbb21e8c762f739288bfb0bef44a = L.circleMarker(
                [-1.298, 36.791],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_2e9e660a718b088f7e1614b7de09d28c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1429db0493af5da186cf22a11a1f12c6 = $(`&lt;div id=&quot;html_1429db0493af5da186cf22a11a1f12c6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spectacular 3 bed apt: Gym heated pool&amp; play area&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.8/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7448/night     &lt;/div&gt;`)[0];
                popup_2e9e660a718b088f7e1614b7de09d28c.setContent(html_1429db0493af5da186cf22a11a1f12c6);



        circle_marker_ce56dbb21e8c762f739288bfb0bef44a.bindPopup(popup_2e9e660a718b088f7e1614b7de09d28c)
        ;




            var circle_marker_2f218d7fa91fd988d31cf8bc24a22796 = L.circleMarker(
                [-1.2666, 36.8058],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bc90373ea44520257d5bc09dc1c27a0e = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3a2fc58c723aa393af7d93184e64202a = $(`&lt;div id=&quot;html_3a2fc58c723aa393af7d93184e64202a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Quiet, modern apartment in the middle of Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6413/night     &lt;/div&gt;`)[0];
                popup_bc90373ea44520257d5bc09dc1c27a0e.setContent(html_3a2fc58c723aa393af7d93184e64202a);



        circle_marker_2f218d7fa91fd988d31cf8bc24a22796.bindPopup(popup_bc90373ea44520257d5bc09dc1c27a0e)
        ;




            var circle_marker_a0d272858132d284dc8662cf20b6e151 = L.circleMarker(
                [-1.2859, 36.7888],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_28b45730fe29e97add98c3a357819bf3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_295f14cbd1390379edde11d26e389ae3 = $(`&lt;div id=&quot;html_295f14cbd1390379edde11d26e389ae3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Comfy 2 Bedroom Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.33/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7275/night     &lt;/div&gt;`)[0];
                popup_28b45730fe29e97add98c3a357819bf3.setContent(html_295f14cbd1390379edde11d26e389ae3);



        circle_marker_a0d272858132d284dc8662cf20b6e151.bindPopup(popup_28b45730fe29e97add98c3a357819bf3)
        ;




            var circle_marker_8b8bbf12d5122e0acc620587ab8c075d = L.circleMarker(
                [-1.2648, 36.7897],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6f0b1658c29af5d210dad7e975b0cf09 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_fa2d7ff713afb0119df9a715a22c6e81 = $(`&lt;div id=&quot;html_fa2d7ff713afb0119df9a715a22c6e81&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Chic 1BR in Westlands- 5 mins from local spots&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.49/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4170/night     &lt;/div&gt;`)[0];
                popup_6f0b1658c29af5d210dad7e975b0cf09.setContent(html_fa2d7ff713afb0119df9a715a22c6e81);



        circle_marker_8b8bbf12d5122e0acc620587ab8c075d.bindPopup(popup_6f0b1658c29af5d210dad7e975b0cf09)
        ;




            var circle_marker_a7803c9ad38eefb2c69708ecbd13628c = L.circleMarker(
                [-1.2653, 36.7962],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6d11c0d9e7107a342ebb4eb2eae09230 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_90082588e74b7c9cda0af9864a20ee39 = $(`&lt;div id=&quot;html_90082588e74b7c9cda0af9864a20ee39&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Erica Residences - Riverside, Deluxe 3BR Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.44/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11958/night     &lt;/div&gt;`)[0];
                popup_6d11c0d9e7107a342ebb4eb2eae09230.setContent(html_90082588e74b7c9cda0af9864a20ee39);



        circle_marker_a7803c9ad38eefb2c69708ecbd13628c.bindPopup(popup_6d11c0d9e7107a342ebb4eb2eae09230)
        ;




            var circle_marker_19c0e709498cdb37a3b3d56be6ccc090 = L.circleMarker(
                [-1.2791, 36.7844],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b29fa80080f09eaba8bcc7768af4ad74 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_bd9e843f53938f9f648c331f727cf99f = $(`&lt;div id=&quot;html_bd9e843f53938f9f648c331f727cf99f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lovely one bedroom apartment free WiFi.&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9582/night     &lt;/div&gt;`)[0];
                popup_b29fa80080f09eaba8bcc7768af4ad74.setContent(html_bd9e843f53938f9f648c331f727cf99f);



        circle_marker_19c0e709498cdb37a3b3d56be6ccc090.bindPopup(popup_b29fa80080f09eaba8bcc7768af4ad74)
        ;




            var circle_marker_8a7ec7369d226a6df4854ea9c80bb62b = L.circleMarker(
                [-1.2775, 36.7855],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a079fc2cf3da1a21ee4d59b242206531 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_2b6d37f5de2eaffa65bc113f2b09f5cc = $(`&lt;div id=&quot;html_2b6d37f5de2eaffa65bc113f2b09f5cc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Studio Apartment Free WiFi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7127/night     &lt;/div&gt;`)[0];
                popup_a079fc2cf3da1a21ee4d59b242206531.setContent(html_2b6d37f5de2eaffa65bc113f2b09f5cc);



        circle_marker_8a7ec7369d226a6df4854ea9c80bb62b.bindPopup(popup_a079fc2cf3da1a21ee4d59b242206531)
        ;




            var circle_marker_48e20e8f3df1ba6363d0110518ad2a70 = L.circleMarker(
                [-1.27, 36.8077],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_41bb841956adbf2ac0ece41465fffb90 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_c4e8666ec3766ee42ac45676ec9e2b91 = $(`&lt;div id=&quot;html_c4e8666ec3766ee42ac45676ec9e2b91&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Trendy 1 bedroom apartment in Westlands Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5693/night     &lt;/div&gt;`)[0];
                popup_41bb841956adbf2ac0ece41465fffb90.setContent(html_c4e8666ec3766ee42ac45676ec9e2b91);



        circle_marker_48e20e8f3df1ba6363d0110518ad2a70.bindPopup(popup_41bb841956adbf2ac0ece41465fffb90)
        ;




            var circle_marker_00f67b78fe8eb42a98b05da82c8df375 = L.circleMarker(
                [-1.317, 36.8422],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_c7f9f2e6c38c57c54be99d433b6937fb = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_816e19737555f7d7fcd7f4d4433ab6cb = $(`&lt;div id=&quot;html_816e19737555f7d7fcd7f4d4433ab6cb&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Budget Studio Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.72/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1954/night     &lt;/div&gt;`)[0];
                popup_c7f9f2e6c38c57c54be99d433b6937fb.setContent(html_816e19737555f7d7fcd7f4d4433ab6cb);



        circle_marker_00f67b78fe8eb42a98b05da82c8df375.bindPopup(popup_c7f9f2e6c38c57c54be99d433b6937fb)
        ;




            var circle_marker_42c02348bf99f0286b10628b8a9df8af = L.circleMarker(
                [-1.2839, 36.7908],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_785eb9785355486c426eeb064192b230 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cf49102a06cdf74fa56e113f3cca04a1 = $(`&lt;div id=&quot;html_cf49102a06cdf74fa56e113f3cca04a1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Purple Stays  3 Bed by Your Host,Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.56/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 15485/night     &lt;/div&gt;`)[0];
                popup_785eb9785355486c426eeb064192b230.setContent(html_cf49102a06cdf74fa56e113f3cca04a1);



        circle_marker_42c02348bf99f0286b10628b8a9df8af.bindPopup(popup_785eb9785355486c426eeb064192b230)
        ;




            var circle_marker_e081db4bf137a2ac9dc4ad8afc4c1b28 = L.circleMarker(
                [-1.278, 36.7846],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_cdf9a3e527c10f3f2cd935ddbc0654f8 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_21413f48493fa221ae5bbb8e5170d3cc = $(`&lt;div id=&quot;html_21413f48493fa221ae5bbb8e5170d3cc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spacious studio Apartment:  FreeWifi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7099/night     &lt;/div&gt;`)[0];
                popup_cdf9a3e527c10f3f2cd935ddbc0654f8.setContent(html_21413f48493fa221ae5bbb8e5170d3cc);



        circle_marker_e081db4bf137a2ac9dc4ad8afc4c1b28.bindPopup(popup_cdf9a3e527c10f3f2cd935ddbc0654f8)
        ;




            var circle_marker_13d2e976535e5dcf8987a76b174a65e7 = L.circleMarker(
                [-1.2722, 36.7984],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6883fe4cd314289b3e27318385351002 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_158d71af7117ca03ac924c3031a98e00 = $(`&lt;div id=&quot;html_158d71af7117ca03ac924c3031a98e00&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Erica Residences - Riverside, Deluxe 4BR Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.62/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 13811/night     &lt;/div&gt;`)[0];
                popup_6883fe4cd314289b3e27318385351002.setContent(html_158d71af7117ca03ac924c3031a98e00);



        circle_marker_13d2e976535e5dcf8987a76b174a65e7.bindPopup(popup_6883fe4cd314289b3e27318385351002)
        ;




            var circle_marker_9a70c9c2ada9743c3fb4641ff62ee44c = L.circleMarker(
                [-1.2659, 36.805],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_dccc4591e5553bbb4355a7f5898c1df4 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_7633fbed4dec69c1fac6e653f66641db = $(`&lt;div id=&quot;html_7633fbed4dec69c1fac6e653f66641db&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Quiet, Central, modern apartment in Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.67/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5906/night     &lt;/div&gt;`)[0];
                popup_dccc4591e5553bbb4355a7f5898c1df4.setContent(html_7633fbed4dec69c1fac6e653f66641db);



        circle_marker_9a70c9c2ada9743c3fb4641ff62ee44c.bindPopup(popup_dccc4591e5553bbb4355a7f5898c1df4)
        ;




            var circle_marker_8676328d25b25ab35b458f431a3531eb = L.circleMarker(
                [-1.2721, 36.7973],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_ce0ba8896d0eb6d638fd8189690a88a5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_25e12d82d076265ee789b7a8a17d30b2 = $(`&lt;div id=&quot;html_25e12d82d076265ee789b7a8a17d30b2&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bellway Suites Kileleshwa-Entire&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.71/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 10742/night     &lt;/div&gt;`)[0];
                popup_ce0ba8896d0eb6d638fd8189690a88a5.setContent(html_25e12d82d076265ee789b7a8a17d30b2);



        circle_marker_8676328d25b25ab35b458f431a3531eb.bindPopup(popup_ce0ba8896d0eb6d638fd8189690a88a5)
        ;




            var circle_marker_ddf58de4cc45be124b906dc0c3ed4329 = L.circleMarker(
                [-1.2794, 36.7856],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_11586512fbbe98deee5e6ffc281a1a85 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d57cc7dbe02169aba7c3670f594c3ce6 = $(`&lt;div id=&quot;html_d57cc7dbe02169aba7c3670f594c3ce6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Two bedroom apartment with balcony&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9839/night     &lt;/div&gt;`)[0];
                popup_11586512fbbe98deee5e6ffc281a1a85.setContent(html_d57cc7dbe02169aba7c3670f594c3ce6);



        circle_marker_ddf58de4cc45be124b906dc0c3ed4329.bindPopup(popup_11586512fbbe98deee5e6ffc281a1a85)
        ;




            var circle_marker_7db678a86b512fea5ce84e5d28c8f5d5 = L.circleMarker(
                [-1.2978, 36.7912],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_50a90760a96d7c31fddd98ed38fd814d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_935558f43ee212c36e003c2ac75ba328 = $(`&lt;div id=&quot;html_935558f43ee212c36e003c2ac75ba328&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Serene King Bed Suite | Near Mall | WiFi | dstv&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6885/night     &lt;/div&gt;`)[0];
                popup_50a90760a96d7c31fddd98ed38fd814d.setContent(html_935558f43ee212c36e003c2ac75ba328);



        circle_marker_7db678a86b512fea5ce84e5d28c8f5d5.bindPopup(popup_50a90760a96d7c31fddd98ed38fd814d)
        ;




            var circle_marker_ddeba9d7e8ff7e367f9b440b75af1e3c = L.circleMarker(
                [-1.3237, 36.8756],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3a06d49ef009abb0088aea84368463db = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5955f707675807e2cde87d0fa4de74f8 = $(`&lt;div id=&quot;html_5955f707675807e2cde87d0fa4de74f8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;JKIA- Gated Apt Near SGR, Imara Mall, Expressway&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.92/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6806/night     &lt;/div&gt;`)[0];
                popup_3a06d49ef009abb0088aea84368463db.setContent(html_5955f707675807e2cde87d0fa4de74f8);



        circle_marker_ddeba9d7e8ff7e367f9b440b75af1e3c.bindPopup(popup_3a06d49ef009abb0088aea84368463db)
        ;




            var circle_marker_3b8c5173fc460900773308726224a4b2 = L.circleMarker(
                [-1.1872, 36.903],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4b0dc0ca7400117905cf54513a4b1874 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dd8f56eb55983fd4da30ebbbe91fdb6b = $(`&lt;div id=&quot;html_dd8f56eb55983fd4da30ebbbe91fdb6b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Private Room in Modern Getaway! (Coin laundry)&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3258/night     &lt;/div&gt;`)[0];
                popup_4b0dc0ca7400117905cf54513a4b1874.setContent(html_dd8f56eb55983fd4da30ebbbe91fdb6b);



        circle_marker_3b8c5173fc460900773308726224a4b2.bindPopup(popup_4b0dc0ca7400117905cf54513a4b1874)
        ;




            var circle_marker_ce6719fe9bc192744733db199c62b12a = L.circleMarker(
                [-1.2821, 36.7916],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1e7f80061bf0327394245096b3481a80 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_3eb9bca7a375f74b5a8433f21b877ac6 = $(`&lt;div id=&quot;html_3eb9bca7a375f74b5a8433f21b877ac6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Purple Stays,2 Bedroom, by Your Host, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.53/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 14802/night     &lt;/div&gt;`)[0];
                popup_1e7f80061bf0327394245096b3481a80.setContent(html_3eb9bca7a375f74b5a8433f21b877ac6);



        circle_marker_ce6719fe9bc192744733db199c62b12a.bindPopup(popup_1e7f80061bf0327394245096b3481a80)
        ;




            var circle_marker_f3f24e09dae77f745456c63e891ba65b = L.circleMarker(
                [-1.2154, 36.8418],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3caf18508c3238c3b15e35a5f1881328 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_fa570fa5e65652421e7ffb2f27674a53 = $(`&lt;div id=&quot;html_fa570fa5e65652421e7ffb2f27674a53&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Executive 1 Bed Apartment, Runda- Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.74/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 7405/night     &lt;/div&gt;`)[0];
                popup_3caf18508c3238c3b15e35a5f1881328.setContent(html_fa570fa5e65652421e7ffb2f27674a53);



        circle_marker_f3f24e09dae77f745456c63e891ba65b.bindPopup(popup_3caf18508c3238c3b15e35a5f1881328)
        ;




            var circle_marker_744d1b0c5d863f44a18d5199dc07513a = L.circleMarker(
                [-1.2177, 36.8916],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bd5393624d629a1a60b2ed00d9594346 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_239f057a383668c17abbea3f8f7cbe21 = $(`&lt;div id=&quot;html_239f057a383668c17abbea3f8f7cbe21&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Strat Penthouse&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.88/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6940/night     &lt;/div&gt;`)[0];
                popup_bd5393624d629a1a60b2ed00d9594346.setContent(html_239f057a383668c17abbea3f8f7cbe21);



        circle_marker_744d1b0c5d863f44a18d5199dc07513a.bindPopup(popup_bd5393624d629a1a60b2ed00d9594346)
        ;




            var circle_marker_a440c2e4ea813e1ea26bc4fd8643f7bf = L.circleMarker(
                [-1.3596, 36.7417],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_7ab72f6981e9bd09a75f4fa1f36d4643 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_d0653c4e825af6e75fee35afe4ad8a74 = $(`&lt;div id=&quot;html_d0653c4e825af6e75fee35afe4ad8a74&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Karen Hardy Executive Homestay&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6190/night     &lt;/div&gt;`)[0];
                popup_7ab72f6981e9bd09a75f4fa1f36d4643.setContent(html_d0653c4e825af6e75fee35afe4ad8a74);



        circle_marker_a440c2e4ea813e1ea26bc4fd8643f7bf.bindPopup(popup_7ab72f6981e9bd09a75f4fa1f36d4643)
        ;




            var circle_marker_8565948faf367015af1a78e2651c767c = L.circleMarker(
                [-1.2776, 36.7842],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b0db9118a91f425f690069b8eeb95ff0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cd5b8f6a93e96056dc60485beea9c4c6 = $(`&lt;div id=&quot;html_cd5b8f6a93e96056dc60485beea9c4c6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Two bedroom apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9600/night     &lt;/div&gt;`)[0];
                popup_b0db9118a91f425f690069b8eeb95ff0.setContent(html_cd5b8f6a93e96056dc60485beea9c4c6);



        circle_marker_8565948faf367015af1a78e2651c767c.bindPopup(popup_b0db9118a91f425f690069b8eeb95ff0)
        ;




            var circle_marker_97188055fb169dd9791cac524ae0e380 = L.circleMarker(
                [-1.2645, 36.7951],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_e235c2dede1e22c92857792ac77fa8f1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1441a5dd051674e304983b45530a2ccd = $(`&lt;div id=&quot;html_1441a5dd051674e304983b45530a2ccd&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;luxurious 4 bedroom Serviced apt Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 17244/night     &lt;/div&gt;`)[0];
                popup_e235c2dede1e22c92857792ac77fa8f1.setContent(html_1441a5dd051674e304983b45530a2ccd);



        circle_marker_97188055fb169dd9791cac524ae0e380.bindPopup(popup_e235c2dede1e22c92857792ac77fa8f1)
        ;




            var circle_marker_cc0e1b4de5d194421c428bebf5dc4864 = L.circleMarker(
                [-1.2226, 36.8957],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_05559079d9618a7dffd69c3af4b96b2f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dc5fa5915143791eaca8828f28bee84e = $(`&lt;div id=&quot;html_dc5fa5915143791eaca8828f28bee84e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Atlanta Penthouse07 Cabin128 then 99-479&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire serviced apartment&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.85/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2592/night     &lt;/div&gt;`)[0];
                popup_05559079d9618a7dffd69c3af4b96b2f.setContent(html_dc5fa5915143791eaca8828f28bee84e);



        circle_marker_cc0e1b4de5d194421c428bebf5dc4864.bindPopup(popup_05559079d9618a7dffd69c3af4b96b2f)
        ;




            var circle_marker_9c81908b16ae476f17db4317cea3cf6b = L.circleMarker(
                [-1.292, 36.7624],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_38aabf287bd6aa7122c2afa935e63140 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_0780b86f279c166751c54ebefd238471 = $(`&lt;div id=&quot;html_0780b86f279c166751c54ebefd238471&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lavi Place Astoria, Lavington, Nairobi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.95/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5216/night     &lt;/div&gt;`)[0];
                popup_38aabf287bd6aa7122c2afa935e63140.setContent(html_0780b86f279c166751c54ebefd238471);



        circle_marker_9c81908b16ae476f17db4317cea3cf6b.bindPopup(popup_38aabf287bd6aa7122c2afa935e63140)
        ;




            var circle_marker_867ff241d064a0e60a586bcc16d5ab9f = L.circleMarker(
                [-1.3272, 36.7818],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_be0f5f4604eda4a0b0dee198728e9bc1 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_00f708b06616d0923c07dfbf8e0e9efb = $(`&lt;div id=&quot;html_00f708b06616d0923c07dfbf8e0e9efb&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spacious minimalist studio next to National park&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire guest suite&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 1962/night     &lt;/div&gt;`)[0];
                popup_be0f5f4604eda4a0b0dee198728e9bc1.setContent(html_00f708b06616d0923c07dfbf8e0e9efb);



        circle_marker_867ff241d064a0e60a586bcc16d5ab9f.bindPopup(popup_be0f5f4604eda4a0b0dee198728e9bc1)
        ;




            var circle_marker_087c188515366ef8265e96bffab5bcc4 = L.circleMarker(
                [-1.3005, 36.7471],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_be665ddd919191e840a76742b4596e0b = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_4e612faad1dce38058e97e58b975c40c = $(`&lt;div id=&quot;html_4e612faad1dce38058e97e58b975c40c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern, beautiful 2 bedroom flat off Ngong Road&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4332/night     &lt;/div&gt;`)[0];
                popup_be665ddd919191e840a76742b4596e0b.setContent(html_4e612faad1dce38058e97e58b975c40c);



        circle_marker_087c188515366ef8265e96bffab5bcc4.bindPopup(popup_be665ddd919191e840a76742b4596e0b)
        ;




            var circle_marker_6a76e4f2a4f3ec0c349bbac539ffe552 = L.circleMarker(
                [-1.2862, 36.7916],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#FF9F1C&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#FF9F1C&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_53121636b51cb690350253b4e0d5560f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_74e4f155f93986877f873c4d3c401cbc = $(`&lt;div id=&quot;html_74e4f155f93986877f873c4d3c401cbc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Lovely 3-bdrm apartment w patio, pool, gym, garden&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 11639/night     &lt;/div&gt;`)[0];
                popup_53121636b51cb690350253b4e0d5560f.setContent(html_74e4f155f93986877f873c4d3c401cbc);



        circle_marker_6a76e4f2a4f3ec0c349bbac539ffe552.bindPopup(popup_53121636b51cb690350253b4e0d5560f)
        ;




            var circle_marker_105a57b7e77fc3df7ff796c3e60f470d = L.circleMarker(
                [-1.3069, 36.8165],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_f4b0540f55b3995a396205373766bf8f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1a293b30d6573da36c8cfe93613f811b = $(`&lt;div id=&quot;html_1a293b30d6573da36c8cfe93613f811b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;ROOFTOP Swimmingpool&amp; patio 2Bed apartmentMadaraka&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5552/night     &lt;/div&gt;`)[0];
                popup_f4b0540f55b3995a396205373766bf8f.setContent(html_1a293b30d6573da36c8cfe93613f811b);



        circle_marker_105a57b7e77fc3df7ff796c3e60f470d.bindPopup(popup_f4b0540f55b3995a396205373766bf8f)
        ;




            var circle_marker_220a591e91d495dc86567e8e8905286a = L.circleMarker(
                [-1.2663, 36.8042],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_0b180d64f90204bb4d23cdca15474b0d = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cfecaed15e8c592eb2a030f2d698ee68 = $(`&lt;div id=&quot;html_cfecaed15e8c592eb2a030f2d698ee68&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;5 St★r 1BR + Gym in UN BlueZone in ❤️ of Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.84/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6929/night     &lt;/div&gt;`)[0];
                popup_0b180d64f90204bb4d23cdca15474b0d.setContent(html_cfecaed15e8c592eb2a030f2d698ee68);



        circle_marker_220a591e91d495dc86567e8e8905286a.bindPopup(popup_0b180d64f90204bb4d23cdca15474b0d)
        ;




            var circle_marker_3a9cc32a4cd1f3d021f3cbc6195d744f = L.circleMarker(
                [-1.2771, 36.7844],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b58fe0ddb6c1e0770df97993a3dbd1b0 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_567a826867a9fd91e50398881759ba8b = $(`&lt;div id=&quot;html_567a826867a9fd91e50398881759ba8b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Spacious two bedroom Apartment in Keleleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9138/night     &lt;/div&gt;`)[0];
                popup_b58fe0ddb6c1e0770df97993a3dbd1b0.setContent(html_567a826867a9fd91e50398881759ba8b);



        circle_marker_3a9cc32a4cd1f3d021f3cbc6195d744f.bindPopup(popup_b58fe0ddb6c1e0770df97993a3dbd1b0)
        ;




            var circle_marker_06edf3c047cbd8bbc422bdb688c05b4f = L.circleMarker(
                [-1.2903, 36.7614],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_249a8413eedd32376501de3fd48635e3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_52ce84f537130ae4c3c11da30590d340 = $(`&lt;div id=&quot;html_52ce84f537130ae4c3c11da30590d340&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Rabella (Beautiful) Homes&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 9146/night     &lt;/div&gt;`)[0];
                popup_249a8413eedd32376501de3fd48635e3.setContent(html_52ce84f537130ae4c3c11da30590d340);



        circle_marker_06edf3c047cbd8bbc422bdb688c05b4f.bindPopup(popup_249a8413eedd32376501de3fd48635e3)
        ;




            var circle_marker_a4980fd3e75f7145f740bc239e0e99c1 = L.circleMarker(
                [-1.2927, 36.7635],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b0c54ab930f028e98b23b68e12177bab = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_88a5637803cbb58bb48943c810be9e0c = $(`&lt;div id=&quot;html_88a5637803cbb58bb48943c810be9e0c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;✨ORAK✨Unwind in an Executive, Sleek Apartment/Pool&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.73/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8390/night     &lt;/div&gt;`)[0];
                popup_b0c54ab930f028e98b23b68e12177bab.setContent(html_88a5637803cbb58bb48943c810be9e0c);



        circle_marker_a4980fd3e75f7145f740bc239e0e99c1.bindPopup(popup_b0c54ab930f028e98b23b68e12177bab)
        ;




            var circle_marker_9b999c35eba6889e2ecf08f28791ab8c = L.circleMarker(
                [-1.2932, 36.7623],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_bbf809b57bef0d69eee01013ba037902 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_69076b9cc7501c44240968455b6ef226 = $(`&lt;div id=&quot;html_69076b9cc7501c44240968455b6ef226&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;luxurious furnished 1 bedroom- with pool and gym&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire vacation home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6990/night     &lt;/div&gt;`)[0];
                popup_bbf809b57bef0d69eee01013ba037902.setContent(html_69076b9cc7501c44240968455b6ef226);



        circle_marker_9b999c35eba6889e2ecf08f28791ab8c.bindPopup(popup_bbf809b57bef0d69eee01013ba037902)
        ;




            var circle_marker_431128470a0d01a2cc0d926eb14869b3 = L.circleMarker(
                [-1.298, 36.7979],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5877ae5a4c3c006f522b22599c6d33fe = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_f5284dbec7dbe88d6a27a7ed17882f04 = $(`&lt;div id=&quot;html_f5284dbec7dbe88d6a27a7ed17882f04&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Luxury cozy studio apartment, Kilimani&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5320/night     &lt;/div&gt;`)[0];
                popup_5877ae5a4c3c006f522b22599c6d33fe.setContent(html_f5284dbec7dbe88d6a27a7ed17882f04);



        circle_marker_431128470a0d01a2cc0d926eb14869b3.bindPopup(popup_5877ae5a4c3c006f522b22599c6d33fe)
        ;




            var circle_marker_ef6c506618b367a7f15cd88b38971faf = L.circleMarker(
                [-1.2715, 36.8141],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_71b04fb810fb2f3702be2ee91baa61ed = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_6345e9ff56d8d82797f41ba971e19e72 = $(`&lt;div id=&quot;html_6345e9ff56d8d82797f41ba971e19e72&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Aoukings Place Home Away from Home&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.86/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2758/night     &lt;/div&gt;`)[0];
                popup_71b04fb810fb2f3702be2ee91baa61ed.setContent(html_6345e9ff56d8d82797f41ba971e19e72);



        circle_marker_ef6c506618b367a7f15cd88b38971faf.bindPopup(popup_71b04fb810fb2f3702be2ee91baa61ed)
        ;




            var circle_marker_78750d6149d2e755ffec3e6b08be9eee = L.circleMarker(
                [-1.2989, 36.7655],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d274433b5b23d26826c3538d9b70f360 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5c4726d305efcd111bffef36875efc6b = $(`&lt;div id=&quot;html_5c4726d305efcd111bffef36875efc6b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Furnished &amp; serviced 2 bed behind Junction Mall&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6899/night     &lt;/div&gt;`)[0];
                popup_d274433b5b23d26826c3538d9b70f360.setContent(html_5c4726d305efcd111bffef36875efc6b);



        circle_marker_78750d6149d2e755ffec3e6b08be9eee.bindPopup(popup_d274433b5b23d26826c3538d9b70f360)
        ;




            var circle_marker_cfa953fb4381c43c53c10b7b9343e4c0 = L.circleMarker(
                [-1.2789, 36.7853],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_1c77329f90689bef11d763c17012d5c6 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_b3ceb5c4cebe4501292b0a7c9010e736 = $(`&lt;div id=&quot;html_b3ceb5c4cebe4501292b0a7c9010e736&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;One bedroom apartment in  kileleshwa&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.25/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6962/night     &lt;/div&gt;`)[0];
                popup_1c77329f90689bef11d763c17012d5c6.setContent(html_b3ceb5c4cebe4501292b0a7c9010e736);



        circle_marker_cfa953fb4381c43c53c10b7b9343e4c0.bindPopup(popup_1c77329f90689bef11d763c17012d5c6)
        ;




            var circle_marker_ea8c939722379b51535005e6e3d8e1a7 = L.circleMarker(
                [-1.2638, 36.7892],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#E63946&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#E63946&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d79cbd182455716dd9eb7093dd2e549c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_52829c81d28c41a1305b412889f7c2b9 = $(`&lt;div id=&quot;html_52829c81d28c41a1305b412889f7c2b9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Bamboo Loft Penthouse, Westlands&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.79/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 22735/night     &lt;/div&gt;`)[0];
                popup_d79cbd182455716dd9eb7093dd2e549c.setContent(html_52829c81d28c41a1305b412889f7c2b9);



        circle_marker_ea8c939722379b51535005e6e3d8e1a7.bindPopup(popup_d79cbd182455716dd9eb7093dd2e549c)
        ;




            var circle_marker_e2963bbcb1dfa9b028ad6b118a27212a = L.circleMarker(
                [-1.2582, 36.8006],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_27e947279067125ba467ded4c5d14aff = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_dcf5234649235db66657faec58133846 = $(`&lt;div id=&quot;html_dcf5234649235db66657faec58133846&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;ACACIA LUXURIOUS ONE BEDROOM APARTMENT&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.21/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5078/night     &lt;/div&gt;`)[0];
                popup_27e947279067125ba467ded4c5d14aff.setContent(html_dcf5234649235db66657faec58133846);



        circle_marker_e2963bbcb1dfa9b028ad6b118a27212a.bindPopup(popup_27e947279067125ba467ded4c5d14aff)
        ;




            var circle_marker_aee1d065c7d98b4cd529665cf87537d9 = L.circleMarker(
                [-1.2564, 36.7991],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_6b615e04f3ea135c7ffdf9984c725ea9 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_46a9b78f102917646dd2be5c328c9608 = $(`&lt;div id=&quot;html_46a9b78f102917646dd2be5c328c9608&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;ACACIA LUXURIOUS ONE BEDROOM APARTMENT&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.33/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5162/night     &lt;/div&gt;`)[0];
                popup_6b615e04f3ea135c7ffdf9984c725ea9.setContent(html_46a9b78f102917646dd2be5c328c9608);



        circle_marker_aee1d065c7d98b4cd529665cf87537d9.bindPopup(popup_6b615e04f3ea135c7ffdf9984c725ea9)
        ;




            var circle_marker_74e52be3020b6d7a58ca4db84bae55b5 = L.circleMarker(
                [-1.2652, 36.7558],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_d9b21399633c157da86a66adc2373960 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_5fb8b9f9f6eb00b28c021e0a7fc8e8f8 = $(`&lt;div id=&quot;html_5fb8b9f9f6eb00b28c021e0a7fc8e8f8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;The Green Room&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2812/night     &lt;/div&gt;`)[0];
                popup_d9b21399633c157da86a66adc2373960.setContent(html_5fb8b9f9f6eb00b28c021e0a7fc8e8f8);



        circle_marker_74e52be3020b6d7a58ca4db84bae55b5.bindPopup(popup_d9b21399633c157da86a66adc2373960)
        ;




            var circle_marker_c98739946cee91e83de2157a3ae92503 = L.circleMarker(
                [-1.3065, 36.8227],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_fbe8526bb747460263168c6d4510e01f = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_72ae0cabc0884dea1297cef6180c3fe7 = $(`&lt;div id=&quot;html_72ae0cabc0884dea1297cef6180c3fe7&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Tana suites! a homely palace&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.78/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2739/night     &lt;/div&gt;`)[0];
                popup_fbe8526bb747460263168c6d4510e01f.setContent(html_72ae0cabc0884dea1297cef6180c3fe7);



        circle_marker_c98739946cee91e83de2157a3ae92503.bindPopup(popup_fbe8526bb747460263168c6d4510e01f)
        ;




            var circle_marker_dd10f5e690f094aa6456b2abae2e8096 = L.circleMarker(
                [-1.2786, 36.7857],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5fe0bc3dc3a02078420c764109af1cf5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_ba43d11213949f85bd490bcc5cb5a0f8 = $(`&lt;div id=&quot;html_ba43d11213949f85bd490bcc5cb5a0f8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;New spacious one bedroom apartment Free WiFi&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.9/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6207/night     &lt;/div&gt;`)[0];
                popup_5fe0bc3dc3a02078420c764109af1cf5.setContent(html_ba43d11213949f85bd490bcc5cb5a0f8);



        circle_marker_dd10f5e690f094aa6456b2abae2e8096.bindPopup(popup_5fe0bc3dc3a02078420c764109af1cf5)
        ;




            var circle_marker_361d6259f3f5248aac5bf0f8a680ad82 = L.circleMarker(
                [-1.2366, 36.8475],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_11c9f3184eee5c430b6a54baa8bfec72 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e9b452b284f787e167b6b9d1cb2a1134 = $(`&lt;div id=&quot;html_e9b452b284f787e167b6b9d1cb2a1134&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;United Nations perimeter.Ideal for expatriates&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire cottage&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 8936/night     &lt;/div&gt;`)[0];
                popup_11c9f3184eee5c430b6a54baa8bfec72.setContent(html_e9b452b284f787e167b6b9d1cb2a1134);



        circle_marker_361d6259f3f5248aac5bf0f8a680ad82.bindPopup(popup_11c9f3184eee5c430b6a54baa8bfec72)
        ;




            var circle_marker_04ed30bf660852ed8046cdc429d4cb65 = L.circleMarker(
                [-1.2903, 36.775],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_b56300c4e524385f76442b34a1c60fe5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_cb72fbdfe39f94ebfd5a2a3946a602a3 = $(`&lt;div id=&quot;html_cb72fbdfe39f94ebfd5a2a3946a602a3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;1.5BR with office, gym, washer &amp; dryer, Netflix&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 5.0/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6159/night     &lt;/div&gt;`)[0];
                popup_b56300c4e524385f76442b34a1c60fe5.setContent(html_cb72fbdfe39f94ebfd5a2a3946a602a3);



        circle_marker_04ed30bf660852ed8046cdc429d4cb65.bindPopup(popup_b56300c4e524385f76442b34a1c60fe5)
        ;




            var circle_marker_949cc2ce09311f6192ceaa7016ac8a23 = L.circleMarker(
                [-1.2978, 36.7912],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_a4abb3c33ae3c5585f16d46b1fdbf4b5 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_1e84739f86ca732d5bad2bd39889536f = $(`&lt;div id=&quot;html_1e84739f86ca732d5bad2bd39889536f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Nairobi  2 bedroom Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.91/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 5229/night     &lt;/div&gt;`)[0];
                popup_a4abb3c33ae3c5585f16d46b1fdbf4b5.setContent(html_1e84739f86ca732d5bad2bd39889536f);



        circle_marker_949cc2ce09311f6192ceaa7016ac8a23.bindPopup(popup_a4abb3c33ae3c5585f16d46b1fdbf4b5)
        ;




            var circle_marker_c0e34fbe413f8413463789a1c2db58ff = L.circleMarker(
                [-1.2747, 36.8181],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_3cf72a2e27867cc7db408616998495e2 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_504236839142a813d29ec18a7401895a = $(`&lt;div id=&quot;html_504236839142a813d29ec18a7401895a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Mvuli Luxury Suites&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.2/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 3711/night     &lt;/div&gt;`)[0];
                popup_3cf72a2e27867cc7db408616998495e2.setContent(html_504236839142a813d29ec18a7401895a);



        circle_marker_c0e34fbe413f8413463789a1c2db58ff.bindPopup(popup_3cf72a2e27867cc7db408616998495e2)
        ;




            var circle_marker_d3d38fbdc24e3d3f967e286a47b8589e = L.circleMarker(
                [-1.2978, 36.7912],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#1F77B4&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#1F77B4&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_591ed121c610209a65ac83b1ea66b15c = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_e6a4400ec15ab84107375cf02d4660d4 = $(`&lt;div id=&quot;html_e6a4400ec15ab84107375cf02d4660d4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Modern 1BR | King Bed | Fast Wi-Fi | Near CBD&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire condo&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.87/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 6573/night     &lt;/div&gt;`)[0];
                popup_591ed121c610209a65ac83b1ea66b15c.setContent(html_e6a4400ec15ab84107375cf02d4660d4);



        circle_marker_d3d38fbdc24e3d3f967e286a47b8589e.bindPopup(popup_591ed121c610209a65ac83b1ea66b15c)
        ;




            var circle_marker_25fe3fb411110aada8cb57eaf66eea3a = L.circleMarker(
                [-1.2667, 36.7353],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_4a9efc5afd92a828c0435b14d2028856 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_2e123e05bece6e2b1b05fc9fe0937b02 = $(`&lt;div id=&quot;html_2e123e05bece6e2b1b05fc9fe0937b02&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Cosy &amp; Airy Studio with Balcony WI-FI and Netflix&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Tiny home&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.93/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2041/night     &lt;/div&gt;`)[0];
                popup_4a9efc5afd92a828c0435b14d2028856.setContent(html_2e123e05bece6e2b1b05fc9fe0937b02);



        circle_marker_25fe3fb411110aada8cb57eaf66eea3a.bindPopup(popup_4a9efc5afd92a828c0435b14d2028856)
        ;




            var circle_marker_3740fdae7c95999f78c0f394f1889574 = L.circleMarker(
                [-1.2725, 36.8206],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_5825e1690b3525c0fb96c78475f875a3 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_06bec791310ab268a12bbeba7901bb0c = $(`&lt;div id=&quot;html_06bec791310ab268a12bbeba7901bb0c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Kazuri Ivy Serene &amp; Spacious Nairobi Apartment&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Entire rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; entire_home&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.83/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 4969/night     &lt;/div&gt;`)[0];
                popup_5825e1690b3525c0fb96c78475f875a3.setContent(html_06bec791310ab268a12bbeba7901bb0c);



        circle_marker_3740fdae7c95999f78c0f394f1889574.bindPopup(popup_5825e1690b3525c0fb96c78475f875a3)
        ;




            var circle_marker_29b7d08cbf19f4892356e0c4c7bfde79 = L.circleMarker(
                [-1.2941, 36.7741],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#2EC4B6&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#2EC4B6&quot;, &quot;fillOpacity&quot;: 0.7, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 4, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_9f59ba5306fc0ae6e73b99c769527a34);


        var popup_983899742eebbab37e3940b4d1a02917 = L.popup({
  &quot;maxWidth&quot;: 300,
});



                var html_310fbbd89bc808f5dbff0b1ed9b16f56 = $(`&lt;div id=&quot;html_310fbbd89bc808f5dbff0b1ed9b16f56&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;     &lt;b style=&#x27;font-size: 14px&#x27;&gt;Airy &amp; Light-Filled: Indoor-Outdoor Home-Stay&lt;/b&gt;&lt;br&gt;     &lt;b&gt;Type:&lt;/b&gt; Private room in rental unit&lt;br&gt;     &lt;b&gt;Room:&lt;/b&gt; private_room&lt;br&gt;     &lt;b&gt;Rating:&lt;/b&gt; 4.89/5.0&lt;br&gt;     &lt;b&gt;Price:&lt;/b&gt; KSh 2931/night     &lt;/div&gt;`)[0];
                popup_983899742eebbab37e3940b4d1a02917.setContent(html_310fbbd89bc808f5dbff0b1ed9b16f56);



        circle_marker_29b7d08cbf19f4892356e0c4c7bfde79.bindPopup(popup_983899742eebbab37e3940b4d1a02917)
        ;



&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



```python
from geopy.geocoders import Nominatim
import pandas as pd
from time import sleep

geolocator = Nominatim(user_agent="airbnb_nairobi_analysis")

def reverse_geocode(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        if location:
            address = location.raw.get('address', {})
            return pd.Series({
                'neighbourhood': address.get('neighbourhood'),
                'suburb': address.get('suburb'),
                'county': address.get('county'),
                'city': address.get('city'),
                'postcode': address.get('postcode')
            })
    except:
        return pd.Series([None]*5)

# Apply to dataset (⚠️ slow — sample first)
df[['neighbourhood', 'suburb', 'county', 'city', 'postcode']] = (
    df[['latitude', 'longitude']]
    .apply(lambda x: reverse_geocode(x[0], x[1]), axis=1)
)

```

    /tmp/ipykernel_54075/626640467.py:25: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      .apply(lambda x: reverse_geocode(x[0], x[1]), axis=1)


After executing the geocoding process, five new location-based columns were added to the dataset: **neighbourhood**, **suburb**, **county**, **city**, and **postcode**. While all five fields contain some missing values, **suburb** stands out as the most reliable geographic indicator, with only a single missing entry and **26 distinct areas** represented. As a result, suburb is used as the primary geographic dimension for analyzing spatial patterns within the dataset.

## Are there neighborhood clusters with similar pricing?


```python
suburbs = df['suburb'].unique()
suburbs
```




    array(['Kilimani division', 'Roysambu division', 'Karen',
           'Mugumo-ini ward', 'Highridge division', 'Karen Hardy',
           'CBD division', 'South B', 'Nyayo Highrise ward', 'Karen ward',
           'Kangemi division', 'Woodley/Kenyatta/Golf Course ward', 'Karen C',
           'Kawangware division', 'Kasarani location', 'Lower Savannah ward',
           'Harambee ward', 'South C ward', 'Imara Daima ward', None,
           'South C', 'Komarock ward', 'Tassia', 'Nairobi West', 'Pipeline',
           'Githurai division', 'Nairobi West ward'], dtype=object)



#### STEP 1: Aggregate pricing by suburb


```python
neighbourhoods_vs_pricing = df.groupby('suburb')['avg_rate_per_year'].mean().sort_values(ascending=True).round(2).reset_index()
neighbourhoods_vs_pricing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>suburb</th>
      <th>avg_rate_per_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nyayo Highrise ward</td>
      <td>1427.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South C</td>
      <td>1717.10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pipeline</td>
      <td>1819.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>South B</td>
      <td>2341.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mugumo-ini ward</td>
      <td>2796.90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nairobi West</td>
      <td>2857.30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CBD division</td>
      <td>3373.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Komarock ward</td>
      <td>4296.90</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Imara Daima ward</td>
      <td>4452.97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Harambee ward</td>
      <td>4460.60</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Githurai division</td>
      <td>5098.80</td>
    </tr>
    <tr>
      <th>11</th>
      <td>South C ward</td>
      <td>5177.70</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kangemi division</td>
      <td>5242.46</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Nairobi West ward</td>
      <td>5552.50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Kawangware division</td>
      <td>5684.97</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Roysambu division</td>
      <td>6030.61</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Woodley/Kenyatta/Golf Course ward</td>
      <td>6614.90</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kasarani location</td>
      <td>6667.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Kilimani division</td>
      <td>7002.30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Tassia</td>
      <td>7173.10</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Highridge division</td>
      <td>8471.39</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Karen C</td>
      <td>8600.96</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Karen</td>
      <td>8606.50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Karen Hardy</td>
      <td>10907.72</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Karen ward</td>
      <td>17108.63</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Lower Savannah ward</td>
      <td>37484.90</td>
    </tr>
  </tbody>
</table>
</div>



#### STEP 2: Prepare data for clustering


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = neighbourhoods_vs_pricing[['avg_rate_per_year']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### STEP 3: Apply KMeans clustering


```python
kmeans = KMeans(n_clusters=3, random_state=42)
neighbourhoods_vs_pricing['price_cluster'] = kmeans.fit_predict(X_scaled)

```

#### STEP 4: Inspect the clusters


```python
neighbourhoods_vs_pricing.sort_values('avg_rate_per_year')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>suburb</th>
      <th>avg_rate_per_year</th>
      <th>price_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nyayo Highrise ward</td>
      <td>1427.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South C</td>
      <td>1717.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pipeline</td>
      <td>1819.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>South B</td>
      <td>2341.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mugumo-ini ward</td>
      <td>2796.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nairobi West</td>
      <td>2857.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CBD division</td>
      <td>3373.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Komarock ward</td>
      <td>4296.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Imara Daima ward</td>
      <td>4452.97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Harambee ward</td>
      <td>4460.60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Githurai division</td>
      <td>5098.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>South C ward</td>
      <td>5177.70</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kangemi division</td>
      <td>5242.46</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Nairobi West ward</td>
      <td>5552.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Kawangware division</td>
      <td>5684.97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Roysambu division</td>
      <td>6030.61</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Woodley/Kenyatta/Golf Course ward</td>
      <td>6614.90</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kasarani location</td>
      <td>6667.00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Kilimani division</td>
      <td>7002.30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Tassia</td>
      <td>7173.10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Highridge division</td>
      <td>8471.39</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Karen C</td>
      <td>8600.96</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Karen</td>
      <td>8606.50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Karen Hardy</td>
      <td>10907.72</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Karen ward</td>
      <td>17108.63</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Lower Savannah ward</td>
      <td>37484.90</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### STEP 5: Visualization


```python
sns.barplot(
    data=neighbourhoods_vs_pricing,
    x='price_cluster',
    y='avg_rate_per_year'
)
plt.title("Average Nightly Rate by Neighborhood Price Cluster")
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_94_0.png)
    


*   **Cluster 0 (Economy/Budget):** This category contains the highest volume of listings, representing the lower-tier accommodations.
    
*   **Cluster 2 (Mid-range):** These listings represent the middle tier of the market.
    
*   **Cluster 1 (High-end/Outlier):** This cluster contains only a single listing, suggesting it is unique or an outlier."

## Do certain neighborhoods favor specific room types?

#### Create a neighborhood × room type distribution


```python
neighborhood_room_type = (
    df
    .groupby(['suburb', 'room_type'])
    .size()
    .reset_index(name='count')
)
```


```python
room_type_pivot = neighborhood_room_type.pivot(
    index='suburb',
    columns='room_type',
    values='count'
).fillna(0)
room_type_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>room_type</th>
      <th>entire_home</th>
      <th>hotel_room</th>
      <th>private_room</th>
    </tr>
    <tr>
      <th>suburb</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CBD division</th>
      <td>8.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Githurai division</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Harambee ward</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Highridge division</th>
      <td>31.0</td>
      <td>1.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>Imara Daima ward</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kangemi division</th>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Karen</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Karen C</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Karen Hardy</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Karen ward</th>
      <td>17.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Kasarani location</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kawangware division</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Kilimani division</th>
      <td>130.0</td>
      <td>0.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>Komarock ward</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Lower Savannah ward</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Mugumo-ini ward</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nairobi West</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nairobi West ward</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Nyayo Highrise ward</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Pipeline</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Roysambu division</th>
      <td>13.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>South B</th>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>South C</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>South C ward</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Tassia</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Woodley/Kenyatta/Golf Course ward</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
room_type_pivot_sorted = (
    room_type_pivot
    .assign(total=room_type_pivot.sum(axis=1))
    .sort_values('total')
    .drop(columns='total')
)

```


```python
room_type_pivot_sorted.plot(
    kind='barh',
    stacked=True,
    figsize=(12, 10)
)
plt.title("Room Type Distribution by Neighborhood")
plt.xlabel("Number of Listings")
plt.ylabel("Suburb")
plt.tight_layout()
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_101_0.png)
    


Neighborhood × Room Type Insights
---------------------------------

### 1\. **Strong geographic concentration**

*   **Kilimani division** overwhelmingly dominates the market, with the highest number of listings by a wide margin.
    
*   **Highridge division** follows as a distant second.
    
*   All other neighborhoods have relatively **low listing density**, many with fewer than 10 listings.
    

**Insight:** Airbnb supply in Nairobi is highly concentrated in a few prime neighborhoods rather than evenly distributed.

### 2\. **Entire homes dominate across neighborhoods**

*   **Entire homes** make up the majority of listings in almost every suburb.
    
*   This dominance is most pronounced in **Kilimani** and **Highridge**, indicating strong demand for full-unit stays in these areas.
    

**Interpretation:** These neighborhoods likely attract:

*   Longer stays
    
*   Families, business travelers, or groups
    
*   Guests prioritizing privacy and space
    

### 3\. **Private rooms play a secondary role**

*   **Private rooms** appear mainly in:
    
    *   Kilimani
        
    *   Highridge
        
    *   Karen-related areas
        
*   Their presence is limited elsewhere.
    

**Interpretation:** Private rooms serve as a **price-sensitive alternative** in high-demand areas but are not a primary offering city-wide.

### 4\. **Hotel rooms are extremely rare**

*   **Hotel room listings are almost nonexistent**, appearing only marginally in a few neighborhoods.
    

**Insight:** Airbnb supply in this dataset is driven by **individual hosts and residential properties**, not traditional hospitality players.

### 5\. **Neighborhood character matters**

*   **Karen ward** shows lower volume but a mix that leans toward **entire homes**, consistent with its low-density, high-end residential profile.
    
*   Peripheral neighborhoods show **minimal diversification**, often only one room type.

## Which areas have the highest revenue potential?


```python
area_revenue = (
    df
    .groupby('suburb')
    .agg(
        avg_annual_revenue=('revenue_per_year', 'mean'),
        avg_nightly_rate=('avg_rate_per_year', 'mean'),
        avg_occupancy=('annual_occupancy', 'mean'),
        listings_count=('listing_id', 'count')
    )
    .sort_values('avg_annual_revenue', ascending=False)
    .round(2)
)

area_revenue.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_annual_revenue</th>
      <th>avg_nightly_rate</th>
      <th>avg_occupancy</th>
      <th>listings_count</th>
    </tr>
    <tr>
      <th>suburb</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Karen ward</th>
      <td>1962535.22</td>
      <td>17108.63</td>
      <td>36.13</td>
      <td>18</td>
    </tr>
    <tr>
      <th>Karen</th>
      <td>811678.25</td>
      <td>8606.50</td>
      <td>35.85</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Nairobi West ward</th>
      <td>712218.00</td>
      <td>5552.50</td>
      <td>37.60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Tassia</th>
      <td>702593.00</td>
      <td>7173.10</td>
      <td>35.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Kangemi division</th>
      <td>661358.00</td>
      <td>5242.46</td>
      <td>21.89</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Karen Hardy</th>
      <td>648590.50</td>
      <td>10907.72</td>
      <td>18.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Kilimani division</th>
      <td>521143.23</td>
      <td>7002.30</td>
      <td>24.41</td>
      <td>151</td>
    </tr>
    <tr>
      <th>Highridge division</th>
      <td>509475.96</td>
      <td>8471.39</td>
      <td>23.61</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Harambee ward</th>
      <td>502857.00</td>
      <td>4460.60</td>
      <td>34.40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lower Savannah ward</th>
      <td>471464.00</td>
      <td>37484.90</td>
      <td>6.50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
area_revenue.head(10)['avg_annual_revenue'] \
    .sort_values() \
    .plot(kind='barh', figsize=(10, 6))

plt.title("Top 10 Areas by Average Annual Revenue")
plt.xlabel("Average Annual Revenue (KSh)")
plt.ylabel("Suburb")
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_105_0.png)
    


#### **1\. The Luxury Tier: Karen’s "High-Value" Dominance**

*   **Performance:** **Karen Ward** is the undisputed market leader, generating nearly **KSh 2,000,000** annually—more than double the next highest contender.
    
*   **The Strategy:** This dominance is driven by high **Average Daily Rates (ADR)** rather than sheer volume. These listings (villas, large homes) cater to the "Long Stay" market (diplomats, expats) where a single booking can secure months of revenue.
    
*   **Verdict:** High entry cost, but unrivaled revenue per booking.
    

#### **2\. The Efficiency Tier: The Hidden Gems (Tassia & Nairobi West)**

*   **Performance:** Surprisingly, lower-middle-income areas like **Nairobi West** (~KSh 700k) and **Tassia** (~KSh 680k) significantly outperform premium hubs.
    
*   **The Strategy:** These areas thrive on a **Volume/Occupancy** model. With lower nightly rates but consistent local demand (traders, short-term work stays), they avoid the "vacancy gaps" that plague seasonal tourist spots.
    
*   **Verdict:** Lower entry cost with higher yield efficiency.
    

#### **3\. The Saturated Tier: The "Popularity Trap" (Kilimani & Highridge)**

*   **Performance:** despite being the most famous rental hubs, **Kilimani** and **Highridge** appear in the bottom half of the top 10 (averaging ~KSh 500k).
    
*   **The Strategy:** These areas suffer from **Oversupply**. The sheer number of apartments creates fierce price competition, diluting the _average_ revenue per host. While top-tier units earn well, the "average" unit struggles to match the returns of less saturated areas.
    
*   **Verdict:** High demand is offset by extreme competition; success here requires a standout product.

# Quality & Features

## How many amenities do high-revenue listings typically have?

#### Step 1: Define high-revenue listings


```python
# Define high-revenue threshold
high_revenue_threshold = df['revenue_per_year'].quantile(0.75)

df['high_revenue'] = (
    df['revenue_per_year'] >= high_revenue_threshold
)
```

#### Step 2: Convert amenities into a count


```python
# Create amenities count
df['amenities_count'] = (
    df['amenities']
    .fillna('')
    .apply(lambda x: len(str(x).split(',')))
)

```

#### Step 3: Compare high vs non-high revenue listings


```python
amenities_vs_revenue = (
    df
    .groupby('high_revenue')['amenities_count']
    .agg(['mean', 'median', 'min', 'max', 'count'])
    .round(1)
)

amenities_vs_revenue

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>min</th>
      <th>max</th>
      <th>count</th>
    </tr>
    <tr>
      <th>high_revenue</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>34.8</td>
      <td>33.0</td>
      <td>11</td>
      <td>78</td>
      <td>225</td>
    </tr>
    <tr>
      <th>True</th>
      <td>38.2</td>
      <td>37.0</td>
      <td>14</td>
      <td>76</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



#### Step 4: Visualization


```python
sns.boxplot(
    data=df,
    x='high_revenue',
    y='amenities_count'
)

plt.xticks([0, 1], ['Lower Revenue', 'High Revenue'])
plt.title("Amenities Count vs Revenue Tier")
plt.ylabel("Number of Amenities")
plt.xlabel("")
plt.show()
```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_116_0.png)
    


From the box plot, **high-revenue listings typically have about** _**35–40 amenities**_.

More precisely:

*   The **median** (the line inside the box) is around **36–37 amenities**
    
*   Most high-revenue listings (the interquartile range) fall roughly between **30 and 42 amenities**
    

So a “typical” high-revenue listing offers **mid-to-high 30s in amenities**, slightly more than lower-revenue listings.

## What's the relationship between listing name length and bookings?

#### Step 1: Define the variables


```python
df['listing_name_length'] = (
    df['listing_name']
    .fillna('')
    .str.len()
)

```

#### Step 2: Measure correlation


```python
correlation = df['listing_name_length'].corr(
    df['reserved_days_in_year']
)

correlation
```




    np.float64(0.10077962524781812)



#### Weak Positive


```python
df['name_length_bin'] = pd.qcut(
    df['listing_name_length'],
    q=4,
    labels=['Short', 'Medium', 'Long', 'Very Long']
)

df.groupby('name_length_bin')['reserved_days_in_year'].mean().sort_values(ascending=False)

```

    /tmp/ipykernel_54075/1443461262.py:7: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby('name_length_bin')['reserved_days_in_year'].mean().sort_values(ascending=False)





    name_length_bin
    Very Long    86.812500
    Long         73.047059
    Short        70.526316
    Medium       65.720000
    Name: reserved_days_in_year, dtype: float64



#### Step 3: Visualize the relationship


```python
sns.scatterplot(
    data=df,
    x='listing_name_length',
    y='reserved_days_in_year',
    alpha=0.6
)

sns.regplot(
    data=df,
    x='listing_name_length',
    y='reserved_days_in_year',
    scatter=False,
    color='red'
)

plt.title("Listing Name Length vs Booked Nights per Year")
plt.xlabel("Listing Name Length (characters)")
plt.ylabel("Booked Nights per Year")
plt.show()

```


    
![png](airbnb_analysis_nairobi_files/airbnb_analysis_nairobi_126_0.png)
    


### Direction of the relationship

*   The **red regression line slopes slightly upward**, indicating a **weak positive relationship** between listing name length and booked nights per year.
    
*   This means that, _on average_, listings with longer names tend to receive **slightly more bookings**, but the effect is **not strong**.
    

### Strength of the relationship

*   The points are **widely scattered** around the line.
    
*   This high dispersion shows that **listing name length alone explains very little** of the variation in bookings.
    
*   Many listings with long names still have low bookings, and some short names perform very well.
    

**Conclusion:** The relationship exists, but it is **weak**.

### Practical interpretation

*   Longer names likely help by:
    
    *   Including keywords (location, amenities, “near CBD”, “luxury”, etc.)
        
    *   Improving clarity and search visibility
        
*   However, bookings are **far more influenced** by:
    
    *   Location
        
    *   Price
        
    *   Reviews
        
    *   Amenities
        
    *   Professional management


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>listing_name</th>
      <th>listing_type</th>
      <th>room_type</th>
      <th>cover_photo_url</th>
      <th>photos_count</th>
      <th>minimum_nights</th>
      <th>cancellation_policy</th>
      <th>professional_management</th>
      <th>registration</th>
      <th>...</th>
      <th>price_range</th>
      <th>neighbourhood</th>
      <th>suburb</th>
      <th>county</th>
      <th>city</th>
      <th>postcode</th>
      <th>high_revenue</th>
      <th>amenities_count</th>
      <th>listing_name_length</th>
      <th>name_length_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75683</td>
      <td>Kiloranhouse Apt Prime Bedroom</td>
      <td>Private room in home</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/5499026/ef...</td>
      <td>13</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Mid-range (5-10k)</td>
      <td>Kilimani ward</td>
      <td>Kilimani division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>30728</td>
      <td>False</td>
      <td>42</td>
      <td>30</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>471581</td>
      <td>Located In a Serene Environment</td>
      <td>Entire cottage</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/6434524/bc...</td>
      <td>37</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Mid-range (5-10k)</td>
      <td>Roysambu location</td>
      <td>Roysambu division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>31224</td>
      <td>True</td>
      <td>24</td>
      <td>31</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>906958</td>
      <td>Makena's Place Karen - Flamingo Room</td>
      <td>Private room in cottage</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/68ecc57f-d...</td>
      <td>29</td>
      <td>1</td>
      <td>Firm</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Mid-range (5-10k)</td>
      <td>None</td>
      <td>Karen</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>00505</td>
      <td>False</td>
      <td>30</td>
      <td>36</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1023556</td>
      <td>Guesthouse Near Nairobi National Park &amp; Airport</td>
      <td>Entire guesthouse</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/ddd8badc-1...</td>
      <td>20</td>
      <td>1</td>
      <td>Flexible</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Budget (0-5k)</td>
      <td>None</td>
      <td>Mugumo-ini ward</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>00517</td>
      <td>False</td>
      <td>33</td>
      <td>47</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1237886</td>
      <td>Hob House</td>
      <td>Room in bed and breakfast</td>
      <td>hotel_room</td>
      <td>https://a0.muscache.com/im/pictures/cbdab7e1-f...</td>
      <td>8</td>
      <td>1</td>
      <td>Flexible</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>Luxury (15k+)</td>
      <td>Highridge location</td>
      <td>Highridge division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>11403</td>
      <td>False</td>
      <td>42</td>
      <td>9</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>42123446</td>
      <td>Mvuli Luxury Suites</td>
      <td>Entire rental unit</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/238557fd-c...</td>
      <td>24</td>
      <td>1</td>
      <td>Firm</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Budget (0-5k)</td>
      <td>Ngara location</td>
      <td>CBD division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>45046</td>
      <td>False</td>
      <td>32</td>
      <td>19</td>
      <td>Short</td>
    </tr>
    <tr>
      <th>296</th>
      <td>42139551</td>
      <td>Modern 1BR | King Bed | Fast Wi-Fi | Near CBD</td>
      <td>Entire condo</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/a10f889a-4...</td>
      <td>27</td>
      <td>1</td>
      <td>Flexible</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Mid-range (5-10k)</td>
      <td>Kilimani ward</td>
      <td>Kilimani division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>30728</td>
      <td>False</td>
      <td>53</td>
      <td>45</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>297</th>
      <td>42187559</td>
      <td>Cosy &amp; Airy Studio with Balcony WI-FI and Netflix</td>
      <td>Tiny home</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/miso/Hosti...</td>
      <td>43</td>
      <td>1</td>
      <td>Moderate</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>Budget (0-5k)</td>
      <td>Kangemi location</td>
      <td>Kangemi division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>29326</td>
      <td>False</td>
      <td>60</td>
      <td>49</td>
      <td>Very Long</td>
    </tr>
    <tr>
      <th>298</th>
      <td>42207619</td>
      <td>Kazuri Ivy Serene &amp; Spacious Nairobi Apartment</td>
      <td>Entire rental unit</td>
      <td>entire_home</td>
      <td>https://a0.muscache.com/im/pictures/hosting/Ho...</td>
      <td>19</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Budget (0-5k)</td>
      <td>Ngara location</td>
      <td>CBD division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>45046</td>
      <td>False</td>
      <td>20</td>
      <td>46</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>299</th>
      <td>42223689</td>
      <td>Airy &amp; Light-Filled: Indoor-Outdoor Home-Stay</td>
      <td>Private room in rental unit</td>
      <td>private_room</td>
      <td>https://a0.muscache.com/im/pictures/fd146f8e-5...</td>
      <td>17</td>
      <td>2</td>
      <td>Moderate</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>Budget (0-5k)</td>
      <td>Kileleshwa location</td>
      <td>Kilimani division</td>
      <td>None</td>
      <td>Nairobi</td>
      <td>54102</td>
      <td>False</td>
      <td>55</td>
      <td>45</td>
      <td>Long</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 52 columns</p>
</div>



# Market Opportunities

## What's underrepresented in the market (low supply, high demand)?

#### Create price bands


```python
price_bins = [0, 3000, 6000, 10000, 15000, df['avg_rate_per_year'].max()]
price_labels = [
    'Low (0–3k)',
    'Lower-Mid (3k–6k)',
    'Mid (6k–10k)',
    'Upper-Mid (10k–15k)',
    'Premium (15k+)'
]

df['price_band'] = pd.cut(
    df['avg_rate_per_year'],
    bins=price_bins,
    labels=price_labels,
    include_lowest=True
)

```

#### Define supply and demand


```python
market_summary = (
    df.groupby(['room_type', 'price_band'])
      .agg(
          listings_count=('listing_id', 'count'),
          avg_occupancy=('annual_occupancy', 'mean'),
          avg_revenue=('revenue_per_year', 'mean')
      )
      .reset_index()
)

```

    /tmp/ipykernel_54075/1390846018.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby(['room_type', 'price_band'])


#### Establish “low supply” and “high demand” thresholds


```python
supply_threshold = market_summary['listings_count'].median()
occupancy_threshold = market_summary['avg_occupancy'].median()
revenue_threshold = market_summary['avg_revenue'].median()

```

#### Flag underrepresented segments


```python
market_summary['underrepresented'] = (
    (market_summary['listings_count'] < supply_threshold) &
    (
        (market_summary['avg_occupancy'] > occupancy_threshold) |
        (market_summary['avg_revenue'] > revenue_threshold)
    )
)

underrepresented_segments = market_summary[
    market_summary['underrepresented']
].sort_values(['avg_occupancy', 'avg_revenue'], ascending=False)

underrepresented_segments

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>room_type</th>
      <th>price_band</th>
      <th>listings_count</th>
      <th>avg_occupancy</th>
      <th>avg_revenue</th>
      <th>underrepresented</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>private_room</td>
      <td>Premium (15k+)</td>
      <td>5</td>
      <td>5.32</td>
      <td>363316.2</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## Are there price gaps where new listings could compete?

#### Measure supply per price band


```python
supply_by_price = (
    df.groupby('price_band')
      .agg(
          listings_count=('listing_id', 'count'),
          avg_price=('avg_rate_per_year', 'mean')
      )
      .reset_index()
)

```

    /tmp/ipykernel_54075/1042693843.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby('price_band')


#### Measure demand/performance per price band


```python
performance_by_price = (
    df.groupby('price_band')
      .agg(
          avg_occupancy=('annual_occupancy', 'mean'),
          avg_revenue=('revenue_per_year', 'mean')
      )
      .reset_index()
)

```

    /tmp/ipykernel_54075/2880606656.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby('price_band')


#### Combine supply + performance


```python
price_gap_analysis = supply_by_price.merge(
    performance_by_price,
    on='price_band'
)

price_gap_analysis.sort_values('listings_count')

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_band</th>
      <th>listings_count</th>
      <th>avg_price</th>
      <th>avg_occupancy</th>
      <th>avg_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Premium (15k+)</td>
      <td>21</td>
      <td>23956.961905</td>
      <td>23.223810</td>
      <td>1.874120e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Upper-Mid (10k–15k)</td>
      <td>30</td>
      <td>11954.396667</td>
      <td>22.013333</td>
      <td>8.627922e+05</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Low (0–3k)</td>
      <td>41</td>
      <td>2346.365854</td>
      <td>23.697561</td>
      <td>1.668768e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lower-Mid (3k–6k)</td>
      <td>97</td>
      <td>4616.963918</td>
      <td>23.797938</td>
      <td>3.463240e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mid (6k–10k)</td>
      <td>111</td>
      <td>7618.214414</td>
      <td>23.310811</td>
      <td>5.501002e+05</td>
    </tr>
  </tbody>
</table>
</div>



#### Identify price gaps


```python
price_gap_analysis['potential_gap'] = (
    (price_gap_analysis['listings_count'] < price_gap_analysis['listings_count'].median()) &
    (price_gap_analysis['avg_occupancy'] > price_gap_analysis['avg_occupancy'].median())
)

price_gap_analysis

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_band</th>
      <th>listings_count</th>
      <th>avg_price</th>
      <th>avg_occupancy</th>
      <th>avg_revenue</th>
      <th>potential_gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low (0–3k)</td>
      <td>41</td>
      <td>2346.365854</td>
      <td>23.697561</td>
      <td>1.668768e+05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lower-Mid (3k–6k)</td>
      <td>97</td>
      <td>4616.963918</td>
      <td>23.797938</td>
      <td>3.463240e+05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mid (6k–10k)</td>
      <td>111</td>
      <td>7618.214414</td>
      <td>23.310811</td>
      <td>5.501002e+05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Upper-Mid (10k–15k)</td>
      <td>30</td>
      <td>11954.396667</td>
      <td>22.013333</td>
      <td>8.627922e+05</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Premium (15k+)</td>
      <td>21</td>
      <td>23956.961905</td>
      <td>23.223810</td>
      <td>1.874120e+06</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Which listing types have declining revenue trends?

Revenue trend analysis typically requires time-series data. As this dataset provides a snapshot of listing performance rather than longitudinal observations, it does not support direct analysis of declining or increasing revenue trends by listing type. Instead, the analysis focuses on comparing current revenue performance across listing categories.
