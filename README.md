## Sales Price Prediction


### 1. Problem Definition
> How well can we predict the future sale price of a bulldozer, given its characteristics previous examples of how much similar bulldozers have been sold for?

### 2. Data

Looking at the [dataset from Kaggle](https://www.kaggle.com/c/bluebook-for-bulldozers/data), there are 3 datasets:
1. **Train.csv** - Historical bulldozer sales examples up to 2011 (close to 400,000 examples with 50+ different attributes, including `SalePrice` which is the **target variable**).
2. **Valid.csv** - Historical bulldozer sales examples from January 1 2012 to April 30 2012 (close to 12,000 examples with the same attributes as **Train.csv**).
3. **Test.csv** - Historical bulldozer sales examples from May 1 2012 to November 2012 (close to 12,000 examples but missing the `SalePrice` attribute, as this is what will be trying to predict).

### 3. Evaluation

For this problem, [Kaggle has set the evaluation metric to being root mean squared log error (RMSLE)](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation). As with many regression evaluations, the goal will be to get this value as low as possible.

To see how well our model is doing, we'll calculate the RMSLE

### 4. Features
Kaggle provides a data dictionary detailing all of the features of the [dataset](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing)


```python
import pandas as pd
df= pd.read_excel("../data/kaggle-bluebook-for-bulldozers-data-dictionary.xlsx")
dfn = df.copy()
del dfn['Unnamed: 2']
dfn
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
      <th>Variable</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SalesID</td>
      <td>unique identifier of a particular sale of a ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MachineID</td>
      <td>identifier for a particular machine;  machin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ModelID</td>
      <td>identifier for a unique machine model (i.e. ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>datasource</td>
      <td>source of the sale record;  some sources are...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>auctioneerID</td>
      <td>identifier of a particular auctioneer, i.e. ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>YearMade</td>
      <td>year of manufacturer of the Machine</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MachineHoursCurrentMeter</td>
      <td>current usage of the machine in hours at tim...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>UsageBand</td>
      <td>value (low, medium, high) calculated compari...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Saledate</td>
      <td>time of sale</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Saleprice</td>
      <td>cost of sale in USD</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fiModelDesc</td>
      <td>Description of a unique machine model (see M...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fiBaseModel</td>
      <td>disaggregation of fiModelDesc</td>
    </tr>
    <tr>
      <th>12</th>
      <td>fiSecondaryDesc</td>
      <td>disaggregation of fiModelDesc</td>
    </tr>
    <tr>
      <th>13</th>
      <td>fiModelSeries</td>
      <td>disaggregation of fiModelDesc</td>
    </tr>
    <tr>
      <th>14</th>
      <td>fiModelDescriptor</td>
      <td>disaggregation of fiModelDesc</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ProductSize</td>
      <td>Don't know what this is</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ProductClassDesc</td>
      <td>description of 2nd level hierarchical groupi...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>State</td>
      <td>US State in which sale occurred</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ProductGroup</td>
      <td>identifier for top-level hierarchical groupi...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ProductGroupDesc</td>
      <td>description of top-level hierarchical groupi...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Drive_System</td>
      <td>machine configuration;  typcially describes wh...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Enclosure</td>
      <td>machine configuration - does machine have an e...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Forks</td>
      <td>machine configuration - attachment used for li...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Pad_Type</td>
      <td>machine configuration - type of treads a crawl...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ride_Control</td>
      <td>machine configuration - optional feature on lo...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Stick</td>
      <td>machine configuration - type of control</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Transmission</td>
      <td>machine configuration - describes type of tran...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Turbocharged</td>
      <td>machine configuration - engine naturally aspir...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Blade_Extension</td>
      <td>machine configuration - extension of standard ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Blade_Width</td>
      <td>machine configuration - width of blade</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Enclosure_Type</td>
      <td>machine configuration - does machine have an e...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Engine_Horsepower</td>
      <td>machine configuration - engine horsepower rating</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Hydraulics</td>
      <td>machine configuration - type of hydraulics</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Pushblock</td>
      <td>machine configuration - option</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Ripper</td>
      <td>machine configuration - implement attached to ...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Scarifier</td>
      <td>machine configuration - implement attached to ...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Tip_control</td>
      <td>machine configuration - type of blade control</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Tire_Size</td>
      <td>machine configuration - size of primary tires</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Coupler</td>
      <td>machine configuration - type of implement inte...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Coupler_System</td>
      <td>machine configuration - type of implement inte...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Grouser_Tracks</td>
      <td>machine configuration - describes ground conta...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Hydraulics_Flow</td>
      <td>machine configuration - normal or high flow hy...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Track_Type</td>
      <td>machine configuration - type of treads a crawl...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Undercarriage_Pad_Width</td>
      <td>machine configuration - width of crawler treads</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Stick_Length</td>
      <td>machine configuration - length of machine digg...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Thumb</td>
      <td>machine configuration - attachment used for gr...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Pattern_Changer</td>
      <td>machine configuration - can adjust the operato...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Grouser_Type</td>
      <td>machine configuration - type of treads a crawl...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Backhoe_Mounting</td>
      <td>machine configuration - optional interface use...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Blade_Type</td>
      <td>machine configuration - describes type of blade</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Travel_Controls</td>
      <td>machine configuration - describes operator con...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Differential_Type</td>
      <td>machine configuration - differential type, typ...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Steering_Controls</td>
      <td>machine configuration - describes operator con...</td>
    </tr>
  </tbody>
</table>
</div>


