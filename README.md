### Rnn Predicor
---

#### TO-DO List
A guideline 
- [ ] **Documentation**
    - [ ] add architecture diagram
    - [ ] explain folder structure
    - [ ] how to run the repository (setup)
    - [ ] create a TO-DO list
- [ ] **Model**
    - [x] implement basic architecture supporting LSTM cell and MLP currently taking as input an independent series of days and outputs a prediction for a specific number of days 
    - [ ] idd model for embedding static data (population information) 
    - [ ] implement a deeper model (add support for multi cell architecture)
    - [ ] find a way to personalize a model for each country (possibilities: train globally and finetune for each country, investigate the "Social" approach.   
    
- [ ] **Data**
    - [x] implent naive handler class for raw **Kaggle Data** that takes data from all countries, merges them(stack), and apply a sliding windows to create batches 
    - [x] naive method for spliting the data (train/test)
    - [ ] investigate the option for doing cross validation
    - [ ] improve how the train/test data is splited (currently randomly select from batches)
    - [ ] investigate if logarithmic scaling can produce results is benefical
    - [x] add data augmentation (like addition, and scaling)
    - [x] dowload and process external data from **rescountries.eu**
    - [ ] add support for population data from **rescountries.eu**
    
- **Utils**
     - [x] add data online visualization of training process (loss, evaluate, learning rate scheduling)
     - [ ] add support for online prediction of all countries 
     
- **Kaggle**
    - [ ] add script for creating submissions
     
---

### Data Analysis
#### SIR (Susceptible Infected Recovered/Deceased) model
 - SIR is the mathematical modeling of infectious disease

#### MultiInput model
 - Generally, researchers use one country as input for prediction its corona spread
 - It is expected that using many countries, grouped by a similarity measure, as input-output will be more certain
 - It will be possible to use differences of spread dynamics as an advantage for the prediction of future values

#### 0day alignment
 - Similar countries having different first infection day will be shifted in time axis to be alignment
 - This will emphasize more obviously the difference between infection grow for the algorithm
 - Typically this method is used for human-readable plots instead of model input

### Additional Data
#### Containment and mitigation measures dataset
- Provides description and start-dates of containment measures against COVID-19
- Correlates with a decreasing number of new cases
- Most promising additional data
- Next step: find categories for description using word clouds or similar

#### Weather data
- [OpenWeatherMap](https://openweathermap.org/) provides historical weather information and forecast
- Next steps: look further into data what kind of data is provided and how to access

#### Public transport data
- Too hard to obtain for each country individually, but very detailed overview is provided for some countries
- Google Transit API could be promising but is aimed at public transport companies to provide data for Google maps
- Next steps: find broader statistics for public transport

#### General country data
- REST API to general information such as area and population: https://restcountries.eu
