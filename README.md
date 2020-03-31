

### Data Analysis
#### SIR (Susceptible Infected Recovered/Deceased) model
 - SIR is mathematical modelling of infectious disease

#### MultiInput model
 - Generally researchers use one country as input for prediction its corona spread
 - It is expected that using many countries, grouped by similarity measure, as input output will be more certain
 - It will be possible to use differences of spread dynamics as advantage for prediction of future values

#### 0day alignment
 - Similar contries having different first infection day will be shifted in time axis to be alignment
 - This will emphasise more obsiously the difference between infection grow for the algorithm
 - Typically this method is used for human readable plots instead of model input

### Additional Data
#### Containment and mitigation measures dataset
- Provides description and start-dates of containment measures against Covid-19
- Correlates with decreasing number of new cases
- Most promising additional data
- Next step: find categories for description using word clouds or similar

#### Weather data
- [OpenWeatherMap](https://openweathermap.org/) provides historic weather information and forecast
- Next steps: look further into data what kind of data is provided and how to access

#### Public transport data
- Too hard to obtain for each country individually, but very detailed overview is provided for some countries
- Google Transit API could be promising but is aimed at public transport companies to provide data for Google maps
- Next steps: find broader statistics for public transport

#### General country data
- REST API to general information such as area and population: https://restcountries.eu