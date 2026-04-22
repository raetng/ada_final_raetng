# Scoping

Problem Scoping: Define the prediction task I am trying to complete here. Also, lay out how I will accomplish this task: which models am I training? Which data am I using (e.g. wind speed, temperature, cloud cover, natural gas price)?

## What is the prediction task I am trying to complete? 
### Can we predict spike magnitude and/or spike occurrence? 
**Spike magnitude: can we predict this?**
- Filter to only spike hours (above 95th percentile)
- Compute correlation between price magnitude and each predictor 
- Fit a simple linear regression between price magnitude and each predictor 
- Create a scatter plot of actual spike price vs predicted spike price from the linear regression 
- Use this to decide if spike magnitude is predictable, or if it is predictable only for certain spike magnitudes (e.g. those between 95th to 99th percentile), or "types" of spikes, e.g. winter cold vs summer heat

**Which factors predict spike magnitude and/or spike occurrence?**
- Drop the least relevant factors from the linear regression run above and run it again 
- Execute the model training plan from the next question 
- Create lag features and run the linear regressions again (temperature change over the past 6 hours, or past week's gas prices)

**Which data am I using?**
- temperature_2m                        
- apparent_temperature                                                             
- relative_humidity_2m                                                             
- dewpoint_2m                                                                    
- precipitation                          
- cloud_cover                                                                    
- wind_speed_10m                                                                   
- wind_speed_100m                       
- wind_direction_10m
- shortwave_radiation                                                              
- direct_normal_irradiance 

## Which models do I plan to train? 
- First, the logistic regression
  - If there is still time, I could consider using a GLM. The fat-tailed, right skewed distribution of the price data means that being able to specify a non-Gaussian error distribution could aid in model accuracy.
- Then, forward stepwise regression
- Next, PCR
- Next, the decision tree
- Then, a KNN classifier
- Last, the random forest   