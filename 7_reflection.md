# Reflection

## What you have achieved so far
- Completed Notebook 03 (remaining from Session 5 checkpoint), plotting the relationship between various weather factors
- Outlined an evaluation framework: chose PR-AUC (average precision) as the primary metric before training any models. Accuracy is rare (~4%) and ROC-AUC is optimistic on imbalanced data, so PR-AUC would be a better reflection of how well a model predicts actual spikes. 
- Trained a simple model - logistic regression with 4 variants, 5-fold CV. Best model selected by mean per-fold PR AUC (L1_balanced) 
- Advanced models -- Random Forest and KNN. Best models of each type selected using mean per-fold PR-AUC
- The ranking of which model is better depends on the metric: logistic regression is better using PR-AUC, and RF is much better at the precision-recall metric. KNN does not do well on any metric. 
- Assessed if models performed better on recent data (yes)

What you are happy with, from your project work so far
- Completed Notebook 03, finally trained some models and got some results
- Completed what I wanted to in this checkpoint 

What you are struggling with, or what challenges you are facing next
- All my models (and notebook 03) were only using data from one hub (HB_Houston). So I need to re-run and re-train models on more data going forward to see if my preliminary conclusions still hold 
- Also, the results so far, while better than random chance, don't fully predict spikes. I don't know if this means that the next stage (predicting the magnitude of the spike instead of just the presence or absence of a spike) will give even worse results 

Anything else you’d specifically like the course staff to focus on in giving you feedback or advice
- How should I deal with the Uri outliers? Removing them altogether seems like its manipulating data, but perhaps I could wait for the results for spike prediction to assess its impact. 
- Splitting spikes by type would introduce a lot more complexity, but the disagreements between folds suggest that it could improve performance of the model too. Not sure if I should proceed with splitting the spikes or just sticking to the binary classification of spike vs no spike that I have so far. 
