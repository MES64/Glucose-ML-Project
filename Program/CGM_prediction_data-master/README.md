# Datasets for CGM prediction

* Included are train and test datasets, with 756 and 324 records each.
* The data was obtained from the paper <https://arxiv.org/abs/2003.01283>
* For each record, you have a 1-day trajectory with 1-minute resolution (hence, 1440 time-points)
* You have three kinds of variables:
  * CGM values (`glucose_readings_*.csv`)
  * Insulin values (`insulin_therapy_*.csv`)
  * Ingested carbs (`meals_carbs_*.csv`)
* The objective is to predict future CGM values given past CGM, insulin, and carbs values
