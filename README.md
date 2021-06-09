# Pet-disease-predictor

del.py  -- __main__ file

data_preprocess - function called by main for preprocessing the data_preprocess

data_preprocess_util - This file conntains all the utilitary functions for data preprocessing

symptoms.csv - Contains all the initial raw training data

new_raw_data.csv - Contains all the data after removing duplicates

new_data - Contains all the raw data after all the preprocessing steps

Dataset.csv - Contains the entire dataset including the generated examples in the trainable pandas format

decision_tree_scratch - Contains the decision tree algorithm

model - Contains the final model(object of decision tree) for classification as a pickle dump file

symp_model - Contains the symptom_list as a pickle dump file

encode_model - The sklearn encoding file for symptoms

Using commadline - python del.py (predict/train) --symptoms(only for predict)"a" "b" "c" (a,b and c are arbitrary symptoms)

disease_obj_list = pickle file containing all the disease objects