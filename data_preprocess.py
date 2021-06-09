import pandas as pd
from disease_prediction.data_preprocess_util import symp_counter , iniData, egGenerater, dataArrange, del_duplicates, symp_process
import csv 
# from data_preprocess_util import iniData 
import numpy as np
# from data_preprocess_util import egGenerater
# from data_preprocess_util import dataArrange 
# from data_preprocess_util import del_duplicates
# from data_preprocess_util import symp_process
import pickle

def data_preprocess() :
    #removing duplicates from the symptoms
    with open('symptoms.csv' , 'r') as csv_file0:
        csv_reader0 = csv.reader(csv_file0)    
        del_duplicates(csv_reader0)


    with open('new_raw_data.csv' , 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #making the symptom list to send from the end point
        symp_list_to_be_sent = []
        for line in csv_reader:
            temp_line = line[1:]
            for i in temp_line:
                if (len(symp_list_to_be_sent) == 0):
                    symp_list_to_be_sent.append(i)
                else:
                    flag = 0
                    for j in symp_list_to_be_sent:
                        if(i.lower() == j.lower()):
                            flag = 1
                    if (flag == 0):
                        symp_list_to_be_sent.append(i)
        with open('symp_list_to_send','wb') as symptom_file:
            pickle.dump(symp_list_to_be_sent,symptom_file)
        csv_file.seek(0)
        #applying remaining processing of symptoms such as removal of caps and spaces
        with open('new_data.csv' , 'w',newline = '') as csv_file1:
            csv_writer = csv.writer(csv_file1)
            for line in csv_reader:
                new_row = []
                for i in line:
                    i = symp_process(i)
                    new_row.append(i)
                csv_writer.writerow(new_row) 


    with open('new_data.csv' , 'r') as csv_file:
        csv_reader = csv.reader(csv_file) 
        symp_list,counter_obj,disease_list,disease_obj = symp_counter(csv_reader)

            
        csv_file.seek(0)
        revised_iniData_matrix,disease_index  = iniData( symp_list, csv_reader)
        #making the actual dataset as a pd dataframe
        revised_iniData= pd.DataFrame(revised_iniData_matrix,columns =  symp_list,index = disease_index)
        # saving the symptom list and disease object as a pickle file
        with open('symp_model','wb') as model_file:
            pickle.dump(symp_list,model_file)
        with open('disease_obj_list','wb') as model_file:
            pickle.dump(disease_obj,model_file)

    # generating new data 
    for i in range(len(disease_obj)):
        #creating 10 examples for each disease
        for j in range(20):
            item = egGenerater(disease_obj[i].selectionPool,disease_obj[i].max_val)
            item_to_append  = dataArrange(symp_list,item) 
            item_df = pd.DataFrame([item_to_append],columns = symp_list,index = [disease_obj[i].name])
            revised_iniData = revised_iniData.append(item_df)
            


    return revised_iniData

