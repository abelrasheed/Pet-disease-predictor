import pandas as pd
import random
import csv

class symp:
    def __init__(self, name = None):
        self.name = name
        self.count = 1
class Disease:
    def __init__(self, name = None, max_val = 0):
        self.name = name
        self.count = 0 #number of examples of this particular disease in the initial dataset
        self.selectionPool = []
        self.max_val = max_val #max number of symptoms for a the given disease( ie it will be the no of symptoms in a tuple with max number of symptoms)

#returns symptom object, symptom list,disease object nd disease list
def symp_counter(csv_reader):
    symp_list = []
    symp_obj = []
    disease_list = []
    disease_obj = []
    for line in csv_reader:
        #Defining the symptoms object and list
        temp_line = line[1:]
        for i in temp_line:
            if i in symp_list:
                for j in symp_obj:
                    if(j.name == i):
                        j.count +=1
            else:
                symp_list.append(i)
                obj = symp(i)
                symp_obj.append(obj)
        #Defining the disease object and list
        if line[0] not in disease_list:
            disease_list.append(line[0])
            obj1 = Disease(line[0],len(line)-1)   
            disease_obj.append(obj1)
        for item in disease_obj:
            if (item.name == line[0]):  
                index_disease_obj = disease_obj.index(item)
                break
        disease_obj[index_disease_obj].count +=1
        disease_obj[index_disease_obj].selectionPool.extend(temp_line)
        if(disease_obj[index_disease_obj].max_val<len(line)-1):
            disease_obj[index_disease_obj].max_val = len(line)-1
    return symp_list,symp_obj,disease_list,disease_obj

def iniData(symp_list,csv_reader):
    temp_matrix = [] #matrix to be returned, in the binary form, symptoms only
    disease_index = [] #name of all diseases in the order as it appears on the temp_matrix( for the corresponding symptoms)
    for line in csv_reader:
        disease_index.append(line[0])
        temp_line = line[1:]
        temp_row = dataArrange(symp_list,temp_line)
        temp_matrix.append(temp_row)  
    return temp_matrix,disease_index

def egGenerater(pool,num):
    count = random.randint(1,num-2)
    eg_list = []
    counter = 0
    while True:
        rand = random.choice(pool)
        if rand in eg_list:
            continue
        else:
            eg_list.append(rand)
        counter += 1
        if(counter == count):
            return eg_list

# Convert data into trainable format
# symp_list is the symptoms list and data_list is the collection of symptoms to be converted to the required format
def dataArrange(symp_list,data_list):
    temp_row= [0]*len(symp_list)
    for i in symp_list:        
        for j in data_list:
            if(i == j):
                t = symp_list.index(i)
                temp_row[t] = 1
                break
    return temp_row

#delets duplicate symptoms
def del_duplicates(csv_reader):

    with open('new_raw_data.csv' , 'w',newline = '') as towrite_file:
        csv_writer = csv.writer(towrite_file)
        for line in csv_reader:
            new_row = []
            for j in line:
                if j not in new_row:
                    new_row.append(j)
            csv_writer.writerow(new_row)

    
def symp_process(entry):
    entry = entry.lower()
    entry = entry.replace(" ","")
    return entry

def del_duplicates_from_symptoms(symptom_list):
    processed = []
    for i in symptom_list:
        if i not in processed:
            processed.append(i)
    return processed
