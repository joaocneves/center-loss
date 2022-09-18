import json
import os
import re
import numpy as np
import time

from scipy.stats import stats

"""
[Name_File][GENDER][AGE][ETHNICITY][FOREHEAD][MOUTH][EYES][GLASSES][SMILING][BEARD][MOUSTACHE][POSE]

Gender
           Male-->0 
           Female-->1

Age          
            Baby-->0                     
            Child-->1                       
            Youth-->2               
            Middle_Aged-->3       
            Senior-->4          

Ethnicity         
            White-->0          
            Black-->1             
            Asian-->2           
            Indian-->3          
            Other_Mixture-->4   

Forehead     
            Fully_Visible-->0  
            Partially_Visible-->1  
            Obstructed-->2  

Mouth        
            Open_Widely-->0    
            Partially_Open-->1   
            Close-->2              

Eyes         
            Open-->0   
            Partially_open-->1 
            Close-->2                  

Glasses     
            No_Glasses-->0   
            Eye_Wear-->1        



Smiling      
            Yes-->0     
            No-->1       


beard
            yes-->0 
            No-->1  

moustache
            yes-->0 
            No-->1  

Pose      
            Frontal-->0   
            Left_Side-->1   
            Right_Side-->2 
"""

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_lfw_softbiometrics_dataset(manual_attributes_file):

    persons = dict()
    img_id = 1
    last_person_name = ''

    atttributes_per_image = dict()
    with open(manual_attributes_file, 'r') as f:

        for line in f.readlines():

            line_parts = line.strip().split()
            img_name = line_parts[0]
            img_name_parts = img_name.split('_')
            person_name = img_name.replace(img_name_parts[-1], '')[:-1]

            atttributes_per_image[os.path.join(person_name, img_name)] = [int(line_parts[i]) for i in range(1, len(line_parts))]

            if not person_name == last_person_name:
                last_person_name = person_name
                persons[person_name] = [[int(line_parts[i]) for i in range(1, len(line_parts))]]
            else:
                persons[person_name].append([int(line_parts[i]) for i in range(1, len(line_parts))])

        return persons, atttributes_per_image

def create_attribute_files_standard_format(persons):

    """
    Creates the files containing the attributes of each image and each person of the dataset

    Inputs:
        persons - dictionary of the list image names per id
            persons[person_id] -> (att_image_1, ..., att_image_n)
            image_info[image_id] -> person_id


    Outputs (complying with the standard format):
        atributes_per_person: dict()
           [person_id] -> dict
                ['mean'] -> 1x40 ndarray with the average value each attribute along the person_id images
                ['majority'] -> 1x40 ndarray with the most frequent value of each attribute along the person_id images
                ['median'] -> 1x40 ndarray with the median value each attribute along the person_id images


        atributes_per_image dict()
           [image_id] -> 1x40 ndarray with the value each attribute for the image
    """

    tic = time.time()
    attributes_per_person = dict()

    for key in persons:

        att_list = persons[key]

        # This code shows that not every person has the same attributes for all images
        # ----------------------------------------------------
        eq_i = [att_list[i] == att_list[i - 1] for i in range(1, len(att_list))]
        if not all(eq_i):
            stop = 1
        # -----------------------------------------------------

        attrs = np.array(att_list)


        attributes_per_person[key] = dict.fromkeys(['mean', 'majority', 'median'], [])
        attributes_per_person[key]['mean'] = np.around(np.mean(attrs, axis=0), decimals=2)
        attributes_per_person[key]['majority'] = stats.mode(attrs, axis=0).mode[0]
        attributes_per_person[key]['median'] = np.median(attrs, axis=0)



    toc = time.time()
    print("[INFO]: Time elapsed - {:.2f}".format((toc - tic) / 60))

    return attributes_per_person


manual_attributes_file = '/home/socialab/Joao/datasets/LFW_SoftBiometrics/files/LFW_ManualAnnotations.txt'

persons, atttributes_per_image = read_lfw_softbiometrics_dataset(manual_attributes_file)
attributes_per_person = create_attribute_files_standard_format(persons)

with open("../../datasets/lfw/attributes_per_person_lfw.json", "w") as file:
    json.dump(attributes_per_person, file, cls=NumpyEncoder)

with open("../../datasets/lfw/attributes_per_image_lfw.json", "w") as file:
    json.dump(atttributes_per_image, file, cls=NumpyEncoder)




