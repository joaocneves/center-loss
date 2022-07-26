import numpy as np

pairs_filename = 'C:\\Users\\joao_\\Dropbox\\LFW_ManualAnnotations.txt'

persons = dict()
img_id = 1
last_person_name = ''

with open(pairs_filename, 'r') as f:

    for line in f.readlines():

        line_parts = line.strip().split()
        img_name_parts = line_parts[0].split('_')
        person_name = line_parts[0].replace(img_name_parts[-1],'')[:-1]

        if not person_name == last_person_name:
            last_person_name = person_name
            persons[person_name] = [[int(line_parts[1]), int(line_parts[2]), int(line_parts[3])]]
        else:
            persons[person_name].append([int(line_parts[1]), int(line_parts[2]), int(line_parts[3])])

person_name_list = []
person_att_list = []
for key in persons:

    att_list = persons[key]

    # This code shows that not every has the same attributes for all images
    # ----------------------------------------------------
    eq_i = [att_list[i] == att_list[i-1] for i in range(1,len(att_list))]
    if not all(eq_i):
        stop = 1
    # -----------------------------------------------------

    att_mat = np.array(att_list)
    att_vec = np.mean(att_mat, axis=0)

    person_name_list.append(key)
    person_att_list.append(att_vec)


with open("datasets\\lfw\\persons_lfw.txt", mode="w") as file:
    file.write("\n".join(person_name_list))

np.save('datasets\\lfw\\atts_lfw.npy', np.array(person_att_list))




