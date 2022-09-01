import os
import re
import numpy as np
import statistics
import time

celeb_attributes_names = ["5_o_Clock_Shadow",
                          "Arched_Eyebrows",
                          "Attractive",
                          "Bags_Under_Eyes",
                          "Bald",
                          "Bangs",
                          "Big_Lips",
                          "Big_Nose",
                          "Black_Hair",
                          "Blond_Hair",
                          "Blurry",
                          "Brown_Hair",
                          "Bushy_Eyebrows",
                          "Chubby",
                          "Double_Chin",
                          "Eyeglasses",
                          "Goatee",
                          "Gray_Hair",
                          "Heavy_Makeup",
                          "High_Cheekbones",
                          "Male",
                          "Mouth_Slightly_Open",
                          "Mustache",
                          "Narrow_Eyes",
                          "No_Beard",
                          "Oval_Face",
                          "Pale_Skin",
                          "Pointy_Nose",
                          "Receding_Hairline",
                          "Rosy_Cheeks",
                          "Sideburns",
                          "Smiling",
                          "Straight_Hair",
                          "Wavy_Hair",
                          "Wearing_Earrings",
                          "Wearing_Hat",
                          "Wearing_Lipstick",
                          "Wearing_Necklace",
                          "Wearing_Necktie",
                          "Young"]

""" 

NOMENCLATURE

image_name: name of the image file ('00001.jpg')
image_id: id of the image (00001)
person_id: id of the person represented in the image (248)

"""


def image_name2image_id(image_name):
    return int(image_name[:-4])


def load_celeba_identities(celeba_identities_files):
    """"
    input: celeba_identities_file - path to the file containing CELEB-A IDs

        identity_CelebA.txt

        image_name_1 person_id_1
        ...
        image_name_n person_id_n


    output: identity_info - dictionary of the list image names per id

        identity_info[person_id] -> (image_name_1, ..., image_name_n)
        image_info[image_id] -> person_id
    """

    identity_info = dict()
    image_info = dict()
    with open(celeba_identities_files) as identities:
        lines = identities.readlines()
        for identity in lines:
            identity = identity.rstrip().lstrip().split()
            # we have 2 infos per line, image name and identity id
            if len(identity) != 2:
                continue
            image_name = identity[0]
            identity_id = int(identity[1])

            if identity_id not in identity_info:
                identity_info[identity_id] = []
            identity_info[identity_id].append(image_name)
            image_info[image_name2image_id(image_name)] = identity_id

    return identity_info, image_info


def load_celeba_bb(celeba_bb_file):
    """"
    input: celeba_bb_file - path to the file containing CELEB-A BBs

        list_bbox_celeba.txt
        N (HEADER)
        image_id x y width height (HEADER)
        image_id_1 x_1 y_1 width_1 height_1
        ...
        image_id_n x_1 y_n width_n height_n


    output: identity_info - dictionary of the bb names per image name

        bb_info[image_id] -> image_bb_1

    """

    bb_info = dict()
    with open(celeba_bb_file) as bb_file:
        lines = bb_file.readlines()
        lines = lines[2:]  # discard header
        for line in lines:
            line_data = line.rstrip().lstrip().split()

            if len(line_data) != 5:
                continue

            image_id = image_name2image_id(line_data[0])
            bb = np.array((int(line_data[1]), int(line_data[2]), int(line_data[3]), int(line_data[4])))

            bb_info[image_id] = bb

    return bb_info


def load_celeba_attrs(attributes_path):
    attibutes_map_list = []
    atttributes_per_image = {}
    num_images = 0
    with open(attributes_path, 'r') as atributes_file:
        lines = atributes_file.readlines()
        assert (len(lines) > 3)

        # first line is the number of images line
        num_images = int(lines[0])
        # second line is the header
        for line in lines[2:]:
            values = re.split(" +", line)
            img_name = values[0]
            attibutes_map = {}
            attributes_arr = []
            for i in range(0, len(celeb_attributes_names)):
                attr = celeb_attributes_names[i]
                value = values[i + 1]

                attibutes_map[attr] = int(value)
                attributes_arr.append(int(value))

            attibutes_map["image"] = img_name
            attibutes_map_list.append(attibutes_map)
            atttributes_per_image[img_name] = np.array(attributes_arr)
    # print("found ", len(celeb_data), ' files')
    # print("expected ", num_images)

    return attibutes_map_list, atttributes_per_image


def get_samples_with_attribute(samples, attrib_name):
    selected_samples = []
    for sample in samples:
        try:
            if sample[attrib_name] == 1:
                selected_samples.append(sample)
        except KeyError:
            print(sample)
    return selected_samples


def get_attr_name(attr_path):
    attr_name_dict = {}
    with open(attr_path, "r") as attr_file:
        lines = attr_file.readlines()

        attr_name = lines[1].rstrip().split()

        for attr in attr_name:
            attr_name_dict[attr] = 0

    return attr_name_dict


def get_stats_per_attribute(attr_path):
    males = 0
    females = 0

    attr_name_dict_male = get_attr_name(attr_path)
    attr_name_dict_female = get_attr_name(attr_path)

    with open(attr_path, "r") as attr_file:
        lines = attr_file.readlines()
        for line in lines[2:]:
            values = line.rstrip().split()
            if int(values[21]) == 1:
                males += 1
                for i, val in enumerate(values[1:]):  # first value is the image name
                    attr_name_dict_male[celeb_attributes_names[i]] += 1 if int(val) == 1 else 0
            else:
                females += 1
                for i, val in enumerate(values[1:]):  # first value is the image name
                    attr_name_dict_female[celeb_attributes_names[i]] += 1 if int(val) == 1 else 0


    print("Males: {0}".format(males))
    print("Females: {0}".format(females))
    max_value = max(attr_name_dict_male.values())
    print("Max value: {0}".format(max_value))
    print([k for k, v in attr_name_dict_male.items() if v == max_value])
    max_value = max(attr_name_dict_female.values())
    print("Max value: {0}".format(max_value))
    print([k for k, v in attr_name_dict_female.items() if v == max_value])

    return males, females, attr_name_dict_male, attr_name_dict_female


def get_attr_per_samples(attr_path, samples):
    attr_list=[]
    with open(attr_path, "r") as attr_file:
        lines = attr_file.readlines()
        for line in lines[2:]:
            values = line.rstrip().split()
            if values[0] in samples:
                attr_list.append([int(v) for v in values[1:]])

    attr_list = np.asarray(attr_list)
    return attr_list


if __name__ == '__main__':
    identity_info, image_info = load_celeba_identities('/home/socialab/Desktop/Joao/datasets/CELEBA/identity_CelebA.txt')
    attibutes_map_list, atttributes_per_image = load_celeba_attrs('/home/socialab/Desktop/Joao/datasets/CELEBA/list_attr_celeba.txt')
    n_identities = len(identity_info.keys())

    tic = time.time()
    person_name_list = []
    person_att_list = []
    persons = list(identity_info.keys())
    persons.sort()
    for key in persons:
        attrs = []
        for img in identity_info[key]:
            attrs.append(atttributes_per_image[img])

        attrs = np.asarray(attrs)
        att_vec = np.mean(attrs, axis=0)

        person_name_list.append(str(key))
        person_att_list.append(att_vec)

    with open("../datasets/celeba/persons_celeba.txt", mode="w") as file:
        file.write("\n".join(person_name_list))

    np.save('../datasets/celeba/atts_celeba.npy', np.array(person_att_list))

    toc = time.time()
    print("[INFO]: Time elapsed - {:.2f}".format((toc-tic)/60))

