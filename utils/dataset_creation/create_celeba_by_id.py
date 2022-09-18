
import re
import os
import shutil

celeb_attributes_names =  ["5_o_Clock_Shadow",
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


def load_celeba_identities(celeba_identities_files):
    identity_info = dict()
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

    return identity_info


def load_celeba_attrs(attributes_path):

    celeb_data = []
    num_images = 0
    with open(attributes_path, 'r') as atributes_file:
        lines = atributes_file.readlines()
        assert(len(lines) > 3)

        # first line is the number of images line
        num_images = int(lines[0])
        # second line is the header
        for line in lines[2:]:
            values = re.split(" +", line)
            img_name = values[0]
            attibutes_map = {}
            for i in range(0, len(celeb_attributes_names)):
                attr = celeb_attributes_names[i]
                value = values[i + 1]

                attibutes_map[attr] = int(value)
            attibutes_map["image"] = img_name

            celeb_data.append(attibutes_map)
    # print("found ", len(celeb_data), ' files')
    # print("expected ", num_images)

    return celeb_data


def get_samples_with_attribute(samples, attrib_name):
    selected_samples = []
    for sample in samples:
        try:
            if sample[attrib_name] == 1:
                selected_samples.append(sample)
        except KeyError:
            print(sample)
    return selected_samples


celeba_path = '/home/socialab/Desktop/Joao/projects/insightface/src/align/celeba-aligned_112/img_align_celeba'

celeba_dict = load_celeba_identities('/home/socialab/Desktop/Joao/datasets/CELEBA/identity_CelebA.txt')

for person_name in celeba_dict:

    images = celeba_dict[person_name]
    person_name = str(person_name)
    os.makedirs(os.path.join('../../datasets/celeba/', person_name + '/'), exist_ok=True)

    for image in images:

        #image = image.replace('.jpg','.png')
        src_path = os.path.join(celeba_path, image)
        dst_path = os.path.join('../../datasets/celeba/', person_name, image)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)


