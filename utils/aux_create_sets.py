import os
import scipy.io

def _create_generic_lfw_set(images_root, images_list, labels):

    set = []
    for img_name, label in zip(images_list, labels):
        img_name_parts = img_name.split('_')
        person_name = img_name.replace(img_name_parts[-1], '')[:-1]

        image_path = os.path.join(images_root, person_name, img_name)

        set.append((image_path, int(label), img_name))

    return set

def create_lfw_blufr_gallery_probes_sets(images_root, blufr_lfw_config_file, closed_set=False):
    mat = scipy.io.loadmat(blufr_lfw_config_file)

    image_list = mat['imageList'][:, 0]
    labels = mat['labels'][:, 0]
    for i in range(10):

        gallery_index = mat['galIndex'][i][0][:, 0] - 1
        probes_index = mat['probIndex'][i][0][:, 0] - 1

        gallery_images_list = [str(image_list[i][0]) for i in gallery_index]
        gallery_labels = [labels[i] for i in gallery_index]
        probes_images_list = [str(image_list[i][0]) for i in probes_index]
        probes_labels = [labels[i] for i in probes_index]

        if closed_set:
            _probes_images_list = [el for i, el in enumerate(probes_images_list) if probes_labels[i] in gallery_labels]
            _probes_labels = [el for i, el in enumerate(probes_labels) if probes_labels[i] in gallery_labels]
            probes_images_list = _probes_images_list
            probes_labels = _probes_labels


        gallery_set = _create_generic_lfw_set(images_root, gallery_images_list, gallery_labels)
        probes_set = _create_generic_lfw_set(images_root, probes_images_list, probes_labels)

    return gallery_set, probes_set