from sklearn.model_selection import train_test_split
import os
import random
import shutil
import cv2


def resize_and_rename_images(root_directory, class_names, lenght, width):
    counter = 0
    for class_name in class_names:
        src = root_directory + str(class_name)+'/'
        image_names = os.listdir(path=src)
        print(type(image_names))
        for image_name in image_names:
            print(src+image_name+'\n')
            image = cv2.imread(src+image_name)
            resized_image = cv2.resize(image, (lenght, width))
            cv2.imwrite('../datasets/resized/'+class_name+'/'+class_name+'_'+str(counter)+'.jpeg', resized_image)
            counter += 1
                

def rename_images(image_dir, new_name_prefix, counting_number=0):
    for image_file in os.listdir(image_dir):
        old_image_path = os.path.join(image_dir, image_file)
        new_image_path = os.path.join(image_dir, new_name_prefix +'_'+ str(counting_number) + '.jpeg')
        os.rename(old_image_path, new_image_path)
        counting_number += 1
        


def calculate_train_val_test_sizes(total_sizes_each_class, val_percetage, test_percetage):
    val_size = [int(x * val_percetage) for x in  total_sizes_each_class]
    test_size = [int(x * test_percetage) for x in  total_sizes_each_class]
    train_size = [(x-y)- z for x, y, z in zip(total_sizes_each_class, val_size, test_size)]
    return train_size, val_size, test_size


        
def split_images_into_train_validation_test(raw_dirctory, target_directory, class_names, val_sizes, test_sizes):
    for class_name in class_names:
        src = raw_dirctory + str(class_name)+'/'
        my_pics = os.listdir(path=src)
        class_index =  class_names.index(class_name)
        for i in range(test_sizes[class_index]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            shutil.copyfile(src=src+pic_name, dst=target_directory+'test/'+class_name+'/' + str(pic_name))
        for i in range(val_sizes[class_index]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            shutil.copyfile(src=src+pic_name, dst=target_directory + 'val/'+class_name+'/' + str(pic_name))
        for i in my_pics:
            pic_name = i
            shutil.copyfile(src=src+pic_name, dst=target_directory + 'train/'+class_name+'/' + str(pic_name))
            

    

