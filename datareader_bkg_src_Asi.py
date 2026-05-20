'''
a data reader module that 
access the simulated Fermi-LAT images 
the original image npy files have shape [5,64,64,1]
first component in the array represents background (0) and four different source components (1-4), h = w = 64
'''

import os
import numpy as np


import pandas as pd


  
###########################
# Read and load npy image #
###########################

def read_npy_data(base_dir:str, channel:int, sqrt:int, train:bool, norm:bool):
    print("read numpy data")
    source_dict, source_files = {}, {}
    total_dict, total_files = {}, {}
    # nums_set = set()

    if train: 
        num_range = set(range(1000, 800000))
    else:
        num_range = set(range(753001, 753407))    


    #for subdir in ['only_sources', 'x_data_new_npy']:
        #folder_path = os.path.join(base_dir, subdir)
    folder_path = base_dir
    print(folder_path)
    #print("before for loop")
    for filename in os.listdir(folder_path):
        #print("in loop")
        if filename.endswith(".npy"):
            #print("Yes")
            # Extract the number from the filename
            number = int(filename.split('_')[2].split('.')[0])
            #print(number)
            if number not in num_range:
                continue
            #print("in range")
            file_path = os.path.join(folder_path, filename)
            im_data = np.load(file_path)
            if channel==0 or channel==1:
                im_data = im_data[:, :, :, channel:channel+1]
            else: im_data = im_data[:, :, :, channel]
            #print(im_data)
            if sqrt==1:
                im_data = np.sqrt(im_data)
            else:
                im_data = im_data
            
            if norm:
                max_value = np.max(im_data)
                im_data = im_data/max_value

            #print(im_data)
            source_dict[number] = im_data
            #print(source_dict)
            source_files[number] = filename
            # if subdir=='only_sources':
            #     source_dict[number] = im_data
            #     source_files[number] = filename
            # elif subdir=='x_data_new_npy':
            #     total_dict[number] = im_data
            #     total_files[number] = filename

    #print("source: ",source_dict.keys())
    #print("total: ",total_dict.keys())
    valid_nums = set(source_dict.keys()) #&  set(total_dict.keys())
    print ('len valid nums: ', len(valid_nums))
    nums_list = sorted(valid_nums)

    source_list = [source_dict[num] for num in nums_list if num in source_dict]
    source_files_list = [source_files[num] for num in nums_list if num in source_files]


    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    # total_list = [total_dict[num] for num in nums_list if num in total_dict]
    # total_files_list = [total_files[num] for num in nums_list if num in total_files]

    # nums_list.sort()

    # return (source_list, total_list, nums_list, source_files_list, total_files_list)
    return (source_list, nums_list, source_files_list)    



####################################
# Read and load npy latent vectors #
####################################

def read_npy_latent(base_dir:str, channel:int):
    print("read numpy data")
    z_dict, z_files = {}, {}
    total_dict, total_files = {}, {}
    # nums_set = set()
 
    num_range = set(range(1000, 800000))
   


    #for subdir in ['only_sources', 'x_data_new_npy']:
        #folder_path = os.path.join(base_dir, subdir)
    folder_path = base_dir
    print(folder_path)
    #print("before for loop")
    for filename in os.listdir(folder_path):
        #print("in loop")
        if filename.endswith(".npy"):
            #print("Yes")
            # Extract the number from the filename
            number = int(filename.split('_')[2].split('.')[0])
            #print(number)
            if number not in num_range:
                continue
            #print("in range")
            file_path = os.path.join(folder_path, filename)
            z_data = np.load(file_path)
            #print(im_data)

            #print(im_data)
            z_dict[number] = z_data
            #print(source_dict)
            z_files[number] = filename
            # if subdir=='only_sources':
            #     source_dict[number] = im_data
            #     source_files[number] = filename
            # elif subdir=='x_data_new_npy':
            #     total_dict[number] = im_data
            #     total_files[number] = filename

    #print("source: ",source_dict.keys())
    #print("total: ",total_dict.keys())
    valid_nums = set(z_dict.keys()) #&  set(total_dict.keys())
    print ('len valid nums: ', len(valid_nums))
    nums_list = sorted(valid_nums)

    z_list = [z_dict[num] for num in nums_list if num in z_dict]
    z_files_list = [z_files[num] for num in nums_list if num in z_files]


    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    # total_list = [total_dict[num] for num in nums_list if num in total_dict]
    # total_files_list = [total_files[num] for num in nums_list if num in total_files]

    # nums_list.sort()

    # return (source_list, total_list, nums_list, source_files_list, total_files_list)
    return (z_list, nums_list, z_files_list)  


