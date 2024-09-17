import os
import glob
import copy
import json

import pandas as pd
import numpy as np
import math

from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

import cv2 as cv
import skimage
import h5py
import alphashape
import trimesh
import matplotlib.pyplot as plt

#########################
###     Metadata      ###
#########################

filetype1 = 'TIF'
filetype2 = 'PNG'
channels1 =     {'ch1':'DAPI','ch2':'GFP','ch3':'mCherry','ch4':'BF'}
channels1_inv = {'DAPI':'ch1','GFP':'ch2','mCherry':'ch3','BF':'ch4'}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size']  = 8

#########################
###     Utils         ###
#########################

def get_pathnames_to_images(data_dir:str, filetype:str='TIF'):
    """
    Get pathnames to all images of the input type in the input data directory

    Input:
        data_dir (str): pathname to a folder with image data.
        filetype (str, optional): filetype. Defaults to TIF.

    Output:
        list[str]: pathnames to individual image files
    """    
    
    # global pathname to any image of input type in the directory
    im_glob = os.path.join(data_dir,'*'+filetype)

    # pathnames to all images of the input type in the directory
    im_list = glob.glob(im_glob)
    
    return im_list

def extract_metadata(file_path:str, channel_dict:dict[str]=channels1) -> pd.DataFrame:
    """
    Based on image file naming conventions, parse a single pathname to an image file to extract image metadata.
    
    ***Customize this function based on file naming conventions as needed***

    Input:
        file_path (str): path to an image of interest
        channel_dict (dict[str], optional):  dict with keys as channel IDs (e.g. 'ch1') and values as channel name (e.g. 'DAPI').
                                            Defaults to channels1.

    Output:
        pd.DataFrame: data frame with image metadata for each image pathname. df columns:
        + FilePath - pathname to image file (str)
        + File - image file name (str)
        + Rat - rat cage ID (int)
        + Section - section ID (int)
        + ChannelID - imaging channel ID (str), e.g. 'ch1', 'ch2', 'ch3', or 'ch4'
        + Channel - imaging channel name (str), e.g. 'DAPI', 'GFP', 'mCherry', or 'BF'
    """ 
    
    # Parse image file path and extract metadata based on file naming conventions
    str_list_1 = file_path.split('\\')
    str_list_2 = str_list_1[-1].split('-')
    str_list_3 = str_list_2[-1].split('.')
    
    # Get file name, channel, section and rat information
    file    = str_list_1[-1] # file name    
    rat     = int(str_list_2[-3][1:]) # rat cage ID
    section = int(str_list_2[-2][1:]) # section ID
    channel_id = str_list_3[-2][:3] # channel ID
    channel = channel_dict[channel_id] # channel name
    
    # Save results as a data frame
    df = pd.DataFrame.from_dict({'FilePath':[file_path],
                                 'File':[file],
                                 'Rat':[rat],
                                 'Section':[section],
                                 'ChannelID':[channel_id],
                                 'Channel':[channel]})
    
    return df

def extract_metadata_all_images(rat:int, channel_dict:dict[str]=channels1, filetype:str=filetype1) -> pd.DataFrame:
    """
    Based on image file naming conventions, parse pathnames to image files to extract image metadata.

    Input:
        rat (int): rat ID of interest. Master rat folder should be parallel to pipeline folder.
                    It should contain tif subdir with stitched images in tif format
        channel_dict (dict[str], optional): dict with keys as channel IDs (e.g. 'ch1') and values as channel name (e.g. 'DAPI').
                                            Defaults to channels1.
        filetype (str, optional): image file extension. Defaults to filetype1.

    Output:
        pd.DataFrame: data frame with image metadata for each image pathname. df columns:
        + FilePath - pathname to image file (str)
        + File - image file name (str)
        + Rat - rat cage ID (int)
        + Section - section ID (int)
        + ChannelID - imaging channel ID (str), e.g. 'ch1', 'ch2', 'ch3', or 'ch4'
        + Channel - imaging channel (str), e.g. 'DAPI', 'GFP', 'NISSL'/'mCherry', or 'BF'
    """
    
    # Define input directory
    data_dir = '../R' + str(rat) + '/tif/'
    
    # Get pathnames to each image in the data direcotry
    im_list = get_pathnames_to_images(data_dir,filetype=filetype)
    
    # Iterate over each image
    for i in range(len(im_list)):

        file_path_i = im_list[i]
        
        # Get current image metadata
        df_i = extract_metadata(file_path_i,channel_dict=channel_dict)
        
        # Store in the master data frame
        if i == 0:
            df = df_i
        else:
            df = pd.concat([df,df_i])
    
    return df

#########################
### Image Conversions ###
#########################

def flip_images(df_meta,axis=1):
    """
    Flip images along the vertical (1) or horizontal (0) axis
    Make sure that all images are greyscale (those that need flipping and those that do no)
    
    Input:
    df_meta (df) -  image metadata
    axis (int): 1 if the sections should be flipped along vertical axis
                2 if the sections should be flipped along horizontal axis
           
    Output:
    None (overwrite images in the original directory)
    """
    
    # Get experiment metadata
    rat = df_meta.Rat.unique()[0]
    exp_metadata_dir = '../R' + str(rat) + '/R' + str(rat) + '-metadata.csv'
    df_meta_exp = pd.read_csv(exp_metadata_dir,comment='#')
    # Get sections that need flipping
    sections_to_flip = df_meta_exp.loc[df_meta_exp.ToFlip == 1,'Section'].values
    
    # Iterate over each section
    sections = df_meta.Section.unique() # all sections
    for section in sections:
        
        df = df_meta[df_meta.Section==section]
        
        # Iterate over each channel
        channels = df.Channel.values
        for channel in channels:
            
            # Read the image
            im_path = df.loc[df.Channel==channel,'FilePath'].values[0]
            im = cv.imread(im_path,-1) # Note that cv imports color images as BGR
            
            # Reduce data to grayscale
            if len(im.shape) > 2: 
                if channel == 'DAPI':
                    im_gray = im[:,:,0] # For blue color image, take the 0th array (Bgr)
                elif channel == 'GFP':
                    im_gray = im[:,:,1] # For green color image, take the 1st array (bGr)
                elif channel == 'mCherry':
                    im_gray = im[:,:,2] # for red color image, take the 2nd array (bgR)
                elif channel == 'BF':
                    im_gray = im # BF images are not colored
            else:
                im_gray=im
            
            # If applicable, flip the image
            if section in sections_to_flip:
                im_gray = cv.flip(im_gray, axis)
                
            # Overwrite the image file even if flipping is not applicable
            # (to make sure that data format is consistent)
            cv.imwrite(im_path,im_gray)
    
    return

def convert_to_h5(df_meta,channel='DAPI',scale_fraction=1.0):
    """
    Convert image files in df_meta to .h5 format and save them to output directory
    
    Input:
    df_meta (df): image metadata, including input image file names and paths to them
    channel (str): name of channel for which file converstion should be performed for
                also the name of the output subdirectory in h5 folder
    scale_fraction (float): <1.0 if image should be reduced in size
                            =1.0 if original image size should be preserved
    
    Output:
    none
    """
    
    # Reduce data to channel of interest
    df_meta = df_meta[df_meta.Channel == channel]
    
    # Define output directory
    rat = df_meta.Rat.unique()[0]
    output_dir = '../R' + str(rat) + '/h5/' + channel + '/'
    
    # Get input image information
    file_paths = df_meta.FilePath.values
    file_names = df_meta.File.values
   
    # Iterate over each image file
    for i in range(len(file_paths)):
                
        # Create a new name for .h5 file
        im_name = file_names[i]
        filetype = im_name.split('.')[-1]
        hf_name = im_name.replace(filetype,'h5')
                
        # Create .h5 file
        im = skimage.io.imread(file_paths[i], as_gray=True)
        width  = int(im.shape[1] * scale_fraction)
        height = int(im.shape[0] * scale_fraction)
        dims = (width, height)
        im_resized = cv.resize(im, dims)
        im_resized = skimage.img_as_ubyte(im_resized)
        
        # Save .h5 file
        hf = h5py.File(os.path.join(output_dir,hf_name), 'w') # create a new h5 file path
        hf.create_dataset('data', data=im_resized) # populate h5 file path with data
        hf.close() # saves h5 file to drive
        
    return

def merge_h5(df_meta_1sec,output_dir,channels_to_merge=['BF','GFP','DAPI']):
    """
    Merge three channel images of a single section into a single h5 RGB image
    
    Input:
    df_meta_1sec (df) - image metadata of just one section
                    containing the 3 files that need to be merged
    output_dir (str): directory where merged h5 files will be saved to
    channels_to_merge (list[str]): the three channels that should be merged into RGB h5 image
                The following order will be assumed:
                R: channels_to_merge[0]
                G: channels_to_merge[1]
                B: channels_to_merge[2]
                len(channels_to_merge)=3
                
    Output:
    None
    """
    # Get paths to all three channel images that need to be combined
    r_dir = df_meta_1sec.loc[df_meta_1sec.Channel==channels_to_merge[0],'FilePath'].values[0]
    g_dir = df_meta_1sec.loc[df_meta_1sec.Channel==channels_to_merge[1],'FilePath'].values[0]
    b_dir = df_meta_1sec.loc[df_meta_1sec.Channel==channels_to_merge[2],'FilePath'].values[0]
    
    # Open all three channel images
    r_img = skimage.io.imread(r_dir, as_gray=True)
    g_img = skimage.io.imread(g_dir, as_gray=True)
    b_img = skimage.io.imread(b_dir, as_gray=True)

    # Create a blank output image that has three channels 
    # and the same number of xy pixels as the original input
    multi_channel_img = np.zeros((r_img.shape[0], r_img.shape[1], 3))

    # Add channels to the output image
    multi_channel_img [:,:,0] = r_img
    multi_channel_img [:,:,1] = g_img
    multi_channel_img [:,:,2] = b_img

    # Save the multi channel .h5 file
    r_name = df_meta_1sec.loc[df_meta_1sec.Channel==channels_to_merge[0],'File'].values[0]
    output_name = r_name[:-5] + 'a.h5' # create output image name
    multi_channel_img_dir = output_dir + output_name # full path to the output image
    with h5py.File(multi_channel_img_dir, 'w') as hf: # create a new h5 file path
         # populate h5 file path with data
        hf.create_dataset('data', data=multi_channel_img, chunks = (multi_channel_img.shape[0],1,3),compression=True)
        
    return

def merge_h5_all_sections(df_meta,main_channel='GFP',channels_to_merge=['BF','GFP','DAPI']):
    """
    For each section in df_meta, merge the images of three channels into a single h5 image file
    
    Input:
    df_meta (df): image metadata (FilePath, File, Rat, Section, Channel)
    main_channel (str): name of main channel in which objects of interest where visualized in
                also the name of output subdirectory in h5 folder
    channels_to_merge (list[str]): the three channels that should be merged into RGB h5 image
                The following order will be assumed:
                R: channels_to_merge[0]
                G: channels_to_merge[1]
                B: channels_to_merge[2]
                len(channels_to_merge)=3
    """
    # Define output directory
    rat = df_meta.Rat.unique()[0]
    output_dir = '../R' + str(rat) + '/h5/' + main_channel + '/'
    
    # Iterate over each section
    sections = df_meta.Section.unique()
    for section in sections:
        # Get all current section data
        df_i = df_meta[df_meta.Section == section]
        # Merge the three channels of interest and export as h5
        merge_h5(df_i,output_dir,channels_to_merge)
        
    return

def convert_for_quickNII(df_meta,scale_fraction=0.2):
    """
    Convert image files in df_meta to .jpeg and saves them to output directory
    Change the order of file name
    Resize images as needed
    
    Input:
    df_meta (df): image metadata, including image file names and paths to them
    scale_fraction (float): <1.0 if image should be reduced in size
                            =1.0 if original image size should be preserved
    
    Output:
    none
    """
    # Define output directory
    rat = df_meta.Rat.unique()[0]
    output_dir = '../R' + str(rat) + '/QuickNII/jpeg/'
    
    # Get input image info
    file_paths = df_meta.FilePath.values
    file_names = df_meta.File.values
    
    # Iterate through each image file
    for i in range(len(file_paths)):
        
        # Create a new .jpeg file name
        # Swap the order from Rat-Section-Channel to Rat-Channel-Section
        im_name = file_names[i]
        name_list = im_name.split('-')
        rat = name_list[0]
        section = name_list[1]
        channel = name_list[2].split('.')[0]
        jpeg_name = rat + '-' + channel + '-' + section + '.jpeg'
                
        # import the original image file
        im = skimage.io.imread(file_paths[i], as_gray=True)
        
        # resize the image
        width  = int(im.shape[1] * scale_fraction)
        height = int(im.shape[0] * scale_fraction)
        dims = (width, height)
        im_resized = cv.resize(im, dims)
        im_resized = skimage.img_as_ubyte(im_resized)
        
        # save image as .jpeg
        skimage.io.imsave(os.path.join(output_dir,jpeg_name),im_resized,check_contrast = False)
        
    return

def downscale(df_meta,output_dir,scale=0.25):
    """
    Reduce the size of input images
    
    Input:
    df_meta (df): image metadata, including input image file names and paths to them
    output_dir (str): directory for converted files
    scale_fraction (float): <1.0 if image should be reduced in size
                            =1.0 if original image size should be preserved
    """
    
    file_paths = df_meta.FilePath.values
    file_names = df_meta.File.values
   
    # Iterate over each image file
    for i in range(len(file_paths)):
        
        # Open current image
        file_path = file_paths[i]
        im = skimage.io.imread(file_path, as_gray=True)
        
        # Resize current image
        width  = int(im.shape[1] * scale)
        height = int(im.shape[0] * scale)
        dims = (width, height)
        im_resized = cv.resize(im, dims)
        im_resized = skimage.img_as_ubyte(im_resized)
        
        # Save resized image
        file_name = file_names[i]
        skimage.io.imsave((output_dir + file_name),
                          im_resized,check_contrast = False)
        
    return

def change_format(input_dir,input_type,output_type):
    """
    Change image format of the images in input_dir
    
    Input:
    input_dir (str): directory to images
    input_type (str): type of input images that need to be converted
    ouptut_type (str): new image type
    """
    
    # get pathnames to images that need to be converted
    im_list = get_pathnames_to_images(input_dir,filetype=input_type)
    
    # Iterate over each image and change its format
    for im_path in im_list:
        im = skimage.io.imread(im_path)
        skimage.io.imsave(im_path.replace(input_type,output_type),
                          im,check_contrast=False)
        
    return

#################################
### Ilastik Output Processing ###
#################################

def process_tissue_masks(rat:int,channel:str='DAPI',hole_thresh:int=100000,object_thresh:int=10000000,remove_from_name:str='_Simple Segmentation_', filetype:str=filetype1) -> None:
    """
    Process tissue mask files:
    |   If applicable: Rename mask files in the input directory (specifically, remove redundant phrases)
    |   close small holes
    |   remove small objects
    |   watershed to remove extra tissue contours
    |   enlarge watershed image and take convex hull
    |   combine convex hull with original image
    |   generate alpha shape on combined image and apply to image
    
    |   Save processed masks files in the output directory

    :param rat_dir: rat ID
    |   Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R1234')
    |   Master rat folder should have IlastikSegmentation subfolder
    |   with further subdfolders to the original and processed Ilastik output:
    |   rat_dir/IlastikSegmentation/[channel] (raw ilastik output to be processed)
    |   rat_dir/IlastikSegmentation/[channel]-processed (processed ilastik output to be generated here)
    :type rat: int
    :param channel: channel name, defaults to 'DAPI'
    :type channel: str, optional
    :param hole_thresh: the largest hole size in pixels that needs to be filled, defaults to 100000
    :type hole_thresh: int, optional
    :param object_thresh: the largest object size in pixels that needs to  be removed as noise, defaults to 10000000
    :type object_thresh: int, optional
    :param shorten_name: True if ilastik mask name should be shortened, defaults to False
    :type shorten_name: bool, optional
    :param remove_from_name:  redundant expression to be removed from mask file names, if applicable. defaults to '_Simple Segmentation'
    :type remove_from_name: str, optional
    :param filetype: mask file type, e.g. '.TIF'. defaults to filetype1
    :type filetype: str, optional
    """    
    
    # Define input and output directories (create them accordingly)
    rat_dir    = '../R' + str(rat)
    input_dir  = rat_dir + '/IlastikSegmentation/' + channel + '\\' # folder with Ilastik output (unprocessed mask files)
    output_dir = rat_dir + '/IlastikSegmentation/' + channel + '-processed\\' # folder where processed masks will be saved
    
    # Get pathnames to mask files
    file_list = get_pathnames_to_images(input_dir,filetype=filetype)
    
    # Process each mask file
    for i in range(len(file_list)):
        
        # Get tissue mask
        file_i = file_list[i]
        mask_i = skimage.io.imread(file_i)
        mask_i = (mask_i == 1)

        # Fill small holes
        mask_i = skimage.morphology.remove_small_holes(mask_i,
                                                       area_threshold=hole_thresh)
        
        # Remove small objects
        mask_i = skimage.morphology.remove_small_objects(mask_i,
                                                         min_size=object_thresh).astype(np.uint8)
        
        # Dilate the mask to avoid edge effects
        mask_i = skimage.morphology.binary_dilation(mask_i,footprint=np.ones((5, 5)))
        
        # Save processed tissue mask
        skimage.io.imsave((output_dir + file_i.replace(remove_from_name,'').split('\\')[-1]),
                          skimage.img_as_ubyte(mask_i*255))
        
    return 

def process_drug_masks(rat:int,channel:str='GFP',fill_holes:bool=True,hole_thresh:int=5,remove_small:bool=True,object_thresh:int=16,remove_from_name:str='_Simple Segmentation',filetype:str=filetype1) -> None:
    """
    Process drug mask files:
    |   If applicable: Rename mask files in the input directory (specifically, remove redundant phrases)
    |   If applicable: Remove small objects
    |   If applicable: Fill small holes
    |   If applicable: apply tissue mask (take bitwise AND)
    |   Save processed masks files in the output directory

    :param rat: rat ID
    |   Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R1234')
    |   Master rat folder should have IlastikSegmentation subfolder
    |   with further subdfolders to the original and processed Ilastik output
    |   rat_dir/IlastikSegmentation/[channel] (raw ilastik output to be processed)
    |   rat_dir/IlastikSegmentation/[channel]-processed (processed ilastik output to be generated here)
    :type rat_dir: int
    :param channel: channel name (e.g. GFP or mCherry), defaults to 'GFP'
    :type channel: str, optional
    :param fill_holes:  True if small holes should be filled. False otherwise. defaults to True
    :type fill_holes: bool, optional
    :param hole_thresh: the largest hole size in pixels that needs to be filled, defaults to 5
    :type hole_thresh: int, optional
    :param remove_small: True if small objects should be removed. False otherwise. defaults to True
    :type remove_small: bool, optional
    :param object_thresh: the largest object size in pixels that needs to  be removed as noise. defaults to 16
    :type object_thresh: int, optional
    :param shorten_name: True if mask file names should be shorten to remove redundant expressions. False otherwise. defaults to True
    :type shorten_name: bool, optional
    :param remove_from_name:redundant expression to be removed from mask file names, if applicable. defaults to '_Simple Segmentation_'
    :type remove_from_name: str, optional
    :param filetype:  mask file type, e.g. '.TIF'. defaults to filetype1
    :type filetype: str, optional
    """    
    
    # Define input and output directories (create them accordingly)
    rat_dir    = '../R' + str(rat)
    input_dir  = rat_dir + '/IlastikSegmentation/' + channel + '\\' # folder with Ilastik output (unprocessed mask files)
    output_dir = rat_dir + '/IlastikSegmentation/' + channel + '-processed\\' # folder where processed masks will be saved
            
    # Get pathnames to mask files
    file_list = get_pathnames_to_images(input_dir, filetype=filetype)
    
    # Process each mask file
    for i in range(len(file_list)):
        
        # Get mask file
        file_i = file_list[i]
        mask_i = skimage.io.imread(file_i).astype(np.uint8)
        mask_i = (mask_i == 1)
        
        # Apply tissue mask to GFP mask
        tissue_file_i = copy.deepcopy(file_i)
        tissue_file_i = tissue_file_i.replace(channel,'DAPI-processed').replace('cha','ch1').replace(remove_from_name,'')
        tissue_mask_i = skimage.io.imread(tissue_file_i).astype(np.uint8)

        mask_i = cv.bitwise_and(tissue_mask_i.astype(np.uint8),mask_i.astype(np.uint8))
        
        # If applicable, fill small holes
        if fill_holes:
            mask_i = skimage.morphology.remove_small_holes(mask_i,area_threshold=hole_thresh).astype(np.uint8)
        # If applicable, remove small objects
        if remove_small:
            mask_i = skimage.morphology.remove_small_objects(mask_i,min_size=object_thresh)
            
        # Save processed mask in the output directory
        skimage.io.imsave((output_dir + file_i.replace(remove_from_name,'').split('\\')[-1]),
                          skimage.img_as_ubyte(mask_i*255),
                          check_contrast=False)
    
    return

def process_drug_masks_final(rat,channel='GFP',tpCol='TruePositive',subdir=''):
    """
    Clean up processed virus+ masks to further remove false positives after DBSCAN clustering.
    Save clean masks in the output directory (initiate it before running this function)

    :param rat: rat ID, e.g. 12345
                Master rat folder should be parallel with pipeline folder and start with R (e.g. ../R1234/)
                It should contain a csv file with object properties and metadata: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
                It should contain a subfolder with processed ilastik output: ../Rxxxxx/IlastikSegmentation/CHANNEL-processed/
                It should contain an empty subfolder to store final masks: ../Rxxxxx/IlastikSegmentation/CHANNEL-final/(optional subfolder)
    :type rat: int
    :param channel: name of channel in which virus+ objects were imaged, defaults to 'GFP'
    :type channel: str, optional
    :param tpCol: Column name in ObjectProps that corresponds to true/false positive assignment of objects
                defaults to 'TruePositive'
    :type tpCol: str, optional
    :param subdir: name of further subdirectory in the final directiry where masks should be saved
                defaults to '' if the final masks should be saved right in the final directory
    :type subdir: str, optional
    :rparam: None
            Save final masks as black and white png images in ../Rxxxxx/IlastikSegmentation/CHANNEL-final/
    """
    
    # Get object props
    props_dir = '../R' + str(rat) + '/R' + str(rat) + '-' + channel + '-ObjectProps.csv'
    df_props  = pd.read_csv(props_dir)
    
    # Get pathnames to input masks
    input_dir = '../R' + str(rat) + '/IlastikSegmentation/' + channel + '-processed/'
    im_list = get_pathnames_to_images(input_dir)
    
    # Define how to modify pathnames to output masks
    old = 'processed'
    new = 'final'
    if subdir != '':
        new += '/'
        new += subdir
    
    # Iterate over each processed mask
    for im_path in im_list:
        
        # Open processed unclustered mask and label objects
        mask_processed = skimage.io.imread(im_path) > 0
        mask_labeled   = skimage.measure.label(mask_processed) # deterministic
        
        # Get the labels of true positive objects
        section = int(im_path.split('-')[-2][1:])
        tp_labels = df_props.loc[((df_props.Section==section) & (df_props[tpCol]==1)),'Label'].values
        
        # Remove false positive objects from the mask (only keep true positives)
        mask_final = np.isin(mask_labeled,tp_labels).astype(np.uint8)
        
        # Save clean mask as png
        skimage.io.imsave(im_path.replace(old,new).replace('tif','png'),
                          skimage.img_as_ubyte(mask_final*255),
                          check_contrast=False)
        
    return

def generate_centroid_masks(rat,channel='GFP',subdir='final',filetype=filetype2):
    """
    Collapse segmented objects into centroids to reduce data size
    Save new masks in the parallel directory wrt to input directory

    :param rat: rat ID, e.g. 12345
                Master rat folder should be parallel with pipeline folder and start with R (e.g. ../R12345/)
                It should contain a subfolder with input masks: ../Rxxxxx/IlastikSegmentation/CHANNEL-subdir/
                It should contain an empty subfolder to store centroid masks: ../Rxxxxx/IlastikSegmentation/CHANNEL-subdir-centroid/
    :type rat: int
    :param channel: name of channel in which the target objects were imaged in, defaults to 'GFP'
    :type channel: str, optional
    :param subdir: name of subdirectory with input masks, defaults to 'final'
                The full name of input subdirectory is CHANNEL-subdir, e.g. GFP-final
    :type subdir: str, optional
    :param filetype: input and output image file type, defaults to filetype2
    :type filetype: str, optional
    """
    
    # Define input and output directories
    input_dir  = '../R' + str(rat) + '/IlastikSegmentation/' + channel + '-' + subdir + '/'
    subdir_new = subdir + '-centroid'
    
    # Get pathnames to input masks
    path_list = get_pathnames_to_images(input_dir,filetype=filetype)
    
    for path in path_list:
        
        # Open the mask, label objects, and get their relevant properties
        mask         = skimage.io.imread(path).astype(np.uint8)
        mask_labeled = skimage.measure.label(mask)
        object_props = skimage.measure.regionprops_table(mask_labeled, properties=('label','centroid'))
            
        # Store object properties as data frame
        df_props = pd.DataFrame(object_props)
        df_props = df_props.rename(columns={'label':'Label',
                                            'centroid-1':'X',
                                            'centroid-0':'Y'})
        
        # Generate a mask with centroid pixels only
        mask_centroid = np.zeros_like(mask)
        x = df_props.X.values
        y = df_props.Y.values
        for i in range(len(x)):
            mask_centroid[int(y[i]),int(x[i])] = 1
            
        # Save centroid mask
        skimage.io.imsave(path.replace(subdir,subdir_new),
                          skimage.img_as_ubyte(mask_centroid*255),check_contrast=False)
    
    return

#######################################
### Needle / Capsule hole detection ###
#######################################

def find_needle_track(rat:int, filetype:str=filetype1)-> None:
    """
    Use this function to find the needle track and save it as a black-and-white mask.
    
    Before running this fuction:
    Mark the entire needle track in sections where it is visible
    Draw the track in blue color on a 1-channel greyscale image and convert to RGB
    Save the file as tif in directory ../Rxxxxx/IlastikSegmentation/DrawNeedleTrack\\
    
    Input:
        rat (int): rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R1234')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            rat_dir/IlastikSegmentation/DrawNeedleTrack (mannually drawn needle track)
            rat_dir/IlastikSegmentation/Hole (subfolder for storing final needle track/hole masks)
        filetype (str, optional): filetype in the input directory. Defaults to filetype1.
    """    
    
    # Define input and output directories (create them accordingly)
    rat_dir = '../R' + str(rat)
    input_dir  = rat_dir + '/IlastikSegmentation/DrawNeedleTrack\\' # folder with images that have needle track drawn
    output_dir = rat_dir + '/IlastikSegmentation/Hole\\' # folder where processed masks will be saved
    
    # Get pathnames to mask files
    file_list = get_pathnames_to_images(input_dir,filetype=filetype)
    
    # Find needle track in each file
    for i in range(len(file_list)):
        
        # Open the image with the needle track / hole drawn onto it
        file_i = file_list[i]
        im_i = skimage.io.imread(file_i)
        im_i_r, im_i_g, im_i_b = im_i[:,:,0], im_i[:,:,1], im_i[:,:,2]
        
        # Generate a mask with a needle track / hole
        out = np.zeros(im_i_r.shape)
        out[np.logical_and(im_i_r == 0, im_i_g == 0, im_i_b == 255)] = 1
        out = skimage.segmentation.clear_border(out)
        
        # Save the mask with the needle track / hole
        skimage.io.imsave((output_dir + file_i.split('\\')[-1]).replace('tif','png'),
                          skimage.img_as_ubyte(out),
                          check_contrast=False)
    
    return

def find_capsule_hole(rat:int, alpha:float=0.001, subsample_power:int=4, filetype:str=filetype1) -> None:
    """
    Generate a black and white mask of the capsule hole(s)
    
    Before running this function:
    Mark capsule hole in sections where it is present
    Mark capsule hole in blue color on a 1-channel greyscale image and convert to RGB
    Small mark is enough - we'll look for overlap to confirm we have grabbed the right hole 
    Save the image with the marge as tif in directory ../Rxxxxx/IlastikSegmentation/MarkCapsuleHole\\

    Input:
        rat (int): rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R1234')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            rat_dir/IlastikSegmentation/MarkCapsuleHole (mannually marked capsule hole)
            rat_dir/IlastikSegmentation/Hole (subfolder for storing final capsule hole masks)
        alpha (float, optional): alphashape parameter that specifies how tightly alpha shape is drawn around the tissue mask
            Defaults to 0.001
        subsample_power (int, optional): for alphashape, pixels in the mask will be subsampled to reduce data size
            subsample_power parameter specifies the power to which pixels will be subsampled
            Defaults to 4 (i.e. every 10**4 pixel will be sampled)
        filetype (str, optional): filetype in the input directory. Defaults to filetype1.
    """    
    
    # Define input and output directories (create them accordingly)
    rat_dir    = '../R' + str(rat)
    input_dir  = rat_dir + '/IlastikSegmentation/MarkCapsuleHole\\' # folder with images that have capsule hole marked down
    output_dir = rat_dir + '/IlastikSegmentation/Hole\\' # folder where processed masks will be saved
    
    # Get pathnames to files with capsule hole markings
    file_list = get_pathnames_to_images(input_dir,filetype=filetype)
    
    # Find needle track in each file
    for i in range(len(file_list)):
        
        # Open the image with the capsule(s) hole marked on it and convert it to a mask
        file_i = file_list[i]
        im_i = skimage.io.imread(file_i)
        mark_i = np.zeros(im_i[:,:,0].shape)
        mark_i[np.logical_and(im_i[:,:,0]==0, im_i[:,:,1]==0, im_i[:,:,2]==255)] = 1
        
        # Open the tissue mask
        mask_i = skimage.io.imread(file_i.replace('MarkCapsuleHole','DAPI-processed'))
        
        ####################################################################################
        ### Draw alpha shape around the tissue mask so that open holes can be identified ###
        ####################################################################################
        # Get cartesian coordinates of tissue pixels in our mask and subsample to reduce data size
        pixelpoints = cv.findNonZero(mask_i)[::10**subsample_power]
        # Change data format
        pixelpoints = np.array([[float(i[0][0]), float(i[0][1])] for i in pixelpoints])
        # Get alpha shape around subsampled pixel points
        alpha_shape = alphashape.alphashape(pixelpoints,alpha=alpha)
        # Change data format
        alpha_shape = np.array([[e[0], e[1]] for e in list(alpha_shape.exterior.coords)], dtype="i")
        # Draw alpha shape around the tissue mask
        mask_i = cv.drawContours(mask_i.astype(np.uint8), [alpha_shape], -1, 1, 10)
        
        ######################################################
        ### Identify holes that correspond to capsule hole ###
        ######################################################
        # Invert tissue mask with alpha shape around it so that holes are now objects and label them
        holes_i = (mask_i == 0).astype(int)
        holes_i = skimage.segmentation.clear_border(holes_i)
        holes_i_labeled = skimage.measure.label(holes_i)
        # Get the labels of hole(s) that overlap(s) with the mark(s)
        labels = set((holes_i_labeled * mark_i).flatten())
        labels.remove(0.0) # remove background label
        labels = list(labels)
        # Generate a mask with capsule hole(s) only
        for i in range(len(labels)):
            hole_i = holes_i_labeled == labels[i]
            if i == 0:
                holes_final = hole_i
            else:
                holes_final = holes_final + hole_i
        
        # Save the mask with the capsule hole(s)
        skimage.io.imsave((output_dir + file_i.split('\\')[-1]).replace('tif','png'),
                          skimage.img_as_ubyte(holes_final),
                          check_contrast=False)
    
    return

##################################
###    Get Object Properties   ###
##################################

def get_object_props(mask_labeled,intensity=np.array([None]),m_order=0.5,lock_xy=True):
    """
    For each object in the labeled_mask, get object properties.
    If intensity image is not provided, get only the basic properties: label, x and y position
    If intensity image is provided, additionally get object intensity specs (mean, min, and media)
    as well as image moment and our defined spatial moment
    
    Input:
    mask_labaled - array, labeled mask
    
    Output:
    df_propts - data frame with object properties
                columns: Label, X, Y
    Input:
    :param mask_labeled: 2D mask with labeled objects (background is 0)
    :type mask_labeled: np.array
    :param intensity: the corresponding intensity image
                    defaults to np.array([None])
    :type intensity: np.array, optional
    :param m_order: the power to which spatial component abs(x-x_i) of the spatial moment should be raised
                    defaults to 0.5
    :type m_order: float, optional
    :param lock_xy: True if relative x and y dimensions should be preserved during normalization
                    False if x and y should be normalized independently
                    defaults to True
    :type lock_xy: bool, optional
    
    Output
    :param df_props: properties of each labeled object
    :type df_props: data frame with columns:
                    Label: object label
                    X: object x position
                    Y: object y position
                    IntensityMean: object mean intensity
                    IntensityMin: object min intensity
                    IntensityMax: object max intensity
                    moments: object moments (default skimage moments)
                    SpatialMoment: object spatial moment (as defined in the manuscript)   
    """
    
    # If intensity image is provided:
    if intensity.any():
        
        # Get object properties    
        object_props   = skimage.measure.regionprops_table(mask_labeled, properties=('label','centroid','intensity_mean','intensity_max','intensity_min','moments'), intensity_image=intensity)  
        
        # Store object properties as data frame
        df_props = pd.DataFrame(object_props)  
        
        # Rename data frame columns 
        df_props = df_props.rename(columns={'label':'Label',
                                            'centroid-1':'X',
                                            'centroid-0':'Y',
                                            'intensity_mean':'IntensityMean',
                                            'intensity_max':'IntensityMax',
                                            'intensity_min':'IntensityMin',
                                            'moments':'moments'})
        
        # Calculate spatial moment
        x   = df_props.X.values
        y   = df_props.Y.values
        M00 = df_props['moments-0-0'].values
        
        x_dist_matrix = abs(x[:,np.newaxis] - x)**m_order
        y_dist_matrix = abs(y[:,np.newaxis] - y)**m_order
        
        spatial_moments = x_dist_matrix*y_dist_matrix*(1/(M00)) # division of zero could be possible
        spatial_moment = [sum(i) for i in spatial_moments]
        
        df_props['SpatialMoment'] = spatial_moment
        
        # Normalize x, y, and spatial moment
        y_max, x_max = mask_labeled.shape
        if lock_xy: # keep relative x and y values constant
            xy_max = np.max([x_max,y_max])
            df_props['X_norm'] = x / xy_max
            df_props['Y_norm'] = y / xy_max
        else: # normalize xy independently
            df_props['X_norm'] = x / x_max
            df_props['Y_norm'] = y / y_max
        df_props['SpatialMoment_norm'] = (spatial_moment - np.min(spatial_moment)) / (np.max(spatial_moment) - np.min(spatial_moment))
    
    # If intensity image is not provided, get only the basic object properties 
    else:
        # Get object properties    
        object_props   = skimage.measure.regionprops_table(mask_labeled, properties=('label','centroid'))
            
        # Store object properties as data frame
        df_props = pd.DataFrame(object_props)
                
        # Rename data frame columns 
        df_props = df_props.rename(columns={'label':'Label',
                                            'centroid-1':'X',
                                            'centroid-0':'Y'})
    
    return df_props

def get_object_props_all_sections(rat,channel='GFP',m_order=0.5,lock_xy=False,channels=channels1_inv):
    """
    For a single channel of interest, get segmented object properties across all sections.

    Input
    :param rat: rat ID
    :type rat: int (e.g. 12345)
    :param channel: channel of interest, defaults to 'GFP'
    :type channel: str, optional
    :param m_order: the power to which spatial component abs(x-x_i) of the spatial moment should be raised
                    defaults to 0.5
    :type m_order: float, optional
    :param lock_xy: True if relative x and y dimensions should be preserved during normalization
                    False if x and y should be normalized independently
                    defaults to False
    :type lock_xy: bool, optional
    :param channels: relationship between channel names and channel IDs, e.g. GFP:ch2
                    defaults channels1_inv
    :type channels: dict, optional
    
    Output:
    :param df_props: properties of each labeled object
                    Save df_props in the master rat directory as csv
    :type df_props: data frame with columns:
                    Rat: rat ID (int)
                    Channel: Channel name (str)
                    Section: section ID (int)
                    Label: object label (int)
                    X: object x position
                    Y: object y position
                    IntensityMean: object mean intensity
                    IntensityMin: object min intensity
                    IntensityMax: object max intensity
                    moments: object moments (default skimage moments)
                    SpatialMoment: object spatial moment (as defined in the manuscript)
    """
    
    # Define directories to folders with object masks and intensity images
    rat_dir  = '../R' + str(rat) + '/'
    mask_dir = rat_dir + 'IlastikSegmentation/' + channel + '-processed/'
    im_dir   = rat_dir + 'tif/'
    
    # Get paths to all masks in the mask direcotry
    file_list = get_pathnames_to_images(mask_dir,filetype='TIF')
    
    # Knowing channel name, get corresponding channel ID
    channelID = channels[channel]
    
    # Iterate over each mask (one mask per section)
    count = 0
    for i in range(len(file_list)):
        
        # Get paths to the mask and the corresponding intensity image
        mask_path = file_list[i]
        mask_name = mask_path.split('\\')[-1]
        im_name   = mask_name.replace('cha',channelID)
        im_path   = im_dir + im_name
        
        # Open the mask and the intensity image
        mask = skimage.io.imread(mask_path).astype(np.uint8)
        im = skimage.io.imread(im_path)
        
        # If the current mask has objects, calculate their props
        if sum(sum(mask)) != 0:
            
            # Label segmented objects
            mask_labeled = skimage.measure.label(mask)
            
            # Get object properties
            df_props_i = get_object_props(mask_labeled,im,m_order,lock_xy)
            
            # Add metadata
            rat_i = int(mask_name.split('-')[0][1:])
            section_i = int(mask_name.split('-')[1][1:])
            df_props_i['Rat'] = rat_i
            df_props_i['Section'] = section_i
            df_props_i['Channel'] = channel
            
            # Add df_props_i to masta data frame
            if count == 0:
                df_props = df_props_i
            else:
                df_props = pd.concat([df_props,df_props_i])
            count += 1
            
    # export df_props as csv
    csv_name = 'R' + str(rat) + '-' + channel + '-' + 'ObjectProps.csv'
    df_props.to_csv((rat_dir + csv_name))
            
    return df_props

##################################################################
### Process seed sections and organize sections for clustering ###
##################################################################

def process_seed_masks(rat,channel='GFP',channels=channels1_inv):
    """
    Before running this function:
    Select seed sections for clustering. At the minimum, at least 1 seed section must be selected.
    At least one selected section must be at be at the center of the cloud (the central section)
    to guide clustering in both directions.
    In FIJI, black out areas outside of true positive cloud (set values to 0) and save in directory
    ../Rxxxxx/IlastikSegmentation/[channel]-seed
    
    In this function:
    Overlay seed masks over the corresponding masks with segmentated objects to identify tp and fp objects.
    Update tp and fp object assignment in ../Rxxxxx/Rxxxxx-[channel]-ObjectProps.csv file.
    Save the final object mask with fp objects removed in dir
    ../Rxxxxx/IlastikSegmentation/[channel]-final
    
    Input:
    :param rat: rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            rat_dir/IlastikSegmentation/[channel]-seed (selected seed sections)
    :type rat: int
    :param channel: channel of interest, defaults to 'GFP'
    :type channel: str, optional
    :param channels: relationship between channel names and channel IDs, e.g. GFP:ch2
                    defaults channels1_inv
    :type channels: dict, optional
    
    Output:
    None
    Save final masks of seed sections (i.e. with false positive noise removed) in direcotry
        ../Rxxxxx/IlastikSegmentation/[channel]-final
    Overwritve csv file with object properties to add tp/fp assignment for seed masks
        ../Rxxxxx/Rxxxxx-[channel]-ObjectProps.csv
    """
    
    # Define directories
    rat_dir   = '../R' + str(rat)
    seed_dir  = rat_dir + '/IlastikSegmentation/' + channel + '-seed/'
    props_dir = rat_dir + '/R' + str(rat) + '-' + channel + '-ObjectProps.csv'
    
    # Get pathnames to seed masks
    seed_path_list = get_pathnames_to_images(seed_dir)
    
    # Get object properties (already saved in master rat direcotory)
    df_props = pd.read_csv(props_dir)
    df_props = df_props.sort_values(['Section','Label'])
    
    channelID = channels[channel]
    
    # Iterate over seed sections
    for i in range(len(seed_path_list)):
        
        # Set direcotories to seed mask, processed mask, and final mask (to be generated)
        seed_path      = seed_path_list[i]
        processed_path = seed_path.replace('seed','processed').replace(channelID,'cha')
        final_path     = seed_path.replace('seed','final').replace(channelID,'cha')
        
        # Open seed mask
        # Recall that seed mask is intensity image that has areas outside the cloud marked as 0
        seed_mask = skimage.io.imread(seed_path)
        seed_mask = seed_mask > 0
        seed_mask_inv = seed_mask == 0
        
        # Open processed ilastik segmentation mask with objects
        mask = skimage.io.imread(processed_path)
        mask = mask > 0
        mask_labeled = skimage.measure.label(mask)
        
        # Overlay seed mask over processed object mask
        mask_tp = seed_mask * mask
        mask_labeled_tp = seed_mask * mask_labeled
        mask_labeled_fp = seed_mask_inv * mask_labeled
        
        # Save the final mask with true positive objects only
        skimage.io.imsave(final_path,skimage.img_as_ubyte(mask_tp*255),check_contrast=False)
        
        # Update df_props with true and false positive object assignment (seed masks only)
        
        # Get labels of true and false positives
        tp = np.unique(mask_labeled_tp)
        tp = np.setdiff1d(tp,[0])
        fp = np.unique(mask_labeled_fp)
        fp = np.setdiff1d(fp,[0])
        # Some objects may be on the boundary - consider those as tp and remove them from fp list
        if any(x in tp for x in fp):
            remove = []
            for a in tp:
                if a in fp:
                    remove.append(a)
            fp = np.setdiff1d(fp,remove)
        
        # Convert tp/fp labels to data frame
        df_tp = pd.DataFrame.from_dict({'Label':tp})
        df_tp['TruePositive'] = 1
        df_fp = pd.DataFrame.from_dict({'Label':fp})
        df_fp['TruePositive'] = 0
        df = pd.concat([df_tp,df_fp])
        df = df.sort_values('Label')

        # Add results to df_props
        section = int(seed_path.split('-')[-2][1:])
        df_props.loc[df_props.Section==section,'TruePositive'] = df.TruePositive.values
    
    # Overwrite object props files to update tp/fp assignment for seed sections
    df_props.to_csv(props_dir)
        
    return

def get_seed_sections(rat,channel):
    """
    Before running this function:
    Select seed sections for clustering. At the minimum, at least 1 seed section must be selected.
    At least one selected section must be at be at the center of the cloud (the central section)
    to guide clustering in both directions.
    In FIJI, black out areas outside of true positive cloud (set values to 0) and save in directory
    ../Rxxxxx/IlastikSegmentation/[channel]-seed
    
    In this function:
    Get the ID of rat brain sections that have been selected as seed points for clustering
    
    :param rat: rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            rat_dir/IlastikSegmentation/[channel]-seed (selected seed sections)
    :type rat: int
    :param channel: channel name, e.g. GFP
    :type channel: str
    :return: rat brain sections that have been selected as seed points for clustering
    :rtype: list[int], minimum length = 1
    """
    
    # Get path to a folder withy seed masks
    seed_dir = '../R' + str(rat) + '/IlastikSegmentation/' + channel + '-seed/'
    
    # Get pathnames to seed masks
    seed_path_list = get_pathnames_to_images(seed_dir)
    
    # Get seed section numbers
    seed_sections = []
    for seed_path in seed_path_list:
        section = int(seed_path.split('-')[-2][1:])
        seed_sections.append(section)
    
    return seed_sections

def get_section_order(all_sections,seed_sections,seed_center):
    """
    Order rat brain sections for DBSCAN clustering.
    
    Strategy:
    When clustering objects in a section, use information from the nearest, already clustered section
    to inform clustering of the present section.
    Seed sections are already clustered manually.
    Start clustering at the central section in both directions.
    If more seed sections have been selected, cluster sections until the next seed section is reached
    and resume the next round of clustering now starting from the next seed section.
    Thus, there will be at least two rounds of clustering: one to the left and one to the right of the center
    There might be more rounds if more seed sections have been selected.
    For each round of clustering, return a list of section IDs arranged in the order of clustering,
    with the first section in this list being the starting seed section.
    
    :param all_sections: 1D list/array of all brain sections for the current rat
    :type all_sections: list/array[int]
    :param seed_sections: 1D list/array of all seed sections (mininum lenght = 1)
    :type seed_sections: list/array[int]
    :param seed_center: ID of the central seed section that was selected in the center of the cloud
    :type seed_center: int
    :return: the order in which the sections must be clustered
            each list in the output list corresponds to a single round of clustering
            the first section in each of these lists is the starting seed section (it will not be clustered)
    :rtype: list[list[int]]
    """

    # Tidy input arrays just in case
    all_sections = np.array(all_sections)
    seed_sections = np.array(seed_sections)
    all_sections.sort()
    seed_sections.sort()
    
    # Split all_sections to left and right tails from the center
    # Keep the central seed section as a starting point for clustering
    ind_center     = np.nonzero(all_sections==seed_center)[0][0]
    left_sections  = all_sections[:(ind_center+1)][::-1] # arrange in discending order
    right_sections = all_sections[ind_center:]
    
    # Initiate output
    ordered_sections = []
    
    # Add arrays of ordered sections
    if len(seed_sections) == 1:
        
        # Only one central seed section has been selected,
        # so there are just 2 rounds of clustering
        # (one in each direction from the center)
        ordered_sections.append(left_sections)
        ordered_sections.append(right_sections)
        
    else:
        
        # More than one seed section has been selected
        # Cluster from the center until the next seed section is reached
        # Then resume starting with the next seed section
        
        # Split seed_sections to left and right seeds from the center
        # Omit the central seed section
        ind_center_seed  = np.nonzero(seed_sections==seed_center)[0][0]
        left_seeds       = seed_sections[:ind_center_seed][::-1] # arrange in descending order
        right_seeds      = seed_sections[(ind_center_seed+1):]
        
        # Order sections to the left of the center
        if len(left_seeds) == 0:
            ordered_sections.append(left_sections)
        else:
            ind_previous = 0 # initiate the keeping track of the previous seed section
            for i in range(len(left_seeds)):
                ind = np.nonzero(left_sections==left_seeds[i])[0][0] # the index of the next seed section
                ordered_sections.append(left_sections[ind_previous:ind]) # store the next stretch of ordered sections
                if i == (len(left_seeds) - 1): # check if this is the last seed section
                    ordered_sections.append(left_sections[ind:]) # if so, add the last stretch (the tail)
                ind_previous = ind # update previous seed section
        
        # Order sections to the right of the center
        if len(right_seeds) == 0:
            ordered_sections.append(right_sections)
        else:
            ind_previous = 0
            for i in range(len(right_seeds)):
                ind = np.nonzero(right_sections==right_seeds[i])[0][0]
                ordered_sections.append(right_sections[ind_previous:ind])
                if i == (len(left_seeds) - 1):
                    ordered_sections.append(right_sections[ind:])
                ind_previous = ind
            
    return ordered_sections

#######################################
### Get spatial features for DBSCAN ###
#######################################

def get_features(df_props,section,version='3D',tp_only=False):
    """
    For the input section, get spatial object features for either 2D or 3D DBSCAN.
    For 2D DBSCAN, get normalized x and y coordinates.
    For 3D DBSCAN, additionally get normalized spatial moment.

    :param df_props: various properties of segmented objects along with metadata
                    minimum object properties: X_norm, Y_norm, SpatialMoment_norm
                    optional object properties: TruePositive assignment
                    minimum metadata: Section
    :type df_props: data frame with segmented object properties
    :param section: section ID of interest
    :type section: int
    :param version: '2D' if features are needed for 2D DBSCAN
                    '3D' if features are needed for 3D DBSCAN
                    defaults to '3D'
    :type version: str, optional
    :param tp_only: True if only the features of true positive objects must be returned
                    (this can only be applied to seed or already clustered sections)
                    defaults to False
    :type tp_only: bool, optional
    :return: either 2D or 3D features for DBSCAN
    :rtype: np.columnstack w/ either 2  (2D DBSCAN) or 3 (3D DBSCAN) columns
    """
    
    # Reduce data to section of interest
    df = df_props[df_props.Section == section]
    
    # If applicable, further reduce data to true positives only
    if tp_only:
        df = df[df.TruePositive==1]
    
    # Get normalized x, y, and spatial moment values
    x = df.X_norm.values
    y = df.Y_norm.values
    m = df.SpatialMoment_norm.values
    
    # Return spatial features for either 2D or 3D DBSCAN 
    if version == '3D':
        features = np.column_stack((x,y,m))
    elif version == '2D':
        features = np.column_stack((x,y))

    return features

def split_features(df_props,section,version='3D',tp_only=False,n_max=1000):
    """
    For the input section, get the spatial features of all segmented objects
    and then split them into bins with maximum bin size of n_max objects.
    
    For 2D DBSCAN, get normalized x and y coordinates.
    For 3D DBSCAN, additionally get normalized spatial moment.
    
    Use this function to cluster objects in a new section before overlaying them
    over already clustered objects from the previous section.
    This way, increase the relative information from the previous, already clustered
    section in the overlay to help inform clustering.

    :param df_props: properties of the segmented objects
                minimum object properties: X_norm, Y_norm, SpatialMoment_norm
                optional object properties: TruePositive assignment
                minimum metadata: Section
    :type df_props: data frame
    :param section: section ID of interest
    :type section: int
    :param version: '2D' if spatial features are needed for 2D DBSCAN
                    '3D' if spatial features are needed for 3D DBSCAN
    :type version: str
    :param tp_only: True if only the features of true positive objects must be returned
                    (this can only be applied to seed or already clustered sections)
                    defaults to False
    :type tp_only: bool, optional
    :param n_max: the maximum number of objects (and their corresponding spatial features) after splitting
                The objects (and their corresponding spatial features) will be split into
                n_total/n_max bins, where n_total is the total number of objects in input section
                defaults to 1000
    :type n_max: int, optional
    :return: split spatial features for either 2D or 3D DBSCAN
            + the number number of objects in the input section
    :rtype: list[np.columnstack], int
    """
    
    # Get spatial features of all segmented objects in the input section
    features = get_features(df_props,section,version,tp_only=tp_only)
    
    # Get the number of bins to split the objects into
    n_total = len(features)
    n_bins  = math.ceil(n_total/n_max)
    
    # If there is just 1 bin, do not split
    if n_bins == 1:
        return [features], n_total
    
    # If there is more than 1 bin, split objects (and their corresponding features)
    elif n_bins > 1:
        split_features = []
        for i in range(n_bins):
            split_features.append(features[i::n_bins,:])
        return split_features, n_total
    
def sample_features(df_props,section,version='3D',tp_only=True,n_sample=100,add_noise=False,noise_scale=0.01,show_sample=False):
    """
    For the input section, get the spatial features of all segmented objects
    and draw a sample from them of size n_sample.
    
    For 2D DBSCAN, get normalized x and y coordinates.
    For 3D DBSCAN, additionally get normalized spatial moment.
    
    Use this function to get information from the previous, already clustered section.
    Subsample clustered objects to reduce their weight in the overlay with the unclustered objects.
    This way, reduce the bias from the previous, already clustered section.
    
    :param df_props: properties of the segmented objects
                minimum object properties: X_norm, Y_norm, SpatialMoment_norm
                optional object properties: TruePositive assignment
                minimum metadata: Section
    :type df_props: data frame
    :param section: section ID of interest
    :type section: int
    :param version: '2D' if spatial features are needed for 2D DBSCAN
                    '3D' if spatial features are needed for 3D DBSCAN
    :type version: str
    :param tp_only: True if only the features of true positive objects must be returned
                    (this can only be applied to seed or already clustered sections)
                    defaults to True
    :type tp_only: bool, optional
    :param n_sample: sample size, defaults to 100
    :type n_sample: int, optional
    :param add_noise: True if noise (jitter) should be added to the sample that is smaller
                than the total number of avalable objects, defaults to False
    :type add_noise: bool, optional
    :param noise_scale: if applicable, the degree of noise
                defined as the standard deviation of normal distribution
                defaults to 0.01
    :type noise_scale: float, optional
    :param show_sample: True if a plot should be generated to show all available objects
                and a sample drawn from them, defaults to False
    :type show_sample: bool, optional
    :return: sampled features for either 2D or 3D DBSCAN
    :rtype: np.columnstack
    """
    
    # Get spatial features of all (TP) segmented objects in the input section
    features = get_features(df_props,section,version=version,tp_only=tp_only)
    
    # Get the total number of objects
    n_total = len(features)
    
    # If the number of all objects exceeds sample size, sample without replacement
    if n_total >= n_sample:
        sample  = features[np.random.choice(n_total, n_sample, replace=False), :]
        if add_noise: # Optionally, add noise
            noise   = np.random.normal(scale=noise_scale,size=np.shape(sample))
            sample += noise
    # If the number of all objects is less than sample size, sample with replacement & add noise
    else:
        sample  = features[np.random.choice(n_total, n_sample, replace=True), :]
        noise   = np.random.normal(scale=noise_scale,size=np.shape(sample))
        sample += noise
    
    # Optionally, plot all data and the sample
    if show_sample:
        
        fig = plt.figure(figsize=(5, 5))
        fig.patch.set_facecolor('white')
        
        if version == '2D':
            ax = fig.add_subplot(111)
            ax.scatter(features[:,0], features[:,1], s=8, c='black', cmap='bone')
            ax.scatter(sample[:,0],   sample[:,1],   s=16, c='red', cmap='bone')
            plt.legend(loc='upper left')
            
        if version == '3D':
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(features[:,0], features[:,1], features[:,2], s=8, c='black', cmap='bone')
            ax.scatter(sample[:,0],   sample[:,1],   sample[:,2],   s=16, c='red', cmap='bone')   
            
        plt.show()
        plt.clf()
        
    return sample

######################################################
### Calculate distance between objects for DBSCAN  ###
######################################################

def euclidean_normalized(a,b):
    """
    Calculate euclidean distance between points a and b
    and normalize it by the sqrt of the dimensional space
    (e.g. sqrt(2) for 2D and sqrt(3))

    :param a: xy coordinates of point a
    :type a: 1D array
    :param b: xy coordinates of point b
    :type b: 1D array
    :return: euclidean distance between points a and b normalized by sqrt of the dimensional space
    :rtype: float
    """
    
    edistance = distance.euclidean(a,b)
    edistance_norm = edistance/(len(a)**0.5)
    return edistance_norm

def euclidean_weighted(a,b):
    """
    Calculate normalized euclidean distance between points a and b
    and then weigh it by the difference in spatial moments between points a and b

    :param a: spatial features (xy coordinates and spatial moment) of point a
    :type a: 1D array of length 3
    :param b: spatial features (xy coordinates and spatial moment) of point b
    :type b: 1D array of length 3
    :return: 3D distance between points a and b that takes into account 
            euclidean distance between xy coordinates of points as well as
            the difference in spatial moments
    :rtype: float
    """
    
    # Split object features into xy coordinate and spatial moment components
    spatial_a = a[:2]
    spatial_b = b[:2]
    moment_a = a[2]
    moment_b = b[2]
    
    # Calculate spatial distance, i.e. euclidean distance between xy coordinates of points
    spatial_distance = euclidean_normalized(spatial_a,spatial_b)
    # Calculate moment distance, i.e. the difference in spatial moments
    moment_distance  = abs(moment_a-moment_b)
    
    # Calculate 3D distance combining spatial distance and moment distance
    distance_3d = (spatial_distance * moment_distance)**0.5
    
    return distance_3d

###################################
### Cluster objects with DBSCAN ###
###################################

def run_db_on_features(features, eps=0.1, min_samples=10, metric='euclidean'):
    """
    Run dbscan of spatial features of segmented objects.

    :param features: spatial features of objects to be clustered
                    At the minimum, spatial features must include xy coordinates.
                    Optionally, they can include spatial moment
    :type features: np.columnstack
    :param eps: DBSCAN parameter describing maximum allowed distance between objects in a cluster
                defaults to 0.1
    :type eps: float, optional
    :param min_samples: DBSCAN parameter describing a minimum cluster size
                defaults to 10
    :type min_samples: int, optional
    :param metric: DBSCAN parameter describing how distance between objects is calculated
                defaults to 'euclidean' (default DBSCAN metric)
                alternatively, use here defined euclidean_weighted function, which can help
                cluster trickier cases but runs much slower than default 'euclidean' metric
    :type metric: str or function name, optional
    :return: cluster labels of input objects
    :rtype: list[int]
    """
    
    # Run dbscan and get cluster labels of input objects
    db = DBSCAN(eps=eps ,min_samples=min_samples, metric=metric).fit(features)
    cluster_labels = db.labels_
    
    return cluster_labels

def run_db(df_props, section, version='3D', eps=0.1, min_samples=10, metric='euclidean'):
    """
    For the section of interest, first get spatial features of segmented objects
    and then run DBSCAN to cluster the objects.

    :param df_props: properties of the segmented objects
                minimum object properties: X_norm, Y_norm, SpatialMoment_norm
                minimum metadata: Section
    :param section: ID of section to be clustered
    :type section: int
    :param version: '2D' if objects should be clustered considering xy coordinate alone
                    '3D' if objects should be clustered additionally considering spatial moment
                    defaults to '3D'
    :type version: str, optional
    :param eps: DBSCAN parameter describing maximum allowed distance between objects in a cluster
                defaults to 0.1
    :type eps: float, optional
    :param min_samples: DBSCAN parameter describing a minimum cluster size
                defaults to 10
    :type min_samples: int, optional
    :param metric: DBSCAN parameter describing how distance between objects is calculated
                defaults to 'euclidean' (default DBSCAN metric)
                alternatively, use here defined euclidean_weighted function, which can help
                cluster trickier cases but runs much slower than default 'euclidean' metric
    :type metric: str or function name, optional
    :return: cluster labels of input objects
    :rtype: list[int]
    """

    # Get spatial features of objects
    features = get_features(df_props,section,version=version)
    
    # Run dbscan and get cluster labels of input objects
    cluster_labels = run_db_on_features(features, eps=eps, min_samples=min_samples, metric=metric)
    
    return cluster_labels

def run_db_all_sections(rat, channel, version, eps=0.12, min_samples=10, metric='euclidean'):
    """
    For the rat of interest, cluster objects in each section using DBSCAN.
    Cluster using fixed input hyperparameters - do not optimize them considering information
    from previous, already clustered section.
    Cluster considering either xy coordinates alone (version='2D') or additionally considering
    spatial moment (version='3D').
    Assume that any clustered object (cluster label != -1) are true positive.

    :param rat: rat ID, e.g. 12345
    :type rat: int
    :param channel: name of channel in which obejcts were visualized, e.g. 'GFP'
    :type channel: str
    :param version: '2D' if objects should be clustered considering xy coordinate alone
                    '3D' if objects should be clustered additionally considering spatial moment
    :type version: str
    :param eps: DBSCAN parameter describing maximum allowed distance between objects in a cluster
                defaults to 0.1
    :type eps: float, optional
    :param min_samples: DBSCAN parameter describing a minimum cluster size
                defaults to 10
    :type min_samples: int, optional
    :param metric: DBSCAN parameter describing how distance between objects is calculated
                defaults to 'euclidean' (default DBSCAN metric)
                alternatively, use here defined euclidean_weighted function, which can help
                cluster trickier cases but runs much slower than default 'euclidean' metric
    :type metric: str or function name, optional
    :return: updated object properties data now specifying cluster and true positive assignment
            New data columns:
            Cluster_xD - cluster assignment for DBSCAN version x (2 or 3)
            TruePositive_xD - true positive assignment for DBSCAN version x (2 or 3)
    :rtype: data frame
    """

    # Get object props data
    props_dir = '../R' + str(rat) + '/R' + str(rat) + '-' + channel + '-ObjectProps.csv' 
    df_props = pd.read_csv(props_dir)
    
    # Iterate over each section & save cluster assignment
    sections = df_props.Section.unique()
    for i in range(len(sections)):
        section_i = sections[i]
        clusters_i = run_db(df_props, section_i, version, eps, min_samples, metric)
        df_props.loc[df_props.Section==section_i,('Cluster_'+version)] = clusters_i
        
    # Find true positives assuming that all clustered objects are true positive
    df_props[('TruePositive_'+version)] = (df_props[('Cluster_'+version)].values > -1).astype(int)

    # Overwrite csv file
    df_props.to_csv(props_dir)
    
    return df_props

def run_idb(df_props,section,ref_section,n_max=1000,n_ref=100,add_noise=True,noise_scale=0.01,version='3D',eps_vals=np.linspace(0.001,0.3,200),min_samples=10,metric='euclidean',thresh=0.8):
    """
    Run iterative DBSCAN on a single section using an already clustered reference section to inform clustering.
    To inform clustering, add a sample of already clustered reference objects to the current object cloud.
    Cluster iteratively with increasing values of eps until the minimum fraction of reference objects are clustered.

    :param df_props: various properties of segmented objects along with metadata
                    minimum object properties: X_norm, Y_norm, SpatialMoment_norm
                    optional object properties: TruePositive assignment
                    minimum metadata: Section
    :type df_props: data frame
    :param section: ID of section to be clustered
    :type section: int
    :param ref_section: ID of reference, already clustered section that will inform clustering
    :type ref_section: int
    :param n_max: the maximum number of objects in the current sectio to be clustered at once
                The objects (and their corresponding spatial features) will be split into
                n_total/n_max bins, where n_total is the total number of objects in input section
                defaults to 1000
    :type n_max: int, optional
    :param n_ref: the number of objects in the reference section to be used to inform clustering
                They will be overlayed over the objects in the current section
                defaults to 100
    :type n_ref: int, optional
    :param add_noise: True if reference objects should be jittered, False otherwise
                Adding noise can help mitigate the effect of slight section misallignment
                defaults to True
    :type add_noise: bool, optional
    :param noise_scale: the extend of noise to be added to the reference section, if applicable
                noise_scale refers to the standard deviation of the normal distribution that will be used
                to randomly add noise to the normalized spatial features 
                defaults to 0.01
    :type noise_scale: float, optional
    :param version: '2D' if objects should be clustered considering xy coordinate alone
                    '3D' if objects should be clustered additionally considering spatial moment
                    defaults to '3D'
    :type version: str, optional
    :param eps_vals: eps values to screen during iterative DBSCAN clustering (spatial features are normalized)
                DBSCAN parameter eps defines maximum distance between objects in a cluster
                defaults to np.linspace(0.001,0.3,200)
    :type eps_vals: np.array or list, optional
    :param min_samples: DBSCAN parameter that defines minimum number of objects in a cluster
                defaults to 10
    :type min_samples: int, optional
    :param metric: metric used to calculate distance between objects
                defaults to 'euclidean' (default DBSCAN metric)
                alternatively, use here defined euclidean_weighted function
    :type metric: str or function name, optional
    :param thresh: minimum number of reference objects to be clustered
                During iterative clustering, increasing eps values are screened until
                the minimum number of reference objects are clustered
                Note: aiming to cluster all reference objects will overestimate eps and grab false positives
                defaults to 0.8
    :type thresh: float, optional
    :return: clusters, true_positives
            cluster - cluster assignment of objects in the current section (-1 = unclustered)
                        note that objects are split into separate bins for clustering, so the same cluster label
                        may not correspond to the same cluster
            true_positives - true positive assignment of objects in the current section
                        1 = true positive, 0 = false positive
    :rtype: np.array (1D), np.array (1D)
    """

    # Get spatial features of objects in the section to be clustered
    # Split the features into bins with up to n_max objects per bin
    # This way, relative information of already clustered reference section increases to help cluster
    X_list, n_total = split_features(df_props,section,
                                     version=version,tp_only=False,n_max=n_max)
    # Get spatial features of objects in the already clustered reference section
    # Sample the features to have n_ref reference objects
    # This way, bias coming from the previous already clustered section decreases
    X_ref           = sample_features(df_props,ref_section,
                                      version=version,tp_only=True,n_sample=n_ref,
                                      add_noise=add_noise,noise_scale=noise_scale)

    # Initiate output (cluster and true positive assignment of objects in the current section)
    clusters = np.zeros(n_total)
    true_positives = np.zeros(n_total)
    
    # Iterate over each bin of split objects in the current section
    n_bins = len(X_list)
    for i in range(n_bins):
        
        # Get spatial features of objects in the current bin
        X_i = X_list[i]
        n_i = len(X_i)
        
        # Overlay objects in the current section with objects in the reference section
        X_overlay = np.vstack((X_i,X_ref))
        
        # Cluster overlaid objects with increasing values of eps
        # until a threshold fraction of reference data is clustered
        fraction = 0
        counter = -1
        while fraction < thresh:
            counter += 1
            eps = eps_vals[counter]
            clusters_overlay = run_db_on_features(X_overlay, eps=eps, min_samples=min_samples, metric=metric)
            clusters_i = clusters_overlay[:n_i]
            clusters_ref = clusters_overlay[n_i:]
            fraction = sum((clusters_ref > -1).astype(int)) / n_ref # fraction of ref objects that were clustered
        
        # Get cluster labels of reference objects that were successfully clustered
        # Assume that these clusters represent true positive clusters
        tp_labels = set(clusters_ref)
        if -1 in tp_labels:
            tp_labels.remove(-1)
        
        # Assume that objects in the current section that clustered with reference objects
        # are true positives
        true_positives_i = np.array([c in tp_labels for c in clusters_i]).astype(int)
        
        # Save results for the current bin
        clusters[i::n_bins] = clusters_i
        true_positives[i::n_bins] = true_positives_i

    return clusters, true_positives

def run_idb_all_sections(rat,channel,seed_center,n_max=1000,n_ref=100,add_noise=True,noise_scale=0.01,version='3D',eps_vals=np.linspace(0.001,0.3,200),min_samples=10,metric='euclidean',thresh=0.8):
    """
    For the input rat, cluster objects in each to separate dense true positive cloud from false positive noise.
    Use information from the last successfully clustered reference section to inform clustering of the present section
    with manually clustered seed sections serving as starting points.
    To use information from the last successfully clustered reference section, seed a sample of objects
    from the reference section onto the present section.
    Then cluster the objects iteratively by screening increasing values of eps until a set threshold of objects from
    the reference section are clustered.
    
    Before running this function:
    Select seed sections for clustering. At the minimum, at least 1 seed section must be selected.
    At least one selected section must be at be at the center of the cloud (the central section)
    to guide clustering in both directions.
    In FIJI, black out areas outside of true positive cloud (set values to 0) and save in directory
    ../Rxxxxx/IlastikSegmentation/[channel]-seed

    :param rat: rat ID, e.g. 12345
                Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R1234')
                It should contain a csv file with object properties and metadata
                The path to this csv file should be:
                '../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
    :type rat: int
    :param channel: name of channel in which the objects were imaged in, e.g. 'GFP'
    :type channel: str
    :param seed_center: ID of the manually clustered seed section that is located at the center of the true positive cloud
                The rest of the sections will be clustered in both directions from the central seed.
                Clustering from cloud center to the tails works best.
    :type seed_center: int
    :param n_max: the maximum number of objects in the current sectio to be clustered at once
                The objects (and their corresponding spatial features) will be split into
                n_total/n_max bins, where n_total is the total number of objects in input section
                defaults to 1000
    :type n_max: int, optional
    :param n_ref: the number of objects in the reference section to be used to inform clustering
                They will be overlayed over the objects in the current section
                defaults to 100
    :type n_ref: int, optional
    :param add_noise: True if reference objects should be jittered, False otherwise
                Adding noise can help mitigate the effect of slight section misallignment
                defaults to True
    :type add_noise: bool, optional
    :param noise_scale: the extend of noise to be added to the reference section, if applicable
                noise_scale refers to the standard deviation of the normal distribution that will be used
                to randomly add noise to the normalized spatial features 
                defaults to 0.01
    :type noise_scale: float, optional
    :param version: '2D' if objects should be clustered considering xy coordinate alone
                    '3D' if objects should be clustered additionally considering spatial moment
                    defaults to '3D'
    :type version: str, optional
    :param eps_vals: eps values to screen during iterative DBSCAN clustering (spatial features are normalized)
                DBSCAN parameter eps defines maximum distance between objects in a cluster
                defaults to np.linspace(0.001,0.3,200)
    :type eps_vals: np.array or list, optional
    :param min_samples: DBSCAN parameter that defines minimum number of objects in a cluster
                defaults to 10
    :type min_samples: int, optional
    :param metric: metric used to calculate distance between objects
                defaults to 'euclidean'
    :type metric: str or function name, optional
    :param thresh: minimum number of reference objects to be clustered
                During iterative clustering, increasing eps values are screened until
                the minimum number of reference objects are clustered
                Note: aiming to cluster all reference objects will overestimate eps and grab false positives
                defaults to 0.8
    :type thresh: float, optional
    :rparam: None
            Update the csv file with object properties with Cluster and TruePositive assignment
    """
    # Get object properties of segmented objects across all sections
    props_dir = '../R' + str(rat) + '/R' + str(rat) + '-' + channel + '-ObjectProps.csv'
    df_props  = pd.read_csv(props_dir)
    
    # Get section information
    all_sections  = df_props.Section.unique() # all imaged brain section for input rat
    seed_sections = get_seed_sections(rat,channel) # of those - seed sections that were "clustered" manually
    section_order = get_section_order(all_sections,seed_sections,seed_center) # the order in which the sections should be clustered
    
    # section_order is a list of lists
    # each sublist starts with the seed section and then lists unclustered sections in the order that they should be clustered
    # iterate over each sublist
    for i in range(len(section_order)):
        
        # Get spatial features of the objects in the seed section (i.e. the starting section in the sublist)
        X_seed = get_features(df_props,section_order[i][0],version=version,tp_only=True)
        
        # If seed section has no true positives, it means it was blacked out to mark the end of the TP cloud
        # Set all objects in remaining sections as false positive
        if len(X_seed) == 0:
            for j in range(1,len(section_order[i])):
                section = section_order[i][j]
                df_props.loc[df_props.Section==section,('Cluster_i'+version)] = -1
                df_props.loc[df_props.Section==section,('TruePositive')] = 0
        
        # If seed section has true positives, cluster the remaining sections in the sublist      
        else:
            j_success = 0 # the index of the last successfully clustered section
            for j in range(1,len(section_order[i])):
                section     = section_order[i][j] # section to be clustered
                ref_section = section_order[i][j_success] # last successfully clustered section
                # Run iterative dbscan
                clusters, true_positives =  run_idb(df_props,section,ref_section,
                                                    n_max=n_max,n_ref=n_ref,add_noise=add_noise,noise_scale=noise_scale,
                                                    version=version,eps_vals=eps_vals,min_samples=min_samples,
                                                    metric=metric,thresh=thresh)
                # Update data with cluster and true positive assignment
                df_props.loc[df_props.Section==section,('Cluster_i'+version)] = clusters
                df_props.loc[df_props.Section==section,('TruePositive')] = true_positives
                # Update the index of the last successfully clustered section
                if sum(true_positives) > 0:
                    j_success = j
    
    # Overwrite csv file
    df_props.to_csv(props_dir)
    
    return

#########################################
###    Compute performance metrics    ###
#########################################

def count_objects_1mask(mask_path,ground_truth_mask_path):
    """
    Count true positive, false positive, and false negative objects in the input mask
    compared to the ground truth mask
    
    Input:
    mask_path - str, path to a mask generated by a model/pipeline
    ground_truth_mask_path - str, path to a mask labeled by an experienced annotator
    
    Output:
    tp - the number of true positives accurately detected in the mask
    fp - the number of true positives false detected in the mask
    fn - the number of true positives that were not detected in the mask
    """
    
    # Open the masks
    try:
        m1 = skimage.io.imread(mask_path)
    except Exception:
        m1 = skimage.io.imread(mask_path.replace('tif','tiff'))

    try:
        m2 = skimage.io.imread(ground_truth_mask_path)
    except Exception:
        m2 = skimage.io.imread(ground_truth_mask_path.replace('tif','tiff'))
        
    # Make sure mask is zeros and ones
    m1 = m1 > 0
    m2 = m2 > 0
    
    # Label the masks (background is labeled 0)
    l1 = skimage.measure.label(m1)
    l2 = skimage.measure.label(m2)
    
    # Get the labels of objects in each mask that overlap with objects in the other mask
    overlap_labels1 = set(l1[m2!=0])
    overlap_labels2 = set(l2[m1!=0])
    
    # Get rid of background label
    overlap_labels1.discard(0)
    overlap_labels2.discard(0)
    
    # Get the mask of true positive objects
    # i.e. objects in the mask that overlap with objects in ground truth mask
    tp_mask = copy.deepcopy(l1)
    tp_mask[np.isin(l1,list(overlap_labels1),invert=True)] = 0
    
    # Get the mask with false positive objects
    # i.e. objects in the mask that do not overlap with any objects in ground truth mask
    fp_mask = copy.deepcopy(l1)
    fp_mask[np.isin(l1,list(overlap_labels1))] = 0
    
    # Get the mask false negative objects
    # i.e. objects in the ground truth mask that do not overlap with any objects in the mask
    fn_mask = copy.deepcopy(l2)
    fn_mask[np.isin(l2,list(overlap_labels2))] = 0
    
    # Count the objects
    tp = len(np.unique(tp_mask)) - 1 # subtract 1 to account for background label
    fp = len(np.unique(fp_mask)) - 1
    fn = len(np.unique(fn_mask)) - 1
        
    return tp, fp, fn

def count_objects(rat,models=['Ilastik1ch','Ilastik3ch','CellProfiler1ch'],filetype=filetype1,channels=channels1):
    """
    Count true positive (TP), false positive (FP), and false negative (FN) objects in computationally generated ROI masks
    compared against the ground truth ROI masks mannually labeled by an experienced annotator.
    
    Iterate over each ROI mask as well as each computational model. Each time, call function count_objects_1mask
    
    GroundTruth and model masks shouldbe stored as (with respect to the location of the pipeline):
    ../Rxxxxx/Model/Rxxxxx-sxxx-chx-roix.TIF
    
    Input:
    rat - int, rat ID
    models - list of str, names of computational models used to segment images.
            Also, names of folders with computationally generated masks.
    filetype - str, file extension of all masks (should be identical)
    channels - dict relating channel ID to channel name, e.g. 'ch2':'GFP'
    
    Output:
    df - data frame with TP, FP, and FN object counts in each ROI along with ROI metadata
    """
    
    # Get paths to ground truth masks in the input dir:
    groundtruth_dir = '../R' + str(rat) + '/ROIs/GroundTruth/'
    groundtruth_glob = os.path.join(groundtruth_dir,'*' + filetype)
    groundtruth_list = glob.glob(groundtruth_glob)
    
    # Iterate over each mask
    for i in range(len(groundtruth_list)):
        
        groundtruth_mask_path_i = groundtruth_list[i]
        
        # Initialize inner loop output
        tp_i = []
        fp_i = []
        fn_i = []
        
        # Iterate over all possible computational models
        for j in range(len(models)):
            
            model_j = models[j]
            
            # Get the path to the model mask
            mask_path_i = groundtruth_list[i].replace("GroundTruth",model_j)
        
            # Count true pos, false pos, and false pos objects
            tp_ij, fp_ij, fn_ij = count_objects_1mask(mask_path_i,groundtruth_mask_path_i)
            tp_i.append(tp_ij)
            fp_i.append(fp_ij)
            fn_i.append(fn_ij)
            
        # Save inner loop results as data frame
        df_i = pd.DataFrame.from_dict({'Model':models,
                                       'TP':tp_i,
                                       'FP':fp_i,
                                       'FN':fn_i})
                
        # Add section and roi info
        # Directory tree: ../Rxxxxx/Model/Rxxxxx-sxxx-chx-roix.TIF
        str_list = groundtruth_mask_path_i.split('-')
        section_i = int(str_list[-3][1:])
        roi_i = int(str_list[-1].split('.')[0][3:])
        df_i['Section'] = section_i
        df_i['ROI'] = roi_i
        
        # Add to master data frame
        if i == 0:
            df = df_i
        else:
            df = pd.concat([df,df_i])
            
    # Add remaining metadata (should identical across all masks)
    # Directory tree: ../Rxxxxx/Model/Rxxxxx-sxxx-chx-roix.TIF
    str_list = groundtruth_mask_path_i.split('\\')[-1].split('-')
    rat = str_list[0]
    channel_ID = str_list[2]
    channel_name = channels[channel_ID]
    df['Rat'] = rat
    df['Channel'] = channel_name
    
    return df

def calc_metrics_1group(tp, fp, fn):
    """
    Calculate precesision, recall, and f1 metrics for input values
    
    Input:
    tp - int, the number of true positive objects in the group
    fp - int, the number of false positive objects in the group
    fn - int, the number of false negative objects in the group
    
    Output:
    precision - float, precision metric = tp/(tp + fp)
    recall - float, recall metric = tp/(tp + fn)
    f1 - float, f1 metric = (2*precision*recall)/(precision + recall)
    """

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2*precision*recall)/(precision + recall)
    
    return precision, recall, f1

def calc_metrics(df,groupingCols=['Rat','Channel','Model']):
    """
    Group data in df based on values in groupingCols
    For each group, calculate precision, recall, and f1
    
    Input:
    df (data frame) - contains TP, FP, and FN object counts along with metadata
    groupingCols (list of str) - how to group data in df, list of column labels in df
    
    Output:
    df_metrics (data frame) - contains Precision, Recall and F1 values for each data group
    """
    
    # Group data in the data frame with tp, fp, and fn object counts
    grouped = df.groupby(groupingCols)
    
    # Iterate over each group to calculate performance metrics
    count = 0
    for name, group in grouped:
        
        # Sum object counts for the current group
        tp = sum(group.TP.values)
        fp = sum(group.FP.values)
        fn = sum(group.FN.values)

        # Calculate performance metrics
        precision, recall, f1 = calc_metrics_1group(tp, fp, fn)
        
        # Save results as data frame
        df_i = pd.DataFrame.from_dict({'Precision':[precision],
                                       'Recall':[recall],
                                       'F1':[f1]})
        
        # Add metadata
        for i, col in enumerate(groupingCols):
            df_i[col] = name[i]
            
        # Add results to the master data frame
        if count == 0:
            df_metrics = df_i
        else:
            df_metrics = pd.concat([df_metrics,df_i])
            
        count += 1
    
    return df_metrics

#################################################################
###  Count virus+ objects across sections and in whole brain  ###
#################################################################

def get_counts(rat,channel='GFP',centroid=False,filetype=filetype2):
    """
    For the input rat and channel, count the number of TP objects in the final masks
    Only consider those sections that were imaged and analyzed

    :param rat: rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            ../Rxxxxx/IlastikSegmentation/CHANNEL-final OR ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid
    :type rat: int
    :param channel: name of channel in which objects were imaged, defaults to 'GFP'
    :type channel: str, optional
    :param centroid: True if objects should be counted in final masks where objects have been reduced to cendroid pixels
                defaults to False
    :type centroid: bool, optional
    :param filetype: filetype of masks in the directory with final masks, defaults to filetype2
    :type filetype: str, optional
    :return: for input rat and channel, the counts of TP objects in each section in the direcotory with final masks
    :rtype: data frame
    """
    
    # Define directory
    mask_dir = '../R' + str(rat) + '/IlastikSegmentation/' + channel + '-final/'
    if centroid:
        mask_dir = mask_dir.replace('final','final-centroid')
    
    # Initialize output to store section ID and the corresponding number of TP objects
    section_list = []
    n_list = []
    
    # Iterate over each section in the direcotory
    path_list = get_pathnames_to_images(mask_dir,filetype=filetype)
    for path in path_list:
        
        # Get section ID
        section = int(path.split('-')[-2][1:])
        
        # Get total number of tp objects in the current section
        mask = skimage.io.imread(path) > 0
        if centroid:
            n = sum(sum(mask))
        else:
            n = np.max(skimage.measure.label(mask))
        
        # Save results
        n_list.append(n)
        section_list.append(section)
    
    # Return results as data frame 
    df_counts = pd.DataFrame.from_dict({'Section':section_list,
                                        'Count':n_list})
    df_counts['Rat'] = rat
    df_counts['Channel'] = channel
    
    return df_counts

def smooth_counts(df_counts, sigma=2.0):
    """
    Using 1D Gaussian filter, smooth raw counts of TP objects across analyzed sections to smooth out random noise in the counts
    Note: Input data frame must contain data for a single rat and a single channel only

    :param df_counts: data frame with raw counts of TP objects across analyzed sections
                Note: input df must only refer to counts for a single rat and a single channel
    :type df_counts: data frame
    :param sigma: the size (standard deviation) of 1D gaussian filter used to smooth raw counts of TP objects in consecutive sections
                defaults to 2.0
    :type sigma: float, optional
    :return: input data frame additionally containing a column with smoothed counts of TP objects in each section
    :rtype: data frame
    """
    
    # Make sure sections are sorted
    df_counts = df_counts.sort_values('Section')
    
    # Smooth the counts using 1D gaussian filter
    counts = df_counts.Count.values
    counts_smooth = ndimage.gaussian_filter1d(counts, sigma) 
    df_counts['Count_smooth'] = counts_smooth
    
    return df_counts

def predict_counts(df_counts):
    """
    Predict TP object counts in the missing sections.
    For this, use linear interpolation between each two consecutive sections with known, smoothed TP object counts.
    Additionally, extrapolate total TP object count in the brain

    :param df_counts: data frame with raw and smoothed counts of TP objects across analyzed sections
                Note: input df must only refer to counts for a single rat and a single channel
    :type df_counts: data frame
    :return: input data frame additionally containing predicted TP object counts in missing sections
                + extrapolated total TP object count in the brain
    :rtype: data frame, float
    """
    
    # Get known values
    known_sections = df_counts.Section.values
    known_counts   = df_counts.Count_smooth.values
    n = len(known_sections)
    
    # Assume that sections away from the first and last analyzed section have no TP objects
    # Update the arrays accordingly    
    known_sections_          = np.zeros((n+2))
    known_sections_[1:(n+1)] = known_sections
    known_sections_[n+1]     = (np.max(known_sections) + np.min(known_sections))
    known_counts_            = np.zeros((n+2))
    known_counts_[1:(n+1)]   = known_counts
    
    # Initialize output to append predicted counts in the missing sections
    all_sections = copy.deepcopy(known_sections_)
    all_counts   = copy.deepcopy(known_counts_)
    
    # Predict counts in missing sections
    for i in range((n+1)):
        
        # Get IDs and smoothed TP objects counts of two consecutive functions with known TP object counts
        x0, y0 = known_sections_[i],   known_counts_[i]
        x1, y1 = known_sections_[i+1], known_counts_[i+1]
        
        # Use linear interporalation to predict counts in the missing sections between those two sections with known counts
        xi_list = np.arange((x0+1),x1,1)
        yi_list = np.zeros_like(xi_list)
        for j in range(len(xi_list)):
            yi_list[j] = y0 + (y1-y0)/(x1-x0)*(xi_list[j]-x0)
        
        # Add results
        all_sections = np.concatenate((all_sections,xi_list))
        all_counts   = np.concatenate((all_counts,yi_list))
    
    # Convert the results to data frame    
    df_counts_all = pd.DataFrame.from_dict({'Section':all_sections,
                                           'Count_smooth':all_counts})
    df_counts_all['Rat'] = df_counts.Rat.unique()[0]
    df_counts_all['Channel'] = df_counts.Channel.unique()[0]
    df_counts_all = df_counts_all.sort_values('Section')
    
    # Extrapolate total TP object counts in the brain
    total_count = np.sum(df_counts_all.Count_smooth.values)
    
    return df_counts_all, total_count

def predict_counts_exp(rats,channels,centroid=False,filetype=filetype2,sigma=2.0):
    """
    For a single experiment, analyze TP virus-infected cell counts across sections and in the entire brain
    For each rat-channel combination, get TP object counts in the analyzed sections,
    smooth raw counts with a 1D Gaussian filter, predict TP object counts in the missing sections
    using linear iterpolation between each two consecutive analyzed sections, and
    extrapolate total TP object count in the brain.

    :param rats: rat IDs of interest that are part of the same experiment
                Each master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
                Each master rat folder should have IlastikSegmentation subfolder with further subdfolders:
                ../Rxxxxx/IlastikSegmentation/CHANNEL-final OR ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid
    :type rats: list(int)
    :param channels: channel names of interest that are part of the same experiment
    :type channels: list(str)
    :param centroid: True if objects should be counted in final masks where objects have been reduced to cendroid pixels
                defaults to False
    :type centroid: bool, optional
    :param filetype: filetype of masks in the directory with final masks, defaults to filetype2
    :type filetype: str, optional
    :param sigma: the size (standard deviation) of 1D gaussian filter used to smooth raw counts of TP objects across consecutive sections
                defaults to 2.0
    :type sigma: float, optional
    :return: df_section_counts, df_total_counts
            (1) data frame with TP object counts in all (i.e. analyzed and missing) sections 
            (2) data frame with total TP object counts for each experimental condition (rat+channel)
    :rtype: data frame x2
    """
    
    # Iterate over all rat-channel combinations for a given experiment
    # For each combination, get raw and predicted counts of TP objects in analyzed and missing sections, respectively
    counter = 0
    for rat in rats:
        for channel in channels:
            
            df_counts_raw    = get_counts(rat,channel=channel,centroid=centroid,filetype=filetype)
            df_counts_smooth = smooth_counts(df_counts_raw,sigma=sigma)
            df_counts_all, _ = predict_counts(df_counts_smooth)
            
            if counter == 0:
                df_section_counts = df_counts_all
            else:
                df_section_counts = pd.concat([df_section_counts,df_counts_all])
            counter += 1
    
    # For each combination, extrapolate total TP object counts in the brain        
    df_total_counts = df_section_counts.groupby(['Rat','Channel'])['Count_smooth'].sum().reset_index()
    
    return df_section_counts, df_total_counts

def validate_total_counts(rat,channel='GFP',centroid=False,filetype=filetype2,sigma=2.0,rates=range(1,10)):
    """
    Understand how predictions in total TP objects counts in the brain vary with subsampling rate.
    For this:
    Subsample the entire dataset at different subsampling rates. At each rate, predict total TP object counts in the brain.
    
    :param rat: rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            ../Rxxxxx/IlastikSegmentation/CHANNEL-final OR ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid
    :type rat: int
    :param channel: name of channel in which objects were imaged, defaults to 'GFP'
    :type channel: str, optional
    :param centroid: True if objects should be counted in final masks where objects have been reduced to cendroid pixels
                defaults to False
    :type centroid: bool, optional
    :param sigma: the size (standard deviation) of 1D gaussian filter used to smooth raw counts of TP objects across consecutive sections
                defaults to 2.0
    :type sigma: float, optional
    :param rates: subsampling rates of interest, defaults to range(1,10)
            1: every analyzed section is "subsampled" (entire dataset)
            2: every 2nd analyzed section is subsampled (half the dataset, 2 possible samples)
            3: every 3rd analyzed section is subsampled (third the dataset, 3 possible samples)
    :type rates: np.array/list(int), optional
    :return: all possible total TP object count predictions in the brain across different subsampling rates
    :rtype: dict
            keys(int): different subsampling rates (1: every analyzed section, 2: every 2nd analyzed section, etc.)
            values(list(float)): all possible total TP object count predictions in the brain at a given subsampling rate
    """
    
    # Get raw counts and analyzed sections
    df_counts_raw = get_counts(rat,channel=channel,centroid=centroid,filetype=filetype)
    df_counts_raw = df_counts_raw.sort_values('Section')
    sections = df_counts_raw.Section.values
    
    # Initialize output to store total predicted TP object count in the brain vs subsampling rate
    # keys: different subsampling rates (int)
    # values: list(int), all possible total TP object count predictions at a given subsampling rate
    # Note that as subsampling rate increases, the number of possible samples also increases, so list length increases
    predict_dict = {}
    
    # Iterate over different subsampling rates
    for rate in rates:
        predict_list =[] # all possible total TP object count predictions at a given subsampling rate
        
        # Iterate over different possible samples at a given sampling rate
        for i in range(rate):
            sample = sections[i::rate]
            
            # Iterate over each section in the sample
            # to get raw TP object counts across section of the present sample
            for j in range(len(sample)):
                inds_j = df_counts_raw.Section == sample[j]
                if j == 0:
                    inds = inds_j
                else:
                    inds = inds | inds_j
            df_sample = df_counts_raw[inds]
            
            # Smooth data considering only the subsampled sections
            df_sample = smooth_counts(df_sample,sigma=sigma)
            # Extrapolate the total TP object count in the brain considering only the subsampled sections
            _, predicted_total = predict_counts(df_sample)
            # Store the prediction for the current sample at the current subsampling rate
            predict_list.append(predicted_total)
        
        # Store all predictions at the current subsampling rate
        predict_dict[rate] = predict_list
    
    return predict_dict

def validate_section_counts(rat,channel='GFP',centroid=False,filetype=filetype2,sigma=2.0,rate=3):
    """
    Understand how predicted TP object counts in the "missing" sections compare against raw counts in the corresponding sections
    at an input subsampling rate. For this:
    Subsample all analyzed sections at the input subsampling rate to generate training datasets. Evaluate each possible sample.
    Using the training dataset, predict TP object counts in the missing sections.
    Compare predictions against raw counts in the remaining validation dataset.

    :param rat: rat ID
            Master rat folder should be parallel with pipeline folder and start with R (e.g. '../R12345')
            Master rat folder should have IlastikSegmentation subfolder with further subdfolders:
            ../Rxxxxx/IlastikSegmentation/CHANNEL-final OR ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid
    :type rat: int
    :param channel: name of channel in which objects were imaged, defaults to 'GFP'
    :type channel: str, optional
    :param centroid: True if objects should be counted in final masks where objects have been reduced to cendroid pixels
                defaults to False
    :type centroid: bool, optional
    :param sigma: the size (standard deviation) of 1D gaussian filter used to smooth raw counts of TP objects across consecutive sections
                defaults to 2.0
    :type sigma: float, optional
    :param rates: subsampling rates of interest, defaults to range(1,10)
            1: every analyzed section is "subsampled" (entire dataset)
            2: every 2nd analyzed section is subsampled (half the dataset, 2 possible samples)
            3: every 3rd analyzed section is subsampled (third the dataset, 3 possible samples)
    :type rates: np.array/list(int), optional
    :param rate: subsampling rate at which section counts should be validated at
            For example, 3 means that every 3rd analyzed section is subsampled as the training dataset,
            leaving the remaining two thirds of sections as the validation dataset.
            At subsampling rate of 3, 3 different training-validation sets are possible.
            defaults to 3
    :type rate: int, optional
    :return: All possible training and validation section combinations and the corresponding raw and predicted TP object counts
            sections_list_traiing    - possible combinations of training sections with raw counts
            counts_list_training      - corresponding raw TP object counts in training sections
            sections_list_validation - possible combinations of validation sections with raw counts
            counts_list_validation   - corresponding raw TP object counts in validation sections
            sections_list_prediction - all sections (training + validation + missing)
            counts_list_prediction   - corresponding predicted TP object counts across all sections
    :rtype: list(list(int))
    """
    
    # Get raw counts in all analyzed sections
    df_counts_raw = get_counts(rat,channel=channel,centroid=centroid,filetype=filetype)
    df_counts_raw = df_counts_raw.sort_values('Section')
    sections = df_counts_raw.Section.values
    
    # Initialize output lists
    sections_list_training = []
    counts_list_training = []
    sections_list_validation = []
    counts_list_validation = []
    sections_list_prediction = []
    counts_list_prediction = []

    # Evaluate each possible training-vs-validation sample for the input subsampling rate
    for i in range(rate):
        
        # Get the current sample of training sections
        training_sections = sections[i::rate]
        
        # Split data frame with counts into training and validations data frames
        for j in range(len(training_sections)):
            inds_j = df_counts_raw.Section == training_sections[j]
            if j == 0:
                inds = inds_j
            else:
                inds = inds | inds_j
        df_training = df_counts_raw[inds]
        df_validation = df_counts_raw[-inds]
        
        # Smooth TP object counts in the training sections
        # and then use them to predict TP object counts in the missing sections
        df_training      = smooth_counts(df_training,sigma=sigma)
        df_prediction, _ = predict_counts(df_training)
        
        # Add results to output lists
        sections_list_training.append(df_training.Section.values)
        counts_list_training.append(df_training.Count.values)
        sections_list_validation.append(df_validation.Section.values)
        counts_list_validation.append(df_validation.Count.values)
        sections_list_prediction.append(df_prediction.Section.values)
        counts_list_prediction.append(df_prediction.Count_smooth.values)
        
    output = [sections_list_training,counts_list_training,
              sections_list_validation,counts_list_validation,
              sections_list_prediction,counts_list_prediction]
        
    return output

#################################################################
### Analyze spatial distribution virus+ objects in the brain  ###
#################################################################

def load_nutil_output(rat,channel):
    """
    Load and tidy nutil output, i.e. the xyz coordinates of mask pixels after alignment to 3D rat brain atlas

    :param rat: rat ID
            Master rat folder should be parallel to pipeline folder and start w/ R, i.e. ../Rxxxxx/
            Master rat folder should contain a subdirectory to nutil output:
            ../Rxxxxx/Nutil/CHANNEL/Coordinates/Rxxxxx-CHANNEL_3D_combined.json (here CHANNEL=Needle,Capsule,Hole)
            ../Rxxxxx/Nutil/CHANNEL/Coordinates/Rxxxxx-CHANNEL-final-centroid_3D_combined.json (here CHANNEL=GFP,mCherry)
    :type rat: int
    :param channel: name of channel in which objects were images
    :type channel: str
    :return: list of xyz pixel coordinates generated by nutil
    :rtype: list
    """
    
    # Define directory with json output
    if channel in ['Needle','Capsule''Hole']:
        # For needle/capsule output, consider all pixles
        path = '../R' + str(rat) + '/Nutil/' + channel + '/Coordinates/R' + str(rat) + '-' + channel + '_3D_combined.json'
    else:
        # For virus+ objects, consider centroids only (less memory/computational power)
        path = '../R' + str(rat) + '/Nutil/' + channel + '-centroid/Coordinates/R' + str(rat) + '-' + channel + '-final-centroid_3D_combined.json'
    
    # Convert nutil json output to a list of pixel coordinates in 3D
    json_file = open(path)
    d = json.load(json_file) # Read in the nutil json
    coordinates=[]
    for part in d: # convert json to list of 3d points
        points = [tuple(x) for x in zip(part['triplets'][0::3], part['triplets'][1::3], part['triplets'][2::3])]
        coordinates.extend(points)
    
    return coordinates

def analyze_distribution(rats,channels,reference_name='Needle',reference_alpha=0.004,point_alpha=0.04,atlas_px=39):
    """
    For a single experiment (same drug channels, single boundary of drug delivery),
    analyze the spatial distribution of virus+ objects. Specifically:
    (1) estimate total virus+ object point cloud volume and
    (2) profile object distance from drug delivery boundary as ECDF.

    :param rats: IDs of rats from a single experimental condition (e.g. either Needle or Capsule)
            Each master rat folder should be parallel to pipeline folder and start w/ R, i.e. ../Rxxxxx/
            Each master rat folder should contain a subdirectory to nutil output:
            ../Rxxxxx/Nutil/CHANNEL/Coordinates/Rxxxxx-CHANNEL_3D_combined.json (here CHANNEL=Needle,Capsule,Hole)
            ../Rxxxxx/Nutil/CHANNEL/Coordinates/Rxxxxx-CHANNEL-final-centroid_3D_combined.json (here CHANNEL=GFP,mCherry)
    :type rats: list(int)
    :param channels: names of channels in which virus+ cells were imaged, e.g. ['GFP','mCherry']
    :type channels: list(str)
    :param reference_name: name of the drug delivery boundary consistent to how nutil files are named
            NOTE: the reference name must be the same across all input rats
            If experiment contains both needle and capsule conditions, consider naming everything as Hole
            defaults to 'Needle'
    :type reference_name: str, optional
    :param reference_alpha: alphashape fitting parameter when drawing alpha shape around the reference
            defaults to 0.004
    :type reference_alpha: float, optional
    :param point_alpha: alphashape fitting parameter when drawing alpha shape around virus+ object point cloud
            defaults to 0.04
    :type point_alpha: float, optional
    :param atlas_px: rat brain atlas pixel size in um, defaults to 39
    :type atlas_px: int, optional
    :return: df_ecdf_master - empirical cumulative distribution of virus+ object distance from the boundary of drug delivery
             df_net_master  - overall spatial distrubution properties, namely:
                            Volume - virus+ object point cloud volume in mm**3
                            Distance50 - 50th percentile of virus+ object distance from drug delivery boundary
                            Distance95 - 95th percentile --//--
                            RawCount   
    :rtype: data frame, data frame
    """

    # Define distance and volume conversations (e.g. atlas pixel size is 39um)
    distance_factor = atlas_px/1000
    volume_factor = distance_factor**3

    counter = 0
    # Iterate over each rat in a single experiment
    for rat in rats:
        
        # Load the reference shape as alphashape object
        # reference = boundary of drug delivery, either needle track or capsule hole
        reference = load_nutil_output(rat,reference_name)
        reference_alphashape = alphashape.alphashape(reference, alpha=reference_alpha)
        
        # Iterate over each drug/virus channel
        for channel in channels:
            
            # Load virus+ object cloud as alphashape object and as trimesh point cloud
            cloud = load_nutil_output(rat,channel)
            cloud_alphashape = alphashape.alphashape(cloud,alpha=point_alpha)
            cloud_points = trimesh.points.PointCloud(cloud)

            # Use alphashape object to estimate object cloud volume
            cloud_volume = volume_factor*cloud_alphashape.volume
            
            # Use point cloud to profile virus+ object distance from drug delivery boundary
            distances = trimesh.proximity.signed_distance(reference_alphashape, cloud_points.vertices) * (-distance_factor)
            distances[distances<0] = 0
            distances = np.sort(distances)
            n = len(distances)
            ecdf = np.arange(1,(n+1)) / n

            # Convert results to data frames
            df_ecdf = pd.DataFrame.from_dict({'Distance':distances,'ECDF':ecdf})
            df_ecdf['Rat'] = rat
            df_ecdf['Channel'] = channel
            
            distance_50 = df_ecdf.loc[df_ecdf.ECDF >= 0.5,'Distance'].values[0]
            distance_95 = df_ecdf.loc[df_ecdf.ECDF >= 0.95,'Distance'].values[0]
            df_net = pd.DataFrame.from_dict({'Rat':[rat],
                                             'Channel':[channel],
                                             'Volume':[cloud_volume],
                                             'Distance50':[distance_50],
                                             'Distance95':[distance_95]})
            
            # Add results to master data frames
            if counter == 0:
                df_ecdf_master = df_ecdf
                df_net_master  = df_net
            else:
                df_ecdf_master = pd.concat([df_ecdf_master,df_ecdf])
                df_net_master  = pd.concat([df_net_master,df_net])
            
            counter += 1
    
    return df_ecdf_master, df_net_master

#################################
###       Plotting tools      ###
#################################

def plot_spatial_features(x,y,z,label_clusters=False,labels=np.array([None]),save=False,output_path=None,show=False):
    """
    Visualize object spatial features in 2D (xy coordinates only) and 3D (additionally display spatial moment) side-by-side.
    Optionally, visualize different clusters in different color.
    Optionally, save the plot to output directory.
    Optionally, display the plot

    :param x: normalized spatial feature x (coordinate x)
    :type x: 1D list(float) or 1D np.array(float)
    :param y: normalized spatial feature y (coordinate y)
    :type y: 1D list(float) or 1D np.array(float)
    :param z: normalized spatial moment
    :type z: 1D list(float) or 1D np.array(float)
    :param label_clusters: True if different clusters should be visualized different colors
                defaults to False
    :type label_clusters: bool, optional
    :param labels: If label_clusters == True, cluster labels of each input object
                defaults to np.array([None])
    :type labels: 1D list(int) or 1D np.array(int), optional
    :param save: True if the plot should be saved, defaults to False
    :type save: bool, optional
    :param output_path: If save == True, output path to the plot, defaults to None
    :type output_path: str, optional
    :param show: True if the plot should should be desplayed, defaults to False
    :type show: bool, optional
    """
    
    # Initialize the figure
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')

    # 2D subplot
    ax = fig.add_subplot(1, 2, 1)
    if label_clusters:
        ax.scatter(x, y, c=labels, s=8, cmap='bone')
    else:
        ax.scatter(x, y, s=8, cmap='bone') 
    ax.axes.set_xlim([0,1]) 
    ax.axes.set_ylim([1,0]) 
    ax.set_facecolor('grey')
    plt.xlabel('x')
    plt.ylabel('y')

    # 3D subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    if label_clusters:
        ax.scatter(x, y, z, c=labels, s=8, cmap='bone')
    else:
        ax.scatter(x, y, z, s=8, cmap='bone')   
    ax.axes.set_xlim3d(left=0, right=1) 
    ax.axes.set_ylim3d(bottom=1, top=0) 
    ax.axes.set_zlim3d(bottom=0, top=1) 
    ax.set_facecolor('grey')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save:
        plt.savefig(output_path)
    if show:
        plt.show()
        
    plt.close()
    
    return 

def plot_spatial_features_all_sections(rat,channel='GFP',label_clusters=False,clusterCol=None,version='3D',save=True,show=False,channels=channels1_inv):
    """
    Visualize object spatial features in 2D (xy coordinates only) and 3D (additionally display spatial moment) side-by-side
    across all sections of the input rat.
    Optionally, visualize different clusters in different color.
    Optionally, save the plot to output directory.
    Optionally, display the plot

    :param rat: Rat ID
            Master rat folder should be parallel to pipeline folder and start w/ R, i.e. ../Rxxxxx/
            Master rat folder should contain a subdirectory to a csv file with object properties:
            ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
            If plots should be saved, master rat folder should contain an output subdirectory:
            ../Rxxxxx/IlastikSegmentation/CHANNEL-DBSCAN/VERSION/
    :type rat: int
    :param channel: name of channel in which objects were imaged, defaults to 'GFP'
    :type channel: str, optional
    :param label_clusters: True if different clusters should be visualized in different color, defaults to False
    :type label_clusters: bool, optional
    :param clusterCol: If different clusters should be visualized in different color,
            the column label in csv input file that contains cluster labels.
            defaults to None
    :type clusterCol: str, optional
    :param version: If different clusters should be visualized in different colors,
            DBSCAN version that was used to cluster the objects and generate cluster labels.
            Output plots will be exported to the final subfolder named 'version'
            defaults to '3D'
    :type version: str, optional
    :param save: True if plots should be saved to output directory, defaults to True
    :type save: bool, optional
    :param show: True if plots should be displayed as they get generated, defaults to False
    :type show: bool, optional
    :param channels: channel name and channel ID relations, defaults to channels1_inv
    :type channels: dict, optional
    """
    
    # Get data for plotting, i.e. the values of object spatial features
    input_dir  = '../R' + str(rat) + '/R' + str(rat) + '-' + channel + '-ObjectProps.csv'
    df_props = pd.read_csv(input_dir)
    
    # Define output directory where to save plots
    output_dir = '../R' + str(rat) + '/IlastikSegmentation/' + channel + '-DBSCAN/' + version + '/'

    # Create a plot for each section
    sections = df_props.Section.unique()
    for section in sections:
    
        # Get values of spatial features
        df = df_props[df_props.Section == section]
        x_norm = df.X_norm.values
        y_norm = df.Y_norm.values
        m_norm = df.SpatialMoment_norm.values
        
        # Define output file name and path
        section_str = str(section)
        if len(section_str) == 1:
            section_str = '00' + section_str
        elif len(section_str) == 2:
            section_str = '0' + section_str
        output_name = 'R' + str(rat) + '-' + 's' + section_str + '-' + channels[channel] + '.PNG'
        output_path = output_dir + output_name
        
        # Optionally, visualize different clusters with different colors
        if label_clusters:
            labels = df[clusterCol].values
        else:
            labels = np.array([None])
        
        plot_spatial_features(x_norm,y_norm,m_norm,label_clusters,labels,save,output_path,show)
    
    return

def plot_total_count_validation(predictions_,min_sampling_rate=3):
    """
    Visualize how predictions in total TP objects counts in the brain vary with subsampling rate.

    :param predictions_: predicted total virus+ cell counts in the brain at different subsampling rates.
            Note that as subsampling rate increases (i.e. as we sample fewer and fewer sections),
            the number of possible samples increases.
            Therefore, as the value of predictions_.keys() increases, the lenth of predictions_.values() lists increases
    :type predictions_: dict{int:list(float)}
    :param min_sampling_rate: the sampling rate considering all imaged and analyzed sections
            For example, 3 means that every 3rd section was imaged and analyzed (3,6,9...)
            The true subsampling rates in predictions_.keys() are then:
                min_subsampling_rate*list(predictions_.keys())
            defaults to 3
    :type min_sampling_rate: int, optional
    """
    
    # Assume that, when all imaged sections are analyzed (sampling rate = 1),
    # correct estimate is obtained for total virus+ cell count in the brain
    true_count = predictions_[1][0]
    
    # Remove this "correct" value plot the rest of the data
    predictions = copy.deepcopy(predictions_)
    del predictions[1]
    n_rates = len(list(predictions.keys()))
    
    # For each sample at each sampling rate, calculate relative error
    REs = []
    for i in range(n_rates):
        # At the current sampling rate, get total count estimates for each sample
        counts = list(predictions.values())[i]
        # Calculate relative error for each sample
        REs_i= [abs(true_count-count_i)/true_count for count_i in counts]
        REs.append(REs_i)

    # Initialize the plot
    fig,ax = plt.subplots(figsize=(3.5,2))
    
    # Add RE group stats at each sampling rate (i.e. box plots)
    b = plt.boxplot(REs, labels=([x*min_sampling_rate for x in list(predictions.keys())]),showfliers=False,zorder=1)
    for meanline in b['medians']:
        meanline.set_color('black')
    # Add a horizontal line marking 10% relative error
    plt.axhline(0.1, color='black', ls='--', alpha=0.5)
    # Add REs of individual samples
    cmap=  ['#4daf4a', '#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00', '#17BECF']
    for i in range(len(REs)):
        y = REs[i]   # current relative errors (multiple values)
        x = list(predictions.keys())[i] # current sampling rate (1 value)
        x = [x-1 for i in y] # extend current sampling rate, one for each sample
        x_jitter = [np.random.normal(i, 0.05) for i in x] # add jitter along x axis
        plt.scatter(x_jitter, y ,s=10, zorder=2, alpha=0.6, c=cmap[i])
        
    #plt.ylim((-0.001,0.2))    
    plt.ylabel('Relative Error in Total Count')
    plt.xlabel('Subsampling Rate')
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    
    return

def plot_section_count_validation(section_counts_validation,perm_ind=1,xmin=3,xmax=120,ymin=0,ymax=1200):
    """
    Visualize how predicted TP object counts in the "missing" sections compare against raw counts in the corresponding sections.
    At a time, visualize a single permution of splitting entire dataset into training and validation datasets.

    :param section_counts_validation: output from function validate_section_counts
    :type section_counts_validation: lis(list(int/float))
    :param perm_ind: permutation index of slipping entire dataset into training and validatin datasets
                defaults to 1
    :type perm_ind: int, optional
    :param xmin: x axis min value, defaults to 3
    :type xmin: int, optional
    :param xmax: x axis max value, defaults to 120
    :type xmax: int, optional
    :param ymin: y axis min value, defaults to 0
    :type ymin: int, optional
    :param ymax: y axis max value, defaults to 1200
    :type ymax: int, optional
    """
    
    # Unpack input
    sections_training = section_counts_validation[0][perm_ind]
    counts_training = section_counts_validation[1][perm_ind] # measured counts in training set
    sections_validation = section_counts_validation[2][perm_ind]
    counts_validation = section_counts_validation[3][perm_ind] # measured counts in validation set
    sections_prediction = section_counts_validation[4][perm_ind]
    counts_prediction = section_counts_validation[5][perm_ind] # predicted counts in all sections
    
    # Check if sections_measured have an extra section wrt to sections_predicted
    # If so, remove those sections and the corresponding section counts
    #extra_inds = []
    #for i in range(len(sections_measured)):
    #    if sections_measured[i] not in sections_predicted:
    #        extra_inds.append(i)
    #sections_measured = np.delete(sections_measured,extra_inds)
    #counts_measured   = np.delete(counts_measured,extra_inds)
    
    # Estimate prediction error
    # For this, get predicted counts only for those sections for which counts have been measured
    counts_training_prediction = []
    for i in range(len(sections_prediction)):
        if sections_prediction[i] in sections_training:
            counts_training_prediction.append(counts_prediction[i])
    # Calculate absolute error between measured and predicted counts and the standard deviation across errors        
    errors = abs(counts_training - counts_training_prediction)
    errors_std = np.std(errors)
    # Estimate the range of predicted counts (1STD: 65% likelihood, 2STD: 95% likelihood)
    val_up1   = counts_prediction + errors_std
    val_up2   = counts_prediction + 2*errors_std
    val_down1 = counts_prediction - errors_std
    val_down2 = counts_prediction - 2*errors_std
    
    # Initialize and populate the figure
    fig, ax = plt.subplots(figsize=(3.5,2))
    # Plot uncertainty in the predictions
    ax.fill_between(sections_prediction,val_up2,val_down2,alpha=0.5, color ='tab:blue')
    ax.fill_between(sections_prediction,val_up1,val_down1,alpha=0.3, color='tab:blue')
    # Plot prediction counts generated using the training dataset
    plt.plot(sections_prediction,counts_prediction, color = 'tab:blue', label='training')
    # Plot actual counts in the validation sections
    plt.scatter(sections_validation,counts_validation, color = 'tab:orange', s=10,label='validation')
    
    # Format
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    plt.yticks([0,500,1000])
    plt.ylabel('Virus+ Cell Count')
    plt.xlabel('Brain Section')
    plt.legend()
    
    return

def plot_ecdf(df_ecdf,channels=['GFP','mCherry'],labels=['AdV','AAV1'],colors=['#0000ff','#00FF00']):
    """
    Plot ECDF of TP object/virus+ cell distance from drug delivery boundary.
    Visualize all rat/channel combinations in input df

    :param df_ecdf: empirical cumulative distribution data along with metadata.
                There must be a separate set of ECDF data for each rat/channel combination, and all combindations will be visualized.
                X-axis values: Distance from the boundary of drug delivery
                Y-axis values: Fraction of TP objects/virus+ cells up to that distance away from the boundary
                Metadata: Rat, Channel
    :type df_ecdf: data frame
    :param channels: names of channels in which virus+ cells were imaged, defaults to ['GFP','mCherry']
    :type channels: list(str), optional
    :param labels: channel labels to be displayed on the plot, defaults to ['AdV','AAV1']
    :type labels: list(str), optional
    :param colors: color to be used to visualize each channel, defaults to ['#0000ff','#00FF00']
                To distinguish the rats, different shades of the same color will be used
    :type colors: list(str), optional
    """
    
    # Get rat IDs; for each rat/channel combination, generate a separate ECDF curve
    rats = df_ecdf.Rat.unique()
    
    # Determine the number of shades that are needed for each channel
    n_rats = len(rats)
    shades = np.linspace((1/n_rats),1,n_rats)
    
    n_channels = len(channels)
    
    # Initiate figure
    fig, ax = plt.subplots(figsize=(3.8,2.5))
    
    # Iterate over each channel/rat combination combination to populate the figure
    for i in range(n_channels):
        
        channel = channels[i]
        label = labels[i]
        color = colors[i]
        
        for j in range(n_rats):
            
            rat = rats[j]
            shade = shades[j]
            
            df = df_ecdf[(df_ecdf.Rat == rat) & (df_ecdf.Channel == channel)]
            x = df.Distance.values
            y = df.ECDF.values
            plt.plot(x, y, label=label, color=color, alpha=shade, linewidth=5)

    # Style
    plt.xticks([0.0,0.5,1.0,1.5,2,2.5,3,3.5,4])
    plt.yticks([0.0,0.25,0.5,0.75,1.0])
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.50
    ax.tick_params(direction="in",width=2, length=5, color='black')
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(2)  
    
    return