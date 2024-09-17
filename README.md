## Purpose
The pipeline is intended to reconstruct and quantify 3D viral vector distribution in rodent brain from 2D images of thin sections.
<br><br>
The pipeline accompanies "Quantitative 3D reconstruction of viral vector distribution in rodent and ovine brain following local delivery" by Poceviciute et al. (2024).
<br><br>
Developers: Roberta Poceviciute with contributions from Kenneth Mitchel, Bionaut Labs, Inc.

## python module versions
        python 3.9.12
        numpy 1.22.4
        pandas 1.4.2
        scipy 1.13.0
        matplotlit 3.5.1
        scikit-learn 1.0.2
        scikit-image 0.20.0
        opencv-python 4.6.0.66
        alphashape 1.3.1
        trimesh 3.18.1
        h5py 3.6.0
        json5 0.9.6
        glob2 0.7

## Image file naming conventions
- Rxxxxx-syyy-chz.TIF == RatID-SectionID-ChannelID.TIF
- Sxxxx-syyy-chz.TIF == SheepID-SectionID-ChannelID.TIF

## Folder organization
Organize pipeline script, data, etc. into folders as follows.
#### Pipeline
    pipeline script, jupyter notebooks, etc.
#### Ilastik
    ilastik models
#### Rxxxxx (separate folder for each animal)
- Rxxxxx-metadata.csv
        at the minimum, columns Rat, Section, ToFlip
- Rxxxxx-CHANNEL-ObjectProps.csv (will  be generated)
        at the minimum, columns Rat, Section, Channel, X_norm, Y_norm, SpatialMoment_norm
- tif
        stitched images in tif format
- h5
    - DAPI
            single-channel DAPI images in h5 format (tissue detection)
    - GFP (same for mCherry)
            three-channel (BF-GFP-DAPI) images in h5 format (virus+ cell detection)
    - mCherry
            three-channel (BF-mCherry-DAPI) images in h5 format (virus+ cell detection)
- IlastikSegmentation
    - DAPI
            ilastik output from tissue segmentation (tissue=1) (tif)
    - DAPI-processed
            processed ilastik output from tissue segmentation (small holes and objects removed) (tif)
    - DrawNeedleTrack / MarkCapsuleHole
            grayscale images of sections where needle / capsule hole is visible (tif)
            fully draw needle track in blue over greyscale images
            mark capsule hole with a small blue stroke over greyscale images
    - Hole
            masks with computationally detected needle/capsule holes (tif)
    - GFP
            ilastik output from virus+ cell segmentation (virus=1) (tif)
    - GFP-processed
            processed ilastik output from virus segmentation (tif)
            (small holes and objects removed, objects outside tissue removed)
    - GFP-seed
            selectdc "manually segmented" GFP greyscale images (tif)
            to manually segment, mark the area outside of true positive cloud in black (pixel value = 0)
            at the minimum, select one seed section in the center of the cloud
    - GFP-DBSCAN
            2D and 3D plots of DBSCAN clustering results (png)
            for visual inspection purposes only
    - GFP-final
            final ilastik output masks with false positive objects identified during clustering removed (png)
    - GFP-final-centroid
            final ilastik output masks with true positive objects reduced to centroid pixels (png)
    - mCherry
            subfolder system is analogous to GFP
- QuickNII
    - jpeg
            stitched images in jpeg format w/ new naming format: Rxxxxx-chz-syyy.JPEG
            QuickNII input
            single channel data is sufficient for alignment
    - aligned
            QuickNII output
- Nutil
    - Hole/Needle/Capsule
            Nutil output for final needle/capsule hole masks
    - GFP-centroid
            Nutil output for final GFP centroid masks
    - mCherry-centroid
            subfolders are analogous to GFP
- ROIs
        For calculation of performance metrics
    - GroundTruth
            black-and-white masks labeled by expert annotator
    - MODEL
            black-and-white masks generated by a model (corresponding ROIs as in GroundTruth folder)

## Step 1: Organize input

1. Move stitched images to tif folder
2. Create Rxxxxx-metadata.csv file and place it in master rat directory. Columns:
        Rat: rat ID (int)
        Section: section ID (int)
        ToFlip: 1 if current section should be flipped, 0 otherwis

## Step 2: Convert files
#### Reduce RGB images to greyscale & flip images that need flipping
    Input folder: ./Rxxxxx/tif
    Output folder: modify in place

#### Convert images from tif to jpeg format for QuickNII
    Change file naming order to Rxxxxx-chz-syyy.jpg
    Input folder: ../Rxxxx/tif/
    Output folder: ../Rxxxx/QuickNII/jpeg/

#### Convert images from tif to h5 for Ilastik
    For DAPI staining of tissue, keep single channel DAPI images
    For GFP/mCherry visualization of virus+ cells, merge three channel images, e.g. BF-GFP-DAPI


```python
import pipeline
```


```python
# Define rat ID
rat = 12345
```


```python
# Extract image metadata from tif file names
df_meta = pipeline.extract_metadata_all_images(rat)

# Reduce RGB images to greyscale & flip images that need flipping
#pipeline.flip_images(df_meta)

# Convert images from tif to jpeg format for QuickNII
pipeline.convert_for_quickNII(df_meta)

# Convert images from tif to h5 for Ilastik
pipeline.convert_to_h5(df_meta,channel='DAPI')
pipeline.merge_h5_all_sections(df_meta,main_channel='GFP',    channels_to_merge=['BF','GFP','DAPI'])
pipeline.merge_h5_all_sections(df_meta,main_channel='mCherry',channels_to_merge=['BF','mCherry','DAPI'])
```

## Step 3: Ilastik Segmentation
#### Training guidelines:
    Use h5 format for training and batch processing.
    Select ~10% of the data representing a wide range of signal and background/signal "phenotypes".
    At first, select all features.
    In training, first label a few representative objects and click on "Suggest Features".
    Identify at least 7 top features.
    Train the model using the suggested features only.
#### DAPI (tissue) specific guidelines
    Files: 1-channel DAPI h5 files
    Labels: Tissue & Background (BG)
#### GFP/mCherry (virus+ cell) specific guidelines
    Files: 3-channel BF-GFP/mCherry-DAPI h5 files
    Labels: GFP/mCherry, Background (BG) & Damage (DMG)
#### Batch processing guidelines
    Batch processing output settings: Format=SimpleSegmentation File type=TIF
    Output files will be saved in the folder with input h5 files.
    Move them to ../IlastikSegmentation/CHANNEL/

## Step 4: Ilastik Output Processing
#### General guidelines
    Remove redundant phrases from output file names
    Remove small objects and close small holes (adjust thresholds as needed)
    Rescale so that objects are now 255 (not 1)
    Input folder: ..Rxxxxx//IlasktikSegmentation/CHANNEL/
    Output folder ..Rxxxxx//CHANNEL-processed/
#### DAPI (tissue) mask processing
    Holes and small objects are relatively large, set thresholds accordingly.
    To detect capsule hole, an alphashape will be drawn around tissue mask, and the mask will be inverted.
    Manually marked image will be used to identify capsule hole.
#### GFP/mCherry (drug) mask processing
    Holes and small objects are relatively small, set thresholds accordingly
    Optionally, overlay processed tissue masks to remove objects outside of the tissue
#### Needle/Capsule hole detection
    Find images where needle/capsule holes are visible (starting images must be greyscale)
    For needle condition: paint over the entire track in blue & save in DrawNeedleTrack subdir.
    Use function find_needle_track.
    For capsule condition: make a small mark on the capsule hole(s) in blue & save in MarkCapsuleHole subdir.
    Use function find_capsule_hole.
    Input folder (needle): ../Rxxxxx/IlastikSegmentation/DrawNeedleTrack
    Input folder (capsule):../Rxxxxx/IlastikSegmentation/MarkCapsuleHole & ../Rxxxxx/IlastikSegmentation/DAPI-processed
    Output folder: ../Rxxxxx/IlastikSegmentation/Hole


```python
pipeline.process_tissue_masks(rat)
pipeline.find_needle_track(rat)
pipeline.process_drug_masks(rat,channel='GFP',    remove_from_name='_Simple Segmentation')
pipeline.process_drug_masks(rat,channel='mCherry',remove_from_name='_Simple Segmentation')
```

## Step 5: Calculate Object Properties
    Calculate object properties for processed drug masks and normalize them
    Key properties: X and Y spatial coordinates & spatial moment (along with metadata)
    Save results as csv file in master rat directory
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-processed/
    Output dir: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv


```python
pipeline.get_object_props_all_sections(rat,channel='GFP')
pipeline.get_object_props_all_sections(rat,channel='mCherry')
```

## Step 6: Process Seed Sections for DBSCAN
#### Identify seed sections
    Seed sections will serve as starting points for object clustering.
    At the minimum, select a single seed section at the center of the virus+ cell cloud.
    The remaining sections will then be clustered starting at the center and towards the tails of the cloud.
    If necessary, additional sections can be selected.
    If virus+ cell cloud ends before the sections are exhausted, also select the first section where virus+ cell cloud is no longer visble.
#### Manually "cluster" seed sections
    To manually "cluster" seed sections, open greyscale images in FIJI.
    Black out areas outside of the true positive cloud (i.e. pixel value = 0).
    Fully black out the first image where virus+ cell cloud is no longer visible.
    Save manually "clustered" images in '../Rxxxxx/IlastikSegmentation/CHANNEL-seed/ folder.
#### Combine processed drug masks and manually "clustered" seed sections
    In processed drug masks, remove objects that lie outside of the virus+ cell cloud, i.e. in the blaced out regions.
    Save the final masks and update csv file with object props with TruePositive object assignment.
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-seed/
    Output folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-final/
    File to modify in place: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv


```python
pipeline.process_seed_masks(rat,channel='GFP')
pipeline.process_seed_masks(rat,channel='mCherry')
```

## Step 7: Cluster Objects with DBSCAN
#### Cluster objects iteratively
    Specify rat, channel, center seed section ID
    Optimize clustering parameters as needed
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-seed/ (seed sections only)
    File to modify in place: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
#### Optionally, visualize clustering results
    Input file: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
    Output folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-DBSCAN/VERSION/
    Example version: 'i3D', i.e. iterative 3D DBSCAN


```python
# Run iterative 3D DBSCAN
pipeline.run_idb_all_sections(rat,'GFP',90)
# Optionally, visualize clustering results for visual inspection
pipeline.plot_spatial_features_all_sections(rat,channel='GFP',
                                            label_clusters=True,clusterCol='TruePositive',version='i3D')
```


```python
# Run iterative 3D DBSCAN
pipeline.run_idb_all_sections(rat,'mCherry',108)
# Optionally, visualize clustering results for visual inspection
pipeline.plot_spatial_features_all_sections(rat,channel='mCherry',
                                            label_clusters=True,clusterCol='TruePositive',version='i3D')
```

## Step 8: Remove Noise
#### Remove false positive objects identified during clustering
    Input file: ../Rxxxxx/Rxxxxx-CHANNEL-ObjectProps.csv
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-processed/
    Output folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-final/
#### Optionally, reduct true positive object masks to centroid pixels
    Reducing objects to centroid pixels speeds up and simplifies Nutil step
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-final/
    Output folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid/


```python
# Remove false positives
pipeline.process_drug_masks_final(rat,channel='GFP')
pipeline.process_drug_masks_final(rat,channel='mCherry')

# Reduce objects to centroid pixels
pipeline.generate_centroid_masks(rat,channel='GFP')
pipeline.generate_centroid_masks(rat,channel='mCherry')
```

## Step 9: Align Sections to Brain Atlas in QuickNII
    See QuickNII documentation

## Step 10: Apply Alignment to Final Masks
    See Nutil documentation
#### Needle/Capsule hole masks
    Use masks as is in png format
#### Drug masks
    To save on comptation, use centroid masks in png format

## Step 11: Extract Quantitative Information
##### Predict virus+ cell counts in missing sections and total virus+ cell counts in the brain
    Smooth raw counts across sections using 1D Gaussian filter
    Then use linear inerpolation between each pair of consecutive analyzed sections to predict virus+ cell counts in missing sections in between
    Sum up section counts to estimate total virus+ cell count in the brain
    Input folder: ../Rxxxxx/IlastikSegmentation/CHANNEL-final/ OR
                  ../Rxxxxx/IlastikSegmentation/CHANNEL-final-centroid/
#### Estimate virus+ cell cloud volume and profile virus+ cell distribution from boundary of drug delivery
    Extract data from Nutil output
    Draw alphashape around needle/capsule/hole pixels to obtain boundary of drug delivery (optimize alpha param as needed)
    Draw alphashape around point cloud to estimate cloud volume (optimize alpha param as needed)
    Use Trimesh PointCloud to profile virus+ cell distance from drug delivery boundary as ECDF
    Input: ../Rxxxxx/Nutil/ with subfolders CHANNEL OR CHANNEL-centroid and Needle OR Capsule OR Hole


```python
# Estimate virus+ cells across sections and in whole brain
df_section_counts, df_total_counts = pipeline.predict_counts_exp([rat],['GFP','mCherry'],centroid=True)
```


```python
# Profile distrubution of virus+ cell distance from boundary of drug delivery as ecdf
# Calculate oveall spatial parameters: volume, distance 50th and 95th percentiles
df_ecdf, df_overall = pipeline.analyze_distribution([rat],['GFP','mCherry'],reference_name='Needle')
pipeline.plot_ecdf(df_ecdf,channels=['GFP','mCherry'],labels=['AdV','AAV1'])
```

## Optionally, Validate Virus+ Cell Counts


```python
# Validate how total virus+ cell number estimate depends on sampling rate
total_counts_validation = pipeline.validate_total_counts(rat,channel='mCherry',centroid=True)
pipeline.plot_total_count_validation(total_counts_validation)
```


```python
# Validation how predicted per-section virus+ cell number compares to actual per-section counts
section_counts_validation = pipeline.validate_section_counts(rat,channel='mCherry',centroid=True)
pipeline.plot_section_count_validation(section_counts_validation)
```

## Optionally, Compute Precision Metrics
    GroundTruth Masks: ../Rxxxxx/ROIs/GroundTruth/
    Model: ../Rxxxxx/ROIs/MODEL/
    Model examples: CellProfiler, Ilastik1ch, Ilastik3ch


```python
df_performance_counts = pipeline.count_objects(rat,models=['Ilastik1ch','Ilastik3ch','CellProfiler1ch'])
df_metrics = pipeline.calc_metrics(df_performance_counts,groupingCols=['Rat','Channel','Model'])
```
