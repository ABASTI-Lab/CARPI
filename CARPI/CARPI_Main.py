# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:49:14 2023

@author: Elvis de Jesus Duran Sierra, PhD
"""

"""------------------------------- LIBRARIES --------------------------------------------------------------------------------------------"""
import os as os
import shutil
import numpy as np
import datetime
import time
import matplotlib
import pydicom # For working with DICOM files and images (pip install pydicom)
from DicomRTTool import DicomReaderWriter # For converting DICOM-RT to NumPy (pip install DicomRTTool)
from radiomics import featureextractor # Import radiomics sub-libraries (within PyRadiomics: pip install pyradiomics)
from CustomModules import * # Custom modules to work with DICOM data
matplotlib.rcParams.update(matplotlib.rcParamsDefault) # Set default matplotlib parameters for optimal graphic report generation

"""--------------------------- ** REQUIRED ** USER-DEFINED VARIABLES --------------------------------------------------------------------"""
input_folder = "" # Path to folder containing all DICOM files
table_name_postgres = "" # Name of the table to be created in PostgreSQL
create_tables = True # If True: Create/Overwrite table in the database; Set to False if an existing table will be populated

"""------------------------------- CARPI OUTPUT DIRECTORIES -----------------------------------------------------------------------------""" 
output_folder = "CARPI Output" # Name of folder where the CARPI outputs will be saved
verify_folder(output_folder) # Create folder if it doesn't exist
output_contour_folder = output_folder + "\\Contours" # Folder containing all DICOM-RT Struct files processed by CARPI
verify_folder(output_contour_folder) # Create folder if it doesn't exist
output_images_folder = output_folder + "\\Images" # Folder containing all DICOM image files processed by CARPI
verify_folder(output_images_folder) # Create folder if it doesn't exist
output_reg_folder = output_folder + "\\Registration" # Folder containing all DICOM registration files processed by CARPI
verify_folder(output_reg_folder) # Create folder if it doesn't exist
output_report_folder = output_folder + "\\Reports" # Folder containing all graphic reports
verify_folder(output_report_folder) # Create folder if it doesn't exist

"""------------------------------- DATABASE SCHEMA AND TABLE CREATION --------------------------------------------------------------------""" 
feature_names = np.load("feature_names.npy").tolist() # Load radiomic feature names to label columns in database
feature_names_format = [name + ' VARCHAR' for name in feature_names] # Format to database schema 
feature_names_format = [name.replace('original_','') for name in feature_names_format]
tables = {
    table_name_postgres:
            ['mrn VARCHAR',
            'acc VARCHAR',
            'contour_name VARCHAR',
            'contour_date TIMESTAMP',
            'contour_author VARCHAR',
            'contour_method VARCHAR',
            'contour_software VARCHAR',
            'contour_original_path VARCHAR',
            'contour_new_path VARCHAR',
            'contour_uid VARCHAR',
            'image_series_uid VARCHAR',
            'image_date TIMESTAMP',
            'sequence_name VARCHAR',
            'scanner_manufacturer VARCHAR',
            'procedure VARCHAR',
            'voxel_x VARCHAR',
            'voxel_y VARCHAR',
            'voxel_z VARCHAR',
            'analysis_date TIMESTAMP',
            'analysis_done INT',
            'time_data FLOAT[]',
            'intensity_data FLOAT[]',
            'perfusion_parameters FLOAT[]'
            ] + feature_names_format
            + ['contour_id SERIAL PRIMARY KEY'], 
            
    table_name_postgres + '_processed_files':
            ['original_file_path VARCHAR NOT NULL',
             'failed_processing INT',
             'file_id SERIAL PRIMARY KEY']
          }
if create_tables: add_tables_to_db(tables) # Create tables in database
tbl_names = list(tables.keys())    
col_names = [[col_name[:col_name.index(' ')] for col_name in tables[name]] for name in tbl_names]

"""------------------------------- DATABASE CONNECTION AND DICOM FILE LOADING ------------------------------------------------------------""" 
# Connect to database
conn, cur = connect_to_db()
# Get all new files to be processed
all_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(input_folder)
             for name in files
             if name.endswith(".dcm")]
# Check which files have not been processed before
cur.execute('SELECT original_file_path FROM {0}'.format(tbl_names[1]))
processed_files = cur.fetchall()
new_files = np.setdiff1d(all_files,processed_files).tolist()
# RTSTRUCT information to be sotred in the database
columns_contour = ['mrn','acc','contour_name','contour_date','contour_author','contour_method','contour_software', 
                   'contour_original_path','contour_new_path','contour_uid','image_series_uid','image_date',
                       'analysis_done']

"""------------------------------- CONTOUR RECORD CREATION IN DATABASE -------------------------------------------------------------------""" 
error_file_record = []
error_file_analysis = []
for filename in new_files:
        # Record file in database
        cur.execute('INSERT INTO {0} ({1}) VALUES(%s,%s)'.format(tbl_names[1],','.join(col_names[1][:-1])),(filename,0))    
        conn.commit()       
        try:
            ds = pydicom.dcmread(filename) 
            print("Processing " + filename)
            mrn = ds.PatientID
            acc = ds.AccessionNumber
            try:
                modality = ds[0x0008, 0x0060].value # Get file type                
                if modality == 'RTSTRUCT': # Check if RTSTRUCT                                                
                    # CHECK if it has been previously recorded in the database
                    cur.execute('SELECT contour_uid FROM {0}'.format(tbl_names[0]))
                    recorded_contour_uids = cur.fetchall()
                    uid_contour = str(ds[0x0008,0x0018].value)                    
                    if (uid_contour,) not in recorded_contour_uids:
                        # get software used to make contour
                        contour_software = ds[0x0008,0x1090].value
                        ## Get details on contours
                        n_contours = ds[0x3006,0x0020].VM
                        contour_date = ds[0x0008, 0x0021].value
                        contour_time = ds[0x0008, 0x0031].value
                        contour_date_time = format_datetime(contour_date+contour_time)
                        study_acc = str(ds[0x0020, 0x0010].value)
                        uid_images = str(ds[0x3006,0x0010][0][0x3006,0x0012][0][0x3006,0x0014][0][0x0020,0x000e].value)                        
                        image_date = format_datetime(ds.StudyDate+ds.StudyTime) # Date of image series acquisition                       
                        # Get data from software-specific locations
                        contour_creator = "unknown"
                        if contour_software == "MIM":
                            contour_creator = str(ds[0x0008, 0x1070].value)                                       
                        # Move contour file
                        new_folder = output_contour_folder+"\\"+str(mrn)+"\\"+str(acc)
                        verify_folder(new_folder)
                        contour_path = new_folder+"\\"+os.path.basename(filename)
                        try: shutil.copy(filename,contour_path)
                        except: print("Could not move file of modality "+ modality + " from " + filename + " to "+ contour_path)
                        # Add contours to database and rename ROIS
                        for i in range(n_contours):
                            original_name = ds[0x3006,0x0020][i][0x3006, 0x0026].value # Original contour name
                            contour_name = contour_creator+'-'+original_name+'-'+str(contour_date_time) # New contour name
                            ds[0x3006,0x0020][i][0x3006, 0x0026].value = contour_name
                            contour_method = ds[0x3006,0x0020][i][0x3006, 0x0036].value                                                                       
                            data = (mrn,acc,contour_name,contour_date_time,contour_creator,contour_method, \
                                    contour_software,filename,contour_path,uid_contour,uid_images,image_date,0)                           
                            sql = 'INSERT INTO {0}({1}) VALUES({2})'.format(tbl_names[0],','.join(columns_contour),','.join(['%s']*len(columns_contour)))
                            cur.execute(sql, data)                       
                            conn.commit() # commit the changes to the database
                        pydicom.dcmwrite(contour_path,ds) # Save file with new ROI names                                                   
                elif modality in ['MR','CT']: # Check if it is and image series file                   
                    series_uid = str(ds.SeriesInstanceUID)
                    # Copy image file to new folder
                    new_folder = output_images_folder+"\\"+str(mrn)+"\\"+str(acc)+"\\"+series_uid
                    verify_folder(new_folder)
                    new_path = new_folder+"\\"+os.path.basename(filename)
                    try: shutil.copy(filename,new_path)
                    except: print("Could not copy file of modality "+ modality + " from " + filename + " to "+ new_path)                   
                elif modality in ['REG','RAW']: # Check if it is a registration file                   
                    new_folder = output_reg_folder+"\\"+str(mrn)+"\\"+str(acc)
                    verify_folder(new_folder)
                    new_path = new_folder+"\\"+os.path.basename(filename)
                    try: shutil.copy(filename,new_path)
                    except: print("Could not copy file of modality "+ modality + " from " + filename + " to "+ new_path) 
                else:
                    error_file_record.append(filename)
                    sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
                    cur.execute(sql, (1,filename))                       
                    conn.commit()                
            except:
                error_file_record.append(filename)
                sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
                cur.execute(sql, (1,filename))                       
                conn.commit()                             
        except Exception as err:
            print(err, filename)
            error_file_record.append(filename)
            sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
            cur.execute(sql, (1,filename))                       
            conn.commit() 
print("___________________ PROCESSING OF DICOM FILES COMPLETED __________________________________________________")


"""------------------------------- RADIOMIC/PERFUSION FEATURE EXTRACTION -----------------------------------------------------------------"""
all_image_folders = [x[0] for x in os.walk(output_images_folder)]
missing_folder = []
# Retrieve contour records
cur.execute('SELECT * FROM {0} WHERE analysis_done = 0'.format(tbl_names[0]))
all_records = cur.fetchall()
# Define the parameter file and prep the extractor
extractor = featureextractor.RadiomicsFeatureExtractor("Params.yaml")
# Columns to write in database
columns_rad = ['sequence_name','scanner_manufacturer','procedure','voxel_x','voxel_y','voxel_z','analysis_date',\
           'analysis_done'] + col_names[0][col_names[0].index('shape_Elongation'):-1] 
columns_rad = [name + '=%s' for name in columns_rad]
columns_tic = ['sequence_name','scanner_manufacturer','procedure','voxel_x','voxel_y','voxel_z','analysis_date',\
           'analysis_done','time_data','intensity_data','perfusion_parameters']
columns_tic = [name + '=%s' for name in columns_tic]
t0 = time.time() # Initial execution time
for contour_id,contour_record in enumerate(all_records): 
    print("+++++++++++++++++++++++++++ Contour "+str(contour_id+1)+" +++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
    mrn = contour_record[col_names[0].index('mrn')]
    acc = contour_record[col_names[0].index('acc')]
    contour_name = contour_record[col_names[0].index('contour_name')]
    original_contour_path = contour_record[col_names[0].index('contour_original_path')]
    contour_file_path = contour_record[col_names[0].index('contour_new_path')]
    image_series_uid = contour_record[col_names[0].index('image_series_uid')]       
    image_series_folder = os.path.join(output_images_folder,str(mrn),str(acc),image_series_uid)   
    if image_series_folder in all_image_folders: # Check if folder exists      
        ds = pydicom.dcmread(os.path.join(image_series_folder,os.listdir(image_series_folder)[0])) # Read ONE image  
        voxel_x = str(ds[0x0028,0x0030].value[0]) # Spatial resolution in mm
        voxel_y = voxel_x
        try: voxel_z = str(ds.SliceThickness.real) # Total slice thickness
        except: voxel_z = str(ds[0x0018,0x0088].value) # Total slice thickness   
        modality= ds[0x0008, 0x0060].value 
        if modality == 'MR': 
            sequence_name = format_series_name(ds.SeriesDescription) 
            try: procedure = ds[0x0040, 0x0254].value
            except: procedure='Unknown'                      
        elif modality == 'CT':
            sequence_name = modality
            procedure = ds.SeriesDescription       
        else:
            sequence_name = modality
        scanner_manufacturer = ds[0x0008,0x0070].value
        analysis_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")           
        # Process images and corresponding contours
        try:
            Dicom_reader = DicomReaderWriter() 
            Dicom_reader.walk_through_folders(image_series_folder) # Read all image files
            Dicom_reader.walk_through_folders(os.path.dirname(contour_file_path)) # Read contour files            
            Dicom_reader.set_contour_names_and_associations([contour_name])
            Dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index
            image_3d = Dicom_reader.ArrayDicom # image 3D array
            mask_3d = Dicom_reader.mask # mask 3D array 
            print("Processing mask: "+ contour_name)
            
            ######################## PERFUSION SUBROUTINE ############################################################################## 
            if sequence_name == 'SAGDYN': 
                print('--Generating TIC')
                t = get_pwi_temporal_data(image_series_folder) # Time points in seconds          
                total_phases = len(t)
                total_num_slices = image_3d.shape[0]
                slices_per_phase = int(total_num_slices/total_phases)
                # Get dimensions of image
                x_dim = image_3d.shape[1]
                y_dim = image_3d.shape[2]
                # Reshape the array to split the phase into a new dimension
                image_4d = np.reshape(image_3d,(total_phases,slices_per_phase,x_dim,y_dim))
                mask_4d = np.reshape(mask_3d,(slices_per_phase,total_phases,x_dim,y_dim))
                # Flatten the mask
                mask_3d = mask_4d.sum(axis=1) # 3D Mask  
                intensity = pwi_intensity(image_4d,mask_3d) # Mean intensity at each time point
                intensity = intensity.astype(t.dtype) # Make intensity array data type match that of t
                _, _, dce_data = dce_curve_fit(t,intensity) # Perfusion TIC parameters
                # Write TIC to database
                data_for_table = (sequence_name,
                                scanner_manufacturer,
                                procedure,
                                voxel_x,
                                voxel_y,
                                voxel_z,
                                analysis_date_time,
                                1,
                                list(t),
                                list(intensity),
                                dce_data)             
                print('--Writing to Database')
                sql = 'UPDATE {0} SET {1} WHERE contour_name=%s'.format(tbl_names[0],','.join(columns_tic))
                cur.execute(sql, data_for_table+(contour_name,))                       
                conn.commit() # commit the changes to the database  
                sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
                cur.execute(sql, (0,original_contour_path))                       
                conn.commit()
                """+++++++++++++++++++++++ PLOT RESULTS +++++++++++++++++++++++"""
                print('--Generating Report')
                generate_results_report(tbl_names[0],contour_name,image_4d,mask_3d,output_report_folder)            
            
            ######################## RADIOMIC FEATURE EXTRACTION ########################################################################    
            else: # DWI / SWI / CT Sequences           
                image_sitk = Dicom_reader.dicom_handle 
                mask_sitk = Dicom_reader.annotation_handle 
                print('--Extracting Radiomic Features')
                result = extractor.execute(image_sitk, mask_sitk)
                if result['original_firstorder_Minimum'] < 0:
                    result['original_firstorder_Minimum'] = 0 # Set the minimum intensity to zero to avoid negative values
                result['original_firstorder_Kurtosis']-= 3 # Correct kurtosis to match IBSI
                dict_names = list(result.keys()) # Extract field names from dictionary
                feature_values = tuple([str(value) for value in result.values()])[dict_names.index('original_shape_Elongation'):] # Extract all radiomic features
                # Add radiomics to database
                data_for_table = (sequence_name,
                                scanner_manufacturer,
                                procedure,
                                voxel_x,
                                voxel_y,
                                voxel_z,
                                analysis_date_time,
                                1)+feature_values
                print('--Writing to Database')
                sql = 'UPDATE {0} SET {1} WHERE contour_name=%s'.format(tbl_names[0],','.join(columns_rad))
                cur.execute(sql, data_for_table+(contour_name,))                       
                conn.commit() # commit the changes to the database
                sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
                cur.execute(sql, (0,original_contour_path))                       
                conn.commit()  
                """+++++++++++++++++++++++ PLOT RESULTS +++++++++++++++++++++++"""
                print('--Generating Report')
                generate_results_report(tbl_names[0],contour_name,image_3d,mask_3d,output_report_folder)             
                print("  kurtosis: " + str(result['original_firstorder_Kurtosis']))
                print("  max: " + str(result['original_firstorder_Maximum']))
                print("  mean: " + str(result['original_firstorder_Mean']))
                print("  median: " + str(result['original_firstorder_Median']))
                print("  min: " + str(result['original_firstorder_Minimum']))
                print("  skewness: " + str(result['original_firstorder_Skewness']))
                print("=================================================================")                                             
        except: # Failed analysis of the contour record
            cur.execute('UPDATE {0} SET sequence_name=%s WHERE contour_name=%s'.format(tbl_names[0]),(sequence_name,contour_name))
            conn.commit() # Record the sequence name of the failed contour record
            error_file_analysis.append(original_contour_path)   
            sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
            cur.execute(sql, (1,original_contour_path))                       
            conn.commit() 
    else: # If no image series folder then remove contour record from database
        cur.execute('DELETE FROM {0} WHERE contour_name = %s'.format(tbl_names[0]),(contour_name,))    
        conn.commit()
        missing_folder.append(image_series_folder)
        sql = 'UPDATE {0} SET failed_processing=%s WHERE original_file_path=%s'.format(tbl_names[1])
        cur.execute(sql, (1,original_contour_path))                       
        conn.commit()        
cur.close()
conn.close()
print("___________________ RADIOMICS COMPUTATION COMPLETED ____________________________________________________\n")
print("--PostgreSQL connection is closed\n")
print('FAILED RECORD FOR DICOM FILES:\n',error_file_record,'\n')
print('FAILED ANALYSIS FOR CONTOUR FILES:\n',error_file_analysis,'\n')
print('MISSING IMAGE SERIES FOLDERS:\n',missing_folder)
print('*** TOTAL ANALYSIS EXECUTION TIME: ' + str(round((time.time() - t0)/60,1)) +' minutes ***')
