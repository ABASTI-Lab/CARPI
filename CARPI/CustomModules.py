# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:44:24 2023

@author: Elvis de Jesus Duran Sierra, PhD
"""

"""------------------------------- LIBRARIES --------------------------------------------------------------------------------------------"""
import os as os
import datetime
import psycopg2 # To work with databases (pip install psycopg2)
from config import config
import numpy as np
import pydicom
import glob
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score
import warnings

"""------------------------------- MODULES ---------------------------------------------------------------------------------------------"""
def connect_to_db():
    
    try:
        # read database configuration
        config_params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**config_params)
        # create a new cursor
        cur = conn.cursor()        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        
    return conn,cur


def add_tables_to_db(tables):
    
    conn, cur = connect_to_db()
    
    for name in list(tables.keys()):
        
        command = 'CREATE TABLE {0} ({1})'.format(name,','.join(tables[name]))
        try:
            cur.execute("DROP TABLE IF EXISTS {0}".format(name)) 
            cur.execute(command) # Create table
            conn.commit() # Commit the changes
            print('--Succesfully Created Table: ' + name)                          
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
    # Close communication with the PostgreSQL database server
    cur.close()
    conn.close()

     
def format_datetime(date_time_str):
    try:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y%m%d%H%M%S.%f')
    except:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
    
    return str(date_time_obj)


def verify_folder(path_to_folder):
    
    # Check to see if folder exists
    folder_exists = os.path.isdir(path_to_folder)
    
    # Create folder if not and set folder_exists=True if successful
    if not folder_exists:
        try:
            os.makedirs(path_to_folder)
            folder_exists = True
        except:
            folder_exists = False
    
    return folder_exists


def format_series_name(sequence_name):

    if 'swi' in sequence_name.lower():
        return 'SWI'
    elif any(s in sequence_name.lower() for s in ['dwi','adc','apparent diffusion coefficient']):
        return 'ADC'
    elif 'sag dyn' in sequence_name.lower():
        return 'SAGDYN'
    elif 't1' in sequence_name.lower():
        return '+C'
    elif 't2' in sequence_name.lower():
        return 'T2'   
    elif 'stir' in sequence_name.lower():
        return 'STIR'
    else:
        return sequence_name
    
    
def get_pwi_temporal_data(series_folder):
    
    phase_number=[]
    image_time_sec=[]
    image_date_time=[]
    trigger_time = []
    time_conversion = False
    all_images = glob.glob(series_folder + "**/*.dcm", recursive=True)
    print('--Extracting Time Data from ' + series_folder + '...')
    for filename in all_images:
        ds = pydicom.dcmread(filename) 
        image_date_time.append(ds.AcquisitionDate + ds.AcquisitionTime) # Date and time of acquired image
        try:
            phase_number.append(ds.TemporalPositionIdentifier) # Usually for GE
        except:
            phase_number.append(ds.AcquisitionNumber) # Siemens and sometimes GE                     
        # Siemens encodes it here
        # GE sometimes doesn't have the trigger time (sometimes variable in the same series)
        try:
            trigger_time.append(ds[0x0018, 0x1060].value/1000) # Trigger time in seconds
        except: 
            trigger_time.append('none')                        
    # If number of unique time points less than number of unique phases
    if len(set(image_date_time)) < len(set(phase_number)):
        image_date_time = trigger_time
    else:
        time_conversion = True                                
    if time_conversion: # Datetime conversion to seconds
        try:
            date_time_list = [datetime.datetime.strptime(x,"%Y%m%d%H%M%S.%f") for x in image_date_time] # Format date-time
        except:
            date_time_list = [datetime.datetime.strptime(x,"%Y%m%d%H%M%S") for x in image_date_time] # Format date-time
        min_date_time = min(date_time_list) # Get first timestamp
        delta_dt = [x-min_date_time for x in date_time_list] # Take difference with first timestamp
        image_time_sec=[x.total_seconds() for x in delta_dt] # Time in seconds with respect to the first datetime 
    else:
        min_time = min(image_date_time) # Get first timestamp
        image_time_sec = [x-min_time for x in image_date_time] # Take difference with first time point                                     
    time_points = list(set(image_time_sec)) # Get unique time points
    time_points.sort() # Ascending order
    # Check if redundant time points
    total_phases = len(time_points)
    total_num_slices = len(image_date_time)
    if (total_num_slices % total_phases) != 0: # If the division is inexact
        indx = np.argwhere(np.diff(time_points) < 2).flatten() # Check for redundant time points
        time_points = np.delete(time_points,indx+1).tolist() # Remove redundant time points
    # Create array of equally spaced time points
    delta_points = np.diff(time_points) # Delta between time points
    dt = stats.mode(delta_points)[0][0] # Most frequent delta
    t = np.arange(len(time_points))*dt
    print('--Finished Extracting Time Data')  

    return t


def log_normal(t,auc,t0,c,mu,sigma):
    
    yy1 = auc/( np.sqrt(2*np.pi)*sigma*(t-t0) )
    yy2 = np.exp( (-(np.log(t-t0)-mu)**2)/(2*sigma**2) )
    y = yy1*yy2 + c
    y[np.argwhere(t<=t0)]=0 # set y(t<=t0)=0
    y = y/max(y) # Normalize values by peak
    
    return y


def dce_curve_fit(t,i):
    
    # Make sure all values > 0
    t = np.array(t)
    i = np.array(i)
    t[np.where(t<0)] = 0
    i[np.where(i<0)] = 0
    i_norm = i/max(i) # Normalize intensity values by peak
    
    # Paramters of the model
    auc = np.trapz(i,t) # Area under curve
    t0 = t[np.where(i>0)[0][0]] # offset time - first time point where intensity > 0
    c = 0 # Baseline intensity offset
    bounds = (0,np.inf)
    maxfev = 800000
    mu = 3 # Mean
    sigma = 1 # Standard deviation
    
    # Non-linear least squares fitting
    warnings.filterwarnings('ignore')
    try:
        p0 = [auc,t0,c,mu,sigma] 
        popt_ln, pcov = curve_fit(log_normal, t, i_norm, p0=p0, bounds=bounds, maxfev = maxfev)
        i_fit = log_normal(t,*popt_ln)
        r2 = np.round(r2_score(i_norm,i_fit),2)      
        # Optimize initial guess for mu
        mu_values = list(range(8)) # Mu values ranging 0 to 7
        r2_scores = [0]*len(mu_values)
        m = 0
        while r2 < 0.98: # If the fitting is bad
            p0 = [auc,t0,c,mu_values[m],sigma] 
            popt_ln, pcov = curve_fit(log_normal, t, i_norm, p0=p0, bounds=bounds, maxfev = maxfev)
            i_fit = log_normal(t,*popt_ln)
            r2 = np.round(r2_score(i_norm,i_fit),2)
            r2_scores[m] = r2
            if m == len(mu_values)-1: # If all mu values have been tested
                mu_opt = mu_values[r2_scores.index(max(r2_scores))] # Optimal mu
                # Non-linear least squares fitting using optimal mu
                p0 = [auc,t0,c,mu_opt,sigma] 
                popt_ln, pcov = curve_fit(log_normal, t, i_norm, p0=p0, bounds=bounds, maxfev = maxfev)
                i_fit = log_normal(t,*popt_ln) 
                break
            m+=1
        dce_data = dce_params(t,i_fit) # TIC kinetic parameters 
    except:
        print('Optimal Parameters NOT Found!!')
        i_fit = [0]*i_norm.size
        dce_data = [0]*7
        
    return i_norm, i_fit, dce_data
  
    
def dce_params(t,i):
   
    dt = t[2]-t[1]
    slopes = np.gradient(i,dt) # Compute gradients in curve
    slopes[0] = np.nan; slopes[-1] = np.nan # Exclude gradient at 1st and last points
    # Time to peak index
    if any(slopes<0): # Check for curve wash out: negative slope is present
        indx = np.argmax(i) # Index of peak intensity
    else:
        indx = np.nanargmax(slopes) # Index of max slope
    
    # DCE Parameters
    ttp = round(t[indx],1) # Time to peak 
    pe = round(i[indx],2) # Peak enhancement
    #rt = ttp - t[np.where(i>0)[0][0]] # Rise time
    w_in = np.nanmax(slopes[:indx+1]) # Wash-in rate
    w_out = np.nanmin(slopes[indx+1:]) # Wash-out rate
    wiAUC = round(np.sum(i[:indx+1])*dt,2) # Wash in area under curve
    woAUC = round(np.sum(i[indx+1:])*dt,2) # Wash out area under curve
    auc = round(wiAUC + woAUC,2) # Total area under curve
    dce_data = [ttp,pe,w_in,w_out,wiAUC,woAUC,auc]
  
    return dce_data
  

def pwi_intensity(image_4d,mask_3d):
    
    total_phases = image_4d.shape[0]
    intensity = [] # Intensity data
    print("--Processing "+str(total_phases)+" phases:")
    for phase in range(total_phases):
        print('Phase '+str(phase+1))
        image_3d = image_4d[phase,:,:,:]
        masked_voxels = np.extract(mask_3d,image_3d)
        intensity.append(masked_voxels.mean())        
    # Subtract signal data from t0 (remove offset)
    intensity = np.array(intensity)-intensity[0]
    
    return intensity


def generate_results_report(tbl_name,contour_name,image_3d,mask_3d,report_path):
    
    # Connect to database
    conn, cur = connect_to_db()
    cur.execute('SELECT * FROM {0} WHERE contour_name=%s'.format(tbl_name),(contour_name,))
    col_names = [desc[0] for desc in cur.description]
    record = cur.fetchone()
    mrn = record[col_names.index('mrn')]
    image_date = record[col_names.index('image_date')].date()
    sequence = record[col_names.index('sequence_name')]
    
    # Colormap for mask display
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[0][-1] = 0
    my_cmap = ListedColormap(my_cmap)
    sns.set_theme(style="whitegrid")

    if sequence == 'SAGDYN': # For Perfusion Graphic Reports
        time_points = np.array(record[col_names.index('time_data')])
        intensity  = np.array(record[col_names.index('intensity_data')])
        i_norm, i_fit, dce_data = dce_curve_fit(time_points,intensity) # Fitted TIC
                        # NOTE: dce_data order of parameters =  [ttp,pe,w_in,w_out,wiAUC,woAUC,auc]       
        # Display Images and Mask
        image_3d = image_3d[-1,:,:,:] # Use last phase of image
        slice_size = [np.sum(mask_3d[i,:,:]) for i in range(mask_3d.shape[0])]
        slice_num = slice_size.index(max(slice_size)) # Largest slice for display
        fsize = 35 # Font size for figures
        fig = plt.figure(figsize=(30,13),dpi=300)
        fig.suptitle('MRN: {0} | Sequence Name: PWI | Image Date: {1}'.format(mrn,image_date),
                     fontweight='bold',fontsize=fsize)
        gs = GridSpec(1, 3, figure=fig)
        # Image + Mask
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(image_3d[slice_num,:,:] ,cmap='Greys_r', aspect='equal')
        ax1.imshow(mask_3d[slice_num,:,:], cmap=my_cmap,alpha=.4, aspect='equal')
        ax1.axis('off')
        ax1.set_title('Contoured Image Slice',fontweight='bold',fontsize=fsize)        
        # TIC Plot
        ax2 = fig.add_subplot(gs[0,1:])
        ax2.plot(time_points,i_norm,'o',color='darkred',linewidth=10,markersize=25)
        ax2.plot(time_points,i_fit,'--',color='darkblue',linewidth=5)
        ax2.set_ylabel('Normalized Intensity (a.u.)',fontweight= 'bold',fontsize=fsize)
        ax2.set_xlabel('Time (sec)',fontweight='bold',fontsize=fsize)   
        ax2.set_title('Time-Intensity Curve',fontweight='bold',fontsize=fsize)
        ax2.legend(['Raw Data','Fitted Model'],prop={'size': 25,'weight':'bold'},loc='upper left',markerscale=0.8)
        txt = 'TTP='+str(dce_data[0])+'\n'+'PE='+str(dce_data[1])+'\n'+r'$\bf{Wi_{R}}$='+'{:.2e}'.format(dce_data[2])+ \
            '\n'+r'$\bf{Wo_{R}}$='+'{:.2e}'.format(dce_data[3])+'\n'+r'$\bf{Wi_{AUC}}$='+str(dce_data[4])+'\n' + \
            r'$\bf{Wo_{AUC}}$='+str(dce_data[5])+'\n'+'AUC='+str(dce_data[-1])
        coords = [max(ax2.get_xticks())-(max(ax2.get_xticks())/3),0]
        ax2.text(coords[0],coords[1],txt,fontsize=30,fontweight='bold')
        plt.xticks(fontsize=fsize,fontweight='bold')
        plt.yticks(fontsize=fsize,fontweight='bold')
        plt.subplots_adjust(wspace=0.35)

    else: # SWI and DWI/ADC Graphic Radiomic Reports
        indx = col_names.index('shape_elongation') # Column index of first radiomic feature
        radiomics = np.around(np.array(record[indx:-1]).astype(float),2)
        feature_names = np.load('feature_names.npy').tolist() # Column names for radiomic data
        feature_names = [name.replace('original_','')[name.replace('original_','').index('_')+1:] for name in feature_names]
        feature_names = [name if len(name)<=10 else name[:10]+'...' for name in feature_names]
        # Remove feature from list: FOR DISPLAY PURPOSES
        radiomics = np.delete(radiomics,feature_names.index('Energy'))
        feature_names.pop(feature_names.index('Energy')) 
        indx1, indx2 = feature_names.index('Range'), feature_names.index('Skewness')
        feature_names[indx1], feature_names[indx2] = feature_names[indx2], feature_names[indx1] # Swap names in list
        radiomics[indx1], radiomics[indx2] = radiomics[indx2], radiomics[indx1] # Swap values in radiomics array
        
        # Display images and mask
        fsize = 40 # Font size for figures
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(23,12),dpi=300)
        fig.suptitle('MRN: {0} | Sequence Name: {1} | Image Date: {2}'.format(mrn,sequence,image_date),\
                     fontweight='bold',fontsize=fsize)
        # Image + Mask
        slice_size = [np.sum(mask_3d[i,:,:]) for i in range(mask_3d.shape[0])]
        slice_num = slice_size.index(max(slice_size)) # Largest slice for display
        ax[0].imshow(image_3d[slice_num,:,:], cmap='Greys_r', aspect='equal')
        ax[0].imshow(mask_3d[slice_num,:,:], cmap=my_cmap,alpha=.4, aspect='equal')
        ax[0].axis('off')
        ax[0].set_title('Contoured Image Slice',fontweight='bold',fontsize=fsize)   
        
        # Radiomics Table
        rad_strings = ["%.2f" % number for number in radiomics] # Convert radiomic data to strings
        n = 5 # Number of table columns
        num_features = 25
        tbl_values = []
        for i in range(0,num_features,n):
            tbl_values.append(feature_names[i:i+n])
            tbl_values.append(rad_strings[i:i+n])
            
        tbl = plt.table(cellText=tbl_values,    
                  cellLoc = 'center', rowLoc = 'center',
                  transform=plt.gcf().transFigure,
                  bbox = ([0.1, -0.5, 0.82, 0.5])) #[x-position, y-position, table width, table height]
        for (row, col), cell in tbl.get_celld().items():
               cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(35)  
     
        # Histogram of Voxels        
        masked_voxels = np.extract(mask_3d,image_3d)
        sns.histplot(data=masked_voxels,stat='percent',ax=ax[1],bins=50)
        ax[1].set_ylabel('Total Voxels (%)',fontweight= 'bold',fontsize=fsize)
        ax[1].set_xlabel('Value',fontweight='bold',fontsize=fsize)
        ax[1].set_title('Histogram of VOI',fontweight='bold',fontsize=fsize)
        plt.xticks(fontsize=32,fontweight='bold')
        plt.yticks(fontsize=32,fontweight='bold')
        plt.subplots_adjust(wspace=0.3)     
    
    # Save Figure to a File
    contour_name = contour_name.replace(':','_') # Replace colon
    contour_name = contour_name.replace('.','_') # Replace dot
    file_name = str(mrn) + '_' + sequence + '_' + str(image_date) + '_' + contour_name
    fig.savefig(report_path + '\\' + file_name, bbox_inches = 'tight')
    plt.close(fig)
    cur.close()
    conn.close()

    return print('--Succesfully Saved Report')


