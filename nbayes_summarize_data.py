# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:16:05 2022

@author: schni
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse


def summarize_class(Class):
    
    class_output=[]
    for column in Class.columns[0:len(Class.columns)-1]:
    
        data_class_column=Class.groupby(column)
        present_column_types=list(data_class_column.groups.keys())
        absent_column_types=list(set(features)-set(present_column_types))
        
        column_dict={}
        keys=features
        for k in keys:
            if k in absent_column_types:
                column_dict[k]=0
            else:
                value=data_class_column.get_group(k)[column]
                prob=round(len(value[~np.isnan(value)])/(len(Class[column][~np.isnan(Class[column])])),3)
                column_dict[k]=prob
        
        class_output.append(column_dict)
   
    return class_output


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute Bayes"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to directory containing training data"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_knn.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    train_dir = args.traindir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)

    # Read the file
    
    data=pd.DataFrame(pd.read_csv("{}/{}".format(args.traindir, 'tumor_info.txt'),sep='\t', header = None))
    data.columns=("clump","uniformity", "marginal","mitoses","Class")
    
    #Create classes and other variables
    features=list(range(1,11))
    
    classes=np.unique(data["Class"][~np.isnan(data["Class"])])
    
    #iterate for each class
    for c in classes: 
        
        # Create the output file 
       
        try:
            file_name = "{}/output_summary_class_{}.txt.".format(args.outdir,c)
            f_out = open(file_name, 'w')
        except IOError:
            print("Output file {} cannot be created".format(file_name))
            sys.exit(1)
    
        # Write header for output file
        f_out.write('{}\t{}\t{}\t{}\t{}\n'.format(
            'Value',
            'clump',
            'uniformity',
            'marginal',
            'mitoses'))
        
        #Select data for only one class
        class_x=(data.iloc[:,4]==c).values
        data_class_x=data.iloc[class_x,:]


        # Run the functions
        class_summary=summarize_class(data_class_x)

        # Transform the results to a string
        output=[]
        for key in features:
            
            f=['{}\t{}\t{}\t{}\t{}'.format(key,class_summary[0][key],class_summary[1][key],class_summary[2][key],class_summary[3][key])]
            output=np.append(output,f)
        
        rows='\n'.join(output)
        
        # Save the output
        f_out.write(rows)
     
        f_out.close()

