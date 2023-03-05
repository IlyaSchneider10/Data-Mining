# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:45:59 2022

@author: schni
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse

def eucidean_d(x, y):
    
    distance= np.sqrt(np.sum((x-y)**2))

    return distance

def knn(matrix, lookup_matrix, test_matrix, k_min, k_max):
    
    train_points_number=matrix.shape[0]
    
    data_points_number=test_matrix.shape[0]
    
    dist_mat=pd.DataFrame(np.zeros((data_points_number,train_points_number)),index=test_matrix.index, columns=matrix.index)
    
    for test_point in range(data_points_number):
        for train_point in range(train_points_number):
            dist_mat.iloc[test_point,train_point]=eucidean_d(matrix.iloc[train_point,:],test_matrix.iloc[test_point,:])
    
    for patient in range(lookup_matrix.shape[0]):
        if (lookup_matrix.iloc[patient]=="+").bool():
            lookup_matrix.iloc[patient]=1
        else:
            lookup_matrix.iloc[patient]=-1
    
    output=[]
    for k in range(k_min, k_max+1):
        classification_matrix=pd.DataFrame(np.zeros((data_points_number,lookup_matrix.shape[1])),index=test_matrix.index, columns=lookup_matrix.columns)
        for ref_point in range(data_points_number):
            nn=dist_mat.iloc[ref_point,:].sort_values(ascending=True).index[0:k]
            outcome=np.sum(lookup_matrix.loc[nn,"ERstatus"])
            
            if outcome<0: 
                classification_matrix.iloc[ref_point,0]=-1
            elif outcome>0:
                classification_matrix.iloc[ref_point,0]=1
            else:
                nn_new=dist_mat.iloc[ref_point,:].sort_values(ascending=True).index[0:k-1]
                outcome_new=np.sum(lookup_matrix.loc[nn_new,"ERstatus"])
                if outcome_new>0:
                    classification_matrix.iloc[ref_point,0]=1
                else:
                    classification_matrix.iloc[ref_point,0]=-1
                
        output.append(classification_matrix)
                
    return output

def compare(ground_truth, knn_output):
    
    for patient in range(ground_truth.shape[0]):
        if (ground_truth.iloc[patient]=="+").bool():
            ground_truth.iloc[patient]=1
        else:
            ground_truth.iloc[patient]=-1
    
    contigency_table=[]
    
    for prediction in range(len(knn_output)):
        
        
        comaprison_table=ground_truth==knn_output[prediction]
        tp=0
        tn=0
        fp=0
        fn=0

        for patient in range(comaprison_table.shape[0]):
                
            if (comaprison_table.iloc[patient]==True).bool():
                if (knn_output[prediction].iloc[patient]==True).bool():
                    tp+=1
                else:
                    tn+=1
            
            else:
                if (knn_output[prediction].iloc[patient]==True).bool():
                    fp+=1
                else:
                    fn+=1
                    
        accuracy=round((tp+tn)/(tp+fp+tn+fn),2)
        precision=round(tp/(tp+fp),2)
        recall=round(tp/(tp+fn),2)
        
        table=[prediction+1,accuracy,precision,recall]
        contigency_table.append(table)
        
    return contigency_table

if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute kNN"
    )
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to directory containing training data"
    )
    parser.add_argument(
        "--testdir",
        required=True,
        help="Path to directory containing test data"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_knn.txt will be created"
    )
    parser.add_argument(
        "--mink",
        required=True,
        help="Minimum k considered"
    )
    parser.add_argument(
        "--maxk",
        required=True,
        help="Maximum k considered"
    )

    args = parser.parse_args()

    # Set the paths
    train_dir = args.traindir
    test_dir = args.testdir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)
    
    # Define the k range
    kmin=int(args.mink)
    kmax=int(args.maxk)

    # Read the file
    train_data=pd.DataFrame(pd.read_csv("{}/{}".format(args.traindir, 'matrix_mirna_input.txt'),sep='\t', header = 0, index_col=0))
    train_lookup=pd.DataFrame(pd.read_csv("{}/{}".format(args.traindir, 'phenotype.txt'),sep='\t', header = 0, index_col=0))
    
    test_data=pd.DataFrame(pd.read_csv("{}/{}".format(args.testdir, 'matrix_mirna_input.txt'),sep='\t', header = 0, index_col=0))
    test_lookup=pd.DataFrame(pd.read_csv("{}/{}".format(args.testdir, 'phenotype.txt'),sep='\t', header = 0, index_col=0))
    
    
    # Create the output file
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    # Write header for output file
    f_out.write('{}\t{}\t{}\t{}\n'.format(
        'Value of k',
        'Accuracy',
        'Precision',
        'Recall'))

    # Run the functions
    knn_predictions=knn(train_data,train_lookup, test_data, kmin, kmax)
    results=compare(test_lookup,knn_predictions)

    # Transform the results to a string
    row_output=[]
    for k_value in range(kmin-1, kmax):
        f=['{}\t{}\t{}\t{}'.format(results[k_value][0],results[k_value][1],results[k_value][2],results[k_value][3])]
        row_output=np.append(row_output,f)
    
    rows='\n'.join(row_output)
    
    # Save the output
    f_out.write(rows)
 
    f_out.close()

