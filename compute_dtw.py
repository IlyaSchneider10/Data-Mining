"""
Homework  : Similarity measures on sets
Course    : Data Mining (636-0018-00L)

Compute all pairwise DTW and Euclidean distances of time-series within
and between groups.
"""
# Author: Xiao He <xiao.he@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import os
import sys
import argparse
import numpy as np

def manhattan_distance(x, y):
    
    answer=np.sum(abs(x-y))
      
    return answer


def constrained_dtw(x, y, w):
    
    #Define distance matrix dimensions
    m=len(x)
    n=len(y)
    
    #Create an empty distance matrix:
    dist_mat=np.full((m+1,n+1),np.inf)
    dist_mat[0,0]=0
    
    #Adress w value:
    if w==float('inf'):
        w=max(m,n)
    else:
        w_constraint=w+1
        

    #Fill in the matrix:
    if m==n:    
        for i in range(m-w):             
            for j in range(w_constraint):
                
                #Calculate distances within the same row applying w constraint
                all_costs_rows=[dist_mat[i,i+j],dist_mat[i+1,i+j],dist_mat[i,i+1+j]]
                cost_row=np.argmin(all_costs_rows)
                
                dtw_row=abs(x[i]-y[i+j])+all_costs_rows[cost_row]
                dist_mat[i+1,i+j+1]=dtw_row
                
                #Calculate distances within the same column applying w constraint
                all_costs_columns=[dist_mat[i+j,i],dist_mat[i+1+j,i],dist_mat[i+j,i+1]]
                cost_column=np.argmin(all_costs_columns)
                
                dtw_column=abs(x[i+j]-y[i])+all_costs_columns[cost_column]
                dist_mat[i+1+j,i+1]=dtw_column
                
        #At the end the w constraint goes to dimensions that exceed the dist_mat dimensions
        #So an empty redion of w x w is left at the left bottom of the dist_mat
        
        #Calculate this w x w empty submatrix
        for ii in range(m-w,m):
            for jj in range(n-w,n):
            
                all_costs=[dist_mat[ii,jj],dist_mat[ii+1,jj],dist_mat[ii,jj+1]]
                cost=np.argmin(all_costs)
                
                dtw=abs(x[ii]-y[jj])+all_costs[cost]
                dist_mat[ii+1,jj+1]=dtw
                
        #Backtracking
    
        final_distance=dist_mat[m,n]
    
    return final_distance


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute distance functions on time-series"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file EGC200_TRAIN.txt"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where timeseries_output.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)

    # Read the file
    data = np.loadtxt("{}/{}".format(args.datadir, 'ECG200_TRAIN.txt'),
                      delimiter=',')

    # Create the output file
    try:
        file_name = "{}/timeseries_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    cdict = {}
    cdict['abnormal'] = -1
    cdict['normal'] = 1
    lst_group = ['abnormal', 'normal']
    w_vals = [0, 10, 25, float('inf')]

    # Write header for output file
    f_out.write('{}\t{}\t{}\n'.format(
        'Pair of classes',
        'Manhattan',
        '\t'.join(['DTW, w = {}'.format(w) for w in w_vals])))

    # Iterate through all combinations of pairs
    for idx_g1 in range(len(lst_group)):
        for idx_g2 in range(idx_g1, len(lst_group)):
            # Get the group data
            group1 = data[data[:, 0] == cdict[lst_group[idx_g1]]]
            group2 = data[data[:, 0] == cdict[lst_group[idx_g2]]]

            # Get average similarity
            count = 0
            vec_sim = np.zeros(1 + len(w_vals), dtype=float)
            for x in group1[:, 1:]:
                for y in group2[:, 1:]:
                    # Skip redundant calculations
                    if idx_g1 == idx_g2 and (x == y).all():
                        continue

                    # Compute Manhattan distance
                    vec_sim[0] += manhattan_distance(x, y)

                    # Compute DTW distance for all values of hyperparameter w
                    for i, w in enumerate(w_vals):
                        vec_sim[i + 1] += constrained_dtw(x, y, w)

                    count += 1
            vec_sim /= count

            # Transform the vector of distances to a string
            str_sim = '\t'.join('{0:.2f}'.format(x) for x in vec_sim)

            # Save the output
            f_out.write(
                '{}:{}\t{}\n'.format(
                    lst_group[idx_g1], lst_group[idx_g2], str_sim)
            )
    f_out.close()
