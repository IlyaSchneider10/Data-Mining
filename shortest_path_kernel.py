"""Skeleton file for your solution to the shortest-path kernel."""

def floyd_warshall(A):
    
    for i in range(A.shape[0]):
        
        for j in range(A.shape[0]):
            
            if i!=j: #consider all the elements that are not on the diagonal
                if A[i,j]==0: #if these elements are 0 substitute for inf
                    A[i,j]=99999
                    
    for k in range(A.shape[0]):
        
        for i in range(A.shape[0]):
                
            for j in range(A.shape[0]):
                
                if A[i,j]>A[i,k]+A[k,j]:
                    
                    A[i,j]=A[i,k]+A[k,j]
    
    
    return A


def sp_kernel(S1, S2):
    
    K=0
    
    for i in range(S1.shape[0]):
        
        for j in range(i+1,S1.shape[0]): #consider only the elements above the diagonal
            
            for ii in range(S2.shape[0]):
                
                for jj in range(ii+1,S2.shape[0]): # the same for the second matrix
                    
                    if S1[i,j]==S2[ii,jj]:
                        #if the values are the same and not equal to inf add 1 to K               
                        K=K+1
                    
    return K




