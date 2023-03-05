# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:37:56 2022

@author: schni
"""

COL1A1="GPAGFAGPPGDA"
COL1A2="PRGDQGPVGRTG"
GPR143="GFPNFDVSVSDM"

def similarity_measure_kernel(seq1, seq2):

    similarity=0
    
    for position1 in range(len(seq1)-2):
        
       s=seq1[position1:position1+3]
       
       for position2 in range(len(seq2)-2):
           
           s_=seq2[position2:position2+3]
           
           if s[0]==s_[0] and s[0]=="G":
               
               for p in range(3):
                   
                  if s[p]==s_[p]:
                      
                      similarity+=1
                  
    return similarity
       