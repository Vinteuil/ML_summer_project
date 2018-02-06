#####################################################                                                                                                                              
# python code to run simulations of flask in a unix #                                                                                                                              
# terminal for a range of values of omega_m and     #                                                                                                                              
# random seeds to create kappa maps                 #                                                                                                                              
#####################################################                                                                                                                              

import os
import numpy as np

#list of values to create simulations of                                                                                                                                           
OMEGA_M_LIST = [0.4, 0.5]
#creating a list of labels                                                                                                                                                         
OMEGA_M_LABELS = []
num_labels = int(len(OMEGA_M_LIST))

count = 0
#iterating through combos of omega_m and random seeds                                                                                                                              
for i in OMEGA_M_LIST:
    OMEGA_M = i
    OMEGA_L = 1 - OMEGA_M
    for j in range(98):
        count = count + 1
        RNDSEED = j
        #creating a string from these values to be executed in terminal                                                                                                            
        COMMAND = ("./bin/flask example.config RNDSEED: %(RNDSEED)s OMEGA_L: %(OMEGA_L)s ELLIPFITS_PREFIX: training_data_3/ellip-%(count)s- OMEGA_m: %(OMEGA_M)s"
                        % {"RNDSEED": RNDSEED, "OMEGA_M": OMEGA_M, "OMEGA_L": OMEGA_L, "count": count})
        os.system(COMMAND)
        OMEGA_M_LABELS.append(i)

        z=[]


        os.system("rm example/*")


        for l in OMEGA_M_LABELS:
            k = int((l * 10) - 2)
            z.append(np.array([int(i == k) for i in range(7)]))


            np.savetxt("labels_one_hot.csv", z, delimiter=",")
