from __future__ import division

import numpy as np
import pickle
import tensorflow as tf
import random
import math
import sys
import os

with open('param.pickle','rb') as l_record:
	para = pickle.load(l_record)

atom_type = para['atom_type']
resi_atom_type = para['resi_atom_type']


def iRMSD_cal(x, protein, ind):
	rpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein+"_r_u.pdb.ms"

	fr = open('re.pdb', "w")

	with open(rpath, "r") as f:
		receptor_pdb = f.readlines()

		i=0
		for re in range(len(receptor_pdb)):
			if(receptor_pdb[re][0:4]=='ATOM' and (receptor_pdb[re][17:20] in resi_atom_type)):
				fr.write("%s%7.3f %7.3f %7.3f%s" %(receptor_pdb[re][0:31], x[i][0], x[i][1], x[i][2], receptor_pdb[re][54:]))
			i+=1

	fr.close()


	rpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein+"_l_u_"+str(ind)+".pdb"

	fr = open('le.pdb', "w")

	with open(rpath, "r") as f:
		receptor_pdb = f.readlines()

		i=0
		for re in range(len(receptor_pdb)):
			if(receptor_pdb[re][0:4]=='ATOM' and (receptor_pdb[re][17:20] in resi_atom_type)):
				fr.write("%s%7.3f %7.3f %7.3f%s" %(receptor_pdb[re][0:31], x[i][0], x[i][1], x[i][2], receptor_pdb[re][54:]))
			i+=1

	fr.close()

	os.system("cp re.pdb /home/cyppsp/cNMA/Example/Example1/Input/"+protein+'_r_u.pdb')
	os.system("cp le.pdb /home/cyppsp/cNMA/Example/Example1/Input/"+protein+'_l_u.pdb')
	os.system("cp /home/cyppsp/project_bayesian/zdock/benchmark/"+protein+"_r_b.pdb /home/cyppsp/cNMA/Example/Example1/Input/")
	os.system("cp /home/cyppsp/project_bayesian/zdock/benchmark/"+protein+"_l_b.pdb /home/cyppsp/cNMA/Example/Example1/Input/")

	os.system("/home/cyppsp/cNMA/Example/Example1/run_irmsd.sh "+protein)

	t=np.loadtxt("/home/cyppsp/cNMA/Example/Example1/Output/iRMSD.txt")

	return t



def energy(x12):

	s=0.
	product = np.matmul(x12*eigval, basis)

	x = np.reshape(product, coor_init.shape) + coor_init

	for i in range(len(x)):
		for j in range(len(x)):
			rij = np.linalg.norm(x[i]-x[j])
			if(rij<9 and rij>0):
				rminij = (r[i]+r[j])/2
				a1 = q[i]*q[j]/(4*rij) + (np.sqrt(e[i]*e[j])) * ((rminij/rij)**12 - (rminij/rij)**6)

				if(rij>7):
					a1 *= (9-rij)**2 * (-12 + 2*rij) / 8

				s+=a1

		#print ("atom: ", i)

	
	with open("irmsd_results", "a") as f:

		f.write("%.3lf %.3lf     " %(s, iRMSD_cal(x, protein, 2)))
		for i in range(len(x12)):
			f.write("%.3f " %(x12[i]))

		f.write('\n')
	#exit(0)
	return s


protein = "2HRK"
x=np.loadtxt("data/"+protein+"_1/coor_init")
q=np.loadtxt("data/"+protein+"_1/q")
r=np.loadtxt("data/"+protein+"_1/r")
e=np.loadtxt("data/"+protein+"_1/e")
eigval=np.loadtxt("data/"+protein+"_1/eigval")
basis = np.loadtxt("data/"+protein+"_1/basis")



eigval = 1.0/np.sqrt(eigval)
coor_init = x








# for i in range(10):
# 	x12=np.random.uniform(-2,2, 12)
# 	print (x12)
# 	product = np.matmul(x12*eigval, basis)

# 	new_coor = np.reshape(product, coor_init.shape) + coor_init

# 	print ("delta new", energy(new_coor, q, r, e) - e0)

initial=[0 for i in range(12)]               # initial starting location [x1,x2...]
e0=energy(initial)


print ("yuanshi", e0)