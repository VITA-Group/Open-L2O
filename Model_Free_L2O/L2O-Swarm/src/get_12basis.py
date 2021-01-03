import numpy as np
import os

path_protein_benchmark = "/home/cyppsp/project_bayesian/zdock/2c/"

protein_list = np.loadtxt("temp", dtype='str')


def c_normal_mode_analysis(protein, rpath, lpath, output):
	os.system("cp "+rpath+" /home/cyppsp/cNMA/Example/Example1/Input/"+protein+'_r_u.pdb')
	os.system("cp "+lpath+" /home/cyppsp/cNMA/Example/Example1/Input/"+protein+'_l_u.pdb')
	os.system("/home/cyppsp/cNMA/Example/Example1/run_example1.sh "+protein)

	os.system("cp /home/cyppsp/cNMA/Example/Example1/basis data/"+output)
	os.system("cp /home/cyppsp/cNMA/Example/Example1/eigval data/"+output)


for i in range(len(protein_list)):
# ligand file
	for j in range(1, 11):
			os.system("mkdir data/"+protein_list[i]+'_'+str(j))

			rpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein_list[i]+"_r_u.pdb.ms"
			lpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein_list[i]+"_l_u_"+str(j)+".pdb"

			c_normal_mode_analysis(protein_list[i], rpath, lpath, protein_list[i]+'_'+str(j))




