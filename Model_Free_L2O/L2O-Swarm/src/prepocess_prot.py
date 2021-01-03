import numpy as np
import os
import pickle

path_protein_benchmark = "/home/cyppsp/project_bayesian/zdock/2c/"

protein_list = np.loadtxt("train_list", dtype='str')

with open('param.pickle','rb') as l_record:
	para = pickle.load(l_record)

atom_type = para['atom_type']
resi_atom_type = para['resi_atom_type']

bench_r=np.loadtxt("/home/cyppsp/project_bayesian/bench_r", dtype='str')
bench_d=np.loadtxt("/home/cyppsp/project_bayesian/bench_d", dtype='str')
bench_m=np.loadtxt("/home/cyppsp/project_bayesian/bench_m", dtype='str')



for i in range(len(protein_list)):
# ligand file
	for j in range(1, 2):
			#os.system("mkdir data/"+protein_list[i]+'_'+str(j))
			

			rpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein_list[i]+"_r_u.pdb.ms"
			lpath="/home/cyppsp/project_bayesian/zdock/2c/"+protein_list[i]+"_l_u_"+str(j)+".pdb"

			with open(rpath, "r") as f:
				receptor_pdb = f.readlines()

			with open(lpath, "r") as f:
				ligand_pdb = f.readlines()

			coor_init=[]
			q=[]
			r=[]
			e=[]

			#---------------------------------------------------------receptor
			for re in range(len(receptor_pdb)):
				
				if(receptor_pdb[re][0:4]=='ATOM' and (receptor_pdb[re][17:20] in resi_atom_type)):

					at=13
					while(receptor_pdb[re][at]!=' '):
						at+=1
						


					atom_name = receptor_pdb[re][13:at]
					resi_name = receptor_pdb[re][17:20]

					

					x=[float(receptor_pdb[re][31:38]), float(receptor_pdb[re][39:46]), float(receptor_pdb[re][47:54])]
					coor_init.append(x)


					q.append(resi_atom_type[resi_name][atom_name][1])
					atom_type_atom = resi_atom_type[resi_name][atom_name][0]

					#print atom_type_atom, atom_type[atom_type_atom]
					r.append(atom_type[atom_type_atom]['rmin'])
					e.append(atom_type[atom_type_atom]['emin'])
				#else:
				#	print ('re', protein_list[i], receptor_pdb[re][17:20])


			#-----------------------------------------------------------ligand
			for re in range(len(ligand_pdb)):
				
				if(ligand_pdb[re][0:4]=='ATOM' and (ligand_pdb[re][17:20] in resi_atom_type)):

					at=13
					while(ligand_pdb[re][at]!=' '):
						at+=1
						


					atom_name = ligand_pdb[re][13:at]
					resi_name = ligand_pdb[re][17:20]

					

					x=[float(ligand_pdb[re][31:38]), float(ligand_pdb[re][39:46]), float(ligand_pdb[re][47:54])]
					coor_init.append(x)


					q.append(resi_atom_type[resi_name][atom_name][1])
					atom_type_atom = resi_atom_type[resi_name][atom_name][0]

					#print atom_type_atom, atom_type[atom_type_atom]
					r.append(atom_type[atom_type_atom]['rmin'])
					e.append(atom_type[atom_type_atom]['emin'])
				#else:
				#	print ('lg', protein_list[i], ligand_pdb[re][17:20])
			
			if(protein_list[i]=='1RLB'):
				coor_init=coor_init[:-1]
				q=q[:-1]
				r=r[:-1]
				e=e[:-1]

			np.savetxt("data/"+protein_list[i]+'_'+str(j)+"/coor_init", coor_init)
			np.savetxt("data/"+protein_list[i]+'_'+str(j)+"/q", q)
			np.savetxt("data/"+protein_list[i]+'_'+str(j)+"/r", r)
			np.savetxt("data/"+protein_list[i]+'_'+str(j)+"/e", e)

			#print len(receptor_pdb)+len(ligand_pdb), len(coor_init), len(r)
	if(len(coor_init)<5000):
		if(protein_list[i] in bench_r):
			print (protein_list[i],',')
		if(protein_list[i] in bench_m):
			print ('medium', protein_list[i], len(coor_init))
		if(protein_list[i] in bench_d):
			print ('diff', protein_list[i], len(coor_init))



