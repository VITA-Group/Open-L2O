import numpy as np
import os
import pickle

resi_atom_type_charge = {
	"ALA":{},
	"ARG":{},
	"ASN":{},
	"ASP":{},
	"CYS":{},
	"GLU":{},
	"GLN":{},
	"GLY":{},
	"HIS":{},
	"ILE":{},
	"LEU":{},
	"LYS":{},
	"MET":{},
	"PHE":{},
	"PRO":{},
	"SER":{},
	"THR":{},
	"TRP":{},
	"TYR":{},
	"VAL":{}
}

with open("/home/cyppsp/param/toph19.inp", "r") as f:
	param=f.readlines()


for i in range(len(param)):
	if(param[i][0:4]=='RESI' and param[i][5:8] in resi_atom_type_charge):
		print (param[i])

		j=i+1
		resi=param[i][5:8]

		while(param[j]!="\n"):
			if(param[j][0:4]=='ATOM'):

				ar = param[j].strip('\n').split()
				print ar, resi
				resi_atom_type_charge[resi][ar[1]]= [ar[2], float(ar[3])]
			j+=1

		#  extra OXT at the end, regard CTER:  O O -0.55

		resi_atom_type_charge[resi]['OXT'] = ['O', -0.55]



atom_type={}

for resi in resi_atom_type_charge:
	for atom_name in resi_atom_type_charge[resi]:
		atom_type[resi_atom_type_charge[resi][atom_name][0]]=1

##----------------------------------------------------6 Carbon atom types-----------------------
atom_type['C']={
	'rmin': 2.1,
	'emin': -0.12
}

atom_type['CR1E']={
	'rmin': 2.1,
	'emin': -0.12
}

atom_type['CH1E']={
	'rmin': 2.365,
	'emin': -0.0486
}

atom_type['CH2E']={
	'rmin': 2.235,
	'emin': -0.1142
}

atom_type['CH3E']={
	'rmin': 2.165,
	'emin': -0.1811
}

atom_type['CTRP']={
	'rmin': 2.100,
	'emin': -0.12
}

##---------------------------------------------------- 3 Oxygen atom types-----------------------

atom_type['O']={
	'rmin': 1.6,
	'emin': -0.1591
}

atom_type['OC']={
	'rmin':  1.6,
	'emin': -0.6469
}

atom_type['OH1']={
	'rmin': 1.6,
	'emin': -0.1591
}

##----------------------------------------------------6 Nitrogen atom types-----------------------------


atom_type['N']={
	'rmin': 1.6,
	'emin': -0.2384
}
atom_type['NH1']={
	'rmin': 1.6,
	'emin': -0.2384
}
atom_type['NH2']={
	'rmin': 1.6,
	'emin': -0.2384
}
atom_type['NH3']={
	'rmin': 1.6,
	'emin': -0.2384
}
atom_type['NR']={
	'rmin': 1.6,
	'emin': -0.2384
}
atom_type['NC2']={
	'rmin': 1.6,
	'emin': -0.2384
}


##--------------------------------------------------2 Silicon atom types---------

atom_type['S']={
	'rmin': 1.89,
	'emin': -0.0430
}
atom_type['SH1E']={
	'rmin': 1.89,
	'emin': -0.0430
}

##--------------------------------------------2 Hydrogen atom types------------------

atom_type['H']={
	'rmin': 0.8,
	'emin': -0.0498
}

atom_type['HC']={
	'rmin': 0.6,
	'emin': -0.0498
}


#  dump parameters into 'param.picke'

with open('param.pickle','wb') as l_record:
  record = {'atom_type':atom_type,\
            'resi_atom_type':resi_atom_type_charge,\
            }
  pickle.dump(record, l_record)
                                



