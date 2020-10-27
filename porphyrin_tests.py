import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr, kendalltau
from math import log10, floor, exp, ceil
import pandas as pd
from flask import Flask,render_template, request, url_for

from porphyrin_tools import *
from os import getcwd
##needs fixing! find a way to distribute together

#These are the precalculated NSD databases of porphyrin compounds, which are formed by an adaptation of the method
#presented in porphyrin_tools.py, using data from the porphyrin subset of the CCDC CSD. A .sd file was extracted
#using Mercury, with the CSD_Materials subset - Search, using the similarity fragemnt of the CuTPP core (CUTPOR02)
#and exporting only the hit fragment. PDB was considerably easier - using the list of porphyrin containing ligand 
#names, the .sd file could be exported automatically. This interface is at 
#"https://www.rcsb.org/pages/download_features#Ligands"
#The relevant CCDC refcodes can be investgated at https://www.kingsbury.id.au/nsd_ccdc

nsdpath = getcwd() + '/data/nsd_min_for_web_20200511.pkl'
nsd_df = pd.read_pickle(nsdpath)
nsdpath_pdb = getcwd() + '/data/nsd_pdb_min_20200522.pkl'
nsd_df_pdb = pd.read_pickle(nsdpath_pdb)

#Clusters were identified using the CCDC Conquest interface, and exporting lists of refcodes (6-letter structure identifiers)
#to compare with the refcode identifiers embedded in the above database. The data-merging of the original NSD database (v201911)
#left some Null values, left over from the need to include several extraneous parameters - so the refcode list was a better choice
#for unambiguous indetification. There are some structures with two different porphyrins which might hit the outliers e.g.
# https://doi.org/10.1002/asia.201600241 but these have been ignored.
#Clusters were defined by the Pearson covariance of the two parameters, which gives a symmetric tensor [[C11,C12],[C12,C22]]
#and the values ('pars') below are in the form [x0,y0,c11,c22,scale,c12] with scale defined as 1.   

### Cluster testing related values

b2u_check_vals = {'pars' : {'H4 acid cluster': [3.0773, 0.94645, -0.29823, 0.07049, 1, -0.12770]},
		'nov' : [7,0,2],
		'lim'  : [0.02,0.005]}
		
b2u2_check_vals = {'names' : ['Likely Dodeca (|B<sub>2u</sub>(1)| > 2)'],
		'nov' : [7,0],
		'lim'  : [2.]}
		
b2g_check_vals = {'pars' : {#'5,15, metal, beta = H':[0.08418010651018837, -0.026778411744277097, 0.04149381712364415, -0.01507673359458361, 1, -0.19287286637293546, 0.09057271167701475, 0.02458884078254685],
							'5,15, metal, beta = non-H':[0.21075549108551414, 0.023933920141517188, 0.10221841949829814, 0.013438533128731273, 1, 0.10392525232738446, 0.19901442413636633, 0.12353032539865172],
							'5,15, freebase, beta = H':[0.2796331868763387, 0.05379403638432011, 0.0005922128644580995, 0.010423122676713428, 1, 0.1820619695556503, 0.2749043100550682, 0.051212207839939246],
							'5,15, freebase, beta = non-H':[0.6071911088525302, 0.02897561012344148, 0.1780476825308557, -0.012238095404704023, 1, -0.0688729413590178, 0.6180045532297789, 0.13583958227054066]
				},
		'nov' : [3,0,2],
		'lim'  : [0.02,0.005]}
				
a2u_check_vals = {'pars' : {'P-Ln-coligand':[0.48569094812634045, 0.1175628286028602, -0.04537143613278038, -0.019756017665943107, 1, 0.04890733363253202, 0.48732830664240334, -0.021572803565188082],
							'TPP-Ln-Pc/P':[0.7885884401688048, 0.04124693320814232, 0.13004548415201533, 0.016810668229109454, 1, -0.09496770869596864, 0.7973665914167872, 0.054681576500455464],
							'OEP-Ln-Pc/P':[0.929994688788826, -0.055799729255368954, 0.33165100527161445, -0.016624880107176058, 1, -0.3959453122199556, 0.9859545583692699, -0.05268889165617063],
							'5,15- di(exo-enyl)chlorins':[0.7140828701984177, 0.22835801682901558, -0.17752495570704085, 0.04153543623170062, 1, -0.19278591061420222, 0.6668412668454056, -0.31105012496639756]},
		'nov' : [3,6,8],
		'lim'  : [0.02, 0.005]}

b1g_check_vals = {'names': ['likely freebase'],
		'nov':[3,7],
		'lim'  : [0.04]}

b1u_check_vals = {'names': ['Ruffled compound'],
		'nov':[7,3],
		'lim'  : [0.8]}

#Points check_ellipsoid() at the right values
mode_dict = {'a2u': a2u_check_vals,
'b2u': b2u_check_vals,
'b2u2': b2u2_check_vals,
'b2g': b2g_check_vals,
'b1g': b1g_check_vals,
'b1u': b1u_check_vals
}

### Symmetry-related data tables
#these are expanded versions of the point group table and symmetry operations retained in the lower symmetry elements,
# represented as a binary string so that they may be multiplied together. As is, the product of rows of the pgt represents
# the symmetry element dictated by the unique string in teh lookup table when these symmetry deviations are taken into
# account. The "Mondrian" method of visualising these symmetry-decompositional thresholds is expected to be published shortly,
# and is available at https://www.kingsbury.id.au/mondrian

### Expanded symmetry-related data tables
#E	2C4 (z)	C2	C'2x C'2y	C''2x+y C''2x-y	i	2S4	h	v1 v2	d1 d2
#             E  C4 C2 Cx Cy Cp Cm i  S4 h  v1 v2 d1 d2   
d4h_pgt   = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],#'A1g'
			 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],#'A2g'
			 [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],#'B1g'
			 [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],#'B2g'
			 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],#'A1u'
			 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],#'A2u'
			 [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],#'B1u'
			 [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],#'B2u'
			 [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],#'Egx'
			 [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],#'Egy'
			 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],#'Egx+y'
			 [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],#'Egx-y'
			 [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],#'Eux'
			 [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],#'Euy'
			 [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],#'Eux+y'
			 [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]#'Eux-y'

#                E C4C2CxCyCpCmi S4h v1v2d1d2   
d4h_lookup = { '[1 1 1 1 1 1 1 1 1 1 1 1 1 1]':'D4h',#a1g
			   '[1 1 1 0 0 0 0 1 1 1 0 0 0 0]':'C4h',#a2g
			   '[1 0 1 1 1 0 0 1 0 1 1 1 0 0]':'D2h',#b1g
			   '[1 0 1 0 0 1 1 1 0 1 0 0 1 1]':'D2h',#b2g
			   '[1 1 1 1 1 1 1 0 0 0 0 0 0 0]':'D4 ',#a1u
			   '[1 1 1 0 0 0 0 0 0 0 1 1 1 1]':'C4v',#a2u
			   '[1 0 1 0 0 1 1 0 1 0 1 1 0 0]':'D2d',#b1u
			   '[1 0 1 1 1 0 0 0 1 0 0 0 1 1]':'D2d',#b2u
			   '[1 0 1 0 0 0 0 1 0 1 0 0 0 0]':'C2h',
			   '[1 0 0 1 0 0 0 1 0 0 1 0 0 0]':'C2h',#egx
			   '[1 0 0 0 1 0 0 1 0 0 0 1 0 0]':'C2h',#egy
			   '[1 0 0 0 0 1 0 1 0 0 0 0 1 0]':'C2h',#egx+y
			   '[1 0 0 0 0 0 1 1 0 0 0 0 0 1]':'C2h',#egx-y
			   '[1 0 1 0 0 0 0 0 0 0 1 1 0 0]':'C2v',
			   '[1 0 1 0 0 0 0 0 0 0 0 0 1 1]':'C2v',
			   '[1 0 0 1 0 0 0 0 0 1 0 1 0 0]':'C2v',#eux
			   '[1 0 0 0 1 0 0 0 0 1 1 0 0 0]':'C2v',#euy
			   '[1 0 0 0 0 1 0 0 0 1 0 0 0 1]':'C2v',#eux+y
			   '[1 0 0 0 0 0 1 0 0 1 0 0 1 0]':'C2v',#eux-y
			   '[1 0 1 0 0 1 1 0 0 0 0 0 0 0]':'D2 ',
			   '[1 0 1 1 1 0 0 0 0 0 0 0 0 0]':'D2 ',
			   '[1 0 1 0 0 0 0 0 1 0 0 0 0 0]':'S4 ',
			   '[1 1 1 0 0 0 0 0 0 0 0 0 0 0]':'C4 ',
			   '[1 0 1 0 0 0 0 0 0 0 0 0 0 0]':'C2 ',
			   '[1 0 0 1 0 0 0 0 0 0 0 0 0 0]':'C2 ',
			   '[1 0 0 0 1 0 0 0 0 0 0 0 0 0]':'C2 ',
			   '[1 0 0 0 0 1 0 0 0 0 0 0 0 0]':'C2 ',
			   '[1 0 0 0 0 0 1 0 0 0 0 0 0 0]':'C2 ',
			   '[1 0 0 0 0 0 0 1 0 0 0 0 0 0]':'Ci ',
			   '[1 0 0 0 0 0 0 0 0 1 0 0 0 0]':'Cs ',
			   '[1 0 0 0 0 0 0 0 0 0 1 0 0 0]':'Cs ',
			   '[1 0 0 0 0 0 0 0 0 0 0 1 0 0]':'Cs ',
			   '[1 0 0 0 0 0 0 0 0 0 0 0 1 0]':'Cs ',
			   '[1 0 0 0 0 0 0 0 0 0 0 0 0 1]':'Cs ',
			   '[1 0 0 0 0 0 0 0 0 0 0 0 0 0]':'C1 ',}

# This symmetry finder operates by the principle outlined above, multiplication of symmetry-access vectors (rows
# of the point-group table) for deviations above a certain threshold (e.g. 0.2 for in-plane and 0.5 for out-of-plane)
# These thresholds are arbitrary, but seem to correlate with symmetry obtained via NMR.
# more info is available in https://dx.doi.org/10.1021/acs.inorgchem.9b01963

def find_symmetry(nsd, thr = (0.2,0.5)): # thr : thresholds
    ipc, opc = nsd[2][:6],nsd[6][:6]
    symms = [1 if float(x)>thr[0] else 0 for x in ipc]+[1 if float(x)>thr[1] else 0 for x in opc]
    ab_syms = [symms[x] for x in [4,5,1,0,11,8,7,6]]
    
    e_syms_o  = [1 if float(x)>thr[1] else 0 for x in [opc[3],opc[4],0,0]]#opc[3]+opc[4],np.abs(opc[3]-opc[4])]]
    e_syms_i  = [1 if float(x)>thr[0] else 0 for x in [ipc[2],ipc[3],0,0]]#ipc[2]+ipc[3],np.abs(ipc[2]-ipc[3])]]
    if ((np.abs(opc[3]-opc[4])<opc[3]) and (np.abs(opc[3]-opc[4])<opc[4])):
        e_syms_o  = [1 if float(x)>thr[1] else 0 for x in [0,0,opc[3]+opc[4],np.abs(opc[3]-opc[4])]]
    if ((np.abs(ipc[2]-ipc[3])<ipc[2]) and (np.abs(ipc[2]-ipc[3])<ipc[3])):
        e_syms_i  = [1 if float(x)>thr[0] else 0 for x in [0,0,ipc[2]+ipc[3],np.abs(ipc[2]-ipc[3])]]
    
    all_syms = ab_syms + e_syms_o + e_syms_i
    all_syms[0] = 1
    pgt_rows = [d4h_pgt[i] for i,x in enumerate(all_syms) if x==1] 
    lookup_row = str(np.prod(pgt_rows,axis = 0))
    symm_guess = str(d4h_lookup.get(str(lookup_row)))
    
    return symm_guess
			   
#generates a simple html table		
tablegen = lambda matrix: '<table>'+'\n'.join(['<tr>'+' '.join(['<td>'+str(x)+'</td>' for x in y])+'</tr>' for y in np.round(matrix,4).tolist()])+'</table>'

#lambdas for the closest match below - these are a little slow, but it's a 66*7000 array, I guess I can't optimise more		
minssq = lambda a1,a2: np.sum([np.square(float(x)-float(y)) for x,y in zip(a1,a2)])
minabs = lambda a1,a2: np.sum([np.abs(float(x)-float(y)) for x,y in zip(a1,a2)])

#these were from Jentzen's paper, basically the elastic strain on a distorted prophyrin from the simple harmonic oscillator
#model. Not supposed to be used for calculation, but might be of interest to strapping types, or could correlate with something
mode_energy_list = [9.1,16.5,41.4,67.3,67.3,238.4,0,0,100.1,147.8,280.5,280.5,293.8,652.9,0,0]
mode_energy_calc = lambda nsd: str(np.round(np.sum(np.multiply(np.square(np.array(nsd[4]+nsd[0]).astype(float)), mode_energy_list)),1))+ ' kJ/mol'
check_NSDt_residuals = lambda nsd: sum([float(nsd[2][7]),float(nsd[6][7])]) < 0.001

#second and third mode frequencies. Currently ignored
Mode_freq_2= [[256, 516, 359, 238, 238, 680],[721, 727, 781, 468, 468], #oop2, oop3
				[428, 757, 413, 413, 701, 769],[893, 989, 739, 739, 994, 1113]]#cm-1 ip2 ip3

#Checks if all ungerade modes are zero i.e. a centrosymmetric porphyrin 
def check_cent(nsd, tval = 0.00):
	uimodes,uomodes = [i for i in range(12,34)],[0,1,2,3,4,5,6,7,8,19,20]
	ipt,opt = nsd[3],nsd[7]
	umod = [float(ipt[i]) for i in uimodes]+[float(opt[i]) for i in uomodes]
	return str(sum([1 for x in umod if np.abs(x) <= tval]) == 33)

#compares the nsd parameters (i.e. molecular conformation) against a precomputed
#database of NSD parameters. This provides links too! Neat!
#essentially, the min_sum and min_ssq (sum of squares) of the differential NSD
#matrices often give a good fit to the dominant mode of distortion, either in or
#out of plane, though A1g has out-of-plane artifacts that are often overfitted
#not planning to go abck and correct for it though
def find_comparison_in_database(nsd, corr_func = minssq, nsd_df = nsd_df):
	tf = lambda x,n: [x+'f'+str(i+1) for i in range(n)]

	op4f_names = tf('b2ut',3)+tf('b1ut',3)+tf('a2ut',3)+tf('egxt',5)+tf('egyt',5)+tf('a1ut',2)+['doptf','doptf_err']
	ip4f_names = tf('b2gt',6)+tf('b1gt',6)+tf('euxt',11)+tf('euyt',11)+tf('a1gt',6)+tf('a2gt',5)+['diptf','diptf_err']

	ip4_firsts = [0]*6 + [6]*6 + [12]*11 + [23]*11 + [45]*6 + [40]*5 + [45,46]
	op4_firsts = [0]*3 + [3]*3 + [6]*3 + [9]*5 + [14]*5 + [19]*2 + [21,22]
	#references to the individual symmetry operations
	
	ip4 = np.array(nsd[3]).astype(float)
	op4 = np.array(nsd[7]).astype(float)
	ip4f = np.array([x * np.sign(y) for x,y in zip(ip4, [ip4[z] for z in ip4_firsts])])
	op4f = np.array([x * np.sign(y) for x,y in zip(op4, [op4[z] for z in op4_firsts])])
	
	#aligns symmetric distortion modes (ignores chirality)
	if corr_func in [minssq,minabs]:   
		correlidx_op = nsd_df['opc'].apply(lambda row: corr_func(row[:-2],op4f[:-2])).idxmin()
		correlidx_ip = nsd_df['ipc'].apply(lambda row: corr_func(row[:-2],ip4f[:-2])).idxmin()
	if corr_func in [pearsonr,spearmanr,kendalltau]:
		correlidx_op = nsd_df['opc'].apply(lambda row: corr_func(row[:-2],op4f[:-2])[0]).idxmax()
		correlidx_ip = nsd_df['ipc'].apply(lambda row: corr_func(row[:-2],ip4f[:-2])[0]).idxmax()
		
	oop_best_corr = nsd_df.iloc[correlidx_op]['NAME']
	ip_best_corr = nsd_df.iloc[correlidx_ip]['NAME']
	
	#for flipping Eg/u x/y when necessary (threshold is 0.1 difference)
	ipc = np.array(nsd[2]).astype(float)[2:4] #these are the Eu values
	opc = np.array(nsd[6]).astype(float)[3:5] #these are the Eg values
	if (((np.abs(ipc[0]-ipc[1])>0.1)>0.1)|(np.abs(opc[0]-opc[1])>0.1)): 
		#compares the two Eg and Eu values to find a significant difference (if (flipped) will be different
		ip4f = np.hstack((ip4f[0:12],ip4f[23:34],ip4f[12:23],ip4f[34:]))
		op4f = np.hstack((op4f[0:9],ip4f[14:19],ip4f[9:14],ip4f[19:]))
		if corr_func in [minssq,minabs]:   
			correlidx_op = nsd_df['opc'].apply(lambda row: corr_func(row[:-2],op4f[:-2])).idxmin()
			correlidx_ip = nsd_df['ipc'].apply(lambda row: corr_func(row[:-2],ip4f[:-2])).idxmin()
		if corr_func in [pearsonr,spearmanr,kendalltau]:
			correlidx_op = nsd_df['opc'].apply(lambda row: corr_func(row[:-2],op4f[:-2])[0]).idxmax()
			correlidx_ip = nsd_df['ipc'].apply(lambda row: corr_func(row[:-2],ip4f[:-2])[0]).idxmax()
		
		oop_best_corr = oop_best_corr + ' / ' + nsd_df.iloc[correlidx_op]['NAME'] + ' (flipped)'
		ip_best_corr  = ip_best_corr + ' / ' + nsd_df.iloc[correlidx_ip]['NAME'] + ' (flipped)'
			
	return [oop_best_corr, ip_best_corr]

#could figure to report the contributing modes of symmetry, but this is easier to see from the Mondrian diagram
def find_symmetry_contributing_modes(nsd, thr = (0.2,0.5)): # thr : thresholds
	ipc,opc = nsd[2][:6],nsd[6][:6]
	op_names = ['b2u','b1u','a2u','egx','egy','a1u']
	ip_names = ['b2g','b1g','eux','euy','a1g','a2g']
	symm_names = [y for x,y in zip(ipc,ip_names) if float(x)>thr[0]]+[y for x,y in zip(opc,op_names) if float(x)>thr[1]]
	return symm_names
	
#Checks if the porphyrin is chiral at the designated threshold. Useful for saddled 5,15s and the like
find_is_chiral = lambda nsd, thr = (0.2,0.5): find_symmetry(nsd, thr) in ['D4 ','D2 ','C4 ','C2 ','C1 ']

#compares the precomputed cluster analysis for porphyrinoid distortion to a two-parameter model, based on a 
#pearson-covariance derived lorentzian ellipsoid. The threshold values are usually 2 sigma.
def check_ellipsoid(nsd, mode):
	lorentzian = lambda xv: 1/(1+xv**2)
	l_func_2d =lambda p,xv,yv: p[4]*(lorentzian(np.sqrt( (((np.cos(p[5])*xv + np.sin(p[5])*yv)-p[0])/p[1])**2 + (((np.cos(p[5])*yv - np.sin(p[5])*xv)-p[2])/p[3])**2 )))
	output = []

	vals = mode_dict.get(mode)
	pars,nov,lim = [vals.get(x) for x in 'pars,nov,lim'.split(',')]
	opt = [float(x) for x in nsd[nov[0]][nov[1]:nov[2]]]
	xval,yval = np.multiply(opt, np.sign(opt[0]))
	for cname,cpars in pars.items():
		cval = l_func_2d(cpars, xval,yval)**2
		for index,limit in enumerate(lim):
			if cval > limit: #arbitrary values
				output.append(cname + ' limit ' + str(index+1))
				break
	return output	

def check_linear(nsd, mode): #checks whether a value is above a threshold (see b1g_check_vals)
	vals = mode_dict.get(mode)
	names,nov,lim = [vals.get(x) for x in 'names,nov,lim'.split(',')]
	output = []
	opt = np.abs(float(nsd[nov[0]][nov[1]]))
	for cname,clim in zip(names, lim):
		if opt >= lim:
			output.append(cname)
	return output
	
#writes html output
check_a2u = lambda nsd: '\n'.join(check_ellipsoid(nsd, 'a2u'))
check_b2u = lambda nsd: '\n'.join(check_ellipsoid(nsd, 'b2u') + check_linear(nsd,'b2u2'))
check_b2g = lambda nsd: '\n'.join(check_ellipsoid(nsd, 'b2g'))
check_b1g = lambda nsd: '\n'.join(check_linear(nsd, 'b1g'))
check_b1u = lambda nsd: '\n'.join(check_linear(nsd, 'b1u'))

def check_a1g(nsd):
	a1g1 = float(nsd[3][34])
	if a1g1 > 0.1:
		return 'Expanded porphyrin core (Zn-like)'
	elif a1g1 > -0.05:
		return 'Median porphyrin core (Cu-like)'
	else:
		return 'Contracted porphyrin core (Ni-like)'
		
def check_a2g(nsd):
	a2g2 = float(nsd[3][41])
	b1u1 = float(nsd[7][4])
	b2u1 = float(nsd[7][1])
	if (a2g2 - (b1u1 * b2u1 *  0.0851))> 0.03:
		return "A2g inconsistent with out-of-plane structure"
	else:
		return ''
def generate_alert_section(nsd):
	return render_template('/alert_section.html',
									 a1 = check_NSDt_residuals(nsd),
									 a2 = find_symmetry(nsd,(0.2,0.5)),
									 a3 = find_symmetry(nsd,(0.03,0.1)),
									 a4 = check_cent(nsd,0.01),
									 a5 = check_cent(nsd,0.1),
									 b2u = check_b2u(nsd),
									 b2g = check_b2g(nsd),
									 b1g = check_b1g(nsd),
									 a2u = check_a2u(nsd),
									 b1u = check_b1u(nsd),
									 a1g = check_a1g(nsd),
									 a2g = check_a2g(nsd),
									 comp1 = find_comparison_in_database(nsd),
									 comp2 = find_comparison_in_database(nsd,minabs),
									 comp3 = find_comparison_in_database(nsd, nsd_df = nsd_df_pdb),
									 comp4 = find_comparison_in_database(nsd,minabs, nsd_df = nsd_df_pdb),
									 
									 sum_energy = mode_energy_calc(nsd))

#extra tables for external plotting
def verbose_output(coords):
	#print(verbose_output(ref_str_mat)) to print the comparison reference structure table
	m3 = np.round(coords,3)
	m4 = np.round(np.hstack((cc_transform_ronly(np.matrix(coords)),cc_transform_thetaonly_deg(np.matrix(coords)))),3)
	return np.hstack((m3,m4)).T

def write_logfile(pdbname, nsd, logname):
	log = open(logname, 'a')
	log.write(str(pdbname) + '\n')
	log.write(str(nsd) + '\n')
	#log.write(str(generate_alert_section(nsd)) + '\n')
	log.close()	
	log2 = open(logname[:-4]+'.csv', 'a')
	vlist = [pdbname] + nsd[0] +nsd[4]
	log2.write(','.join([str(x) for x in vlist]) + '\n')
	log2.close()	

def esd_gen(num, err):
	# puts two floats (e.g. 2.1022, 0.02) into standard notation (e.g. "2.10(2)")
	if (err < 0.0001) and (num < 2):
		return str(round(num,4))
	if (err < 0.0001) and (num >= 2):
		return str(round(num,2))
	if err > 1.95:
		num = int(num)
	if err*10**(-int(floor(log10(abs(err))))) < 1.95:
		return str(round(num, 1-int(floor(log10(abs(err)))))) + '(' + str(ceil(err*10**(1-int(floor(log10(abs(err))))))) + ')'
	else:	
		return str(round(num, -int(floor(log10(abs(err)))))) + '(' + str(int(round(err*10**(-int(floor(log10(abs(err)))))))) + ')'

#charts predefined bond distances, angles, plane deviations, rotation angles, etc. Useful for padding a report.
#errors reports the mean values and the standard deviation of these values - can be much more or less than
#the error on the individual bond distances. Keith mode just reports all the values without averaging.
def bdba_extract_array(coords, errors = True, keith = False):
	nca	   = np.array([x-1 for x in [21,1,21,4,22,6,22,9,23,11,23,14,24,16,24,19]]).reshape(8,2)
	cacb   = np.array([x-1 for x in [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]]).reshape(8,2)
	cacm   = np.array([x-1 for x in [4,5,5,6,9,10,10,11,14,15,15,16,19,20,20,1]]).reshape(8,2)
	cbcb   = np.array([x-1 for x in [2,3,7,8,12,13,17,18]]).reshape(4,2)
	
	canca  = np.array([x-1 for x in [1,21,4,6,22,9,11,23,14,16,24,19]]).reshape(4,3)
	ncacb  = np.array([x-1 for x in [21,1,2,21,4,3,22,6,7,22,9,8,23,11,12,23,14,13,24,16,17,24,19,18]]).reshape(8,3)
	ncacm  = np.array([x-1 for x in [21,1,20,21,4,5,22,6,5,22,9,10,23,11,10,23,14,15,24,16,15,24,19,20]]).reshape(8,3)
	cacbcb = np.array([x-1 for x in [1,2,3,2,3,4,6,7,8,7,8,9,11,12,13,12,13,14,16,17,18,17,18,19]]).reshape(8,3)
	cbcacm = np.array([x-1 for x in [20,1,2,3,4,5,5,6,7,8,9,10,10,11,12,13,14,15,15,16,17,18,19,20]]).reshape(8,3)
	cacmca = np.array([x-1 for x in [19,20,1,4,5,6,9,10,11,14,15,16]]).reshape(4,3)
	
	nnadj  = np.array([x-1 for x in [21,22,22,23,23,24,24,21]]).reshape(4,2)
	nnopp  = np.array([x-1 for x in [21,23,22,24]]).reshape(2,2)

	pyrrs  = np.array([x-1 for x in [1,2,3,4,21,6,7,8,9,22,11,12,13,14,23,16,17,18,19,24]]).reshape(4,5)
	
	delns  = np.array([21,22,23,24])-1
	delcas = np.array([1,4,6,9,11,14,16,19])-1
	delcbs = np.array([2,3,7,8,12,13,17,18])-1
	delcms = np.array([5,10,15,20])-1
	del24  = np.array(range(24))
	
	bdlist  = [nca,cacb,cacm,cbcb,nnadj,nnopp]
	balist  = [canca,ncacb,ncacm,cacbcb,cbcacm,cacmca]
	dellist = [delns,delcas,delcbs,delcms,del24]
	bdnames = 'nca,cacb,cacm,cbcb,nnadj,nnopp'.split(',')
	banames = 'canca,ncacb,ncacm,cacbcb,cbcacm,cacmca'.split(',')
	delnames= 'delns,delcas,delcbs,delcms,del24'.split(',')
	
	dist_from_pts  = lambda v1,v2: np.sqrt(np.sum(np.square(v2-v1)))
	ang_from_pts   = lambda v1,v2,v3: np.degrees(np.arccos(np.float(np.dot((v1-v2), (v3-v2).T))/np.product([dist_from_pts(x,y) for x,y in [(v2,v1),(v2,v3)]])))
	ang_from_vec   = lambda v1,v2: np.degrees(np.arccos(np.float(np.dot((v1), (v2).T))/np.product([np.sqrt(np.sum(np.square(x))) for x in [v1,v2]])))
	normalise_atoms= lambda xyzmat: np.matrix([np.array(x-np.mean(x)).flatten() for x in xyzmat.T]).T
	
	round_to_2 = lambda x: round(x, 1-int(floor(log10(abs(x)))))
	round_to_4 = lambda x: str(round(x, 3-int(floor(log10(abs(x))))))

	idents = 'nca,cacb,cacm,cbcb,cacbcb,ncacb,ncacm,canca,cbcacm,cacmca,del24,delns,delcas,delcbs,delcms,pyr_ang,nnadj,nnopp'.split(',')
	
	outdict = {}
		
	for name,idxs in zip(bdnames,bdlist):
		pars = [dist_from_pts(coords[a],coords[b]) for a,b in idxs]
		[outdict.update({x:y}) for x,y in [[name, pars],[name +'_mean', np.mean(pars)],[name +'_std', np.std(pars)],[name +'_format', esd_gen(np.mean(pars),np.std(pars))]]]
	for name,idxs in zip(banames,balist):
		pars = [ang_from_pts(coords[a],coords[b],coords[c]) for a,b,c in idxs]
		[outdict.update({x:y}) for x,y in [[name, pars],[name +'_mean', np.mean(pars)],[name +'_std', np.std(pars)],[name +'_format', esd_gen(np.mean(pars),np.std(pars))]]]
	
	mpln_quat = minimize(mpln_fit, [1,0,0,0], coords).x
	mpln_vec  = rot_atoms([[0,0,1]],mpln_quat)
	coords_mpln = rot_atoms(coords,mpln_quat)
	
	pyr_angles = []
	for idxs in pyrrs:
		py_coords = np.array([coords[x] for x in idxs])
		norm_py_c = normalise_atoms(py_coords)
		pypl_quat = minimize(mpln_fit, [1,0,0,0], norm_py_c).x
		py_vec	= rot_atoms([[0,0,1]],pypl_quat)
		pang	  = ang_from_vec(mpln_vec, py_vec)
		
		pyr_angles.append(min([pang, 180 - pang]))
	
	[outdict.update({x:y}) for x,y in [['pyr_ang', pyr_angles],
									   ['pyr_ang_mean', np.mean(pyr_angles)],
									   ['pyr_ang_std', np.std(pyr_angles)],
									   ['pyr_ang_format', esd_gen(np.mean(pyr_angles),np.std(pyr_angles))]]]
	
	for name,idxs in zip(delnames,dellist):
		pars = [np.abs(coords_mpln[a,2]) for a in idxs]
		[outdict.update({x:y}) for x,y in [[name, pars],[name +'_mean', np.mean(pars)],[name +'_std', np.std(pars)],[name +'_format', esd_gen(np.mean(pars),np.std(pars))]]]
		if name == 'del24':
			[outdict.update({x:y}) for x,y in [[name, pars],[name +'_mean', np.mean(pars)],[name +'_std', np.std(pars)],[name +'_format', esd_gen(np.mean(pars),0.00001)]]]
	
	if keith:
		return ['\n'.join([str(round_to_4(x)) for x in outdict.get(name)]) for name in idents]
	elif errors:
		return [outdict.get(name+'_format') for name in idents]
	else:
		return [round_to_4(outdict.get(name+'_mean')) for name in idents]
	

