
from numpy import sin, cos, pi, square, array, hstack, sum, dot, mean, min, where, round, add, subtract, ndarray
from numpy.linalg import norm

from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment, minimize, basinhopping

import pandas as pd

from symmetry_dicts import *

s2c = lambda th1, ph1: array((sin(ph1) * cos(th1), sin(ph1) * sin(th1), cos(ph1)))
#spherical to cartesian
c2q = lambda om, vec: hstack([array([cos(om/2)]), (vec * sin(om/2))])
#cartesian to quaternion
#s2q = lambda th, ph, om: c2q(om, s2c(th, ph))
s2q = lambda th, ph, om: array([sin(om/2)*sin(ph)*cos(th), sin(om/2)*sin(ph)*sin(th), sin(om/2)*cos(ph), cos(om/2)])
#spherical to quaternion

atom_dist   = lambda a1,a2: norm(array(a1) - array(a2))

def mat_dist(m1,m2):
    costmat   = array([[norm(v1 - v2) for v1 in m1] for v2 in m2])
    row_ind, col_ind = linear_sum_assignment(costmat)
    return sum(square(m1[col_ind]-m2[row_ind]))

def costmat_gen(m1,m2):
    return square(array([[atom_dist(val_1, val_2) for val_1 in m1] for val_2 in m2]))

fit_atoms_rt = lambda p,ats: R.from_quat(p[3:7]).apply(ats-p[0:3])

def mir_sq(atom_matrix, theta, phi, rotation = None, properness = None):
    refvec = s2c(theta, phi)
    refmat = atom_matrix-2*array([dot(x, refvec)*refvec for x in atom_matrix])
    costmat   = costmat_gen(atom_matrix,refmat)
    row_ind, col_ind = linear_sum_assignment(costmat)
    dev = costmat[row_ind, col_ind].sum()
    return dev

def rot_sq(atom_matrix, theta, phi, rotation, properness = 1):
    #omega     = 
    rot_obj   = R.from_quat(s2q(theta, phi, (2*pi)/rotation))
    #rotmat    = atom_matrix.copy()
    rotmat    = R.apply(rot_obj,atom_matrix)
    if not properness:
        refvec    = s2c(theta, phi)
        rotmat    = rotmat-2*array([dot(x, refvec)*refvec for x in rotmat])
    costmat   = costmat_gen(atom_matrix,rotmat)
    row_ind, col_ind = linear_sum_assignment(costmat)
    dev = costmat[row_ind, col_ind].sum()
    return dev

def inv_sq(atom_matrix):
    costmat   = costmat_gen(atom_matrix,atom_matrix*-1)
    row_ind, col_ind = linear_sum_assignment(costmat)
    dev = costmat[row_ind, col_ind].sum()
    return(dev)

def import_pdb_linked(filenm, query_atoms = False):
    flist = open(filenm,'r').readlines()
    atoms = array([(l[31:39],l[39:47],l[47:55]) for l in flist if l.startswith('ATOM') 
                      or l.startswith('HETATM')]).astype(float)
    atom_serials =  array([l.split()[1] for l in flist if (l.startswith('ATOM') 
                      or l.startswith('HETATM'))])
    atoms = atoms - mean(atoms, axis = 0)
    
    if query_atoms:
        hit_atom_serials = array([l.split()[1] for l in flist if (l.startswith('ATOM') 
                      or l.startswith('HETATM')) and l.split()[-1] in query_atoms])
        hit_atoms = array([atoms[where(atom_serials == x)].flatten() for x in hit_atom_serials])
        return(hit_atoms,False)
    else:
        clist = [[int(x)-1 for x in l.split()[1:]] for l in flist if l.startswith('CONECT')]
        clist2 = []
        [[clist2.append([l[0],x]) for x in l[1:]] for l in clist]
        atom_tups = array([(atoms[i0],atoms[i1]) for i0,i1 in clist2])
        return(atoms,atom_tups)
        
def import_pdb_ats(filenm, query_atoms = False):
    flist = open(filenm,'r').readlines()
    atoms = array([(l[31:39],l[39:47],l[47:55]) for l in flist if l.startswith('ATOM') 
                      or l.startswith('HETATM')]).astype(float)
    atom_serials =  array([l.split()[1] for l in flist if (l.startswith('ATOM') 
                      or l.startswith('HETATM'))])
    atoms = atoms - mean(atoms, axis = 0)
    
    if query_atoms:
        hit_atom_serials = array([l.split()[1] for l in flist if (l.startswith('ATOM') 
                      or l.startswith('HETATM')) and l.split()[-1] in query_atoms])
        hit_atoms = array([atoms[where(atom_serials == x)].flatten() for x in hit_atom_serials])
        return(hit_atoms)
    else:
        hit_atom_serials = array([l.split()[1] for l in flist if (l.startswith('ATOM') 
                      or l.startswith('HETATM'))])
        hit_atoms = array([atoms[where(atom_serials == x)].flatten() for x in hit_atom_serials])
        return(hit_atoms)

def check_val_from_dict(ats, operation):
    vals = operations_dict.get(operation)
    if vals[0][0].lower() == 'rotation':
        return sum([rot_sq(ats, x[1], x[2], x[3]) for x in vals])
    if vals[0][0].lower() == 'improperrotation':
        return sum([rot_sq(ats, x[1], x[2], x[3],False) for x in vals])
    if vals[0][0].lower() == 'inversion':
        return sum([inv_sq(ats)])
    if vals[0][0].lower() == 'mirror':
        return sum([mir_sq(ats, x[1], x[2]) for x in vals])

def remove_symm(atom_matrix, typo, theta, phi, rotation):
    
    omega     = (2*pi)/rotation
    rot_obj   = R.from_quat(s2q(theta, phi, omega))
    rotmat    = atom_matrix.copy()
    
    if typo.lower() in ['improperrotation','mirror']:
        refvec    = s2c(theta, phi)
        rotmat    = rotmat-2*array([dot(x, refvec)*refvec for x in rotmat])
    if typo.lower() in ['rotation','improperrotation']:
        rotmat    = R.apply(rot_obj,rotmat)
    if typo.lower() == 'inversion':
        rotmat = array([[-x for x in y] for y in atom_matrix])
        
    costmat   = square(array([[norm(val_1 - val_2) for val_1 in atom_matrix] for val_2 in rotmat]))
    row_ind, col_ind = linear_sum_assignment(costmat)
    
    a1,a2 = atom_matrix[col_ind], rotmat[row_ind]
    return (a1-a2), add(a1,a2)/2
    #difference vectors, new atoms

def yield_model(ats, point_group_name,bhopping= False):
    ats = ats - mean(ats,axis = 0)
    ops = point_group_dict.get(point_group_name)
    quat_i,quat_2,quat_3 = [1,0,0,1],[0,1,0,1],[0,0,1,1]
    fitfunc = lambda p: sum([check_val_from_dict(R.from_quat(p).apply(ats), x) for x in ops])
    if bhopping: fits = [basinhopping(fitfunc,q,niter=10) for q in [quat_i,quat_2,quat_3]]
    else: fits = [minimize(fitfunc,q) for q in [quat_i,quat_2,quat_3]]
    fitv = min([x.fun for x in fits])
    fit  = [x for x in fits if x.fun == fitv][0]
    total_symm_output = [ats*0, R.from_quat(fit.x).apply(ats),'ident']
    for x in ops:
        for typo, t,p,r in operations_dict.get(x):
            total_symm_output = list(remove_symm(total_symm_output[1],typo, t,p,r))+['Total']
    return total_symm_output[1]

def trim_to_model(m1,m2):
    costmat   = array([[norm(v1 - v2) for v1 in m1] for v2 in m2])
    row_ind, col_ind = linear_sum_assignment(costmat)
    return m1[col_ind]

def yield_vectors_by_matrix(ats,point_group_name,model = False,bhopping= False):
    #point groups which are currentyl supported: 'C2v'
    ops = point_group_dict.get(point_group_name.capitalize())
    pgt = pgt_dict.get(point_group_name.capitalize())
    ops_order = ordered_ops_dict.get(point_group_name.capitalize())
    
    init_params = [[0,0,0]+x for x in [[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,0,0,-1],[0,1,0,-1],[0,0,1,-1]]]
    #quarter rotation about the three principal axes. Gives appropriate start points for 
    #avoiding all falling into the same local minimum, a bit hackneyed
    
    #fits a set query atom matrix. necessary for A / A1 / A1g fitting. Else generates a model on the fly
    #query atoms should ideally be unsubsituted or computationally optimised, are pre-aligned 
    #and should be used throughout any analysis and published alongside.
    if isinstance(model,ndarray): 
        fitfunc = lambda p: mat_dist(fit_atoms_rt(p,ats), model)
    else: 
        fitfunc = lambda p: sum([check_val_from_dict(fit_atoms_rt(p,ats), x) for x in ops_order])
    
    #gives a basinhopping mode, if the function isn't fitting the right atoms
    #is significantly slower due to algorithm constraints, shouldn't be necessary in the final version
    if bhopping: fits = [basinhopping(fitfunc,q,niter=3) for q in init_params]
    else: fits = [minimize(fitfunc,q) for q in init_params]
    
    #the fit structure is to find the minimum axis for summation of the total distortion attributed to all symmetric modes. 
    #doesn't take into account translation, which is on the list of things to do
    #quaternions don't have the same edge arguments as rot coords, thus don't fall into the same local minima so easily.
    #there's an Acta Cryst A on the subject.
    
    fitv = min([x.fun for x in fits])
    fit  = [x for x in fits if x.fun == fitv][0]
    
    atom_matrices,output = [],[]
    for name, row in pgt.items():
        #rotates atoms into the appropriate reference frame. will likely replace with rotation/translation for accuracy's sake
        atoms = fit_atoms_rt(fit.x,ats)
        if isinstance(model,ndarray): 
            atoms = trim_to_model(atoms,model)
        #E subgroups have to be treated seperately, as they contain subgroups which have been grouped in the abvove tables, and have to be ungrouped
        #This basically gets a list of poerations to be applied
        #-old version, delete if needed
        #if name.startswith('E'):
        #    ops_in_row = e_group_subdict.get(point_group_name)
        #else:
        #    ops_in_row = [x for x,y in zip(ops_order,row) if (y==1)]
        ops_in_row = [x for x,y in zip(ops_order,row) if (y==1)]
        #This applies each of the operations and generates a new list of atoms without that symmetry
        #doing this for all ops in a row gives the residual symmetry element that is the atom matrix minus the irreducible representations
        for op in ops_in_row:
            for typo,t,p,r in operations_dict.get(op):
                atoms = remove_symm(atoms,typo,t,p,r)[1]
        atom_matrices.append((name, atoms))
    
    atom_mat_dict = dict(atom_matrices)
    #this adds in the totally symmetric version of the target compound as the 'model' if no target is applied.
    #thus, the A / A1 / A1g totally symmetric mode will be 0, but the others will be exactly the same. 
    if type(model) in [bool]: 
        model = atom_matrices[0][1]
        #requires the A1 to be first in the dictionary. Not sure how {}.items handles stuff, but could scramble
    
    for name,atoms in atom_matrices:
         #attempts to find the appropriate named subgroup 'model' for some E groups
        if point_group_name in ['D4h','D2d']:
            if e_group_parent.get(point_group_name).get(name) in atom_mat_dict.keys():
                model = atom_mat_dict.get(e_group_parent.get(point_group_name).get(name))
                print(name,e_group_parent.get(point_group_name).get(name))
        
        #atom asignment to the model.
        costmat   = array([[norm(val_1 - val_2)**2 for val_1 in atoms] for val_2 in model])
        row_ind, col_ind = linear_sum_assignment(costmat)
        a1,a2 = atoms[col_ind], model[row_ind]
        #this is where the magic happens. The rounding is just a formality for testing
        output.append([name,round(mean([norm(x) for x in subtract(a1,a2)]),3),
                     (a1-a2),a1])
        #Adds back the A1 modes for cycles past #1, otherwise they'd all have additional total symmetry distortion - not helpful. 
        model = output[0][3]
        
    return output

def return_simple_nsd(ats,point_group_name,model = False,bhopping= False):
    output = yield_vectors_by_matrix(ats,point_group_name,model,bhopping)
    return [[x[0],x[1]*len(x[2])] for x in output]

def find_symmetry_arbitrary(ops,pgt,lookup_table):
    if (ops == []) or (ops == None): ops = [list(pgt.keys())[0]]
    return lookup_table.get(str(product([pgt.get(x) for x in ops],axis=0).tolist()))

def make_html_table(arr): 
    r1 = '\n'.join(['<tr>']+['<th>'+ x[0] +'</th>' for x in arr]+['</tr>'])
    r2 = '\n'.join(['<tr>']+['<td>'+ "%.3f" % x[1]+'</td>' for x in arr]+['</tr>'])
    return "<table id='tableip'>"+r1+r2+'</table>'

def import_sd_file(filepath, model, symmetry, verbose = False):
    
    sd_file = open(filepath,'r').read()
    sd_list = [x.split('\n') for x in sd_file.split('$$$$\n')]
    n_atoms = int(sd_list[0][3].split()[0])
    
    structdb = []
    for data in sd_list[:-1]:
        n,x,y,z,c = [data[0]],[],[],[],[]
        for i in data[4:4+n_atoms]:
            #always starts at 4; contains n-4 atoms for [4,n] as query
            j = i.split()
            x.append(float(j[0]))
            y.append(float(j[1]))
            z.append(float(j[2]))
            c.append(j[3])
        structdb.append(n+[x-mean(x),y-mean(y),z-mean(z),c])
        
    structures_db = pd.DataFrame(structdb)
    structures_db.columns = ['NAME', 'xdata','ydata','zdata','atom']
    namelist,coordslist,nsdlist = [],[],[]

    for row in structures_db.itertuples():
        xyz_matrix = hstack([row[2:5]]).T
        ccdc_ident = row[1]
        nsd = yield_vectors_by_matrix(xyz_matrix, symmetry, model)

        namelist.append(ccdc_ident)
        coordslist.append(xyz_matrix)
        nsdlist.append(nsd)
        if verbose:
            print(str(row[0]) +':'+ ccdc_ident)

    nsd_df = pd.DataFrame([namelist,nsdlist,coordslist]).transpose()
    nsd_df.columns = ['NAME', 'NSD_matrix', 'coords_matrix']
    return nsd_df

def split_nsd_df(df):
    irreps = [x[0] for x in df['NSD_matrix'].values[0]]
    for i,x in enumerate(irreps):
        df[x] = [y[i][1] for y in df['NSD_matrix']]
        df[x+'m'] = [y[i][2] for y in df['NSD_matrix']]
    return df
