import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos, pi
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment, minimize, basinhopping

from symmetry_dicts import *
from nsd import *

nms = 'B$_{2g}$,B$_{1g}$,E$_{u}$x,E$_{u}$y,A$_{1g}$,A$_{2g}$,B$_{2u}$,B$_{1u}$,A$_{2u}$,E$_{g}$x,E$_{g}$y,A$_{1u}$'.split(',')
symm_typog_lookup = {'D4h':'D$_{4h}$','D4 ':'D$_{4 }$','D2 ':'D$_{2 }$','D2d':'D$_{2d}$','D2h':'D$_{2h}$','S4 ':'S$_{4 }$','C4 ':'C$_{4 }$','C4h':'C$_{4h}$','C1 ':'C$_{1 }$','Ci ':'C$_{i }$','C2h':'C$_{2h}$','Cs ':'C$_{s }$','C2 ':'C$_{2 }$','C2v':'C$_{2v}$','C4v':'C$_{4v}$',
                    'D4':'D$_{4 }$','D2':'D$_{2 }$','S4':'S$_{4 }$','C4':'C$_{4 }$','C1':'C$_{1 }$','Ci':'C$_{i }$','Cs':'C$_{s }$','C2':'C$_{2 }$'}
good_cmaps_with_sns = 'Spectral,RdYlBu,RdYlBu_r,rainbow,Blues,Greys,sns_RdBu_r,sns_colorblind,sns_Set2,sns_Set3'.split(',') + ['sns_ch:-1.5,-1,light=0.98,dark=.3','sns_ch:0.3,-0.5,light=0.98,dark=.3']
good_cmaps = 'Spectral,RdYlBu,RdYlBu_r,rainbow,Blues,Greys,RdBu_r,Set2,Set3'.split(',')
hatch_picks = ['.', '..','/','//', '\\', '\\\\', '*','**','-','--', '+', '++', '|', '||', 'x', 'xx','o','oo','O', None]

op_lims = lambda nsd: np.sort(np.array([float(x) for x in nsd[6][:6] if float(x) > 0.01]))
ip_lims = lambda nsd: np.sort(np.array([float(x) for x in nsd[2][:6] if float(x) > 0.01]))

def return_simple_nsd(ats,point_group_name,model = False,bhopping= False):
    output = yield_vectors_by_matrix(ats,point_group_name,model,bhopping)
    return [[x[0],x[1]*len(x[2])] for x in output]

def find_symmetry_arbitrary(ops,pgt,lookup_table):
    if (ops == []) or (ops == None): ops = [list(pgt.keys())[0]]
    return lookup_table.get(str(np.product([pgt.get(x) for x in ops],axis=0).tolist()))

def mondrian_arbitrary(nsd, symmetry, imagepath):#, cmap = random.choice(good_cmaps),linewidth = 3,bleed = 0.0001,
            #random_order = False, regular_axes = True, hatchwork = False):
    regular_axes = True
    hatchwork = False
    linewidth = 3
    bleed = 0.0001
    cmap = 'Spectral'
    nsd = dict(nsd)

    if symmetry == 'D4h':
        egv = [nsd.get(x) for x in 'Egx,Egy,Egx+y,Egx-y'.split(',')]
        euv = [nsd.get(x) for x in 'Eux,Euy,Eux+y,Eux-y'.split(',')]
        if np.argmax(egv)<1.5:
            nsd['Egx+y'],nsd['Egx-y'] = 0,0      
        else:
            nsd['Egx'],nsd['Egy'] = 0,0
        if np.argmax(euv)<1.5:
            nsd['Eux+y'],nsd['Eux-y'] = 0,0      
        else:
            nsd['Eux'],nsd['Euy'] = 0,0

    if symmetry == 'D2d':
        print(nsd.items())
        ev = [nsd.get(x) for x in 'Ex,Ey,Ex+y,Ex-y'.split(',')]
        if np.argmax(ev)<1.5:
            nsd['Ex+y'],nsd['Ex-y'] = 0,0      
        else:
            nsd['Ex'],nsd['Ey'] = 0,0
        
    lookup_dict = mondrian_lookup_dict.get(symmetry)
    orientation_dict = mondrian_orientaion_dict.get(symmetry)
    pgt = pgt_dict.get(symmetry)
    
    #takes the total distortion as the starting point, ignores zeros 
    vvs = np.array([x for y,x in nsd.items() if ((orientation_dict.get(y)=='v') and (x>0.001))]).astype(float)
    vns = np.array([y for y,x in nsd.items() if ((orientation_dict.get(y)=='v') and (x>0.001))])
    hvs = np.array([x for y,x in nsd.items() if ((orientation_dict.get(y)=='h') and (x>0.001))]).astype(float)
    hns = np.array([y for y,x in nsd.items() if ((orientation_dict.get(y)=='h') and (x>0.001))])
    vss,hss = np.sort(vvs),np.sort(hvs)
    #Sets the borders of the plot
    if regular_axes: xmin,xmax,ymin,ymax = -1.5, 1, -1.5, 1 
    else:
        if hss.size > 0: xmin, xmax = np.log10(np.min(ops))-0.5,np.log10(np.max(ops))+0.5
        else: xmin,xmax = 0.01, 1.0
        if vss.size > 0: ymin, ymax = np.log10(np.min(ips))-0.5,np.log10(np.max(ips))+0.5
        else: ymin, ymax = 0.01, 1.0
        
    #defines the 'flat' panes of the plot, by making points at corners of a rectangle. 
    #This is more efficient than making a grid of the entire area, as symmetry doesn't change within panels.
    x_op = np.sort(np.hstack([10**xmin, hss - bleed, hss + bleed, 10**xmax]))
    y_ip = np.sort(np.hstack([10**ymin, vss - bleed, vss + bleed, 10**ymax]))
    
    #gernerates an ordered list of symmetry operations
    sym1 = [find_symmetry_arbitrary([n for v,n in zip(vvs,vns) if v>y]+[n for v,n in zip(hvs,hns) if v>x],pgt,lookup_dict) for x,y in zip(*[z.flatten() for z in np.meshgrid(x_op,y_ip)])]
    print(sym1)
    sym2,sym3 = np.unique(sym1,return_index=True)
    try: sym3.sort()
    except: pass
    syms = [sym1[x] for x in sym3]
       
    ludict = {y:x+0.5 for x,y in enumerate(syms)}
    
    #if str(cmap).startswith('sns_'): 
    #    cmap_o = ListedColormap(sns.color_palette(cmap[4:], len(syms)))
    #else: 
    #    cmap_o = cmap
    cmap_o = cmap
    if hatchwork == True:
        hats = hatch_picks.copy()
        random.shuffle(hats)
    elif hatchwork == False: hats = [None]
    else: hats = hatchwork
        
    nx, ny = np.meshgrid(x_op, y_ip)
    nz = np.array([ludict.get(x) for x in sym1]).reshape(len(y_ip),len(x_op))
    
    #plot setup
    fig,ax = plt.subplots(figsize = (9,7))
    
    #The primary plot
    im = ax.contourf(nx, ny, nz, cmap = cmap_o, corner_mask = False, levels = np.abs(len(syms)-2), hatches = hats)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    #The colourbar
    cbar = fig.colorbar(im, drawedges = True)
    for index, value in enumerate(syms): cbar.ax.text(1.66, (index+0.5)/(len(syms)), symm_typog_lookup.get(value), ha='center', va='center')
    cbar.ax.get_yaxis().set_ticks([])
    cbar.outline.set_linewidth(linewidth)
    cbar.dividers.set_linewidth(linewidth)
    
    #plots horizonatal and vertical lines, indicating symmetry operations above threshold
    for x in hvs: ax.plot([x,x],[np.min(ny),np.max(ny)], 'black', lw = linewidth)
    for y in vvs: ax.plot([np.min(nx),np.max(nx)],[y,y], 'black', lw = linewidth)
    
    #Labels on horizontal/vertical lines.
    [ax.text(1.055 , 1-(ymax-np.log10(value))/(ymax-ymin), name, ha='right',  va='center', transform=ax.transAxes) for name, value in zip(vns,vvs)]
    [ax.text(1-(xmax-np.log10(value))/(xmax-xmin), 1.01, name, ha='center',  va='bottom', rotation = 45, transform=ax.transAxes) for name, value in zip(hns,hvs)]
    
    ax.set_title('"Mondrian" Symmetry Plot', pad = 30)
    ax.set_ylabel('Sum Distortion ($\\AA$)')
    ax.set_xlabel('Sum Distortion ($\\AA$)')
    [i.set_linewidth(linewidth) for i in ax.spines.values()]
    [i.set_linewidth(linewidth) for i in cbar.ax.spines.values()]
    fig.savefig(imagepath, dpi = 300)
    return cmap

def make_html_table(arr): 
    r1 = '\n'.join(['<tr>']+['<th>'+ x[0] +'</th>' for x in arr]+['</tr>'])
    r2 = '\n'.join(['<tr>']+['<td>'+ "%.3f" % x[1]+'</td>' for x in arr]+['</tr>'])
    return "<table id='tableip'>"+r1+r2+'</table>'
