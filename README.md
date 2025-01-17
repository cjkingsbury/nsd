# nsd
General tools for the Normal-coordinate Structural Decomposition of molecules, or generally point graphs in 3 dimensions with near-symmetry.

Coordinate vector decomposition, i.e. decomposition of molecules into numerical arguments is the ideal method of data reduction for correlating with photophysical 
measurements and performing machine-learning and data science 

Assists in the numerical encoding of arbitrary near-symmetry molecular conformation as a set of easily grokkable symmetry arguments related to the 
dissymetric components in each possible sub-point-group from encoding by the irreducible representations. Tools for database analysis and principal component
analysis are included, and these should, in the ideal case, by iteration approximate the n-th eigenvectors of the vibrational calculated structure, such that 
these may be used interchangably. The first principal components generally account for approx. < 90% of the total distortion of the molecule in the symmetric mode 
for simple molecules.

A particularly useful tool for database analysis have been the 2-d plots of alignment with principal component vs not, which generally show clusters of similar 
molecular conformations. See ("The Shape of Porphyrins"; Kingsbury, C. J., Senge, M. O., Article in preparation) for more indication of the kind of analysis
which can be performed.

In general, it is best to trim the input pdb file of the molecule of interest to simply include the relevant atoms of the aromatic unit 
(i.e. just the BODIPY C<sub>9</sub>N<sub>2</sub>BF<sub>2</sub> or porphyrin C<sub>20</sub>N<sub>4</sub> core) to obtain meaningful data. Fitting a molecule to a
precomputed model (included in model_xyzs.py, able to be added to) is significantly faster than finding by symmetry operations, and gives results that may
be compared between molecules, especially in the cases where orientation is not defined by the point group (i.e. D<sub>2</sub>h, where x, y, and z are interchangeable,
or D<sub>4</sub>h, where B<sub>1g</sub> and B<sub>2g</sub> are dependent on alignment and could be confused)

The Neoplasic symmetry plots are a cutesy graphic which can be used to spice up a report. Make sure the structure is correctly assigned by the program 
(see graphs in the verbose output, algorithm is not infallible and can break from things like 180&#176; twisting) and try to use a model for computational 
speed and consistency of output. The algorithm requires expanded per-element point-group tables and comparative lookup for each of the subsymmetric point 
groups, and is therefore somewhat less generally applicable than something like the "Symmetrize" routine (R. J. Largent, W. F. Polik, J. R. Schmidt, 
J. Comput. Chem. 2012, 33, 1637–1642. DOI: 10.1002/jcc.22995), but I'm working on getting more point-groups integrated and on a faster and more generally
applicable method to do so.

The porphyrin-tools are one specific use-case for a symmetrical distortion review methodology, which requres the use of precomputed vector-sets to generate the 
tables of structural distortion along symmetric lines - J. A. Shelnutt's NSD method, readapted for the analysis of chemical databases. We can show here that the 
precomputation of vectors is not necessary for the analysis of distortion vectors, and that a method similar to NSD can be performed with any near-symmetric molecule,
allowing for easy parameter reduction.


This package is a set of functions and equations to assist in the analysis of molecular conformation from theoretical or crystallographic data sets. 

<h3>Porphyrin-tools</h3>
Based on the methodology of Jentzen, the analysis of porphyrin conformation occurs as a least-squares reduction of a molecule, provided in the .pdb format, or a database of porphyrin components, provided as a .sd file. pdb and sd files can be generated by Mercury, from the CCDC team.

<h3>Porphyrin-tests</h3>
This is the implementation of cluster analysis and the data-science equations, as well as the methods of extracting useable data easily from a data-set of a porphyrin molecule. The data generated is shown in the "Verbose" data set generated by kingsbury.id.au/nsd

<h3>nsd</h3>
This is the general case of structural decomposition of near-symmetric molecules by symmetry, applied to isolated chromophores of structures from either calculated or measured conformations. Intended to allow for the general parameterisation of the crystalline solid state, this allows for simple, relatively consistent and chemically relevant data reduction for machine learning processes. In essence, a molecule is split into the vectors which comprise the deviation from symmetric representation along the irreducible representation mode (e.g. "Agm"), and are assigned a sum value of the normal values of this vector collection ("Ag"). The vector field can be collated with other representative vectors using the methods in nsd-principal-components.

<h3>nsd-mondrian</h3>
The implementation of the mondrian representation plots for arbitrary molecular motifs comprising near-symmetric molecules or point collections in the point groups C2v, C2h, D2d, D2h and D4h. New symmetric groups will be added for future releases.

<h3>model_xyzs</h3>
A small set of idealised molecular fragments, with names and a set dictionary. Add any that you may need to use by using the create_model routine in nsd on a relatively undistorted molecule.

<h3>nsd-principal-components</h3>
Under development. Scikit-learn's sklearn.decomposition.PCA is a nice way of taking a database of molecules and generating reasonable normal-coordinates which might be
used for otehr purposes. This is potentially a method of using previously reported crystallographic data to fit crystal structures from data-poor sources, such as PD.
