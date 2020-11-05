# nsd
General tools for the Normal-coordinate Structural Decomposition of molecules, or generally point graphs in 3 dimensions with near-symmetry.

Assists in the numerical encoding of arbitrary near-symmetry molecular conformation as a set of easily grokkable symmetry arguments related to the 
dissymetric components in each possible sub-point-group from encoding by the irreducible representations. Tools for database analysis and principal component
analysis are included, and these should, in the ideal case, by iteration approximate the n-th eigenvectors of the vibrational calculated structure, such that 
these may be used interchangably. The first principal components generally account for approx <90% of the total distortion of the molecule in the symmetric mode 
for simple molecules.

A particularly useful tool for database analysis have been the 2-d plots of alignment with principal component vs not, which generally show clusters of similar 
molecular conformations. See ("The Shape of Porphyrins"; Kingsbury, C. J., Senge, M. O., Article in preparation) for more indication of the kind of analysis
which can be performed.

In general, it is best to trim the input pdb file of the molecule of interest to simply include the relevant atoms of the aromatic unit 
(i.e. just the BODIPY C9N2BF2 or porphyrin C20N4 core) to obtain meaningful data. Fitting a molecule to a precomputed model (included in model_xyzs.py, 
able to be added to) is significantly faster than finding by symmetry operations, and gives results that may be compared between molecules, especially in the cases 
where orientation is not defined by the point group (i.e. D2h, where x, y, and z are interchangeable, or D4h, where B1g and B2g are dependent on alignment and could be confused)

The Neoplasic symmetry plots are a cutesy graphic which can be used to spice up a report. Make sure the structure is correctly assigned by the program 
(see graphs in the verbose output, algorithm is not infallible and can break from things like 180deg. twisting) and try to use a model for computational 
speed and consistency of output. The algorithm requires expanded per-element point-group tables and comparative lookup for each of the subsymmetric point 
groups, and is therefore somewhat less generally applicable than something like the "Symmetrize" routine, but I'm working on getting more point-groups integrated
and on a faster and more generally applicable method to do so.

The porphyrin-tools are one specific use-case for a symmetrical distortion review methodology, which requres the use of precomputed vector-sets to generate the 
tables of structural distortion along symmetric lines - J. A. Shelnutt's NSD method, readapted for the analysis of chemical databases. We can show here that the 
precomputation of vectors is not necessary for the analysis of distortion vectors, and that a method similar to NSD can be performed with any near-symmetric molecule,
allowing for easy parameter reduction.
