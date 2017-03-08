import quippy
import time
import os
import numpy as np
import cPickle as pickle
import sys
import itertools
import gzip
from permanent import rematch
from collections import defaultdict
import multiprocessing
import Queue

def convert(nmax, lmax, nspecies, rawsoap):
    """ Converts from flat quippy SOAP format to a multidimensional array representation
    with dimension (nspecies, nspecies, nmax**2*(lmax+1))"""
    soaps = np.zeros((nspecies, nspecies, nmax*nmax*(lmax+1)))
    vector_idx = np.zeros((nspecies, nspecies))

    idx_soap = 0
    isqrttwo = 1.0/np.sqrt(2.0)
    for s1 in xrange(nspecies):
        for n1 in xrange(nmax):
            for s2 in xrange(s1+1):
                for n2 in xrange(nmax if s2<s1 else n1+1):
                    for l in xrange(lmax+1):
                        if (s1 != s2):
                            # multiplication by isqrttwo undo the normalisation
                            soaps[s2, s1, l+(lmax+1)*(n2+nmax*n1)] = rawsoap[idx_soap] * isqrttwo
                            soaps[s1, s2, l+(lmax+1)*(n1+nmax*n2)] = rawsoap[idx_soap] * isqrttwo
                        else:
                            # diagonal species (s1=s2) have only half of the elements.
                            # TODO: PBJ: I don't know why this is the case?
                            # this is tricky. we need to duplicate diagonal blocks "repairing" these to be full.
                            # this is necessary to enable alchemical similarity matching, where we need to combine
                            # alpha-alpha and alpha-beta environment fingerprints
                            val = rawsoap[idx_soap] * (1 if n1==n2 else isqrttwo)
                            soaps[s2, s1, l+(lmax+1)*(n2+nmax*n1)] = val
                            soaps[s2, s1, l+(lmax+1)*(n1+nmax*n2)] = val
                        idx_soap += 1
    return soaps

def map_array(ain, aout, specin, specout):
    """ Writes multidimensional array ain into the array aout
    specin is a dictionary specifying the number of atoms of each species for ain
    specout is the corresponding dictionary for aout.
    specout must always have at least the same number of atoms of a given species
        as specin. For example """

    in_spec = sorted(specin.keys())
    out_spec = sorted(specout.keys())

    # Map from input species to output species index
    sp_map = np.array([out_spec.index(k) for k in in_spec])

    # Outstruct is a list of output atoms.
    # e.g. if out_spec={1:4, 6:2}, outspec is [1,1,1,1,6,6]
    outstruct = [[z]*specout[z] for z in out_spec]
    outstruct = [itm for sublist in outstruct for itm in sublist]

    # atm_map maps input atom indices to output atom indices
    atm_map = []
    for z in in_spec:
        offset = outstruct.index(z)
        atm_map += [offset+i for i in range(specin[z])]

    aout[np.ix_(atm_map, sp_map, sp_map)] = ain


def tensordot(arrA, arrB, alchemAB):
    dotprod = np.tensordot(arrA, arrB, axes=([3],[3]))
    dotprod = np.tensordot(dotprod, alchemAB, axes=([1,4],[0,1]))
    return np.tensordot(dotprod, alchemAB, axes=([1,3],[0,1]))

def kernel(a, b, alchem, gamma=0.5):
    """ Compute kernel of molecule a and b of type SoapMol,
    using alchemical rules alchem
    gamma is the regularisation parameter for the rematch algorithm """

    # Union of species of both molecules as a list
    zspecies = sorted(list(set(a.species.keys()+b.species.keys())))
    nsp = len(zspecies) # Shorthand for number of species

    # Create dictionary mapping species to number of atoms,
    # (union of both molecules)
    vals = []
    for z in zspecies:
        if z in a.species and z in b.species:
            val = max(a.species[z], b.species[z])
        elif z in a.species:
            val = a.species[z]
        else:
            val = b.species[z]
        vals.append((z,val))
    union_species = dict(vals)

    nenv = sum(union_species.values())

    # Make a mapping from atom idx to (species, species_atom_idx)
    nspecies = [(i, union_species[z]) for i,z in enumerate(zspecies)]
    idx_to_spec = [zip([i]*nz,range(nz)) for i, nz in nspecies]
    idx_to_spec = [itm for sublist in idx_to_spec for itm in sublist]

    flen = a.envs.shape[3] # Length of vector of Power Spectrum coefficients
    arrA = np.zeros((nenv, nsp, nsp, flen), dtype=float)
    arrB = np.zeros((nenv, nsp, nsp, flen), dtype=float)

    for env_idx in range(nenv):
        sp_idx, atm_idx = idx_to_spec[env_idx]
        # Missing atoms are set to isolated species
        # TODO: Not sure why we set them to 1?
        #arrA[env_idx, sp_idx, sp_idx, 0] = 1
        #arrB[env_idx, sp_idx, sp_idx, 0] = 1

    map_array(a.envs, arrA, a.species, union_species)
    map_array(b.envs, arrB, b.species, union_species)

    # alchemical matrix for species
    alchemAB = np.zeros((nsp,nsp), float)
    for sA in xrange(nsp):
        for sB in xrange(sA+1):
            alchemAB[sA,sB] = alchem[(zspecies[sA],zspecies[sB])]
            alchemAB[sB,sA] = alchemAB[sA,sB]

    kk = tensordot(arrA, arrB, alchemAB)

    return rematch(kk, gamma, 1e-6)

class PickableMol():
    """ Contains the necessary attributes from SoapMol to compute kernel.
        This object must be pickable to be sent through the multiprocessing
        queue. """
    def __init__(self, soapmol):
        self.species = soapmol.species
        self.envs = soapmol.envs

class SoapMol():
    def __init__(self, alchem_rules, atoms):
        self.alchem_rules = alchem_rules
        self.species, self.envs = self._get_environments(atoms)
        self._normalise()

        assert sum(self.species.values()) == self.envs.shape[0]

    def _normalise(self):
        """ Normalise each local environment such that the kernel
        with a single environment with itself is unity """
        # alchemical matrix for species
        nsp = len(self.species.keys())
        zspecies = sorted(self.species.keys())
        alchemAB = np.zeros((nsp,nsp), float)
        for sA in xrange(nsp):
            for sB in xrange(sA+1):
                alchemAB[sA,sB] = self.alchem_rules[(zspecies[sA],zspecies[sB])]
                alchemAB[sB,sA] = alchemAB[sA,sB]

        for i in range(self.envs.shape[0]):
            norm = np.tensordot(self.envs[i], self.envs[i], axes=([2],[2]))
            norm = np.tensordot(norm, alchemAB, axes=([1,3],[0,1]))
            norm = np.sqrt(np.tensordot(norm, alchemAB, axes=([0,1],[0,1])))
            self.envs[i] *= 1.0/norm

    #def _get_environments(self, atoms, coff=3.0, cotw=0.5, nmax=8, lmax=6, gs=0.5, cw=1.0):
    def _get_environments(self, atoms, coff=4.0, cotw=0.5, nmax=10, lmax=8, gs=0.5, cw=0.0):
        """ Compute environments for each atom
        Returns tuple consisting of
        a) Dictionary giving number of atoms for a given species
        b) An array of partial power spectra of dimension
           (natoms, nspecies, nspecies, nmax**2*(lmax+1) """

        env_list = []

        at = atoms.copy()

        # Set cutoff radius for each atom
        at.set_cutoff(coff);
        at.calc_connect(); # calculate connections (not sure why this is necessary)

        # Create dictionary giving number of atoms of a given species
        species = dict([(i,len(list(j))) for (i,j) in itertools.groupby(sorted(at.z))])

        # Specify species in format suitable for quippy descriptor
        # e.g. if the molecule contains Hydrogen and Carbon lspecies is
        # "n_species=2 species_Z={1 6}"
        lspecies = ("n_species=%d " % len(species.keys())) + \
                "species_Z={" + " ".join(map(str, sorted(species.keys()))) + "}"

        for sp in sorted(species.keys()):
            # Create descriptor for current species
            #quippy_str = "soap central_weight="+str(cw)+"  covariance_sigma0=0.0 atom_sigma="+str(gs)+" cutoff="+str(coff)+" cutoff_transition_width="+str(cotw)+" n_max="+str(nmax)+" l_max="+str(lmax)+' '+lspecies+' Z='+str(sp)
            quippy_str = "soap central_weight="+str(cw)+"  covariance_sigma0=0.0 atom_sigma="+str(gs)+" cutoff="+str(coff)+" cutoff_transition_width="+str(cotw)+" n_max="+str(nmax)+" l_max="+str(lmax)+' '+lspecies+' Z='+str(sp)+' xml_version=0'
            desc = quippy.descriptors.Descriptor(quippy_str)

            # Create output array
            psp = quippy.fzeros((desc.dimensions(),desc.descriptor_sizes(at)[0]))
            # Compute power spectrum
            desc.calc(at,descriptor_out=psp)

            # Transpose such that each row of psp is the power specturm of the
            # environment as seen from one atom of species sp
            # Each row is a flattened array with "original" dimension
            # (s1, n1, s2, n2, l)
            # where
            # s1 is index of species alpha [0:nspecies-1]
            # n1 is the index of spherical harmonics basis [0:nmax-1]
            # s2 is index of species beta [0:s1]
            # n2 is the index of spherical harmonics basis [0:nmax-1] if s2<s1, else [0:n1]
            # l  is the index of spherical harmonics basis [0:lmax]
            #   when s1=s2 the power spectrum is half as long
            psp = np.array(psp.T)
            for soap in psp:
                env_list.append(convert(nmax, lmax, len(species.keys()), soap))

        # Convert list of 3d arrays to single 4d array
        outarray = np.array(env_list)
        assert outarray.shape[0] == len(at.z)
        assert outarray.shape[0] == sum(species.values())

        return species, outarray

def worker(job_queue, res_queue, alchem_rules, descriptors):
    while 1:
        for row, col in job_queue.get(True):
            if row is None or col is None:
                return
            mola = descriptors[row]
            molb = descriptors[col]
            res = kernel(mola, molb, alchem_rules)
            res_queue.put((row, col, res))

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return itertools.izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def main():
    with open("alchemy.pickle","rb") as alchem_file:
        alchem_rules = pickle.load(alchem_file)
    #from collections import defaultdict
    #alchem_rules = defaultdict(lambda: 1) # TODO: WARNING. Using 1 as alchem rule
    #print "ALCHEM RULES ARE 1"

    xyz_file = sys.argv[1]
    kernel_file = os.path.splitext(xyz_file)[0] + ".k.pkz"

    # Load xyz file with molecules and convert to quippy format
    all_mols = quippy.AtomsList(xyz_file)

    # Compute descriptor for each molecule
    print "Computing descriptors"
    descriptors = []
    for mol in all_mols:
        descriptors.append(PickableMol(SoapMol(alchem_rules, mol)))

    # Create kernel matrix
    N = len(all_mols)
    kernel_matrix = np.ones((N, N)) * np.nan

    # Setup multiprocessing
    job_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    num_processes = 4
    chunk_size = 100
    workers = [
            multiprocessing.Process(
                target=worker, args=(job_queue, res_queue, alchem_rules, descriptors))
            for _ in range(num_processes)
            ]

    # First compute unnormalised diagonal (used for normalisation)
    for (row, col) in zip(range(N), range(N)):
        job_queue.put([(row, col),])

    # Start all workers
    for w in workers:
        w.start()

    # Retrieve results
    num_jobs_finished = 0
    print "Computing diagonal"
    while num_jobs_finished < N:
        row, col, res = res_queue.get()
        kernel_matrix[row,col] = res
        num_jobs_finished += 1

    # Compute off-diagonal elements (with normalisation)
    print "Computing off-diagonal elements"

    # Submit jobs to queue
    for chunk in grouper(chunk_size, itertools.combinations(range(N), 2), (None, None)):
        job_queue.put(chunk)

    # Retrieve results
    num_jobs_finished = 0
    total_jobs = len(list(itertools.combinations(range(N), 2)))
    while num_jobs_finished < total_jobs:
        row, col, res = res_queue.get()
        inorm = 1.0/np.sqrt(kernel_matrix[row,row]*kernel_matrix[col,col])
        kernel_matrix[row,col] = res*inorm
        kernel_matrix[col,row] = res*inorm
        num_jobs_finished += 1
        if (num_jobs_finished % 1000) == 0:
            print "%s job %d/%d complete (%d%%)" %\
                (time.strftime("%Y-%m-%d %H:%M"),
                num_jobs_finished,
                total_jobs,
                100.0*float(num_jobs_finished)/float(total_jobs))

    print "Done computing off-diagonal, saving kernel"
    # Diagonal is normalised to unity
    kernel_matrix[np.diag_indices(N)] = 1.0

    # Save kernel matrix to file
    with gzip.open(kernel_file, "wb") as f:
        pickle.dump(kernel_matrix, f)

    print "Joining workers"
    for w in workers:
        job_queue.put([(None, None),])
    for w in workers:
        w.join()



if __name__ == "__main__":
    main()
