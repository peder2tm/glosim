#!/usr/bin/env python
# Computes the matrix of similarities between structures in a xyz file
# by first getting SOAP descriptors for all environments, finding the best
# match between environments using the Hungarian algorithm, and finally
# summing up the environment distances.
# Supports periodic systems, matching between structures with different
# atom number and kinds, and sports the infrastructure for introducing an
# alchemical similarity kernel to match different atomic species

# import sys, os, pickle
import sys, os
import cPickle as pickle
import gc 
from lap.lap import best_pairs, best_cost, lcm_best_cost
from lap.perm import xperm, mcperm, rematch
import numpy as np
from environments import environ, alchemy, envk
import quippy
__all__ = [ "structk", "structure" ]

   
class structure:
   def __init__(self, salchem=None):
      self.env={}
      self.species={}
      self.zspecies = []
      self.nenv=0  
      self.alchem=salchem
      if self.alchem is None: self.alchem=alchemy()
      self.globenv = None
      
   def getnz(self, sp):
      if sp in self.species:
         return self.species[sp]
      else: return 0
      
   def getenv(self, sp, i):
      if sp in self.env and i<len(self.env[sp]):
         return self.env[sp][i]
      else: 
         return environ(self.nmax,self.lmax,self.alchem,sp)  # missing atoms environments just returned as isolated species!
         
   def ismissing(self, sp, i):
      if sp in self.species and i<self.species[sp]:
         return False
      else: return True
   
      
   def parse(self, fat, coff=5.0, cotw=0.5, nmax=4, lmax=3, gs=0.5, cw=1.0, nocenter=[], noatom=[], kit=None):
      """ Takes a frame in the QUIPPY format and computes a list of its environments. """
      
      # removes atoms that are to be ignored
      at = fat.copy()
      nol = []
      for s in range(1,at.z.size+1):
         if at.z[s] in noatom: nol.append(s)
      if len(nol)>0: at.remove_atoms(nol)
      
      self.nmax = nmax
      self.lmax = lmax
      self.species = {}
      for z in at.z:      
         if z in self.species: self.species[z]+=1
         else: self.species[z] = 1
            
      self.zspecies = self.species.keys();
      self.zspecies.sort(); 
      lspecies = 'n_species='+str(len(self.zspecies))+' species_Z={ '
      for z in self.zspecies: lspecies = lspecies + str(z) + ' '
      lspecies = lspecies + '}'
   
      at.set_cutoff(coff);
      at.calc_connect();
      
      self.nenv = 0
      for sp in self.species:
         if sp in nocenter: 
            self.species[sp]=0
            continue # Option to skip some environments
         
         # first computes the descriptors of species that are present
         desc = quippy.descriptors.Descriptor("soap central_weight="+str(cw)+"  covariance_sigma0=0.0 atom_sigma="+str(gs)+" cutoff="+str(coff)+" cutoff_transition_width="+str(cotw)+" n_max="+str(nmax)+" l_max="+str(lmax)+' '+lspecies+' Z='+str(sp) )   
         try:
            psp =np.asarray(desc.calc(at,desc.dimensions(),self.species[sp])).T
         except TypeError:
            psp = quippy.fzeros((desc.dimensions(),desc.descriptor_sizes(at)[0]))
            desc.calc(at,descriptor_out=psp)
            psp = np.array(psp.T)

         # now repartitions soaps in environment descriptors
         lenv = []
         for p in psp:
            nenv = environ(nmax, lmax, self.alchem)
            nenv.convert(sp, self.zspecies, p)
            lenv.append(nenv)
         self.env[sp] = lenv
         self.nenv += self.species[sp]
         
      # adds kit data   
      if kit is None: kit = {}
      
      for sp in kit:         
         if not sp in self.species: 
            self.species[sp]=0
            self.env[sp] = []
         for k in range(self.species[sp], kit[sp]):            
            self.env[sp].append(environ(self.nmax,self.lmax,self.alchem,sp))
            self.nenv+=1
         self.species[sp] = kit[sp]          
      
      self.zspecies = self.species.keys()
      self.zspecies.sort() 
      
      # also compute the global (flattened) fingerprint
      self.globenv = environ(nmax, lmax, self.alchem)
      
      for k, se in self.env.items():
         for e in se:
            self.globenv.add(e)
      # divides by the number of atoms in the structure
      for sij in self.globenv.soaps:  self.globenv.soaps[sij]*=1.0/self.nenv
      # self.globenv.normalize()  #if needed, normalization will be done later on.....
      

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)

#def gstructk(strucA, strucB, alchem=alchemy(), periodic=False):
#    
#   return envk(strucA.globenv, strucB.globenv, alchem) 
def compile_tensordot():
    import theano
    import theano.tensor as T
    arrA = T.tensor4("arrA", dtype=theano.config.floatX)
    arrB = T.tensor4("arrB", dtype=theano.config.floatX)
    alchemAB = T.matrix("alchemAB", dtype=theano.config.floatX)
    dotprod = T.tensordot(arrA, arrB, axes=([3],[3]))
    dotprod = T.tensordot(dotprod, alchemAB, axes=([1,4],[0,1]))
    kk = T.tensordot(dotprod, alchemAB, axes=([1,3],[0,1]))
    return theano.function([arrA, arrB, alchemAB], outputs=kk, name="tensordot")

def numpy_tensordot(arrA, arrB, alchemAB):
    dotprod = np.tensordot(arrA, arrB, axes=([3],[3]))
    dotprod = np.tensordot(dotprod, alchemAB, axes=([1,4],[0,1]))
    return np.tensordot(dotprod, alchemAB, axes=([1,3],[0,1]))

#tensordot = compile_tensordot()
tensordot = numpy_tensordot



def structk(strucA, strucB, alchem=alchemy(), periodic=False, mode="match", fout=None, peps=0.0, gamma=1.0, zeta=1.0, xspecies=False):
    # computes the SOAP similarity KERNEL between two structures by combining atom-centered kernels
    # possible kernel modes include:
    #   average :  scalar product between averaged kernels
    #   match:     best-match hungarian kernel
    #   permanent: average over all permutations
      
    # top up missing atoms with isolated environments
    # first checks which atoms are present
    zspecies = sorted(list(set(strucB.zspecies+strucA.zspecies)))
    nspecies = []
    nenv = 0
    for z in zspecies:
        nz = max(strucA.getnz(z),strucB.getnz(z))
        nspecies.append((z,nz)) 
        nenv += nz

    # Make a mapping from atom idx to (species, species_idx)
    idx_to_spec = [zip([z]*nz,range(nz)) for z, nz in nspecies]
    idx_to_spec = [itm for sublist in idx_to_spec for itm in sublist]

    nsp = len(zspecies)
    # alchemical matrix for species
    alchemAB = np.empty((nsp,nsp), dtype=float)
    for sA in xrange(nsp):
        for sB in xrange(sA+1):
            alchemAB[sA,sB] = alchem.getpair(zspecies[sB],zspecies[sA])
            alchemAB[sB,sA] = alchemAB[sA,sB]
         
    #prepares the lists of pairs to avoid calling many times getpair further down the line
    #for i1 in xrange(nsp):
    #    s1 = zspecies[i1] 
    #    for i2 in xrange(nsp):
    #        s2 = zspecies[i2] 
    #        a=envA.getpair(s1,s2)
    #        b=envB.getpair(s1,s2)
    #        if i1==0 and i2==0:
    #            arrA = np.empty((nenv, nsp, nsp, a.shape[0]), float)
    #            arrB = np.empty((nenv, nsp, nsp, a.shape[0]), float)
    #        arrA[i1,i2,:] = a
    #        arrB[i1,i2,:] = b

    #first = True
    #for za, nza in nspecies:      
    #    for ia in xrange(nza):
    #        envA = strucA.getenv(za, ia)         
    #        ikb = 0
    #        for zb, nzb in nspecies:
    #            a = envA.getpair(za, zb)
    #            for ib in xrange(nzb):
    #                envB = strucB.getenv(zb, ib)
    #                b = envB.getpair(za, zb)
    #                if first:
    #                    arrA = np.empty((nenv, nsp, nsp, a.shape[0]), float)
    #                    arrB = np.empty((nenv, nsp, nsp, a.shape[0]), float)
    #                    first = False
    #                arrB[ikb,i1,i2,:] = b

    #                ikb+=1
    #        arrA[ika, i1,i2,:] = a
    #        ika+=1

    for ik in range(nenv):
        i = idx_to_spec[ik]
        envA = strucA.getenv(i[0], i[1])         
        envB = strucB.getenv(i[0], i[1])         
        for i1 in range(nsp):
            s1 = zspecies[i1]
            for i2 in range(nsp):
                s2 = zspecies[i2]
                a=envA.getpair(s1,s2)
                b=envB.getpair(s1,s2)
                if i1==0 and i2==0 and ik == 0:
                    arrA = np.empty((nenv, nsp, nsp, a.shape[0]), dtype=float)
                    arrB = np.empty((nenv, nsp, nsp, a.shape[0]), dtype=float)
                arrA[ik,i1,i2,:] = a
                arrB[ik,i1,i2,:] = b


    kk = tensordot(arrA, arrB, alchemAB)

    #dotprod = np.tensordot(arrA, arrB, axes=([3],[3]))
    #dotprod = np.tensordot(dotprod, alchemAB, axes=([1,4],[0,1]))
    #kk = np.tensordot(dotprod, alchemAB, axes=([1,3],[0,1]))

    #kk = np.zeros((nenv,nenv),float)
    #for ika in range(nenv):      
    #    ai = idx_to_spec[ika]
    #    envA = strucA.getenv(ai[0], ai[1])         
    #    for ikb in range(nenv):
    #        bi = idx_to_spec[ikb]
    #        envB = strucB.getenv(bi[0], bi[1])
    #        res = envk(envA, envB, alchem)**zeta              
    #        kk[ika,ikb] = res              

    #ika = 0
    #ikb = 0  
    #for za, nza in nspecies:      
        #for ia in xrange(nza):
            #envA = strucA.getenv(za, ia)         
            #ikb = 0
            #for zb, nzb in nspecies:
                #for ib in xrange(nzb):
                    #envB = strucB.getenv(zb, ib)
                    #res = envk(envA, envB, alchem)**zeta              
                    #kk[ika,ikb] = res              
                    #ikb+=1
            #ika+=1
      
    # Now we have the matrix of scalar products. 
    # We can first find the optimal scalar product kernel
    # we must find the maximum "cost"
    if mode == "rematch":
        cost=rematch(kk, gamma, 1e-6)  # hard-coded residual error for regularized gamma
        # print cost, kk.sum()/(nenv*nenv), envk(strucA.globenv, strucB.globenv, alchem)
    else: raise ValueError("Unknown global fingerprint mode ", mode)
         
    return cost,kk

#def structk(strucA, strucB, alchem=alchemy(), periodic=False, mode="match", fout=None, peps=0.0, gamma=1.0, zeta=1.0, xspecies=False):
#   # computes the SOAP similarity KERNEL between two structures by combining atom-centered kernels
#   # possible kernel modes include:
#   #   average :  scalar product between averaged kernels
#   #   match:     best-match hungarian kernel
#   #   permanent: average over all permutations
#      
#   # average kernel. quick & easy!   
#   if mode=="fastavg":
#       genvA=strucA.globenv
#       genvB=strucB.globenv        
#       return envk(genvA, genvB, alchem)**zeta, 0
#   elif mode=="fastspecies": 
#       # for now, only implement standard Kronecker alchemy
#       senvB = environ(strucB.nmax, strucB.lmax, strucB.alchem)
#       kk = 0
#       for za in strucA.zspecies:    
#         if not za in strucB.zspecies: continue         
#         senvA = environ(strucA.nmax, strucA.lmax, strucA.alchem)
#         for ia in xrange(strucA.getnz(za)):
#            senvA.add(strucA.getenv(za, ia))
#         senvB = environ(strucB.nmax, strucB.lmax, strucB.alchem)   
#         for ib in xrange(strucB.getnz(za)):
#            senvB.add(strucB.getenv(za, ib))
#         kk += envk(senvA, senvB, alchem)**zeta
#       
#       kk/=strucA.nenv*strucB.nenv
#       return kk,0
#         
#       #  for zb, nzb in nspeciesB:
#       #         for ib in xrange(nzb):
#       #            return envk(genvA, genvB, alchem), 0
#
#   nenv = 0
#   
#   if periodic: # replicate structures to match structures of different periodicity
#      # we do not check for compatibility at this stage, just assume that the 
#      # matching will be done somehow (otherwise it would be exceedingly hard to manage in case of non-standard alchemy)
#      nspeciesA = []
#      nspeciesB = []
#      for z in strucA.zspecies:
#         nspeciesA.append( (z, strucA.getnz(z)) )
#      for z in strucB.zspecies:
#         nspeciesB.append( (z, strucB.getnz(z)) )
#      nenv=nenvA = strucA.nenv
#      nenvB = strucB.nenv            
#   else:   
#      # top up missing atoms with isolated environments
#      # first checks which atoms are present
#      zspecies = sorted(list(set(strucB.zspecies+strucA.zspecies)))
#      nspecies = []
#      for z in zspecies:
#         nz = max(strucA.getnz(z),strucB.getnz(z))
#         nspecies.append((z,nz)) 
#         nenv += nz
#      nenvA = nenvB = nenv
#      nspeciesA = nspeciesB = nspecies   
#         
#   np.set_printoptions(linewidth=500,precision=4)
#
#   kk = np.zeros((nenvA,nenvB),float)
#   ika = 0
#   ikb = 0  
#   for za, nza in nspeciesA:      
#      for ia in xrange(nza):
#         envA = strucA.getenv(za, ia)         
#         ikb = 0
#         for zb, nzb in nspeciesB:
#            for ib in xrange(nzb):
#               envB = strucB.getenv(zb, ib)
#               if alchem.mu > 0 and (strucA.ismissing(za, ia) ^ strucB.ismissing(zb, ib)):
#                   # includes a penalty dependent on "mu", in a way that is consistent with the definition of kernel distance
#                   kk[ika,ikb] = exp(-alchem.mu)
#               else:
#                  if za == zb or not xspecies:  #uncomment to zero out kernels between different species
#                    kk[ika,ikb] = envk(envA, envB, alchem)**zeta              
#                  else: kk[ika,ikb] = 0
#               ikb+=1
#         ika+=1
#   aidx = {}
#   ika=0
#   for za, nza in nspeciesA: 
#      aidx[za] = range(ika,ika+nza)
#      ika+=nza
#   ikb=0
#   bidx = {}
#   for zb, nzb in nspeciesB: 
#      bidx[zb] = range(ikb,ikb+nzb)
#      ikb+=nzb
#
#   if fout != None:
#      # prints out similarity information for the environment pairs
#      fout.write("# atomic species in the molecules (possibly topped up with dummy isolated atoms): \n")      
#      for za, nza in nspeciesA:
#         for ia in xrange(nza): fout.write(" %d " % (za) )
#      fout.write("\n");
#      for zb, nzb in nspeciesB:
#         for ib in xrange(nzb): fout.write(" %d " % (zb) )
#      fout.write("\n");
#      
#      fout.write("# environment kernel matrix: \n")      
#      for r in kk:
#         for e in r:
#            fout.write("%20.14e " % (e) )
#         fout.write("\n")
#      #fout.write("# environment kernel eigenvalues: \n")      
#      #ev = np.linalg.eigvals(kk)
#      #for e in ev:
#      #    fout.write("(%8.4e,%8.4e) " % (e.real,e.imag) )
#      #fout.write("\n");
#         
#       
#
#      
#   # Now we have the matrix of scalar products. 
#   # We can first find the optimal scalar product kernel
#   # we must find the maximum "cost"
#   if mode == "match":
#        if periodic and nenvA != nenvB:
#            nenv = lcm(nenvA, nenvB)
#            hun = lcm_best_cost(1-kk)
#        else:
#            hun=best_cost(1.0-kk)        
#        cost = 1-hun/nenv
#   elif mode == "permanent":
#        # there is no place to hide: cross-species environments are not necessarily zero 
#        if peps>0: cost = mcperm(kk, peps)
#        else: cost = xperm(kk)
#            
#        cost = cost/np.math.factorial(nenv)/nenv        
#   elif mode == "rematch":
#       cost=rematch(kk, gamma, 1e-6)  # hard-coded residual error for regularized gamma
#       # print cost, kk.sum()/(nenv*nenv), envk(strucA.globenv, strucB.globenv, alchem)
#   elif mode == "average":
#       cost = kk.sum()/(nenvA*nenvB)
#       # print 'elem: {}'.format(kk.sum()) 
#       # print 'elem norm: {}'.format(cost) 
#       # print 'avg norm: {}'.format((nenvA*nenvB)) 
#       
#   else: raise ValueError("Unknown global fingerprint mode ", mode)
#   
#         
#   return cost,kk


class structurelist(list):
    def __init__(self, basedir="tmpstructures"):
        self.basedir=basedir
        # create the folder if it is not there        
        if not os.path.exists(basedir):os.makedirs(basedir)
        self.count=0
        
    def exists(self, index):
        # return true if the file associated with index exists, false otherwise
        f=self.basedir+'/sl_'+str(index)+'.dat'
        return os.path.isfile(f)
    # @profile
    def append(self, element):
        #pickle the element for later use
        ind=self.count
        f=self.basedir+'/sl_'+str(ind)+'.dat'
        file = open(f,"wb")
        gc.disable()
        pickle.dump(element, file,protocol=pickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL is 2 in py 2.7
        file.close()
        gc.enable()
        self.count+=1

        
    # @profile
    def __getitem__(self, index):
        f = self.basedir+'/sl_'+str(index)+'.dat'
        try:
            file = open(f,"rb")
        except IOError:
            raise IOError("Cannot load descriptors for index %d" % (index) )
        gc.disable()
        l = pickle.load(file)
        file.close()
        gc.enable()
        return l
