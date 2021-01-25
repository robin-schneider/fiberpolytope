import numpy as np
from string import ascii_uppercase
from string import ascii_lowercase

import logging
import matplotlib.pyplot as plt

import itertools as it

from scipy.special import comb

from sage.all import *

#from sage.geometry.point_collection import PointCollection
from sage.geometry.lattice_polytope import LatticePolytope
from sage.interfaces.singular import singular

#sage -preparse fiberpolytope.sage
#mv fiberpolytope.sage.py fiberpolytope.py


logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('fiberpolytope')

class fiberpolytope:
    
    def __init__(self, vertices, log=20):
        
        # we first tried to inherit from the polytope class, but that
        # broke some functions, such as 3d plot, ... .
        self.polytope = LatticePolytope(vertices)

        logger.setLevel(level=int(log))

        self.p_array = np.array(self.polytope.points().column_matrix().T)
        self.pp_array = np.array(self.polytope.polar().points().column_matrix().T)


        # number of monomials is equal to number of points in polar
        # number of variables is equal to number of points in polytope - origin
        self.nMonomials = len(self.pp_array)
        self.nVariables = len(self.p_array)-1
        self.ambient_dimension = self.polytope.dim()
        self.dim = self.ambient_dimension-1

        origin = np.where(np.sum(self.p_array == np.zeros(self.ambient_dimension), axis=1) == self.ambient_dimension)[0][0]
        self.PC = PointConfiguration(self.p_array.tolist(), fine=True, star=origin)
        self.PC.set_engine('TOPCOM')

        # we should make this more general to also include other ambient spaces for the fibrations
        self.x_vars = list(var('x%d' % i) for i in range(1, self.nVariables+1))
        # complex moduli
        self.c_moduli = list(var('c%d' % i) for i in range(self.nMonomials))
        # s coefficients in the Weierstrass model
        self.s_coeff = list(var('s%d' % i) for i in range(1,11))
        # d coefficients, where s_i = some factorization in the x_vars * d_i
        self.d_coeff = list(var('d%d' % i) for i in range(1,11))
        # labeled base variables
        self.base_vars = []

        self.variables = self.x_vars
        self.fibration_vars = list(var('u v w t'))

        #wsf
        self.wsf = 0
        self.wsf_list = np.array([])
        self.s = {}
        self.d = {}
        if self.ambient_dimension > 3:
            self._find_hodge()

        # TO DO: make these to property methods
        
        # monomial basis of surface
        self.monomial_basis = self._make_surface_basis()
        
        # active fibration, we want to study
        self.active_fibration = np.array([])
        self.active_fibration_type = 'not defined'
        self.active_base = np.array([])
        self.active_blow_ups = np.array([])

        # Pauls notation
        self.cubic_notation = np.array([[3,0,0], [2,1,0], [1,2,0], [0,3,0], [2,0,1], [1,1,1], [0,2,1], [1,0,2], [0,1,2], [0,0,3]])
        self.p1xp1_notation = np.array([[2,0,2,0],[2,0,1,1],[2,0,0,2],[0,0,0,0],[1,1,2,0],[1,1,1,1],[1,1,0,2],[0,2,2,0],[0,2,1,1],[0,2,0,2]])
        self.p112_notation = np.array([[4,0,0,2],[3,1,0,2],[2,2,0,2],[1,3,0,2],[0,4,0,2],[2,0,1,1],[1,1,1,1],[0,2,1,1],[0,0,2,0],[0,0,0,0]])

        # triangulation stuff
        self.triangulation = []
        self.TV = 0
        self.HH = 0
        self.c1 = 0
        self.triple = 0
        self.c2 = 0

    def _find_hodge(self):
        r""" finds Hodge numbers by reading in Palp computation."""
        s = 0
        e = 0
        rel = False
        palp_string = self.polytope.polar().poly_x('g')
        for c, i in zip(palp_string, range(len(palp_string))):
            if c == 'H':
                s = i+2
                rel = True
            if c == ',' and rel:
                e = i
        self.h11 = int(palp_string[s:e])
        s = 0
        e = 0
        rel = False
        palp_string = self.polytope.poly_x('g')
        for c, i in zip(palp_string, range(len(palp_string))):
            if c == 'H':
                s = i+2
                rel = True
            if c == ',' and rel:
                e = i
        self.h21 = int(palp_string[s:e])

    def set_fibration(self, fibration, base, key, label=True):
        r"""sets the fibration, which you want to investigate further.
        
        Parameters
        ----------
        fibration : int_array[nFibration_coordinates]
            all fibration ambient coordinates
        base : int_array[nBase_coordinates]
            all base coordinates
        key : str
            Specify the fibration ambient space
             1) 'cubic' for P2 2) 'P1xP1' or 3) 'P112'
        label: bool
            Relabels base variables, by default=True

        Raises
        ------
        RuntimeError
            if a not supported key is given.
        """
        # specify fibration
        #self.active_fibration_type = key
        #self.active_fibration = np.array(fibration)

        if key == 'cubic' or key == 'P1xP1' or key == 'P112':
            if key == 'P112':
                self.active_fibration = np.array(fibration)[:-1]
                self.p112_blow_up = fibration[-1]
                blow_ups = []
                for i in range(self.nVariables):
                    if i not in fibration and i not in base:
                        blow_ups += [i]
                self.set_blow_ups([fibration[-1]]+blow_ups)
            else:
                self.active_fibration = np.array(fibration)
                blow_ups = []
                for i in range(self.nVariables):
                    if i not in fibration and i not in base:
                        blow_ups += [i]
                self.set_blow_ups(np.array(blow_ups))
            self.active_fibration_type = key
            self._set_base(base)
            self.label_base()
            # sort variables accordingly
            self._sort_variables()
        else:
            logger.error('Only cubic, P1xP1, P112 fibrations supported.\n'+
                        'Specify the key correctly.')

    def hodge_numbers(self):
        r"""Determines the Hodge numbers, by constructing 
        the codimension 1,2 faces and counting of interior points.

        Uses Sage, takes long, only works for 4d polytopes."""
        if self.dim != 3:
            logger.warning('Only implemented for 4d polytopes.')
            return 0

        h11 = self.polytope.npoints()-5
        for f in self.polytope.faces(3):
            npoints = f.interior_points()
            h11 -= len(npoints)
        logger.info('Toric contribution to h11: {}'.format(h11))
        for f in self.polytope.faces(2):
            npoints = f.interior_points()
            if len(npoints) != 0:
                ndpoints = f.dual().interior_points()
                if len(ndpoints) != 0:
                    h11 = h11 + len(npoints)*len(ndpoints)
        return h11

    def is_h11_toric(self):
        r"""Determines if the contribution to h11
        only comes from Toric divisos.

        Uses Sage, takes long, only works for 4d polytopes."""
        h11 = self.polytope.npoints()-5
        for f in self.polytope.faces(3):
            npoints = f.interior_points()
            h11 -= len(npoints)
        if h11 == self.h11:
            return True
        else:
            logger.info('True h11 = {}, toric h11 = {}'.format(self.h11, h11))
            return False

    def vertices_to_variables(self):
        r"""Returns a list of tuples with
        variable and corresponding vertex of the underlying
        polytope.
        
        Returns
        -------
        list[(var, ray)]
            list of tuples
        """
        dictionary = []

        j = 0
        for i in range(len(self.p_array)):
            if not np.array_equal(self.p_array[i], np.zeros(self.ambient_dimension)):
                dictionary += [(self.variables[j], self.p_array[i])]
                j += 1
        return dictionary

    def wsfmodel(self):
        r"""Determines the WSF models in terms of
        fibration coordinates and coefficients s_i.
        
        Returns
        -------
        sage_poly
            WSF polynomial
        """

        if self.active_fibration_type == 'not defined':
            logger.error('Specify active fibration first.')

        s_monomials, _ = self.find_s(monomial_basis=True)

        wsf_list = []
        if self.active_fibration_type == 'cubic':
            for monomials, s, cubics in zip(s_monomials, self.s_coeff[0:len(s_monomials)], self.cubic_notation):
                if len(monomials) == 0:
                    continue
                mon = s
                #factorization = np.zeros(len(self.active_blow_ups))
                for e in self.active_blow_ups:
                    min = np.min(monomials[:,e])
                    max = np.max(monomials[:,e])
                    assert min == max
                    #factorization[j] = min
                    if min != 0:
                        mon *= self.variables[e]**min
                
                for v, j in zip(self.active_fibration, cubics):
                    mon *= self.variables[v]**j
                #print(mon)
                wsf_list += [mon]
                # need to have the same order as before..


        if self.active_fibration_type == 'P112':
            #return self._find_p112_f(s1, s2, s3, s5, s6, s7, s8, s9, s10)
            logging.warning('Not implemented')

        if self.active_fibration_type == 'P1xP1':
            #return self._find_p1xp1_f(s1, s2, s3, s4, s5, s6, s7, s8, s9)
            logging.warning('Not implemented')

        wsf = 0
        for monomial in wsf_list:
            wsf += monomial

        self.wsf = wsf
        self.wsf_list = wsf_list

        return wsf

    def label_base(self):
        r"""Assumes a trivial projection onto two zero
        coordinates. Then goes through all base variables
        and labels them according to what they project on.

        Notation:
            Each base rays is denoted with a capital letter,
            A, B, ... having two indices. The first denoting if
            its a basis vector or a multiplicity of one, the second just
            counts all variables.

            Example:
                Say after projection we have the following rays
                [(1,0), (2,0), (1,0), (0,-1), (-1,-1)]
                the algorithm would label them as
                [A10, A20, A11, B10, C10]

            In case there are more than 26 unique rays in the base,
            the numeration continues with aA, aB, ... .
        
        Returns
        -------
        list
            of relabeld base variables
        """
        # First we recover the trivial projection
        fibrations = self.p_array[self.active_fibration]
        projection = np.where(np.all(fibrations == np.zeros(fibrations.shape[1]), axis=0))[0]
        # next we need to find all rescaled points.
        sub_system = self.p_array[self.active_base]
        sub_system = sub_system[:, projection]
        # first, we consider only the unique points
        unique = np.unique(sub_system, axis=0).astype(np.float32)
        # brute force finding higher order vectors
        order = np.ones(len(unique))
        basis_vectors = np.copy(unique)
        np.seterr(divide = 'ignore')
        logger.warning('There might be some warnings about division by zero, do not mind them. All is fine.')
        #np.seterr(invalid = 'ignore')
        # we run over all vectors - the last to comapre them with each other
        for i in range(len(unique)-1):
            # we find the slope via divison
            # here is some divison by zero.. which numpy handles well with nans
            tmp = unique[i+1:]/unique[i]
            # when there is a nan we have 0/0 and this is always fine
            # thus set to the mean, since if slope is constant, 
            # so is the mean after division
            x, y = np.where(np.isnan(tmp))
            tmp[x,y] = 0
            for xi, yi in zip(x, y):
                # here need to be a bit careful if base is 3d or higher
                # since then there can be more than one zero
                nonzero = np.nonzero(tmp[xi])[0]
                mean = np.sum(tmp[xi, nonzero])/len(nonzero)
                tmp[xi,yi] = mean
            # we compare all cols to the first
            first_col = tmp[:,0]
            zeros = tmp.T-first_col
            zeros = zeros.T
            multiples = np.where(np.all(zeros == np.zeros(zeros.shape[1]), axis=1))
            # we look for zero rows, as in those we have a constant slope for all entries in our vectors
            if len(multiples[0]) != 0:
                for m in multiples[0]:
                    #we require the slope to be positive
                    if first_col[m] > 0:
                        # unique[i] is the base but is m also the largest we have found so far?
                        if first_col[m] > 1 and first_col[m] > order[m]:
                            # is it integer?
                            if np.allclose(np.round(first_col[m])*unique[i], unique[m+i+1]):
                                order[m+1+i] = np.round(first_col[m])
                                basis_vectors[m+i+1] = unique[i]
                        elif first_col[m] < 1 and 1/first_col[m][0] > order[i]:
                            #unique[i] is a mutiple, but is unique[m+i+1] the base?
                            #is it integer?
                            if np.allclose(np.round(1/first_col[m])*unique[i], unique[m+i+1]):
                                order[m+i+1] = np.round(1/first_col[m])
                                basis_vectors[m+i+1] = unique[i]
        np.seterr(divide = 'warn')
        #np.seterr(invalid = 'warn')
        nunique = np.unique(basis_vectors, axis=0)
        nLetters = len(nunique)
        if nLetters > 26:
            logger.warning('I am running out of letters here, your base is messy.')

        # create an array which tracks the unique base vector and its multiplicity
        labels = np.zeros((len(sub_system),2))
        for v1, li in zip(sub_system, labels):
            for v2, j in zip(unique, range(len(unique))):
                # use all close since float, int comparison
                if np.allclose(v1, v2):
                    li[1] = order[j]
                    for v3, k in zip(nunique, range(nLetters)):
                        if np.array_equal(basis_vectors[j], v3):
                            li[0] = k
                            break
                    break
        labels = labels.astype(np.int16)
        label_tracker = np.zeros((nLetters, np.max(order).astype(np.int16)), dtype=np.int16)

        self.base_vars = []
        for l in labels:
            if l[0] < 26:
                self.base_vars += [var(ascii_uppercase[l[0]]+str(l[1])+str(label_tracker[l[0], l[1]-1]))]
                label_tracker[l[0], l[1]-1] += 1
            else:
                d = l[0] % 26
                n = int(l[0] / 26)
                self.base_vars += [var(ascii_lowercase[n]+ascii_uppercase[d]+str(l[1])+str(label_tracker[l[0], l[1]-1]))] 
                label_tracker[l[0], l[1]-1] += 1
        return self.base_vars

    def blow_down_coordinates(self, fibration=True, base=True, higher=False):
        r"""Blows down coordinates specified by the Boolean arguments.
        The blown down coordinates are internally set to 1.
        
        Parameters
        ----------
        fibration : bool, optional
            fibration blow ups, by default True
        base : bool, optional
            blow ups of the base rays, by default True
        higher : bool, optional
            blow ups of the higher multiplicity base rays, by default False
        
        Returns
        -------
        list
            list of new variables
        """
        # we blow down the fibration blow ups
        if fibration:
            for e in self.active_blow_ups:
                self.variables[e] = 1

        # we blow all but one coordinates that are projected onto the
        # same base rays down
        if base:
            for x, i in zip(self.variables, range(self.nVariables)):
                string = str(x)
                if i < 26:
                    if len(string) >=3:
                        if string[2] != str(0):
                            self.variables[i] = 1
                else:
                    if len(string) >=4:
                        if string[3] != str(0):
                            self.variables[i] = 1
        if higher:
            for x, i in zip(self.variables, range(self.nVariables)):
                string = str(x)
                if i <26:
                    if len(string) >=3:
                        if string[2] != str(0) or string[1] != str(1):
                            self.variables[i] = 1
                else:
                    if len(string) >=4:
                        if string[3] != str(0) or string[2] != str(1):
                            self.variables[i] = 1

        return self.variables

    def _set_base(self, base):
        r"""Sets the base of our fibration.
        Introduces base coordinates by taking into count the projection.
        
        Parameters
        ----------
        base : int_array[nBase_coordinates]
            all base coordinates
        """
        # set base
        self.active_base = np.array(base)
    
    def set_blow_ups(self, indices):
        r"""specifies blow up coordinates
        
        Parameters
        ----------
        indices : int_array[nBlow_up_coordinates]
            all blow up coordinates
        """
        # set base
        self.active_blow_ups = np.append(self.active_blow_ups, indices).astype(np.int16)
        self.blow_up_vars = list(var('e%d' % i) for i in range(1,len(self.active_blow_ups)+1))

    def clear_fibration(self):
        r"""Resets all active calculation to the base Polytope
        """
        self.active_fibration = np.array([])
        self.active_fibration_type = 'not defined'
        self.active_base = np.array([])
        self.active_blow_ups = np.array([])
        self.f = 0
        self.g = 0
        self.Delta = 0
        self.s = {}
        self.d = {}
        self.base_vars = []
        logger.info('All active fibrations cleared.')

    def give_current_fibration(self):
        r"""Prints all active results into the console
        """
        logger.info('Current active fibration is of type: {}.'.format(self.active_fibration_type))
        logger.info('The defining ambient coordinates have the indices {}.'.format(self.active_fibration))
        logger.info('The base space is given by the following coordinates {}.'.format(self.active_base))
        logger.info('These {} coordinates can be rescaled to 1.'.format(self.active_blow_ups))

    def set_singular_ring(self, coeff='d'):
        r"""Specifies the singular ring.
        This function needs to be called BEFORE doing any 
        symbolic calculations and AFTER specifying the active
        fibration.

        Use coeff to specify the variables and coefficients you want 
        to work with.

        Options
        -------
        's': s_i coefficients
        'd': refined (factorized) d_i coefficients
        'dv': d coefficients are promoted to variables
        'all': s_i, d_i are promoted to variables and c_i coeff
        'c': c_i coefficients
            Working with cmoduli is very Memory constraint and not recommended.

        Parameters
        ----------
        coeff : str , optional
            Pick 's', 'd', 'dv', 'all' or 'c', default = 'c'.
        
        Returns
        -------
        singular_exprs
            singular ring with nMonomial coefficients in
            nVariables char=0 and default dp sorting.
        """

        variables = []
        for x in self.variables:
            if x != 1:
                variables += [x]

        if coeff == 's':
            return singular.ring('(0, '+str(tuple(self.s_coeff))[1:], str(tuple(variables)) , 'dp')
        elif coeff == 'd':
            return singular.ring('(0, '+str(tuple(self.d_coeff))[1:], str(tuple(variables)) , 'dp')
        elif coeff == 'dv':
            return singular.ring(0, str(tuple(self.d_coeff))[:-1]+str(tuple(variables))[1:], 'dp')
        elif coeff == 'all':
            return singular.ring('(0, '+str(tuple(self.c_moduli))[1:], str(tuple(self.s_coeff))[:-1]+str(tuple(self.d_coeff))[1:-1]+str(tuple(variables))[1:], 'dp')
        elif coeff == 'c':
            return singular.ring('(0, '+str(tuple(self.c_moduli))[1:], str(tuple(variables)) , 'dp')
        
    def _make_surface_basis(self):
        r"""We use the Batyrev construction to find
        the surface from polytope and polar polytope.
        
        Returns
        -------
        int_array[nMonomial, nVariables]
            monomial basis of the surface.
        """
        # Batyrev
        monomial_basis = np.zeros((self.nMonomials, self.nVariables), dtype=np.int16)

        # introduce a new p_array for the case where sage adds the origin not at the end.
        p_array = np.delete(self.p_array, np.where(np.all(self.p_array == 0, axis=-1))[0], 0)
        for i in range(self.nMonomials):
            for j in range(self.nVariables):
                monomial_basis[i][j] = np.dot(p_array[j], self.pp_array[i])+1

        return monomial_basis
    
    def monomial_factorize(self, basis):
        r"""Checks if a polynomial, given in terms of its 
        monomials can be factorized with respect to any of the
        variables.
        
        Parameters
        ----------
        basis : int_array[nTerms, nVariables]
            Monomial basis of the polynomial.
        
        Returns
        -------
        bool_array[nVariables]
            True if factorizable in the corresponding variable.
        """
        non_zero = np.absolute(np.sign(basis))
        factors = np.sum(non_zero, axis=0)
        return factors > len(non_zero)

    def find_subpolytopes(self):
        r"""Finds all subpolytopes.
        Only supported for 4d polytopes.
        A straightforward translation of the julia code to python
        from 1809.05160. The Julia code is due to Huang and Taylor 
        and can be found here:
            http://ctp.lns.mit.edu/wati/data/fibers/fibers.jl

        Returns
        -------
            nested list of vertices corresponding to the subpolytopes
        """
        s = np.zeros(3, dtype=np.int8)
        short = np.zeros((len(self.p_array), 3, self.ambient_dimension), dtype=np.int16)
        self._find_inner(short, s)
        #and type 1: P1xP1, 2:P2, 3:P112
        polar_subs = self._find_2dfiber(short, s)
        subs = []
        for entry in polar_subs:
            new_sub = np.array(self._find_full_subpolytopes(entry[0]))
            new = True
            for entry2 in subs:
                if entry2.shape == new_sub.shape:
                    if np.array_equal(entry2, new_sub):
                        new = False
                        break
            if new:
                subs += [new_sub]
        return subs

    def _find_full_subpolytopes(self, fib_vec):
        """give it a set of vectors found with find_subpolytopes"""
        full = []
        matrix = np.zeros((len(fib_vec)+1,4))
        for j in range(len(fib_vec)):
            matrix[j] += fib_vec[j]
        for v in self.p_array:
            matrix[-1] = -1*v
            r = np.linalg.matrix_rank(matrix)
            if r == 2:
                full += [v]
        return full

    def _find_inner(self, short, s):
        s[:] = [0,0,0]
        # we compute the inner product of all vertices and points
        for i in range(len(self.p_array)):
            m = np.max([np.dot(self.p_array[i], self.pp_array[j]) for j in range(len(self.pp_array))])
            # if the max is below 4 we have identified either
            # Imax = 2: p2,Imax = 1: p1p1 or Imax = 3: p112
            if m < 4 and m != 0:
                # and add the corresponding point to short
                short[s[m-1], m-1] += self.p_array[i]
                # we save the multiplicity in s
                s[m-1] += 1
        return None

    def _member(self, v, l, size, multiple=1):
        for i in range(size):
            if np.array_equal(v, multiple*l[i]):
                return True
        return False

    def _find_2dfiber(self, short, s):
        subpolytopes = []
        # we are given all points and their multiplicity in s
        # First we run over the P1P1 cases
        for i in range(s[0]):
            # We check if the opposite is in the same list, since P1P1
            if self._member(-1*short[i,0], short[:,0], s[0]):
                for j in range(i+1, s[0]):
                    # We check if any other point also has its opposite in the list
                    # and that this point is not the opposite itself
                    if (self._member(-1*short[j,0], short[:, 0], s[0]) and not
                            np.array_equal(short[j, 0], -1*short[i, 0])):
                        subpolytopes += [[[short[i,0], -1*short[i,0], short[j,0], -1*short[j,0]], 1]]
        # Next the P2 case; we run over all points which have a small inner product
        for i in range(s[1]):
            # we run over all other points in short
            for j in range(i+1, s[1]):
                # now we check if the sum of the two points
                # also has a short inner product of 1 or 2, then P2
                # there are three combinations, once the unit*unit and twice unit*diag
                if (self._member(-1*short[i, 1]-short[j, 1], short[:, 0], s[0]) or
                        self._member(-1*short[i, 1]-short[j, 1], short[:, 1], s[1])):
                    subpolytopes += [[[short[i,1], short[j,1], -1*short[i, 1]-short[j, 1]], 2]]
        # Finally P112 case; we again run over the combination of all points
        for i in range(s[2]):
            for j in range(i+1, s[2]):
                # we check if the sum has inner product = 1 with multiplicity 2
                if self._member(-1*short[i, 2]-short[j, 2], short[:, 0], s[0], 2):
                    subpolytopes += [[[short[i,2], short[j,2], -1*short[i, 2]-short[j, 2]], 3]]
        return subpolytopes

    def _find_cubic_fibrations(self):
        r"""Finds P2 ambient spaces in the surface basis.
        
        Returns
        -------
        int_array[nP2, 3]
            all P2s and their corresponding coordinates 
            that can be identified in the surface.
        """
        # we brute force run over all combinations of three variables
        # and see if they only occur as a cubic together
        fibrations = []
        found = True
        for i,j,k in it.combinations(range(self.nVariables), r=3):
            found = True
            for n in range(self.nMonomials):
                degree = self.monomial_basis[n][i]+self.monomial_basis[n][j]+self.monomial_basis[n][k]
                if degree != 3:
                    found = False
                    break
            if found:
                fibrations += [[i,j,k]]
        return np.array(fibrations)

    def _find_p112_fibrations(self):
        r"""Finds P112 ambient spaces in the surface basis.
        
        Returns
        -------
        int_array[nP112, 4]
            all P112s and their corresponding coordinates 
            that can be identified in the surface. The fourth
            coordinate is the blow up e.
        """
        # we brute force run over all combinations of three variables
        # and see if they only occur as a quadratic together
        fibrations = []
        found = True
        # need to take permutations, as we know change the weight of one variable
        for i,j,k,e in it.permutations(range(self.nVariables), r=int(4)):
            found = True
            # sometimes there are false positives with zero vector
            # so explicitly ask for it to not be here.
            skip = False
            for v in self.p_array[[i,j,k,e]]:
                if np.array_equal(v, np.zeros(len(v))):
                    skip = True
                    pass
            if skip:
                pass
            for n in range(self.nMonomials):
                degree = self.monomial_basis[n][i]+self.monomial_basis[n][j]+2*self.monomial_basis[n][k]
                # also check if the blow up is included
                degree_2 = self.monomial_basis[n][k]+self.monomial_basis[n][e]
                if degree != 4:
                    found = False
                    break
                if degree_2 != 2:
                    found = False
                    break
            if found:
                fibrations += [[i,j,k,e]]
        # the last variable k/w will always be the one coming with 2 degree
        return np.array(fibrations)

    def _find_p1xp1_fibrations(self):
        r"""Finds P1xP1 ambient spaces in the surface basis.
        
        Returns
        -------
        int_array[nP1xP1, 4]
            all P1xP1s and their corresponding coordinates 
            that can be identified in the surface.
        """
        # we brute force run over all combinations of 2x2 variables
        # and see if they only occur as a quadratics together
        fibrations = []
        found = True
        for i,j in it.combinations(range(self.nVariables), r=2):
            found = True
            for n in range(self.nMonomials):
                degree = self.monomial_basis[n][i]+self.monomial_basis[n][j]
                if degree != 2:
                    found = False
                    break
            if found:
                fibrations += [[i,j]]

        fibrations = np.array(fibrations)
        # now check if there are two disjoint sets.
        nVariables = np.unique(fibrations)
        if len(nVariables) < 4:
            return np.array([])

        # find all distinct P1xP1 pairs
        # the indices in fibration are already ordered according to 
        # combinations()
        distinct = []
        for p1 in fibrations:
            for p2 in fibrations:
                if not np.any(p1 == p2):
                    distinct += [[p1,p2]]

        return np.array(distinct)
    
    def monomials_to_string(self, monomials, variables, coeff):
        r"""Transforms a polynomial in monomial basis to a string,
        which can be read in to singular.
        Requires the variables and coefficients for each monomial
        as seperate arguments.
        
        Parameters
        ----------
        monomials : int_array[nTerms, nVariables]
            Monomials basis of the Polynomial
        variables : sage_vars_array[nVariabels]
            Sage variabels or in principle any list of strings
            representing the variables in the output string
        coeff : sage_vars_array[nTerms]
            Sage variabels or in principle any list of strings
            representing the coefficeints in the output string  

        Returns
        -------
        str_array[nTerms]
            All monomials as strings.
        """
        if monomials.shape[0] == 0:
            return '0'

        nMonomials, nVariables = monomials.shape
        singular_string = ''

        for i in range(nMonomials):
            if coeff[i] != 0:
                singular_string += str(coeff[i])
                for j in range(nVariables):
                    if monomials[i][j] != 0:
                        singular_string += '*'+str(variables[j])+'^'+str(monomials[i][j])
                if i != nMonomials-1:
                    singular_string += '+'

        return singular_string
    
    def find_s(self, monomial_basis=False):
        r"""Finds the coefficients in the defining equation of the
        ative fibration.

        Important: Requires P.set_singular_ring() to be called before.
        
        Parameters
        ----------
        monomial_basis : bool, optional
            True for a monomial_basis, by default False
        
        Returns
        -------
        singular_array[nTerms]
            A list of singular expressions for every coefficient s_i.
        
        Raises
        ------
        RuntimeError
            Gives error if no fibration has been specified previously.
        """
        fibration = False
        if self.active_fibration_type == 'cubic':
            # find cubic coefficients in monomial basis
            fibration = True
            s_monomials, s_monomials_coeff = self._find_cubic_s_monomials()
            
        if self.active_fibration_type == 'P1xP1':
            # find cubic coefficients in monomial basis
            fibration = True
            s_monomials, s_monomials_coeff = self._find_p1xp1_s_monomials()    
            
        if self.active_fibration_type == 'P112':
            # find cubic coefficients in monomial basis
            fibration = True
            s_monomials, s_monomials_coeff = self._find_p112_s_monomials()    
            
        if not fibration:
            logger.error("No fibration specified. Use polytope.set_fibration(fibration) first.")

        self.s = {'monomials': s_monomials, 'monomials_c': s_monomials_coeff}

        
        s_string = [self.monomials_to_string(s_monomials[i], self.variables, s_monomials_coeff[i])
                                for i in range(len(s_monomials))]

        self.s['string'] = s_string
        
        # if only monomial basis is requested
        if monomial_basis:
            return s_monomials, s_monomials_coeff
        # now return Singular expression.
        # this only works if the user has already started a singular session.
        # we always need to return singular evaluations, since the class
        # can not start its own singular kernel. At least I haven't figured out how.
        return [singular.new(si) for si in s_string]

    def find_d(self):
        r"""Finds the factorization of the s_i into d_i and 
        base variables.
        
        Returns
        -------
        list
            singular expressions for each di
        """
        if not self.s:
            self.find_s(monomial_basis=True)

        # reset d just in case.
        self.d = {}
        
        d_monomials = [np.copy(self.s['monomials'][i]) for i in range(len(self.s['monomials']))]
        d_string = ['' for _ in range(len(d_monomials))]
        d_factor = np.zeros((len(d_monomials), self.nVariables))
        for i in range(len(d_monomials)):
            if len(d_monomials[i]) == 0:
                d_string[i] = '0'
                continue
            # already write the d coefficient
            tmp = np.zeros(self.nVariables, dtype=np.int16)
            d_string[i] += str(self.d_coeff[i])
            # next do the factorization
            di = d_monomials[i]
            for v in range(self.nVariables):
                min = np.min(di[:,v])
                if min != 0:
                    tmp[v] += min
                    di[:,v] -= min
                    d_string[i] += '*'+str(self.variables[v])+'^'+str(min)
            d_factor[i] += tmp

        self.d['monomials_c'] = d_factor 
        self.d['monomials'] = d_monomials
        self.d['string'] = d_string

        return [singular.new(di) for di in d_string]

    def find_nd_base(self, ambient):
        r"""Given an ambient space of a fibration, finds 
        suitable base spaces of dim n.
        
        Only works for obvious ambients and trivial projections,
        i.e. looks at the coordinates, where the ambient is zero.

        It collects all those rays and returns them.

        Parameters
        ----------
        ambient : int_array[nFibration_coordinates]
            All fibration coordinates.
        
        Returns
        -------
        int_array[nBase, nBase_coordinates]
            All fitting base spaces.
        """

        # find if the fibration already lies on a plane aligned with one of the axis
        coordinates = np.where(np.sum(np.absolute(np.sign(self.p_array[ambient])), axis=0) == 0)[0]
        if coordinates.shape[0] == 0:
            return np.array([])
        # find every variables that is non trivial in the base.
        base = []
        for i in range(self.nVariables):
            # we check if the variable is not part of the fibration
            if i not in ambient:
                # we run over all perpendicular base space axis
                # and check if i contributes here
                # if it contributes somewhere we add
                if np.sum(self.p_array[i][coordinates]) != 0:
                    base += [i]
        return np.array(base)

    def _find_fibration_blow_ups(self, base_known = True):
        r"""Given an active fibration, returns all coordinates
        that are not part of the base and ambient fibration space.

        According to Paul, we can set them to 1 since they 
        correspond to blow ups in the fiber.

        Parameters
        ----------
        base_known : bool, optional
            If False: Looks for 2d reflexive subpolytopes,
            by default True
        
        Returns
        -------
        int_array[nBlow_up_coordinates]
            All coordinate \ {Ambient_fiber_coord, Base_coord}.
        """

        if self.active_base.shape[0] == 0:
            logger.error('Specify base first.')

        # find all coordinates which we can set to one, i.e
        # all not in basis and not in fibration
        if base_known:
            one = []
            for i in range(self.nVariables):
                if i not in self.active_base and i not in self.active_fibration:
                    one += [i]

            #self.active_blow_ups = np.array(one)
            return np.array(one)
        # we need to find the 2d reflexive polytope defined by
        # self.active_fibration. Since the fibration contains
        # at least three points we can find the hypersurface
        # and all points there on.

        #all_points = []
        #for 

    def find_fibrations(self, symbolic_output = False):
        r"""
        Finds fibrations and a possible set of suitable base coordinates.

        The ambient fibration coordinates are found by studying 
        the defining CY surface and its monomial degrees.

        The potential base coordinates are found by looking for all
        rays with non trivial charges in the direction
         with trivial entries for the fiber coordinates.
        
        Parameters
        ----------
        sybolic_output : bool, optional
            If True: return a symbolic dictionary. IMPLEMENT THIS
            
        Returns
        -------
        dict{'cubic': list, 'P1xP1': list, 'P112': list}
            A dictionary with a list of tuples of np.arrays.
            Each tuple consists of 
            (fibration_ambient_coordinates, suitable base spaces)
        """

        ambient_fibrations = []
        ambient_fibrations += [self._find_cubic_fibrations()]
        ambient_fibrations += [self._find_p1xp1_fibrations()]
        ambient_fibrations += [self._find_p112_fibrations()]

        d = {}
        d['cubic'] = [(f, self.find_nd_base(f)) for f in ambient_fibrations[0]]
        d['P1xP1'] = [(f, self.find_nd_base(f)) for f in ambient_fibrations[1]]
        d['P112'] = [(f, self.find_nd_base(f)) for f in ambient_fibrations[2]]

        return d

    def set_triangulation(self, triangulation = [], i=-1, smooth=False):
        r"""Sets active triangulation

        Parameters
        ----------
        triangulation : sage.triangulation
            obtained from a Point Collection.
            Better stick with intern notation

        i : integer
            index of triangulation in triangulationlist
            WSFpolytope uses topcom which changes order
            Only works first time being called

        smooth : bool
            if True requires triangulation to be smooth. default = False
        """
        # define TV
        if len(triangulation) == 0:
            if i == -1:
                if smooth:
                    logger.info('No triangulation specified, trying to find first smooth triangulation.')
                else:
                    logger.info('No triangulation specified, picking first triangulation.')
            else:
                logger.info('Picking {}-th triangulation.'.format(i))
            for j, t in enumerate(self.PC.triangulations()):
                if i >= 0 and j != i:
                    pass
                self.triangulation = t
                self.TV = ToricVariety(self.triangulation.fan(), str(self.variables)[1:-1])
                if smooth:
                    if self.TV.is_smooth():
                        logger.info('Smooth triangulation found.')
                        break
                else:
                    break
        else:
            self.triangulation = triangulation
            self.TV = ToricVariety(self.triangulation.fan(), str(self.variables)[1:-1])
            #j = -1
        if not self.TV.is_smooth():
            logger.warning('TV is not smooth.')
        
        self.HH = self.TV.cohomology_ring()
        self.c1 = self.HH(-self.TV.K())
        self.c2 = self.TV.Chern_class(2)

        #generator of cohomology
        #VB = TV.toric_divisor_group().gens()
        #D = [HH(VB[i]) for i in range(len(VB))]

        #generator of kählercone
        self.J = []
        for D in self.TV.Kaehler_cone().rays():
            self.J += [self.HH(D.lift())]

        #return j

    #def triangulate(self):
    #
    #    yield self.PC.triangulations()

    def find_kollar(self, triangulation = [], r_coeff = -1):
        r"""Finds Kollar divisors with positive coefficients for the 
        generator of the Kähler cone. Only works for 3-folds.

        

        Parameters
        ----------
        triangulation : sage.triangulation
            use the internal points to not mess with notation, by default []
        r_coeff : int, optional
            range of coefficients, by default -1 
            (if -1 only checks the J divisors spanning the Kähler cone)

        Returns
        -------
        list/arrays[int]
            list of divisors satisfying Kollar criteria.
        """

        # define TV
        if len(triangulation) != 0 or len(self.triangulation) == 0:
            self.set_triangulation(triangulation)

        Kollar = []
        # There are three necessary checks
        # run over the divisor basis

        if r_coeff > 0:
            tuples = it.product(range(r_coeff), repeat=len(self.J))
            n_conf = (r_coeff+1)**len(self.J)
            if n_conf > 5000:
                logger.warning('Warning: you are scanning over {} configurations.'.format(n_conf))
        else:
            tuples = np.eye(len(self.J), dtype=np.int)

        for coeff in tuples:

            K = 0
            for i, c in enumerate(coeff):
                K += c*self.J[i]
            
            if self.is_kollar(K):
                Kollar += [K]

        logger.info('The following divisors satisfy the three necessary Kollar conditions:\n {}'.format(Kollar))
        return Kollar

    def is_kollar(self, divisor):
        r"""Determines if a divisor satisfies the three Kollar criteria.
        1) D^3 = 0
        2) D * c2 =/= 0
        3) D^2*D_i >= 0
        Assumes that CICY is favourable.

        Parameters
        ----------
        divisor : sage.toric_divisor
            Divisor with integer coefficients for each divisor
            descending from the projective ambient spaces

        Returns
        -------
        bool
            True if it satisfies all criteria
        """
        
        #if len(triangulation) != 0:
        #    self.set_triangulation(triangulation)

        # D^3 = 0
        if self.TV.integrate(self.c1*divisor*divisor*divisor) != 0:
            return False
        
        # D * c2 =/= 0
        if self.TV.integrate(self.c1*divisor*self.c2) == 0:
            return False

        # D^2*D_i >= 0
        for d in self.J:
            if self.TV.integrate(self.c1*divisor*divisor*d) < 0:
                return False
        
        return True

    def triple_intersection(self, triangulation = []):
        """ give triangulation or not """
        # define TV; maybe generalize that so we dont have to always repeat.
        if len(triangulation) != 0 or self.triangulation == 0:
            self.set_triangulation(triangulation)

        self.triple = np.zeros((len(self.J), len(self.J), len(self.J)), dtype=np.int32)
        for i, j, k in it.product(range(len(self.J)), repeat=3):
            self.triple[i,j,k] += self.TV.integrate(self.c1*self.J[i]*self.J[j]*self.J[k])

        return self.triple

    def find_K(self, I, triple):
        r"""We compute:
        K^I = \sum_{i,j,k \in I} triple[i,j,k]

        Parameters
        ----------
        I : list
            list of Kähler indices send to inf
        triple : np.array[h11, h11, h11]
            triple intersection numbers

        Returns
        -------
        int
            K^I
        """
        K = 0
        for tuples in it.product(I, repeat=3):
            K += triple[tuples]
        return K

    def find_KI(self, I, triple):
        r"""We compute:
        K^I_I = \sum_{i,j \in I} triple[i,j,I]

        Parameters
        ----------
        I : list
            list of Kähler indices send to inf
        triple : np.array[h11, h11, h11]
            triple intersection numbers

        Returns
        -------
        np.array[h11]
            K^I_I
        """
        KI = np.zeros(len(triple))
        for i in range(len(triple)-len(I)):
            if i not in I:
                for tuples in it.product(I, repeat=2):
                    KI[i] += triple[tuples[0], tuples[1], i]
        return KI

    def find_KIJ(self, I, triple):
        r"""We compute:
        K^I_IJ = \sum_{i \in I} triple[i,I,J]

        Parameters
        ----------
        I : list
            list of Kähler indices send to inf
        triple : np.array[h11, h11, h11]
            triple intersection numbers

        Returns
        -------
        np.array[h11, h11]
            K^I_IJ
        """
        if len(I) == len(triple):
            return np.array(0)
        KIJ = np.zeros((len(triple), len(triple)))
        for i in range(len(triple)):
            for j in range(len(triple)):
                for k in I:
                    KIJ[i, j] += triple[k, i, j]
        return KIJ

    def find_type(self, I, triple):
        r"""We compute the type of sending I to inf

        Parameters
        ----------
        I : list
            list of Kähler indices send to inf
        triple : np.array[h11, h11, h11]
            triple intersection numbers

        Returns
        -------
        str
            type
        """
        rank_1 = np.linalg.matrix_rank(self.find_K(I, triple))
        rank_2 = np.linalg.matrix_rank(self.find_KI(I, triple))
        rank_3 = np.linalg.matrix_rank(self.find_KIJ(I, triple))
        if rank_1 == 0:
            if rank_2 == 1:
                logger.debug('Type III_{} with ({})'.format(rank_3-2, [rank_1, rank_2, rank_3]))
                return "III_"+str(rank_3-2)
            else:
                logger.debug('Type II_{} with ({})'.format(rank_3, [rank_1, rank_2, rank_3]))
                return "II_"+str(rank_3)
        else:
            logger.debug('Type IV_{} with ({})'.format(rank_3, [rank_1, rank_2, rank_3]))
            return "IV_"+str(rank_3)

    def enhancement_diagram(self,  triple, fname, alpha=0.5):
        r""" Computes enhancement diagramm as in 1910.02963.

        Parameters
        ----------
        triple : np.array(len(M.J), len(M.J), len(M.J))
            triple intersection number of current triangulation
        fname : str
            file name to save figure to. Use .png as matplolib .pdf is having issues with sage constants
        alpha : float, optional
            alpha value of connecting lines, by default 0.5

        Returns
        -------
        matplotlib.pyplot.ax
            plot of the enhancement diagram.
        """
        fig, ax = plt.subplots()
        x_coordinates = []
        y_coordinates = []
        first_node = "I_0"
        largest = int(comb(len(triple), int(len(triple)/2))/2)
        fig.set_size_inches(largest, self.h11)
        # create j_list for linear dependency
        involved_variables = sum(self.J).variables()
        j_list = np.zeros((len(self.J),len(involved_variables)))
        for i in range(len(self.J)):
            coeffs = self.J[i].lift().coefficients()
            for k, v1 in enumerate(self.J[i].variables()):
                for j, v2 in enumerate(involved_variables):
                    if v1 == v2:
                        j_list[i,j] = coeffs[k]
        #start generating points
        tuples = [[]]
        for i in range(self.h11+1):
            y_coordinates += [np.array(int(comb(len(triple),i))*[len(triple)-i])]
            x_new = np.array(range(int(comb(len(triple),i))))
            x_coordinates += [x_new-np.mean(x_new)]
            if i == 0:
                ax.scatter(x_coordinates[i], y_coordinates[i], color='blue')
                ax.annotate(first_node, (x_coordinates[i][0]+0.05, y_coordinates[i][0]+0.05), color='red')
            else:
                tuples += [list(it.combinations(range(len(triple)), i))]
                for j, t in enumerate(it.combinations(range(len(triple)), i)):
                    # check if linear dependent
                    if np.linalg.matrix_rank(j_list[np.array(t)]) == i:
                        # create good subset of triple
                        complement = np.delete(np.arange(len(self.J)), list(t))
                        for comp in it.combinations(complement, self.h11-len(t)):
                            indices = list(t)+list(comp)
                            tmp_rank = np.linalg.matrix_rank(j_list[indices])
                            if tmp_rank == self.h11:
                                new_triple = triple[indices]
                                new_triple = new_triple[:, indices]
                                new_triple = new_triple[:, :, indices]
                                str_type = self.find_type(np.arange(len(t)), new_triple)
                                break
                        if str_type[0:3] == "III":
                            ax.scatter(x_coordinates[i][j], y_coordinates[i][j], color='red')
                            ax.annotate(str_type, (x_coordinates[i][j]+0.05, y_coordinates[i][j]+0.05), color='red')
                        else:
                            ax.scatter(x_coordinates[i][j], y_coordinates[i][j], color='black')
                            ax.annotate(str_type, (x_coordinates[i][j]+0.05, y_coordinates[i][j]+0.05), color='black')
                        for l in range(len(x_coordinates[i-1])):
                            if len(tuples[i-1]) != 0:
                                if len(set(tuples[i-1][l]).intersection(set(tuples[i][j]))) == len(tuples[i-1][l]):
                                    if str_type[0:3] == "III":
                                        ax.plot([x_coordinates[i-1][l], x_coordinates[i][j]], [y_coordinates[i-1][l], y_coordinates[i][j]], color='red', alpha=alpha)
                                    else:
                                        ax.plot([x_coordinates[i-1][l], x_coordinates[i][j]], [y_coordinates[i-1][l], y_coordinates[i][j]], color='black', alpha=alpha)
                            else:
                                if str_type[0:3] == "III":
                                    ax.plot([x_coordinates[i-1][l], x_coordinates[i][j]], [y_coordinates[i-1][l], y_coordinates[i][j]], color='red', alpha=alpha)
                                else:
                                    ax.plot([x_coordinates[i-1][l], x_coordinates[i][j]], [y_coordinates[i-1][l], y_coordinates[i][j]], color='black', alpha=alpha)
                                    
        plt.axis('off')
        plt.savefig(fname)
        return ax

    def exists_type_III(self, triple):
        r"""Determines if there exists a type III in 
        the enhancement diagram.

        Parameters
        ----------
        triple : np.array(len(M.J), len(M.J))
            triple intersection numbers of current triangulation

        Returns
        -------
        bool
            True if type III exists
        """
        good_tuples = [[i] for i in range(len(triple))]
        # check for linear dependencies
        involved_variables = sum(self.J).variables()
        j_list = np.zeros((len(self.J),len(involved_variables)))
        for i in range(len(self.J)):
            coeffs = self.J[i].lift().coefficients()
            for k, v1 in enumerate(self.J[i].variables()):
                for j, v2 in enumerate(involved_variables):
                    if v1 == v2:
                        j_list[i,j] += coeffs[k]
        for i in range(1, self.h11+1):
            # scan over good tuples combinations
            tmp_tuples = []
            for t in good_tuples:
                if np.linalg.matrix_rank(j_list[np.array(t)]) == i:
                    str_type = self.find_type(t, triple)
                    if str_type[0:3] == "III":
                        return True
                    if str_type[0:3] == "II_":
                        tmp_tuples += [t]
            #massage tmp_tuples to good_tuples
            if len(tmp_tuples) == 0:
                # no more good tuples left
                return False
            good_tuples = []
            for t in tmp_tuples:
                for j in range(self.h11):
                    if not j in t:
                        good_tuples += [t+[j]]
            good_tuples = np.unique(np.sort(np.array(good_tuples), axis=-1), axis=0).tolist()
            if len(good_tuples) == 0:
                return False
        return False

    def trace_type_III(self, triple):
        good_tuples = [[i] for i in range(len(triple))]
        type_ii_trace = []
        type_iii_trace = []
        # check for linear dependencies
        involved_variables = sum(self.J).variables()
        j_list = np.zeros((len(self.J),len(involved_variables)))
        for i in range(len(self.J)):
            coeffs = self.J[i].lift().coefficients()
            for k, v1 in enumerate(self.J[i].variables()):
                for j, v2 in enumerate(involved_variables):
                    if v1 == v2:
                        j_list[i,j] += coeffs[k]
        for i in range(1, self.h11+1):
            # scan over good tuples combinations
            tmp_tuples = []
            for t in good_tuples:
                if np.linalg.matrix_rank(j_list[np.array(t)]) == i:
                    str_type = self.find_type(t, triple)
                    if str_type[0:3] == "III":
                        tmp_tuples += [t]
                        type_iii_trace += [t]
                    if str_type[0:3] == "II_":
                        tmp_tuples += [t]
                        type_ii_trace += [t]
            #massage tmp_tuples to good_tuples
            if len(tmp_tuples) == 0:
                # no more good tuples left
                return {'II': type_ii_trace, 'III': type_iii_trace}
            good_tuples = []
            for t in tmp_tuples:
                for j in range(self.h11):
                    if not j in t:
                        good_tuples += [t+[j]]
            good_tuples = np.unique(np.sort(np.array(good_tuples), axis=-1), axis=0).tolist()
            if len(good_tuples) == 0:
                return {'II': type_ii_trace, 'III': type_iii_trace}
        return {'II': type_ii_trace, 'III': type_iii_trace}

    def _sort_variables(self):
        r"""Sorts and relabels the internal variables
        array to represent Base and fiber coordinates.
        """
        if self.active_fibration.shape[0] == 0:
            logger.error("No active fibration.")
        
        e = 0
        f = 0
        b = 0
        for i in range(self.nVariables):
            if i in self.active_fibration:
                self.variables[i] = self.fibration_vars[f]
                f += 1
            elif i in self.active_blow_ups:
                self.variables[i] = self.blow_up_vars[e]
                e += 1
            elif i in self.active_base:
                self.variables[i] = self.base_vars[b]
                b += 1
            else:
                self.variables[i] = self.x_vars[i]

    def _find_cubic_s_monomials(self):
        r"""Determines all s_i coefficients for a cubic
        fibration in terms of monomials basis.

        The notation is consistent with
        1408.4808.
        
        Returns
        -------
        list(int_array[nMonomials])
            List of monomials for every coefficients.
        
        Raises
        ------
        RuntimeError
            The active fibration needs to be of Type cubic.
        """
        if self.active_fibration_type != 'cubic':
            logger.error("No active cubic fibration.")
        
        # we run over all monomials and save them in corresponding s1, s2, ...
        u = self.active_fibration[0]
        v = self.active_fibration[1]
        w = self.active_fibration[2]

        # make copy which only includes the s monomials
        s_monomials = np.copy(self.monomial_basis)
        s_monomials[:, u] = 0
        s_monomials[:, v] = 0
        s_monomials[:, w] = 0

        # set all fibration blow ups to one
        #if self.active_blow_ups.shape[0] != 0:
        #    for i in self.active_blow_ups:
        #        s_monomials[:, i] = 0

        s = [[] for _ in range(10)]
        s_mod = [[] for _ in range(10)]
        for i in range(self.nMonomials):
            # make consisten way with pauls notation
            cont = [self.monomial_basis[i][u], self.monomial_basis[i][v],
                                                self.monomial_basis[i][w]]
            # check with Pauls notation
            j = 0
            for k in range(len(self.cubic_notation)):
                if np.array_equal(self.cubic_notation[k], cont):
                    j = k
                    break
            # fill the s_j
            s[j] += [s_monomials[i]]
            s_mod[j] += [self.c_moduli[i]]
        s = [np.array(monomials, dtype=np.int16) for monomials in s]
        s_mod = [np.array(moduli, dtype=object) for moduli in s_mod]
        return s, s_mod

    def _find_p1xp1_s_monomials(self):
        r"""Determines all s_i coefficients for a P1xP1
        fibration in terms of monomials basis.

        The notation is consistent with
        1408.4808.
        
        Returns
        -------
        list(int_array[nMonomials])
            List of monomials for every coefficients.
        
        Raises
        ------
        RuntimeError
            The active fibration needs to be of Type P1xP1.
        """
        if self.active_fibration_type != 'P1xP1':
            logger.error("No active P1xP1 fibration.")
        
        # we run over all monomials and save them in corresponding s1, s2, ...
        u = self.active_fibration[0][0]
        v = self.active_fibration[0][1]
        w = self.active_fibration[1][0]
        t = self.active_fibration[1][1]
        
        # make copy which only includes the s monomials
        s_monomials = np.copy(self.monomial_basis)
        s_monomials[:, u] = 0
        s_monomials[:, v] = 0
        s_monomials[:, w] = 0
        s_monomials[:, t] = 0

        # set all fibration blow ups to one
        #if self.active_blow_ups.shape[0] != 0:
        #    for i in self.active_blow_ups:
        #        s_monomials[:, i] = 0

        # there are 9 terms, but to be consistent with the rest we want a 10d
        # array for the moment. where b4 in Pauls notation or in our s4 = 0. see 3.12
        nTerms = 10
        s = [[] for _ in range(nTerms)]
        s_mod = [[] for _ in range(nTerms)]
        for i in range(self.nMonomials):
            # make consisten way with pauls notation
            cont = [self.monomial_basis[i][u], self.monomial_basis[i][v],
                     self.monomial_basis[i][w], self.monomial_basis[i][t]]
            #Pauls notation
            j = 0
            for k in range(len(self.p1xp1_notation)):
                if np.array_equal(self.p1xp1_notation[k], cont):
                    j = k
                    break
            # fill the s_j
            s[j] += [s_monomials[i]]
            s_mod[j] += [self.c_moduli[i]]
        s = [np.array(monomials, dtype=np.int16) for monomials in s]
        s_mod = [np.array(moduli, dtype=object) for moduli in s_mod]
        return s, s_mod

    def _find_p112_s_monomials(self):
        r"""Determines all s_i coefficients for a P112
        fibration in terms of monomials basis.

        The notation is consistent with
        1408.4808.
        
        Returns
        -------
        list(int_array[nMonomials])
            List of monomials for every coefficients.
        
        Raises
        ------
        RuntimeError
            The active fibration needs to be of Type P112.
        """
        if self.active_fibration_type != 'P112':
            logger.error("No active P112 fibration.")
        
        # we run over all monomials and save them in corresponding s1, s2, ...
        u = self.active_fibration[0]
        v = self.active_fibration[1]
        w = self.active_fibration[2]

        # make copy which only includes the s monomials
        s_monomials = np.copy(self.monomial_basis)
        s_monomials[:, u] = 0
        s_monomials[:, v] = 0
        s_monomials[:, w] = 0
        s_monomials[:, self.p112_blow_up] = 0

        # set all fibration blow ups to one
        #if self.active_blow_ups.shape[0] != 0:
        #    for i in self.active_blow_ups:
        #        s_monomials[:, i] = 0

        # there are 9 terms, but to be consistent with the rest we want a 10d
        # array for the moment. where d10 in Pauls notation or in our s10 = 0. see 3.17
        nTerms = 10
        s = [[] for _ in range(nTerms)]
        s_mod = [[] for _ in range(nTerms)]
        for i in range(self.nMonomials):
            # make consisten way with pauls notation
            cont = [self.monomial_basis[i][u], self.monomial_basis[i][v],
                        self.monomial_basis[i][w], self.monomial_basis[i][self.p112_blow_up]]
            #Pauls notation
            j = 0
            for k in range(len(self.p112_notation)):
                if np.array_equal(self.p112_notation[k], cont):
                    j = k
                    break
            # fill the s_j
            s[j] += [s_monomials[i]]
            s_mod[j] += [self.c_moduli[i]]
        s = [np.array(monomials, dtype=np.int16) for monomials in s]
        s_mod = [np.array(moduli, dtype=object) for moduli in s_mod]
        return s, s_mod

    def find_f(self, s):
        r"""Given the coefficients s_i of an active fibration,
        determines f.
        
        Parameters
        ----------
        s : list(10singular_exprs)
            List of 10 singular epressions, for the coeff s_i.
        
        Returns
        -------
        singular_expr
            A singular polynomial.
        """
        #define the s
        s1 = s[0]
        s2 = s[1]
        s3 = s[2]
        s4 = s[3]
        s5 = s[4]
        s6 = s[5]
        s7 = s[6]
        s8 = s[7]
        s9 = s[8]
        s10 = s[9]

        if self.active_fibration_type == 'cubic':
            return self._find_cubic_f(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
        
        if self.active_fibration_type == 'P1xP1':
            return self._find_p112_f(s1, s2, s3, s5, s6, s7, s8, s9, s10)

        if self.active_fibration_type == 'P112':
            return self._find_p1xp1_f(s1, s2, s3, s4, s5, s6, s7, s8, s9)

    def find_g(self, s):
        r"""Given the coefficients s_i of an active fibration,
        determines g.
        
        Parameters
        ----------
        s : list(10singular_exprs)
            List of 10 singular epressions, for the coeff s_i.
        
        Returns
        -------
        singular_expr
            A singular polynomial.
        """
        #define the s
        s1 = s[0]
        s2 = s[1]
        s3 = s[2]
        s4 = s[3]
        s5 = s[4]
        s6 = s[5]
        s7 = s[6]
        s8 = s[7]
        s9 = s[8]
        s10 = s[9]

        if self.active_fibration_type == 'cubic':
            return self._find_cubic_g(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
        
        if self.active_fibration_type == 'P1xP1':
            return self._find_p112_g(s1, s2, s3, s5, s6, s7, s8, s9, s10)

        if self.active_fibration_type == 'P112':
            return self._find_p1xp1_g(s1, s2, s3, s4, s5, s6, s7, s8, s9)

        logger.error('Only cubic, P112, P1xP1 fibrations are implemented.')

    def _find_cubic_f(self, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):

        # copy from Paul
        self.f = (-1/48*s6^4 + 1/6*s5*s6^2*s7 - 1/3*s5^2*s7^2 - 1/2*s4*s5*s6*s8
                    + 1/6*s3*s6^2*s8 + 1/3*s3*s5*s7*s8 - 1/2*s2*s6*s7*s8 + s1*s7^2*s8
                    - 1/3*s3^2*s8^2 + s2*s4*s8^2 + s4*s5^2*s9 - 1/2*s3*s5*s6*s9 
                    + 1/6*s2*s6^2*s9 + 1/3*s2*s5*s7*s9 - 1/2*s1*s6*s7*s9 
                    + 1/3*s2*s3*s8*s9 - 3*s1*s4*s8*s9 - 1/3*s2^2*s9^2 + s1*s3*s9^2
                    + s3^2*s5*s10 - 3*s2*s4*s5*s10 - 1/2*s2*s3*s6*s10
                    + 9/2*s1*s4*s6*s10 + s2^2*s7*s10 - 3*s1*s3*s7*s10)
        return self.f
    
    def _find_cubic_g(self, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
        
        # copy from Paul
        self.g =  (1/864*s6^6 - 1/72*s5*s6^4*s7 + 1/18*s5^2*s6^2*s7^2 - 
                   2/27*s5^3*s7^3 + 1/24*s4*s5*s6^3*s8 - 1/72*s3*s6^4*s8 - 
                   1/6*s4*s5^2*s6*s7*s8 + 1/36*s3*s5*s6^2*s7*s8 + 
                   1/24*s2*s6^3*s7*s8 + 1/9*s3*s5^2*s7^2*s8 - 1/6*s2*s5*s6*s7^2*s8 - 
                   1/12*s1*s6^2*s7^2*s8 + 1/3*s1*s5*s7^3*s8 + 1/4*s4^2*s5^2*s8^2 - 
                   1/6*s3*s4*s5*s6*s8^2 + 1/18*s3^2*s6^2*s8^2 - 
                   1/12*s2*s4*s6^2*s8^2 + 1/9*s3^2*s5*s7*s8^2 - 
                   1/6*s2*s4*s5*s7*s8^2 - 1/6*s2*s3*s6*s7*s8^2 + s1*s4*s6*s7*s8^2 + 
                   1/4*s2^2*s7^2*s8^2 - 2/3*s1*s3*s7^2*s8^2 - 2/27*s3^3*s8^3 + 
                   1/3*s2*s3*s4*s8^3 - s1*s4^2*s8^3 - 1/12*s4*s5^2*s6^2*s9 + 
                   1/24*s3*s5*s6^3*s9 - 1/72*s2*s6^4*s9 + 1/3*s4*s5^3*s7*s9 - 
                   1/6*s3*s5^2*s6*s7*s9 + 1/36*s2*s5*s6^2*s7*s9 + 
                   1/24*s1*s6^3*s7*s9 + 1/9*s2*s5^2*s7^2*s9 - 1/6*s1*s5*s6*s7^2*s9 - 
                   1/6*s3*s4*s5^2*s8*s9 - 1/6*s3^2*s5*s6*s8*s9 + 
                   5/6*s2*s4*s5*s6*s8*s9 + 1/36*s2*s3*s6^2*s8*s9 - 
                   3/4*s1*s4*s6^2*s8*s9 + 1/18*s2*s3*s5*s7*s8*s9 - 
                   3/2*s1*s4*s5*s7*s8*s9 - 1/6*s2^2*s6*s7*s8*s9 + 
                   5/6*s1*s3*s6*s7*s8*s9 - 1/6*s1*s2*s7^2*s8*s9 + 
                   1/9*s2*s3^2*s8^2*s9 - 2/3*s2^2*s4*s8^2*s9 + s1*s3*s4*s8^2*s9 + 
                   1/4*s3^2*s5^2*s9^2 - 2/3*s2*s4*s5^2*s9^2 - 1/6*s2*s3*s5*s6*s9^2 + 
                   s1*s4*s5*s6*s9^2 + 1/18*s2^2*s6^2*s9^2 - 1/12*s1*s3*s6^2*s9^2 + 
                   1/9*s2^2*s5*s7*s9^2 - 1/6*s1*s3*s5*s7*s9^2 - 
                   1/6*s1*s2*s6*s7*s9^2 + 1/4*s1^2*s7^2*s9^2 + 1/9*s2^2*s3*s8*s9^2 - 
                   2/3*s1*s3^2*s8*s9^2 + s1*s2*s4*s8*s9^2 - 2/27*s2^3*s9^3 + 
                   1/3*s1*s2*s3*s9^3 - s1^2*s4*s9^3 - s4^2*s5^3*s10 + 
                   s3*s4*s5^2*s6*s10 - 1/12*s3^2*s5*s6^2*s10 - 
                   3/4*s2*s4*s5*s6^2*s10 + 1/24*s2*s3*s6^3*s10 + 5/8*s1*s4*s6^3*s10 - 
                   2/3*s3^2*s5^2*s7*s10 + s2*s4*s5^2*s7*s10 + 
                   5/6*s2*s3*s5*s6*s7*s10 - 3/2*s1*s4*s5*s6*s7*s10 - 
                   1/12*s2^2*s6^2*s7*s10 - 3/4*s1*s3*s6^2*s7*s10 - 
                   2/3*s2^2*s5*s7^2*s10 + s1*s3*s5*s7^2*s10 + s1*s2*s6*s7^2*s10 - 
                   s1^2*s7^3*s10 + 1/3*s3^3*s5*s8*s10 - 3/2*s2*s3*s4*s5*s8*s10 + 
                   9/2*s1*s4^2*s5*s8*s10 - 1/6*s2*s3^2*s6*s8*s10 + 
                   s2^2*s4*s6*s8*s10 - 3/2*s1*s3*s4*s6*s8*s10 - 
                   1/6*s2^2*s3*s7*s8*s10 + s1*s3^2*s7*s8*s10 - 
                   3/2*s1*s2*s4*s7*s8*s10 - 1/6*s2*s3^2*s5*s9*s10 + 
                   s2^2*s4*s5*s9*s10 - 3/2*s1*s3*s4*s5*s9*s10 - 
                   1/6*s2^2*s3*s6*s9*s10 + s1*s3^2*s6*s9*s10 - 
                   3/2*s1*s2*s4*s6*s9*s10 + 1/3*s2^3*s7*s9*s10 - 
                   3/2*s1*s2*s3*s7*s9*s10 + 9/2*s1^2*s4*s7*s9*s10 + 
                   1/4*s2^2*s3^2*s10^2 - s1*s3^3*s10^2 - s2^3*s4*s10^2 + 
                   9/2*s1*s2*s3*s4*s10^2 - 27/4*s1^2*s4^2*s10^2)
        return self.g
    
    def _find_p1xp1_f(self, b1, b2, b3, b5, b6, b7, b8, b9, b10):

        self.f = 1/48*(-(-4 * b1 * b10 + b6^2 - 4 *(b5 * b7 + b3 * b8 + b2 * b9))^2 + 
                    24 * (-b6 * (b10 * b2 * b5 + b2 * b7 * b8 + b3 * b5 * b9 + b1 * b7 * b9) + 
                        2 *(b10 * (b1 * b5 * b7 + b2^2 * b8 + b3 * (b5^2 - 4 * b1 * b8) + 
                            b1* b2 * b9) + b7 * (b1 * b7 * b8 + b2 * b5 * b9) + 
                            b3 *(b5* b7 *b8 + b2* b8* b9 + b1 *b9^2))))

        return self.f

    def _find_p1xp1_g(self, b1, b2, b3, b5, b6, b7, b8, b9, b10):

        self.g = 1/864*((-4 *b1* b10 + b6^2 - 4* (b5* b7 + b3 *b8 + b2* b9))^3 - 
                    36 *(-4* b1* b10 + b6^2 - 
                        4 *(b5* b7 + b3* b8 + b2* b9))*(-b6* (b10 *b2 *b5 + b2* b7* b8 + 
                            b3* b5 *b9 + b1* b7 *b9) + 
                        2 *(b10* (b1* b5* b7 + b2^2* b8 + b3 *(b5^2 - 4* b1* b8) + 
                            b1 *b2 *b9) + b7* (b1 *b7* b8 + b2* b5* b9) + 
                            b3* (b5* b7* b8 + b2* b8* b9 + b1* b9^2))) + 
                    216* ((b10* b2* b5 + b2* b7 *b8 + b3* b5* b9 + b1 *b7* b9)^2 - 
                        4 *(b2* b3 *b5 *b7* b8* b9 + 
                            b1^2 *b10* (-4* b10 *b3* b8 + b7^2* b8 + b3* b9^2) + 
                            b10* (b3^2* b5^2* b8 + b2^2* b5* b7* b8 + 
                            b2* b3* (-b5* b6* b8* + b2* b8^2 + b5^2* b9)) + 
                            b1* (b10^2 *(b3* b5^2 + b2^2* b8) + b2* b7^2* b8* b9 + 
                            b3^2* b8* b9^2 + b3* b7* (b7* b8^2 - b6* b8* b9 + b5* b9^2) + 
                            b10* (-4* b3^2* b8^2 + b3* b6* (b6* b8 - b5* b9) + 
                                b2* b7* (-b6* b8 + b5* b9))))))
        return self.g

    def _find_p112_f(self, s1, s2, s3, s4, s5, s6, s7, s8, s9):


        self.f = 1/48*(-24* s9* (-2* s5* s6^2 + s4 *s6 *s7 - 2* s3* s6* s8 + s2* s7* s8 - 
                    2* s1* s8^2 - 2* s2* s4* s9 + 8* s1* s5* s9) - (s7^2 - 
                    4* (s6* s8 + s3* s9))^2)

        return self.f

    def _find_p112_g(self, s1, s2, s3, s4, s5, s6, s7, s8, s9):

        self.g = 1/864*(36* s9* (-2* s5* s6^2 + s4* s6* s7 - 2* s3* s6* s8 + s2* s7* s8 - 
                2* s1* s8^2 - 2* s2* s4* s9 + 8* s1* s5* s9) *(s7^2 - 
                4* (s6* s8 + s3* s9)) + (s7^2 - 4* (s6* s8 + s3* s9))^3 + 
                216* s9^2* (4* s2* s5* s6* s7 - 4* s1* s5* s7^2 + s2^2* s8^2 + 
                    s4* (-2* s2* s6* s8 + 4* s1* s7* s8) - 4* s2^2* s5* s9 + 
                    s4^2* (s6^2 - 4* s1* s9) - 
                    4* s3* (s5* s6^2 + s1* s8^2 - 4* s1* s5* s9)))

        return self.g


    def find_discriminant(self, f=None, g=None):
        if f is None:
            f = self.f
        if g is None:
            g = self.g
        self.Delta = 4*self.f^3+27*self.g^2
        return self.Delta
