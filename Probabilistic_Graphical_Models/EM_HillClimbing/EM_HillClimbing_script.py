# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:01:13 2016

@author: Andres
"""
from __future__ import division
import numpy as np
import itertools
from collections import defaultdict
from collections import Counter
import pdb
import warnings
import matplotlib.pyplot as plt
import time
import copy
import math
import os
#to use in default constructor in Factor class
def creatEmptyCPT(numvars):
    rlz = list(itertools.product([0, 1], repeat=numvars))
    return {k:0 for k in rlz}

#create necessary classes
class Factor(object):
    #a factor is merely a cpt (represented by a python dictionary) with some additional attributes.
    def __init__(self, order, children=None, query='', cpt=None):
        if cpt is None:
            self.cpt = creatEmptyCPT(len(order))
        else:
            self.cpt = cpt
        if children is None:
            self.children = ''
        else:
            self.children = children  
        self.query = query
        self.parents = order.replace(query, '')
        self.order = order
        
    def __constructMap(self,oldorder,neworder):
        map_ = []
        for o in oldorder:
            map_.append(neworder.index(o))
        return map_
    
    def __mapKey(self,oldkey,keymap):
        #        pdb.set_trace()
        newkey = tuple(oldkey[i] for i in keymap)
        return newkey  
    
    def setorder(self,neworder):
        map_ = self.__constructMap(self.order,neworder)
        temp = {}
        for k in self.cpt.keys():
            newkey = self.__mapKey(k,map_)
            temp[newkey] = self.cpt[k]
        self.cpt = copy.deepcopy(temp)
        self.order = neworder
        
    def addparent(self,newparent):
        neworder = self.order + newparent
        return  Factor(order=neworder, children=self.children, query=self.query)
    
    def addchild(self,newchild):
        return Factor(order=self.order, query=self.query, children = self.children+newchild)
    
    def removeparent(self,whichparent):
        if whichparent not in self.parents:
            raise ValueError('Tried to remove node not in Factor.parents')
        neworder = self.order.replace(whichparent, '')
        return Factor(order=neworder, query=self.query, children=self.children)
    
    def removechild(self,whichchild):
        if whichchild not in self.children:
            raise ValueError('Tried to remove node not in Factor.children')
        newchildren = self.children.replace(whichchild, '')
        return Factor(order=self.order, query=self.query, children=newchildren)
    
    def clearpars(self):
        return Factor(children=self.children, query=self.query, order=self.order)        
        


        
        
#Functions for manipulating Factors
def listOrders(factorlist): # returns a list that containes the orders of each variable in factorlist
    try: return [f.order for f in factorlist]
    except:     pdb.set_trace() 
# a function to multiply factors:
def facProd(factorlist):
    outputorder,occurences = strUnion(listOrders(factorlist))
    scope_cardinality = len(outputorder)
    rlz = list(itertools.product([0, 1], repeat=scope_cardinality))
    product = {k:1 for k in rlz}
    for r in rlz:
        for indx,f in enumerate(factorlist):
            product[r] *= f.cpt[ tuple(r[x] for x in occurences[indx]) ]
    return Factor(cpt=product,order=outputorder)
# quotient of two factors
def facQuot(numerator,denominator):
    outputorder,occurences = strUnion(listOrders([numerator, denominator]))
    scope_cardinality = len(outputorder)
    rlz = list(itertools.product([0, 1], repeat=scope_cardinality))
    quotient = {}
    for r in rlz:
        num = numerator.cpt[ tuple(r[x] for x in occurences[0]) ]
        den = denominator.cpt[ tuple(r[x] for x in occurences[1]) ]
        if (num==0) & (den==0):
            warnings.warn('Numerator and Denominator both zero, setting quotient equal to 0')
            quotient[r] = 0
        elif den==0:
            warnings.warn('Division by zero: setting quotient equal to 1')
            quotient[r] = 1
        else:
            quotient[r] = num/float(den)
    return Factor(cpt=quotient,order=numerator.order, query=numerator.query, children=numerator.children)

def facSum(factorlist):
    # we assume all input factors have keys with the same order
    # note this function will return a counter in the cpt instead of a dictionary.
    # as far as i can tell, returning a Counter instead of a dict will not have bad ramifications for this particular app
    sum_ =  Counter({key:0 for key in factorlist[0].cpt.keys()})
    for f in factorlist:
        sum_ += Counter(f.cpt)
    return Factor( cpt=sum_, order=factorlist[0].order, query=factorlist[0].query, children=factorlist[0].children )      
    
def marginalize(factor,variable): #marginalize factor w.r.t. variable
    if variable not in factor.order:
        return factor
    varind = factor.order.index(variable)
    term1 = {tuple(k for i,k in enumerate(key) if i!=varind):value for key,value in factor.cpt.iteritems() if key[varind] == 1}
    term2 = {tuple(k for i,k in enumerate(key) if i!=varind):value for key,value in factor.cpt.iteritems() if key[varind] == 0}
    fac1 = Factor(cpt=term1, order=''.join([s for s in factor.order if s!=variable]), query = factor.query )
    fac2 = Factor(cpt=term2, order=''.join([s for s in factor.order if s!=variable]), query = factor.query )
    return facSum([fac1, fac2])      

#Miscellaneous functions for string set operations
def strUnion(stringlist): #returns union of input strings, and occurence list
    union = stringlist[0]
    occurences = [ range(len(stringlist[0])) ]
    #occurences is a list that has the same length as stringlist.
    #the kth element of occurences will be the indices in union
    # of each character in the kth element of stringlist
    for i,s in enumerate(stringlist):
        if i==0: 
            continue
        occind = []        
        for c in s:#for each character in the string s
            if c not in union:
                union += c
                occind.append(len(union)-1)
            else:
                occind.append(union.index(c))
        occurences.append(occind) 
    # for each element of string
    return union,occurences


def stringDiff(string1, string2): #sett difference between string1 and string2
    diff = ''
    for s1 in string1:
        if s1 not in string2:
            diff += s1
    for s2 in string2:
        if s2 not in string1:
            diff += s2
    return diff    
    
#Helper functions for E and M steps of EM algorithm
def get_xu_joint_given_o(fulljoint, query_node, parents, observed_nodes, varorder):
    union  = strUnion([query_node,parents,observed_nodes])[0]
    diff = stringDiff(union,varorder)
    numerator = copy.deepcopy(fulljoint)
    for d in diff:
        numerator = copy.deepcopy(marginalize(numerator, d))
    diff = stringDiff(observed_nodes, varorder)
    denominator = copy.deepcopy(fulljoint)
    for d in diff:
        denominator = copy.deepcopy(marginalize(denominator, d))
    return facQuot(numerator, denominator)

def getPseudoKey(realization, realizationorder, factororder):
    pseudokey = ()
    for x in factororder: 
        index = realizationorder.index(x)
        pseudokey += (realization[index],)
    return pseudokey
  
def formPseudocounts(N,THETA):
    pseudo = {}
    for i in THETA.keys():
        pseudo[i] = Factor(order=THETA[i].order, query=THETA[i].query) 
    return pseudo
        
def indexString(string, indices):
    return ''.join([string[i] for i in indices])
    
def whichComplete(datarow,dataorder):
    completeind = np.where(datarow!=-1)[0]            
    return indexString(dataorder,completeind), tuple(datarow[completeind])
              
def contradicts(key, keyorder, otherkey, otherorder):
#do the realizations contained in key contradict the realizations in otherkey?
    for i,k in enumerate(keyorder):
        if k in otherorder:
            index = otherorder.index(k)
            if key[i] != otherkey[index]:
                return True
    return False   

def match(facorder, tomatch, tomatchkey):
    # return only elements of key that correspond to elements of facorder to which elements of tomatch correspond, arranged accordingly
    newkey = ()
    neworder = ''
    for s in facorder:
        if s not in tomatch:
            continue
        index = tomatch.index(s)
        newkey += (tomatchkey[index],)
        neworder += s
    return newkey, neworder
        
    
########## functions for E-step and M-step, Kl divergence, log likelihood ############
def Estep(THETA,N,M, varorder):
    pseudocounts = formPseudocounts(N,THETA)
    joint = facProd(THETA.values())
    for m in xrange(M):
        #perform inference for each parameter
        curr_row = data1[m,:]
        completenames, completevals = whichComplete(datarow=curr_row, dataorder=varorder)
        #completenames are the names of the observed variables, competevals are their realizations
        for v in varorder:
            joint_xu_given_o = get_xu_joint_given_o(joint, THETA[v].query, THETA[v].parents, completenames, varorder)
            for k,i in joint_xu_given_o.cpt.iteritems():
                if not contradicts(k, joint_xu_given_o.order, completevals, completenames):
                    try:
                        xpar_key, xpar_order = match(THETA[v].order, joint_xu_given_o.order, k ) 
                    except:
                        pdb.set_trace()
                    pseudokey = getPseudoKey(xpar_key, xpar_order, THETA[v].order)
                    #xpar stands for "x and its parents"
                    pseudocounts[v].cpt[pseudokey] += i
    return pseudocounts
    
    
def Mstep(pseudocounts):
    est = {}
    for k,v in pseudocounts.iteritems():
        est[k] = facQuot(v, marginalize(v,v.query)) 
    return est


def KLdiv(P,Q): #P is true, Q is estimate  
    #P and Q are factor objects
    try:
        if P.order!=Q.order:
            raise ValueError('Order of variables in cpt dictionaries do not match. \
                             Use Factor.setorder()to match orders and then try again.')
    except:
        pdb.set_trace()
    div = 0
    for k in Q.cpt.keys():
        if (Q.cpt[k]==0) | (P.cpt[k]==0) :
            continue
        div += P.cpt[k]*math.log(P.cpt[k]/Q.cpt[k])
    return div
    
def LogLik(THETA, D , Dorder):
    LL = 0
    joint = facProd(THETA.values())
    for m in xrange(D.shape[0]):
        rlz = D[m,:]
        completenames, completevalues = whichComplete(rlz, Dorder)
        missing = stringDiff(completenames, Dorder)
        prod = copy.deepcopy(joint)
        for i in missing:
            prod = marginalize(prod, i)
        prod.setorder(completenames) #member function of Factor class
        LL += math.log(prod.cpt[completevalues])
        #        if LL==0:
        #            pdb.set_trace()
    return LL
            
################FUNCTIONS FOR HILL CLIMBING####################3
                    
def clearTHETA(graph): #reset parameters to zero while preserving structure
    cleared = copy.deepcopy(graph)
    for k in graph.keys():
        cleared[k] = graph[k].clearpars()
    return cleared
        
    
def estTHETA(data, dataorder, graph): 
    #graphs a dictionary of factors, keyed by each factors query node
    M = data.shape[0]   
    cnts = clearTHETA(graph)
    #one theta per node
    for m in xrange(M):
        for n in graph.keys():   
            #incremenet count for parameter to which data[:,m] corresponds
            rlz = match(graph[n].order, dataorder , data[m,:])[0]
            cnts[n].cpt[rlz] += 1
    return Mstep(cnts)

            
def scoreBIC(data, dataorder, graph):
    #graph is a dictionary of factors
    #suppose we have the links and corresponding parameters THETA
    M  = data.shape[0]
    firstterm = 0
    secondterm = 0
    for theta in graph.values():
        secondterm += math.log(M)/2 * 2**(len(theta.order)-1)
        #2**(len(THETA[n].order)-1) is the number of independent parameters in G(X_n^m)
        for m in xrange(M):
            datarlz = match(theta.order, dataorder, data[m,:])[0]
            firstterm += math.log(theta.cpt[datarlz])
    return firstterm - secondterm

def scoreML(data, dataorder, graph): #maximum likelihood score
    score = 0
    for m in xrange(data.shape[0]):
        for x in graph.values():
            datarlz = match(x.order, dataorder, data[m,:])[0]
            score += math.log(x.cpt[datarlz])
            #except: pdb.set_trace()
    return score

def replaceFactors(factordict, newfactors):
    temp = copy.deepcopy(factordict)
    if len(newfactors)==1:
        #if newfactors.query not in factordict.keys(): 
            #print 'Len==1: cannot replace key that is not in both dicts'
            #pdb.set_trace()
        temp[newfactors.query] = newfactors
        return factordict
    for f in newfactors:
        #if f.query not in factordict.keys(): 
            #print 'Len>1: cannot replace key that is not in both dicts'
            #pdb.set_trace()
        temp[f.query] = f
    return temp


def reverseFilialEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for c in queryfactor.children:
        f1 = THETA[c].removeparent(queryfactor.query).addchild(queryfactor.query)
        #except: pdb.set_trace()
        f2 = queryfactor.addparent(c).removechild(c)
        newthetas.append( estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))) )
    return newthetas

def reverseParentalEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for p in queryfactor.parents:
        f1 = queryfactor.removeparent(p).addchild(p)
        f2 = THETA[p].addparent(queryfactor.query).removechild(queryfactor.query)
        newthetas.append( estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))) )
    return newthetas

def removeFilialEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for c in queryfactor.children:
        f1 = THETA[c].removeparent(queryfactor.query)
        f2 = queryfactor.removechild(c)
        newthetas.append( estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))) )
    return newthetas

def removeParentalEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for p in queryfactor.parents:
        f1 = queryfactor.removeparent(p)
        f2 = THETA[p].removechild(queryfactor.query)
        newthetas.append( estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))) )
    return newthetas
    
def addFilialEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for node in [k for k in THETA.keys() if k!=queryfactor.query]:
        if node in queryfactor.children+queryfactor.parents:
            continue
        else:
            f1 = queryfactor.addchild(node)
            f2 = THETA[node].addparent(queryfactor.query)
            newthetas.append(estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))))
    return newthetas
   
def addParentalEdges(queryfactor, THETA, data, dataorder):
    newthetas = []
    for node in [k for k in THETA.keys() if k!=queryfactor.query]:
        if node in queryfactor.children+queryfactor.parents:
            continue
        else:
            f1 = queryfactor.addparent(node)
            f2 = THETA[node].addchild(queryfactor.query)
            newthetas.append(estTHETA(data, dataorder, replaceFactors(THETA, (f1,f2))))
    return newthetas
   
def recursegraph(facdict, currentnode, nodesvisited):
    if currentnode not in nodesvisited:
        if facdict[currentnode].children=='': return True
        nodesvisited += currentnode
        recursed =  []        
        for c in facdict[currentnode].children:
            recursed.append(recursegraph(facdict, c, nodesvisited))
        return True if all(r==True for r in recursed) else False         
    else: return False

def isdag(facdict):
    for node in facdict.keys():
        if recursegraph(facdict, node, '')==False:
            return False
    return True
    
def argmaxBIC(graphlist, data, dataorder, oldgraph,oldscore):
    max_G = oldgraph
    max_score = oldscore
    for G in graphlist:
        score = scoreBIC(data, dataorder, G)
        if score>max_score:
            max_G = G
            max_score = score
    return max_G, max_score
                
   
def getOptChange(node, THETA, data, dataorder, oldscore): #theta is a dictionary of factors key by each factors query node 
    THETAcands = [] #THETA candidates
    THETAcands += reverseParentalEdges(THETA[node], copy.deepcopy(THETA), data, dataorder)
    THETAcands += removeParentalEdges(THETA[node], copy.deepcopy(THETA), data, dataorder)
    THETAcands += addParentalEdges(THETA[node], copy.deepcopy(THETA), data, dataorder)
    THETAcands += reverseFilialEdges(THETA[node], copy.deepcopy(THETA), data, dataorder)
    THETAcands += removeFilialEdges(THETA[node], copy.deepcopy(THETA), data, dataorder)
    THETAcands += addFilialEdges(THETA[node], copy.deepcopy(THETA), data, dataorder )
    THETAcands = [theta for theta in THETAcands if isdag(theta)]
    if len(THETAcands)==0: pdb.set_trace()
    return argmaxBIC(THETAcands, data, dataorder, THETA, oldscore)
    
     
def Climb(data, dataorder, initialgraph, eps):
    G = estTHETA(data, dataorder, initialgraph)
    BIC = [scoreBIC(data, dataorder, G)]
    ML = [scoreML(data, dataorder, G)]
    oldbic = 99999999999999999999 #solely for the purposes of entering while loop
    t = 0
    while abs(BIC[-1]-oldbic)>eps: 
        t0 = time.time()
        oldbic = BIC[-1]
        for node in dataorder:
            G, bic = getOptChange(node, G, data, dataorder, BIC[-1])
            BIC.append(bic)
            ML.append(scoreML(data, dataorder,G))
        t += 1
        print 'Elapsed time:', time.time()-t0, 'Iteration:', t
    return BIC, ML, G
        
def displayStruct(graph): # display structure to see if Hill Climbing learned the correct structure
    for k,v in graph.iteritems():
        print k,'->', 'parents:', v.parents, ', children:', v.children 
    
##############################EM SCRIPT#########################################    

##CREATE INITIAL GUESS PARAMETERS
        # I recommend initializing the initial guesses only once and then uncommenting for further experimentation, as it takes up time.

np.random.seed(101)
rand1 = float(np.random.rand(1))
rand2 = float(np.random.rand(1))
rand31 = float(np.random.rand(1))
rand32 = float(np.random.rand(1))
rand33 = float(np.random.rand(1))
rand34 = float(np.random.rand(1))
rand41 = float(np.random.rand(1))
rand42 = float(np.random.rand(1))
rand51 = float(np.random.rand(1))
rand52 = float(np.random.rand(1))

# P(B)
cpt1 = {(1,):rand1, (0,):1-rand1}
theta1 = Factor(cpt=cpt1, order='b', query='b')

#P(E)
cpt2 = {(1,):rand2, (0,):1-rand2}
theta2 = Factor(cpt=cpt2, order='e', query='e')

# P(A|B,E)
num_vars = 3
rlz = list(itertools.product([0, 1], repeat=num_vars)) #rlz (realizations) is a list of binary tuples
#cardinality of each tuple in rlz is the cardinality of the scope of the corresponding cpt 
cpt3 = {}
for r in rlz:
    if r==(1,1,1):
        cpt3[r] = rand31
    if r==(1,0,1):
        cpt3[r] = rand32
    if r==(0,1,1):
        cpt3[r] = rand33
    if r==(0,0,1):
        cpt3[r] = rand34
    if r==(1,1,0):
        cpt3[r] = 1-rand31
    if r==(1,0,0):
        cpt3[r] = 1-rand32
    if r==(0,1,0):
        cpt3[r] = 1-rand33
    if r==(0,0,0):
        cpt3[r] = 1-rand34
theta3 = Factor(cpt=cpt3, order='bea', query='a')

# P(J|A)
num_vars = 2
rlz = list(itertools.product([0, 1], repeat=num_vars))
cpt4 = {}
for r in rlz:
    if r==(1,1):
        cpt4[r] = rand41
    if r==(0,1):
         cpt4[r] = rand42
    if r==(1,0):
         cpt4[r] = 1-rand41
    if r==(0,0):
         cpt4[r] = 1-rand42
theta4 = Factor(cpt=cpt4, order='aj', query='j')

# P(M|A)
num_vars = 2
rlz = list(itertools.product([0, 1], repeat=num_vars))
cpt5 = {}
for r in rlz:
    if r==(1,1):
        cpt5[r] = rand51
    if r==(0,1):
        cpt5[r] = rand52
    if r==(1,0):
        cpt5[r] = 1-rand51
    if r==(0,0):
        cpt5[r] = 1-rand52
theta5 = Factor(cpt=cpt5, order='am', query='m')

#CREATE TRUE PARAMETERS
cpt1 = {(1,):0.01, (0,):0.99} #P(B)
thetatrue1 = Factor(cpt=cpt1, order='b', query='b')

cpt2 = {(1,):0.02, (0,):0.98} #P(E)
thetatrue2 = Factor(cpt=cpt2, order='e', query='e')

num_vars = 3 # P(A|B,E)
rlz = list(itertools.product([0, 1], repeat=num_vars)) #rlz (realizations) is a list of binary tuples
#cardinality of each tuple in rlz is the cardinality of the scope of the corresponding cpt 
cpt3 = {}
for r in rlz:
    if r==(1,1,1):
        cpt3[r] = 0.95
    if r==(1,0,1):
        cpt3[r] = 0.94
    if r==(0,1,1):
        cpt3[r] = 0.29
    if r==(0,0,1):
        cpt3[r] = 0.001
    if r==(1,1,0):
        cpt3[r] = 0.05
    if r==(1,0,0):
        cpt3[r] = 0.06
    if r==(0,1,0):
        cpt3[r] = 0.71
    if r==(0,0,0):
        cpt3[r] = 0.999
thetatrue3 = Factor(cpt=cpt3, order='bea', query='a')

num_vars = 2 # P(J|A)
rlz = list(itertools.product([0, 1], repeat=num_vars))
cpt4 = {}
for r in rlz:
    if r==(1,1):
        cpt4[r] = 0.9
    if r==(0,1):
         cpt4[r] = 0.05
    if r==(1,0):
         cpt4[r] = 0.1
    if r==(0,0):
         cpt4[r] = 0.95
thetatrue4 = Factor(cpt=cpt4, order='aj', query='j')


num_vars = 2 # P(M|A)
rlz = list(itertools.product([0, 1], repeat=num_vars))
cpt5 = {}
for r in rlz:
    if r==(1,1):
        cpt5[r] = 0.7
    if r==(0,1):
        cpt5[r] = 0.01
    if r==(1,0):
        cpt5[r] = 0.3
    if r==(0,0):
        cpt5[r] = 0.99
thetatrue5 = Factor(cpt=cpt5, order='am', query='m')


####load data1#####
os.chdir('DIRECTORY WHERE DATA FILES LIVE')
data1 = np.loadtxt('data1.txt',delimiter=' ')
M,N = data1.shape
ordvars = 'beajm'
if M!=5000:
    raise ValueError('Sample Size Not equal to 5000')

#define parents, group factors, define query nodes, set true parameters
parents = ['', '', 'be', 'a', 'a']
THETA = {'b':theta1, 'e':theta2, 'a':theta3, 'j':theta4, 'm':theta5}
THETAtrue = {'b':thetatrue1, 'e':thetatrue2, 'a':thetatrue3, 'j':thetatrue4, 'm':thetatrue5}

#implement EM algorithm
T = 15
KL = np.zeros(T)
for t in xrange(T):
    t0  = time.time()
    pseudos = Estep(THETA, N, M, ordvars)
    THETA = Mstep(pseudos)
    for k in THETA.keys():
        P = THETAtrue[k]
        Q = THETA[k]
        KL[t] += KLdiv(P,Q)
    print time.time()-t0, KL[t]

   
plt.figure(7)
plt.clf()
plt.plot(np.arange(T)+1, KL, 'r' )
plt.xlabel('Iterations')
plt.ylabel('KL Divergence')
plt.suptitle('KL Divergence vs Iterations\ninitializing parameters randomly in U(0,1)')

loglik = 0
newloglik = 9999 
eps= 0.01
t = 0
T=10
LL = np.zeros(T)
#while abs(newloglik - loglik)>eps:
while t < T:
    t0 = time.time()
    loglik = newloglik
    pseudos = Estep(THETA, N, M, ordvars)
    THETA = Mstep(pseudos)
    newloglik = LogLik(THETA, data1, ordvars)
    LL[t] = newloglik
    t += 1
    print time.time() - t0, t, newloglik


##################HILL CLIMBING SCRIPT##################################   
#we represent the graph as a dictionary with values being the cpts (represented as Factor objects)
#and keys being the corresponding query nodes
queries = 'beajm'
orders = ['b', 'e', 'a', 'j', 'm']
children = ['', '', '', '', '']
initialguess =  {q:Factor(order=orders[i], query=q, children=children[i]) for i,q in enumerate(queries)} 
#initialize to a empty BN with no links between nodes  
eps = 0.01
data2 = np.loadtxt('data2.txt',delimiter=' ')
dataorder = 'beajm'
BIC, ML, G = Climb(data2, dataorder, initialguess, eps)    
displayStruct(G)

plt.figure(7)
plt.clf()

p1, = plt.plot(np.arange(len(BIC)), BIC,'r')
p2, = plt.plot(np.arange(len(ML)), ML, 'b')

itMarkerx = [1,6,11]
itMarkerBICy = [BIC[x] for x in itMarkerx]
itMarkerMLy = [ML[x] for x in itMarkerx]
plt.plot(itMarkerx, itMarkerBICy,'r*')
plt.plot(itMarkerx, itMarkerMLy,'b*')

plt.suptitle('Hill Climbing Algorithm Results')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.legend([p1,p2],['BIC','Maximum Likelihood'], loc=0)  

