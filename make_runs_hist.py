#from __future__ import division
from numba import jit,int32,autojit
import random, math
import numpy as np
from scipy import stats

#@autojit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result
    
arr = np.arange(5*2)
arr = arr.reshape((5, 2))
print arr
print sum2d(arr)    

def throw_coins(n):
    return np.random.randint(2, size=(n))
  
#@jit('int32(int32)')
#@autojit
def get_longest_heads_run(coins):
    """coins is array of 0,1 where 1 => heads, 0 => tails
    """
    #coins = np.random.randint(2, size=(n))
    n = coins.size
        
    # diffs contains
    #   +1 = start of run of heads
    #   -1 = start of run of tails
    #    0 = continuation of run
    diffs = np.zeros(n)
    diffs[0] = 2*coins[0]-1
    diffs[1:] = np.diff(coins)
    
    heads_starts = np.where(diffs == 1)[0]
    # heads runs end where tails runs start
    heads_ends = np.where(diffs == -1)[0]
     
    if heads_ends.size == 0:
        # All heads
        longest = n
    elif heads_starts.size == 0:
        # All tails
        longest = 0    
    else:
        # We have some heads and some tails
        # if coins[0]==1 then runs are heads,tails
        # if coins[0]==0 then runs are tails,heads 
        if coins[0] == 0:
            heads_ends = heads_ends[1:]
                
        if heads_ends.size == 0 or heads_ends[-1] < heads_starts[-1]:
            heads_ends = np.append(heads_ends, [n])
            
        heads_runs = heads_ends - heads_starts
        longest = np.max(heads_runs)
         
    return longest

coins = throw_coins(3)
longest = get_longest_heads_run(coins)

def get_hist(arr, num_elems):
    hist,bin_edges = np.histogram(arr, bins=num_elems+1, range=[-0.5,num_elems+0.5])
    assert len(hist) == num_elems + 1
    assert len(bin_edges) == num_elems + 2
    last = len(hist - 1)
    for last in range(len(hist), 1, -1):
        if hist[last-1] != 0:
            break
    #print num_elems
    #print arr
    #print hist        
    return hist[:last]
    
   
if False:   
    def test(a, p):
        r = get_longest_heads_run(np.array(a)) 
        print r
        assert r == p
        print '-' * 80
        
    test([0,0], 0)
    test([0,1], 1)
    test([1,0], 1)
    test([1,1], 2)
    test([0,1,1], 2)  
    exit()
   
#@autojit
def make_hist(n, m):
    for f in range(1, m):
        longest_runs = np.zeros((n), dtype=np.int32)
        for i in range(n):
            coins = throw_coins(f)
            longest_runs[i] = get_longest_heads_run(coins)
        hist = get_hist(longest_runs, f) 
        #print type(hist)
        aa = np.arange(hist.size)
        #print type(aa)
        number_heads = np.dot(aa, hist)
        print f, float(number_heads)/float(n),
        print hist.size,
        print hist
        #summary = ', '.join(['%d' % i for int(i) in list(hist)])
        #print '%4d, %.3f, %3d : %s' % (f,
        #    float(number_heads)/float(n), hist.size, ', '.join(['%d' % i for i in hist]))
       
N = 2000
M = 10
make_hist(N, M)

