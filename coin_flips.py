from __future__ import division
import random, math
import numpy as np
from scipy import stats

DEBUG = False

def get_mode(arr, num_elems):

    int_mode = stats.mode(arr)[0][0]
    hist,bin_edges = np.histogram(arr, bins=num_elems+1, range=[-0.5,num_elems+0.5])
    #print num_elems
    #print hist
    #print bin_edges
    assert len(hist) == num_elems + 1
    assert len(bin_edges) == num_elems + 2

    mode_index = -1
    start = bin_edges[0]
    for i, end in enumerate(bin_edges[1:]):
        if start <= int_mode < end:
            mode_index = i
            break
        start = end    
    assert mode_index >= 0
    
    numerator = float(mode_index)
    denominator = 1.0
    rfac = lfac = 0.0
    if mode_index > 0:
        lfac = float(hist[mode_index -1]) / float(hist[mode_index]) 
        #lfac = lfac ** 2
        numerator += float(mode_index -1) * lfac **2
        denominator += lfac    
    if mode_index < num_elems:
        rfac = float(hist[mode_index +1]) / float(hist[mode_index]) 
        #rfac = rfac ** 2
        numerator += float(mode_index +1) * rfac
        denominator += rfac    
    
    mode = numerator/denominator
    #print int_mode, mode, (mode_index -1,lfac), (mode_index, 1.0), (mode_index +1,rfac),
    
    #return mode
    return (mode_index-1,lfac), (mode_index,1.0), (mode_index+1,rfac)
    
def get_raw_pred(r):
    return (2.0 ** (r+1)) - 2.0    
    
def get_raw_run(pred):    
    """p = (2 ** (r+1)) - 2
       r = log(p + 2, 2) - 1   
    """
    return math.log(pred + 2.0, 2) - 1.0
    
def get_pred(cpts):
    #return get_raw_pred(sum(f * r for r,f in cpts)/sum(f for r,_ in cpts))
    return sum(f * get_raw_pred(r) for r,f in cpts)/sum(f for r,_ in cpts)  
    
    
def throw_coins(n):
    return np.random.randint(2, size=(n))
    
def longest_heads_run(coins):
    # 1 = heads, 0 = tails
    n = coins.shape[0]
    # +1 = start of run of heads
    # -1 = start of run of tails
    #  0 = continuation of run
    diffs = np.zeros(n)
    diffs[0] = 2*coins[0]-1
    diffs[1:] = np.diff(coins)
    
    if DEBUG:
        print n
        print 'coins', coins, coins.shape
        print 'diffs', diffs, diffs.shape
    
    heads_starts = np.where(diffs == 1)[0]
    # heads runs end at tails runs
    heads_ends = np.where(diffs == -1)[0]
    
    if DEBUG:
        print 'heads_starts', heads_starts, heads_starts.shape
        print 'heads_ends  ', heads_ends, heads_ends.shape
     
    if heads_ends.shape[0] == 0:
        longest = n
    elif heads_starts.shape[0] == 0:
         longest = 0    
    else:
        # We have some heads and some tails
        # if coins[0]==1 then runs are heads,tails
        # if coins[0]==0 then runs are tails,heads 
        if coins[0] == 0:
            heads_ends = heads_ends[1:]
            if DEBUG:
                print 'heads_ends  ', heads_ends, heads_ends.shape
          
        assert heads_starts.shape[0] > 0
                
        if heads_ends.shape[0] == 0 or heads_ends[-1] < heads_starts[-1]:
            heads_ends = np.append(heads_ends, [n])
        heads_runs = heads_ends - heads_starts
        if DEBUG:
            print 'heads_runs', heads_runs
        longest = np.max(heads_runs)
        
    if DEBUG:
        print 'longest', longest
        
        
    return longest
   
if False:   
    def test(a, p):
        r = longest_heads_run(np.array(a)) 
        print r
        assert r == p
        print '-' * 80
        
    test([0,0], 0)
    test([0,1], 1)
    test([1,0], 1)
    test([1,1], 2)
    test([0,1,1], 2)  
    exit()
   
def make_counts(n2, flips):
    """Repeat n2 times:
        For f in flips:
            throw coin f times
       Return counts matrix
        Row = one instance of flips: n rows
    """
    counts = np.zeros((n2, len(flips)))
    for i in range(n2):
        for j,f in enumerate(flips):
            coins = throw_coins(f)
            r = longest_heads_run(coins)
            counts[i,j] = r 
    return counts
    
W = 4
H = 4   
n = 20000
m = W * H

def num_flips(k):
    i = k // W
    j = k % W
    n = W * (2 ** i)
    nW = 2 ** (1.0/W) # n // W
    return int(n * (nW**j)) # j * nW
    #return 2 * 2 ** j

if False:
    for k in range(m):
        print num_flips(k)
        
if True:
    for k in range(10**8):
        counts = np.zeros((n))
        f = num_flips(k)
        for i in range(n):
            coins = throw_coins(f)
            counts[i] = longest_heads_run(coins)
        cpts = get_mode(counts, f)
        pred = get_pred(cpts)
        pred_r = get_raw_run(f)
        print '%4d,%6.3f,%6.3f,%s' % (f, 
            pred_r, math.log(2 * f, 2),
            ','.join('%d,%3d,%5.3f' % (i,get_raw_pred(i),x) for i,x in cpts))
      

    
counts = make_counts(n, [num_flips(j) for j in range(m)])    
print 'counts', counts.shape
for j in range(m):
    f = num_flips(j)
     
    # Use mode, not average, of jth column
    cpts = get_mode(counts[:,j], f)
    pred = get_pred(cpts)
    pred_r = get_raw_run(f)
    
    print '%4d: %6.3f %6.3f (%s)' % (f, 
        pred_r, math.log(2 * f, 2),
        '; '.join('%d,%3d,%5.3f' % (i,get_raw_pred(i),x) for i,x in cpts))
 
    # http://www.stanford.edu/~henrya/math151/Hw8Sol.pdf
    # http://people.ccmr.cornell.edu/~ginsparg/INFO295/mh.pdf
    # http://stats.stackexchange.com/questions/22938/expected-number-of-coin-tosses-to-get-n-consecutive-given-m-consecutive
    #pred = (1.0 - 0.5 ** r) / (0.5 ** r) / 0.5
    #pred =  ((0.5 ** -r) - 1) /0.5
    #pred = (2.0 ** (r+1)) - 2.0
    
    # http://www.csun.edu/~hcmth031/tspolr.pdf
    # r = log(n/q, 1/p) = log(2n, 2)
    # n = 2 ** r
    if False:
        print '%4d %7.1f %5.2f ' % (f, pred, (pred - f)/f) #, stats.mode(counts[:,j])

if True:    
    import matplotlib.pyplot as plt
    # assert m == 9    

    plt.figure(1)
    for j in range(m):    
        f = num_flips(j)
        r = stats.mode(counts[:,j])[0][0]
        plt.subplot(H, W, j+1)
        plt.hist(counts[:,j], bins=100)
        plt.xlabel('%d flips, mode=%.1f plot=%d' % (f, r, j+1))
    plt.show()

    plt.figure(2)
    for k in range(9):
        n = 100 * 2 **k
        counts = make_counts(n, [32])    
        r = stats.mode(counts[:,0])[0][0]
        plot_n = 331 + k
        print plot_n
        plt.subplot(plot_n)
        plt.hist(counts[:,0], bins=100)
        plt.xlabel('%d runs, %d flips, mode=%.1f plot=%d' % (n, f, r, plot_n))
    plt.show()

num_trials = 1000
results = {}
print '-' * 80
for k in range(9):
    n = 100 * 2**k
    results[k] = { 'n':n, 'correct': 0 }
    for trial in range(num_trials):
        counts = make_counts(n, [32])    
        r = stats.mode(counts[:,0])[0][0]
        pred = 2 ** (r+1) - 2
        #print pred,
        if pred == 30:
            results[k]['correct'] += 1
    print k, results[k]        
     
    

    