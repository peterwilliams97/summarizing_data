from __future__ import division
import time
import numpy as np
            
def test_mean(func, num_elems, int_type=None):
    """Test performance of average calculating function mean
        on a numpy array of num_elems and dtype int_type.
    """    
    arr = np.arange(num_elems, dtype=int_type)
    mean = (num_elems-1)/2
    t0 = time.time()
    n = 1
    while True:
        res = func(arr)
        dt = time.time() - t0
        if dt >= 1.0:
            break
        n = n + 1    
    print '%s, size=%d, type=%s: average time=%g sec' % (
        func.func_name, arr.size, arr.dtype, dt/n) 
    if res != mean:
        print '    *** res=%g != mean=%g' % (res, mean)

def fold_mean(it):
    tally = lambda x, y: (x[0]+y, x[1]+1)
    div = lambda x: x[0] / float(x[1])
    mean_f = lambda it: div(reduce(tally, it, (0, 0)))
    return mean_f(it)

def lazy_mean(some_iter):
    i, tot = 0, 0.0
    for s in some_iter:
        tot += s
        i += 1
    return tot / i

def numpy_mean(arr):
    return np.average(arr)   
  
for int_type in 'int64', 'int32': 
    for num_elems in 5000,50000,500000,5000000:
        print '-' * 80
        for func in numpy_mean,lazy_mean,fold_mean:
            test_mean(func, num_elems, int_type)
    print '=' * 80        
       
     
