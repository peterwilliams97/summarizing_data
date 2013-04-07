from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time

if False:
    import numpy as np
    a,b,c = 1, np.arange(2), np.arange(4).reshape((2,2))
    for x in (a*2,b*2,c*2):  print x

def posix2dt(posix_string):
    """2013-03-27 09:43:34"""
    #print '!!', posix_string
    tm = time.strptime(posix_string, '%Y-%m-%d %H:%M:%S')
    #print '@@', tm
    dt = datetime.datetime(*tm[:6])
    #print '$$', dt
    return dt

if False:    
    print posix2dt.__doc__  
    print posix2dt(posix2dt.__doc__) 
    exit() 

if False:    
    log = pd.read_csv('print_logs_by_printer.csv',
        index_col=0,
        parse_dates=0,
        date_parser=posix2dt
    )
    print log.columns
    print '-' * 80
    #for c in log.columns: print log[c]
    #print    
    print log.index
    print log[['Total Pages', 'Total Color Pages']]
    print '-' * 80
        
    store = pd.HDFStore('peter.store')
    store['log'] = log
    del log

store = pd.HDFStore('peter.store')    
log = store['log']
print log

plt.plot(log.index, log['Total Pages'], label='Total Pages')
plt.plot(log.index, log['Total Color Pages'], label='Total Color Pages')
plt.legend()
plt.show()

  