import copy
import time

a = {'i':i for i in range(8000)}
stime = time.time()
b = copy.deepcopy(a)
total = time.time()-stime
print(total)
