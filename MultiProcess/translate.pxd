

import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t INT_t

cdef class Translation:
    cdef list vocab
    cdef INT_t [:] tgt
    cdef INT_t [:] output
    cdef INT_t last_word_id
    cdef INT_t len_output
    
    