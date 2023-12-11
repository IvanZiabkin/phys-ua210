"""Computes all data"""

import advection as a
import burgers_new as b
import kdv_new as k
import kdv2waves as k22
import special_kdv as s


u_burgers = b.compute_burgers()
u_kdv2 = k22.compute_kdv2()
u_kdv = k.compute_kdv(0.001)
u_kmu = k.compute_kdv(0.01)
u_s = s.compute_special()
u_a = a.compute_advect()



#note: two files not in here: convergence and dual. They are plotted directly.
#this is an artifact of the way we worked on the code and 
#it's too close to the deadline to untangle it, apologies