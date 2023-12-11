"""Plot all outputs"""
import compute_all as c
import advection as a
import burgers_new as b
import kdv_new as k
import kdv2waves as k22
import special_kdv as s
import convergence as convergence
import kdvdualwaves as du

a.plot_advect(c.u_a)
b.plot_burgers(c.u_burgers)
k.plot_kdv(c.u_kdv)
k.plot_kdv(c.u_kmu)
k22.plot_kdv2(c.u_kdv2)
s.plot_special(c.u_s)
convergence.conv()
du.dual()