#!/usr/bin/env python3

from sympy import *
x,y,xb,yb,zb = symbols('x y xb yb zb')
G = 1/sqrt((x-xb)**2+(y-yb)**2+zb**2)
for q in range(8):
    for p in range(8-q):
        print('(', p,',',q,'): ',diff(G,x,p,y,q).subs(x,0).subs(y,0))
