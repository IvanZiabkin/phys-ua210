#Problem 1

def f(x): 
    return x**4-2*x+1
N = 10
a=0
b=2
h=(b-a)/N
s=0.5*f(a)+0.5*f(b)
for k in range(1, N):
    s+=f(a+k*h)
end10 = s*h

N = 20
h=(b-a)/N
s=0.5*f(a)+0.5*f(b)
for k in range(1, N):
    s+=f(a+k*h)
end20 = s*h
print(end10)
print(end20)
print((1/3)*(end20-end10))