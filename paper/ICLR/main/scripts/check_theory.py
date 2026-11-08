"""独立有限枚举检查；不将数值检查冒充定理的形式化证明。"""
from functools import lru_cache
from itertools import product
from math import comb

def tail(n,p,t):
 return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(max(0,t),n+1))

for B in range(2,13):
 for n in range(1,B):
  for p in [.1,.4,.6,.9]:
   r=B-n
   mixed=sum(comb(n,k)*p**k*(1-p)**(n-k)*tail(r,p,B//2+1-k) for k in range(n+1))
   assert abs(mixed-tail(B,p,B//2+1))<1e-12

loss=.4**4*tail(28,.6,17)
assert loss>0
print(f'Binary counterexample: SC={tail(32,.6,17):.9f}, PC={tail(32,.6,17)-loss:.9f}, loss={loss:.9f}')

# Three-category history-dependent kernels. Category 0 is correct, ties lose.
J=5
base=(.55,.3,.15)
def kernel(h):
 if not h:return base
 return (.8,.1,.1) if h[-1]==0 else (.2,.65,.15)
def success(h):return float(h.count(0)>max(h.count(1),h.count(2)))
@lru_cache(None)
def value(h,pc=False):
 if len(h)==J:return success(h)
 p=kernel(h) if pc else base
 return sum(p[a]*value(h+(a,),pc) for a in range(3))
states={():1.};delta=0.;tv=0.
for j in range(J):
 nxt={}
 for h,mass in states.items():
  p=kernel(h)
  delta+=mass*sum((p[a]-base[a])*value(h+(a,)) for a in range(3))
  tv+=mass*sum(abs(p[a]-base[a]) for a in range(3))/2
  for a in range(3):nxt[h+(a,)]=mass*p[a]
 states=nxt
actual=value((),True)-value(())
assert abs(actual-delta)<1e-12 and abs(actual)<=min(1,tv)+1e-12
print(f'Three-category adaptive enumeration: delta={actual:.9f}, telescoping sum={delta:.9f}, TV sum={tv:.9f}')
print('PASS: convolution identities and finite adaptive decomposition checks')

# 穷举三答案计数及余下投票：严格锁定条件与边界。
checked=0
for counts in product(range(5),repeat=3):
 leader=max(range(3),key=lambda a:counts[a])
 if sum(c==counts[leader] for c in counts)!=1:continue
 runner=max(counts[a] for a in range(3) if a!=leader)
 for remaining in range(5):
  margin=counts[leader]-runner
  if margin>remaining:
   for suffix in product(range(3),repeat=remaining):
    total=[counts[a]+suffix.count(a) for a in range(3)]
    assert total[leader]>max(total[a] for a in range(3) if a!=leader)
   checked+=1
  else:
   competitor=max((a for a in range(3) if a!=leader),key=lambda a:counts[a])
   assert (counts[competitor]+remaining>counts[leader])==(margin<remaining)
print(f'PASS: irreversible plurality and boundary enumeration ({checked} locked states/horizons)')

# 二项投票响应的一阶导数与二阶曲率，直接求和交叉检查。
for r in range(2,13):
 for t in range(1,r+1):
  for p in [.1,.3,.6,.9]:
   deriv=sum(comb(r,k)*p**k*(1-p)**(r-k)*(k/p-(r-k)/(1-p)) for k in range(t,r+1))
   formula=r*comb(r-1,t-1)*p**(t-1)*(1-p)**(r-t)
   assert abs(deriv-formula)<1e-10
   second=sum(comb(r,k)*p**k*(1-p)**(r-k)*((k/p-(r-k)/(1-p))**2-k/p**2-(r-k)/(1-p)**2) for k in range(t,r+1))
   curvature=formula*((t-1)/p-(r-t)/(1-p))
   assert abs(second-curvature)<1e-9
mixture=sum(comb(4,k)*.6**k*.4**(4-k)*(.65*tail(28,1.,17-k)+.35*tail(28,0.,17-k)) for k in range(5))
assert abs(mixture-.65)<1e-12
assert (4*.6+28*.65)/32>.6 and mixture<tail(32,.6,17)
print(f'PASS: binomial curvature and shared-prefix counterexample (shared={mixture:.6f}, independent={tail(32,.6,17):.6f})')
