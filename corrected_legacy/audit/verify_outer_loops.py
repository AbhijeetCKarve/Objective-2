"""Checks the three MAML outer loops found in the old notebooks, on the same model and data.
A: uploaded 04_maml_training.ipynb (learn2learn, first_order=True)
B: Kaggle copy 04-maml-training (3).ipynb (deepcopy, backward on the copy)
C: 06_smap_maml_clean.ipynb corrected loop
D: compares the A and C meta-gradients.
Needs learn2learn 0.2.0. It no longer builds on Python 3.11, but its MAML code is pure Python:
download the sdist and pass a folder containing learn2learn/{__init__,_version}.py,
learn2learn/utils and learn2learn/algorithms (with algorithms/__init__.py importing only
base_learner and maml).  Usage: python verify_outer_loops.py <that folder>"""
import sys, copy, torch, torch.nn as nn, numpy as np
sys.path.insert(0, sys.argv[1]); import learn2learn as l2l
torch.manual_seed(0)
class Enc(nn.Module):
    def __init__(s,d=25): super().__init__(); s.lstm1=nn.LSTM(d,64,batch_first=True); s.lstm2=nn.LSTM(64,32,batch_first=True); s.fc=nn.Linear(32,16)
    def forward(s,x): o,_=s.lstm1(x); _,(h,_)=s.lstm2(o); return s.fc(h.squeeze(0))
class Dec(nn.Module):
    def __init__(s,d=25): super().__init__(); s.lstm1=nn.LSTM(16,32,batch_first=True); s.lstm2=nn.LSTM(32,64,batch_first=True); s.fc=nn.Linear(64,d)
    def forward(s,z): r=z.unsqueeze(1).repeat(1,30,1); o,_=s.lstm1(r); o,_=s.lstm2(o); return s.fc(o)
class AE(nn.Module):
    def __init__(s): super().__init__(); s.encoder=Enc(); s.decoder=Dec()
    def forward(s,x): return s.decoder(s.encoder(x))
crit=nn.MSELoss()
tasks=[(torch.rand(20,30,25),torch.rand(20,30,25)) for _ in range(4)]
def snap(m): return [p.detach().clone() for p in m.parameters()]
def report(name, before, model):
    after=snap(model); d=sum((a-b).norm()**2 for a,b in zip(after,before))**0.5
    grads=[p.grad for p in model.parameters()]
    nz=sum(1 for g in grads if g is not None and g.abs().sum()>0)
    print(f"{name:34s} params with non-zero grad: {nz}/{len(grads)} | meta-param change L2: {float(d):.3e}")

# A) uploaded 04_maml_training.ipynb (learn2learn, first_order=True)
base=AE(); maml=l2l.algorithms.MAML(base, lr=0.01, first_order=True)
opt=torch.optim.Adam(maml.parameters(), lr=1e-3); b=snap(maml); ml=0
for s,q in tasks:
    L=maml.clone(); L.train()
    for _ in range(5): L.adapt(crit(L(s),s))
    ml=ml+crit(L(q),q)
ml=ml/4; opt.zero_grad(); ml.backward(); torch.nn.utils.clip_grad_norm_(maml.parameters(),1.0); opt.step()
report("A 04 learn2learn (uploaded)", b, maml)

# B) 04-maml-training (3).ipynb on Kaggle (deepcopy + backward on copy)
torch.manual_seed(0); model=AE(); opt=torch.optim.Adam(model.parameters(), lr=1e-3); b=snap(model); ml=torch.tensor(0.0)
for s,q in tasks:
    L=copy.deepcopy(model); L.train(); io=torch.optim.SGD(L.parameters(),lr=0.01)
    for _ in range(10): io.zero_grad(); l=crit(L(s),s); l.backward(); io.step()
    ml=ml+crit(L(q),q)
ml=ml/4; opt.zero_grad(); ml.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
report("B 04-maml-training(3) Kaggle", b, model)

# C) 06 corrected core (autograd.grad on copy, written onto meta .grad)
torch.manual_seed(0); model=AE(); opt=torch.optim.Adam(model.parameters(), lr=1e-3); b=snap(model); acc=None
for s,q in tasks:
    L=copy.deepcopy(model); L.train(); io=torch.optim.SGD(L.parameters(),lr=0.01)
    for _ in range(10): io.zero_grad(); l=crit(L(s),s); l.backward(); io.step()
    g=torch.autograd.grad(crit(L(q),q), L.parameters()); acc=[x.detach() for x in g] if acc is None else [a+x.detach() for a,x in zip(acc,g)]
opt.zero_grad()
for p,a in zip(model.parameters(),acc): p.grad=a/4
torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
report("C 06 corrected FOMAML", b, model)

# D) Do A and C compute the same first-order gradient? (same init, same data, same 10 steps)
torch.manual_seed(0); base=AE(); ref=copy.deepcopy(base)
maml=l2l.algorithms.MAML(base, lr=0.01, first_order=True)
s,q=tasks[0]; L=maml.clone()
for _ in range(10): L.adapt(crit(L(s),s))
crit(L(q),q).backward(); gA=[p.grad.clone() for p in maml.parameters()]
L2=copy.deepcopy(ref); io=torch.optim.SGD(L2.parameters(),lr=0.01)
for _ in range(10): io.zero_grad(); crit(L2(s),s).backward(); io.step()
gC=torch.autograd.grad(crit(L2(q),q), L2.parameters())
rel=max(float((a-c).norm()/(c.norm()+1e-12)) for a,c in zip(gA,gC))
print(f"D max relative difference between A and C meta-gradients: {rel:.2e}")
