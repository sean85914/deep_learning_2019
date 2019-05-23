import sys
import torch
from model import *
from trainer import Trainer
  
fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init)

if len(sys.argv) == 2: # pretrained provided
  g.load_state_dict(torch.load(sys.argv[1]))

trainer = Trainer(g, fe, d, q)
trainer.train()
