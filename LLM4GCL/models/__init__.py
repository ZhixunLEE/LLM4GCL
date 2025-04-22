from LLM4GCL.backbones import *

from .base import BaseModel

from .GNN.BareGNN import BareGNN
from .GNN.EWC import EWC
from .GNN.LwF import LwF
from .GNN.cosine import cosine

from .LM.RoBERTa import RoBERTa
from .LM.LLaMA import LLaMA
from .LM.SimpleCIL import SimpleCIL
from .LM.InstructLM import InstructLM

from .GLM.LM_emb import LM_emb
from .GLM.GraphPrompter import GraphPrompter
from .GLM.ENGINE import ENGINE

from .GLM.SimGCL import SimGCL