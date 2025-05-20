from LLM4GCL.backbones import *

from .base import BaseModel

from .GNN.BareGNN import BareGNN
from .GNN.EWC import EWC
from .GNN.LwF import LwF
from .GNN.cosine import cosine
from .GNN.TEEN import TEEN
from .GNN.TPP import TPP

from .LM.RoBERTa import RoBERTa
from .LM.LLaMA import LLaMA
from .LM.SimpleCIL import SimpleCIL
from .LM.InstructLM import InstructLM
from .LM.GPT import GPT
from .LM.BERT import BERT

from .GLM.LM_emb import LM_emb
from .GLM.GraphPrompter import GraphPrompter
from .GLM.ENGINE import ENGINE
from .GLM.LLaGA import LLaGA
from .GLM.GraphGPT import GraphGPT
from .GLM.SimGCL import SimGCL