from LLM4GCL.backbones import *

from .base import BaseModel

from .GNN.BareGNN import BareGNN
from .GNN.JointGNN import JointGNN
from .GNN.EWC import EWC
from .GNN.MAS import MAS
from .GNN.GEM import GEM
from .GNN.LwF import LwF
from .GNN.cosine import cosine
from .GNN.ERGNN import ERGNN
from .GNN.SSM import SSM
from .GNN.CaT import CaT
from .GNN.DeLoMe import DeLoMe
from .GNN.TPP import TPP

from .LM.BareLM import BareLM
from .LM.SimpleCIL import SimpleCIL
from .LM.OLoRA import OLoRA

from .Graph_LLM.LM_emb import LM_emb
from .Graph_LLM.GraphPrompter import GraphPrompter
from .Graph_LLM.ENGINE import ENGINE