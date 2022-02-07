"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, settings):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """

        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = settings.input_size
        self.num_outputs = settings.output_size
        self.controller_size = settings.controller_size
        self.controller_layers = 1
        self.num_heads = 1
        self.N = settings.m # M and N are intentionally reversed from settings
        self.M = settings.n

        # Create the NTM components
        memory = NTMMemory(self.N, self.M)
        controller = LSTMController(self.num_inputs + self.M*self.num_heads, self.controller_size, self.controller_layers)
        heads = nn.ModuleList([])
        for i in range(self.num_heads):
            heads += [
                NTMReadHead(memory, self.controller_size),
                NTMWriteHead(memory, self.controller_size)
            ]

        self.ntm = NTM(self.num_inputs, self.num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
