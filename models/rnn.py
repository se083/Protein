import torch
import torch.nn as nn

def rnn_forward(z, o_len, rnn, last):
    state = (z.unsqueeze(dim = 1), z.unsqueeze(dim = 1))
    #output, state = self.rnn(x, state)
    #output = output[:, -1]
    output = []
    for i in range(o_len):
        x, state = rnn(x, state)
        x = x[:, -1]
        x = last(x)
        x = x.unsqueeze(dim=1)
        output.append(x)
    output = torch.cat(output, dim=1)
    return output

class VaeRNNDecoder(nn.Module):
    def __init__(self, layer_sizes, output_shape, **kwargs):
        super(VaeRNNDecoder, self).__init__()
        #self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        #self.fc_last = nn.Linear(layer_sizes[-2],layer_sizes[-1])
        #self.sigmoid = nn.Sigmoid()
        #self.unflatten = UnFlatten(output_shape)
        self.t_len, t_channels, latent_size = layer_sizes[0]
        self.o_len, o_channels = output_shape    
        hidden_channels = layer_sizes[1]
        self.hidden_size = latent_size
        self.num_layers = 1
        self.rnn = nn.LSTM(
            input_size = t_channels, 
            hidden_size = self.hidden_size,
            num_layers = self.num_layers, #can play around with later
            batch_first = True,
            dropout = 0,
            bidirectional = False
        )
        self.last = nn.Linear(self.hidden_size, o_channels)
        #self.block = deconv_decoder(layer_sizes, output_shape)

    def forward(self, z):
        return rnn_forward(z, self.o_len, self.rnn, self.last)
    