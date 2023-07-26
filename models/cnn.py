import torch.nn as nn

def conv_block(in_size, out_size):
    conv1 = nn.Conv1d(
        in_channels = in_size, 
        out_channels = out_size, 
        kernel_size = 5,
        stride = 1,
        padding = 0
    )
    act1 = nn.ReLU()
    max1 = nn.MaxPool1d(
        kernel_size = 4,
        stride = 2
    )    
    layers = [conv1, act1, max1]
    return nn.Sequential(*layers)

def deconv_block(in_size, out_size, upsample = True, kernel_size = 5):
    conv1 = nn.ConvTranspose1d(
        in_channels = in_size, 
        out_channels = out_size, 
        kernel_size = kernel_size,
        stride = 1,
        padding = 0
    )
    act1 = nn.ReLU()
    layers = [conv1, act1]
    if upsample:
        max1 = nn.Upsample(
            scale_factor = 2,
            mode = 'linear' # adjust later
        )    
        layers.append(max1)
    return nn.Sequential(*layers)

def deconv_decoder(layer_sizes, output_shape): 
    
    t_len, t_channels = layer_sizes[0]
    o_len, o_channels = output_shape    
    input_channels = t_channels
    hidden_channels = layer_sizes[1]
    in_len = t_len

    blocks = []

    while 2 * in_len + 8 <= o_len:
        block = deconv_block(
            input_channels, 
            hidden_channels
        )
        blocks.append(block)
        in_len = 2 * in_len + 8
        input_channels = hidden_channels


    diff_len = o_len - in_len

    remaining_blocks = diff_len // 4
    remainder = diff_len % 4

    for _ in range(remaining_blocks):
        block = deconv_block(
            input_channels, 
            hidden_channels, 
            upsample=False
        )
        blocks.append(block)

    if remainder > 0:
        block = deconv_block(
            input_channels, 
            hidden_channels, 
            upsample=False, 
            kernel_size=remainder+1
        )
        blocks.append(block)


    f_layer = nn.ConvTranspose1d(
        in_channels=input_channels,
        out_channels=o_channels,
        kernel_size=1
    )

    blocks.append(f_layer)
    
    return nn.Sequential(*blocks)

class VaeCNNDecoder(nn.Module):
    def __init__(self, layer_sizes, output_shape, **kwargs):
        super(VaeCNNDecoder, self).__init__()
        #self.fc_blocks = nn.Sequential(*[fc_block(in_size, out_size, **kwargs) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:-1])])
        #self.fc_last = nn.Linear(layer_sizes[-2],layer_sizes[-1])
        self.sigmoid = nn.Sigmoid()
        #self.unflatten = UnFlatten(output_shape)
        self.block = deconv_decoder(layer_sizes, output_shape)

    def forward(self, x):
        #x = self.fc_blocks(x)
        #x = self.fc_last(x)
        x = self.sigmoid(x)
        #return self.unflatten(x)
        x = x.transpose(1, 2)
        x_y = self.block(x)
        x_y = x_y.transpose(1,2)
        return x_y