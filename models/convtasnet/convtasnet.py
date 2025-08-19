from .utils import *
from models import register_model

class Separation_TasNet(nn.Module):
    '''
       TasNet Separation part
       LayerNorm -> 1x1Conv -> 1-D Conv .... -> output_data
    '''

    def __init__(self, repeats=3, conv1d_block=8, in_channels=64, out_channels=128,
                 out_sp_channels=512, kernel_size=3, norm='gln', causal=False, num_spks=2):
        super(Separation_TasNet, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1d_list = self._Sequential(
            repeats, conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.PReLu = nn.PReLU()
        self.norm = select_norm('cln', in_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, num_spks*in_channels, 1)
        self.activation = nn.Sigmoid()
        self.num_spks = num_spks

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_lists = [Conv1D(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_lists)
    
    def _Sequential(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = [self._Sequential_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeats_lists)

    def forward(self, x):
        """
           Input:
               x: [B x C x T], B is batch size, T is times
           Returns:
               x: [num_spks, B, N, T]
         """
        # B x C x T
        x = self.norm(x)
        x = self.conv1x1(x)
        # B x C x T
        x = self.conv1d_list(x)
        # B x num_spks*N x T
        x = self.PReLu(x)
        x = self.end_conv1x1(x)
        # num_spks x B x N x T
        x = torch.chunk(x, self.num_spks, dim=1)
        x = self.activation(torch.stack(x, dim=0))
        return x


@register_model("CONVTASNET")
class Conv_TasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gln",
                 num_spks=2,
                 activate="relu",
                 causal=False):
        super(Conv_TasNet, self).__init__()
        self.encoder = Encoder(kernel_size=L, out_channels=N)
        self.separation = Separation_TasNet(repeats=R, conv1d_block=X, in_channels=N,
                                            out_channels=B, out_sp_channels=H, kernel_size=P,
                                            norm=norm, causal=causal, num_spks=num_spks)
        self.decoder = Decoder(
            in_channels=N, out_channels=1, kernel_size=L, stride=L//2)
        self.num_spks = num_spks

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, T]
        """
        # B x T -> B x C x T
        x_encoder = self.encoder(x)
        # B x C x T -> num_spks x B x N x T
        x_sep = self.separation(x_encoder)
        # [B x N x T, B x N x T]
        audio_encoder = [x_encoder*x_sep[i] for i in range(self.num_spks)]
        # [B x T, B x T]
        audio = [self.decoder(audio_encoder[i]) for i in range(self.num_spks)]
        return audio


if __name__ == "__main__":
    # 简单自检 (可选执行)
    model = Conv_TasNet()
    dummy = torch.randn(2, 16000)
    out = model(dummy)
    print(f"Outputs: {len(out)} tensors, shape[0]={out[0].shape}")
