import torch
from torch import nn
from .mask.mask import Mask

class STDIL(nn.Module):
    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)
        self.backend = STDIL_1(**backend_args)
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
        checkpoint_dict = torch.load(self.pre_trained_smae_path)
        self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        short_term_history = history_data     # [B, L, N, 1]

        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = self.tmae(long_history_data[..., [0]])
        hidden_states_s = self.smae(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        out_len=1
        hidden_states = hidden_states[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)

        return y_hat

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

def nconv1(x, A):
    return torch.einsum('bcnt,nm->bcmt', (x, A)).contiguous()

class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.conv = Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv1(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv1(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(device, logits, temperature, eps=1e-10):
    sample = sample_gumbel(device, logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(device, logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(
        device, logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Graph_Generator(nn.Module):
    def __init__(self, device, channels, num_nodes, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.node = num_nodes
        self.device = device
        self.fc0 = nn.Linear(channels, num_nodes)
        self.fc1 = nn.Linear(num_nodes, 2 * num_nodes)
        self.fc2 = nn.Linear(2 * num_nodes, num_nodes)
        self.diffusion_conv = Diffusion_GCN(channels, channels, dropout, support_len=1)

    def forward(self, x, adj):
        x = self.diffusion_conv(x, [adj])
        x = x[-1, :, :, :]
        x = x.sum(2)
        x = x.permute(1, 0)
        x = self.fc0(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.log(F.softmax(x, dim=-1))
        x = gumbel_softmax(self.device, x, temperature=0.5, hard=True)
        mask = torch.eye(x.shape[0], x.shape[0]).bool().to(device=self.device)
        x.masked_fill_(mask, 0)
        return x


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))



class IDGCN(nn.Module):
    def __init__(self, device, channels, splitting=True, num_nodes=170, dropout=0.25, pre_adj_len=1):
        super(IDGCN, self).__init__()

        device = device
        self.dropout = dropout
        self.pre_adj_len = pre_adj_len
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.pre_graph = []
        self.split = Splitting()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3
        pad_r = 3

        apt_size = 10
        aptinit = []
        self.pre_adj_len = 1
        nodevecs = self.svd_init(apt_size, aptinit)
        self.nodevec1, self.nodevec2 = [
            Parameter(n.to(device), requires_grad=True) for n in nodevecs]
        self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device),
                                     requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device),
                                     requires_grad=True).to(device)

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 5), dilation=1, stride=1, groups=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels,
                      kernel_size=(1, 3), dilation=1, stride=1, groups=1),
            nn.Tanh()
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

        self.graph_generator = Graph_Generator(
            device, channels, num_nodes)

        self.diffusion_conv = Diffusion_GCN(
            channels, channels, dropout, support_len=self.pre_adj_len)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(
            p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        x1 = self.conv1(x_even)
        learn_adj = self.graph_generator(x1, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x1 = x1+self.diffusion_conv(x1, [dadj])
        d = x_odd.mul(torch.tanh(x1))
        x2 = self.conv2(x_odd)
        learn_adj = self.graph_generator(x2, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x2 = x2+self.diffusion_conv(x2, [dadj])
        c = x_even.mul(torch.tanh(x2))

        x3 = self.conv3(c)
        learn_adj = self.graph_generator(x3, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x3 = x3+self.diffusion_conv(x3, [dadj])
        x_odd_update = d - x3
        x4 = self.conv4(d)
        learn_adj = self.graph_generator(x4, adaptive_adj)
        dadj = learn_adj*self.a+adaptive_adj*(1-self.a)
        x4 = x4+self.diffusion_conv(x4, [dadj])
        x_even_update = c + x4
        return (x_even_update, x_odd_update, dadj)

class IDGCN_Tree(nn.Module):
    def __init__(self, device, num_nodes, channels, num_levels, dropout, pre_adj_len=1):
        super().__init__()
        self.levels = num_levels
        self.pre_graph = []

        self.IDGCN1 = IDGCN(splitting=True, channels=channels, device=device,
                            num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN2 = IDGCN(splitting=True, channels=channels, device=device,
                            num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)
        self.IDGCN3 = IDGCN(splitting=True, channels=channels, device=device,
                            num_nodes=num_nodes, dropout=dropout, pre_adj_len=pre_adj_len)

        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, adj):
        x_even_update1, x_odd_update1, dadj1 = self.IDGCN1(x)
        x_even_update2, x_odd_update2, dadj2 = self.IDGCN2(x_even_update1)
        x_even_update3, x_odd_update3, dadj3 = self.IDGCN3(x_odd_update1)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        adj = dadj1*self.a+dadj2*self.b+dadj3*self.c
        adj = dadj1
        return concat0, adj

class STDIL_1(nn.Module):

    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):

        super(STDIL_1, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field

        self.num_levels = 2
        apt_size = 10
        self.pre_graph = []
        pre_adj_len = 1
        residual_channels = 64
        dropout = 0.2
        num_nodes = num_nodes
        # device = 'cpu'
        device = 'cuda'
        self.pre_adj_len = len(self.pre_graph) + 1
        self.a = nn.Parameter(torch.rand(1).to(
            device=device), requires_grad=True).to(device)
        self.tree = IDGCN_Tree(
            device=device,
            channels=residual_channels,
            num_nodes=num_nodes,
            num_levels=self.num_levels,
            dropout=dropout,
            pre_adj_len=self.pre_adj_len,

        )
        self.diffusion_conv = Diffusion_GCN(
            residual_channels, residual_channels, dropout, support_len=self.pre_adj_len)
        self.start_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0))
        self.start_conv2 = nn.Conv2d(64, 32, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))

    def forward(self, input, hidden_states):

            input = input.transpose(1, 3)
            input = nn.functional.pad(input,(1,0,0,0))
            input = input[:, :2, :, :]
            in_len = input.size(3)
            if in_len<self.receptive_field:
                x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
            else:
                x = input
            x = self.start_conv(x)
            x = self.start_conv1(x)

            skip = 0

            adaptive_adj = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

            skip1 = x
            x, dadj = self.tree(x, adaptive_adj)
            x = skip1 + x
            adj = self.a * adaptive_adj + (1 - self.a) * dadj
            adj = self.pre_graph + [adj]
            gcn = self.diffusion_conv(x, adj)
            x = gcn + x
            x = self.start_conv2(x)

            new_supports = None
            if self.gcn_bool and self.addaptadj and self.supports is not None:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]

            for i in range(self.blocks * self.layers):
                residual = x
                filter = self.filter_convs[i](residual)
                filter = torch.tanh(filter)
                gate = self.gate_convs[i](residual)
                gate = torch.sigmoid(gate)
                x = filter * gate

                s = x
                s = self.skip_convs[i](s)
                try:
                    skip = skip[:, :, :,  -s.size(3):]
                except:
                    skip = 0
                skip = s + skip

                if self.gcn_bool and self.supports is not None:
                    if self.addaptadj:
                        x = self.gconv[i](x, new_supports)
                    else:
                        x = self.gconv[i](x,self.supports)
                else:
                    x = self.residual_convs[i](x)

                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[i](x)

            hidden_states_t = self.fc_his_t(hidden_states[:,:,:96])
            hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)
            skip = skip + hidden_states_t
            hidden_states_s = self.fc_his_s(hidden_states[:,:,96:])
            hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)
            skip = skip + hidden_states_s

            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)
            x = x.squeeze(-1).transpose(1, 2)
            return x