# cornell eaval
# python3 evaluate.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/OW_mish_jac/epoch_41_iou_0.90 --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --iou-eval 
# python3 evaluate.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/allpwlayer_coernell_Grconv/epoch_20_iou_0.98 --dataset cornell --dataset-path /home/ericlab/cornell_grasp --iou-eval
# cornel train 
# python3 train_network.py --dataset cornell --dataset-path /home/ericlab/cornell_grasp --description training_cornell
# python3 train_network3.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 

#  python3 eval_pt.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/221209_1321_training_jacquard/epoch_98_iou_0.91.pt --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --iou-eval 
#  python3 eval2.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --iou-eval 
#  python3 eval2.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/221209_1321_training_jacquard/epoch_98_iou_0.91.pt --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --iou-eval 
# python3 eval_OW.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/OW_graduation2/epoch_92_iou_0.92 --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --iou-eval 
# python3 eval_OW.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/graduation_cornell_iw/epoch_98_iou_0.86--dataset cornell --dataset-path /home/ericlab/cornell_grasp --iou-eval 
# python3 train_network_ow.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 
# python3 train_network_iw.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 
# python3 train_origin_ow.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 
# python3 train_origin_iw.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 

#  python3 train_network_ow.py --dataset jacquard --dataset-path /home/ericlab/jacquard_dataset --description training_jacquard 

# python3 eval_OW.py --network /home/ericlab/gr_convnet/robotic-grasping/logs/graduation_cornell_iw/epoch_98_iou_0.86--dataset cornell --dataset-path /home/ericlab/cornell_grasp --iou-eval 

import torch.nn as nn
import torch.nn.functional as F
import torch


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class SE(nn.Module):
     def __init__(self, inp, oup):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(inp,8),
                nn.Mish(inplace=True),
                nn.Linear(8,oup),
                nn.Sigmoid()
        )

     def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class attention(nn.Module):
     def __init__(self, inp, oup):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(inp,inp//16),
                nn.ReLU(inplace=True),
                nn.Linear(inp//16,oup),
                nn.Sigmoid()
        )

        self.conv=nn.Conv2d(inp,1,1)
        self.ac=nn.Sigmoid()

    
     def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.mul(x,y)

        z = self.conv(x)
        z = self.ac(z)
        z = torch.mul(x,z)
        return  y + z

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)

        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class ResidualBlock_relu(nn.Module):
    """
    A residual block with dropout option
    """
# SE,relu
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock_relu, self).__init__()
        self.se = SE_relu(128, 128)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.se(x_in)
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class ResidualBlockMish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlockMish, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class ResidualBlockMish_SE(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlockMish_SE, self).__init__()

        self.se = SE(128, 128)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.se(x_in)  
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        return x + x_in



class SE_identity(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SE_identity, self).__init__()

        self.se = SE(128, 128)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x1 = self.se(x_in) 
        x2 = self.bn1(self.conv1(x_in))
        x2 = self.act(x2)
        x2 = self.bn2(self.conv2(x2))
        x2 = x2 + x_in
        return x1 + x2

class ResidualBlock_sepa(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock_sepa, self).__init__()
        
        self.pw1 = nn.Conv2d(in_channels, 64, 1,1)
        self.conv1 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.act = nn.Mish(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pw2 = nn.Conv2d(64,out_channels,1,1)

    def forward(self, x_in):
        x = self.pw1(x_in)
        x = self.bn1(self.conv1(x))
        x = self.act(x)
        x = self.bn2(self.conv2(x))
        x = self.pw2(x)

        return x + x_in

class MBC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel = 3):
        super(MBC, self).__init__()
        # pw
        self.conv1 = nn.Conv2d(in_channels, 64, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.Mish(inplace=True)
        # dw
        self.conv2 = nn.Conv2d(64, 64, kernel, 1 , bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.Mish(inplace=True)

        self.se = SE(64, 64)
        # SElayer
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.liner1 = nn.Linear(64, 64),
        # self.act3 = nn.Mish()
        # self.liner2 = nn.Linear(64, 64),
        # self.act4 = nn.Sigmoid()

        # pw
        self.conv3 = nn.Conv2d(64, 128, 1, 1,  padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        # pw
        x = self.act1(self.bn1((self.conv1(x_in))))
        # dw
        x = self.act2(self.bn2(self.conv2(x)))
        # SElayer
        x = self.se(x)
        # x = self.act4(self.liner2(self.act3(self.liner1(self.avg_pool(x)))))
        # pw-Linear
        x = self.bn3(self.conv3(x))

        return x + x_in




    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.mish(self.conv(x))], 1)


class RDN(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDN, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    
    def forward(self, x_in):
        x1 = self.se(x_in) 
        x2 = self.bn1(self.conv1(x_in))
        x2 = self.act(x2)
        x2 = self.bn2(self.conv2(x2))
        x2 = x2 + x_in
        return x1 + x2