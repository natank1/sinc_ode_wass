# https://github.com/mravanelli/SincNet/blob/master/dnn_models.py
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch_for_privae.ode_helper as ode
from torch_for_privae.resnet_block import Resnet_block
from torch_for_privae.sinc_model import SincNet


class resnet_arc(nn.Module):
    def __init__(self, d_args,inp_d, device, inp_data, vanilla_kl, target_f,  use_ode=False):

        super(resnet_arc, self).__init__()
        self.device = device



        #downsampling
        self.sinc= SincNet(inp_d)
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.nrm = ode.norm(64)
        self.conv1 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.lrel = nn.ReLU(inplace=True)


        self.first_bn = nn.BatchNorm2d(num_features=d_args['filts'][0])

        self.first_conv = nn.Conv2d(in_channels=d_args['in_channels'],
                                    out_channels=d_args['filts'][0],
                                    kernel_size=d_args['kernels'][0],
                                    padding=(1, 3),
                                    stride=d_args['strides'][0])
        self.first_lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.block0 = Resnet_block(nb_filts=d_args['filts'][1],
                                     kernels=d_args['kernels'][1],
                                     strides=d_args['strides'][1],
                                     first=True,
                                     downsample=True)

        self.block1 = self._make_layer(nb_blocks=d_args['blocks'][0],
                                       nb_filts=d_args['filts'][2],
                                       kernels=d_args['kernels'][2],
                                       strides=d_args['strides'][2])

        self.block2 = self._make_layer(nb_blocks=d_args['blocks'][1],
                                       nb_filts=d_args['filts'][3],
                                       kernels=d_args['kernels'][3],
                                       strides=d_args['strides'][3])

        self.block3 = self._make_layer(nb_blocks=d_args['blocks'][2],
                                       nb_filts=d_args['filts'][4],
                                       kernels=d_args['kernels'][4],
                                       strides=d_args['strides'][4])

        self.block4 = self._make_layer(nb_blocks=d_args['blocks'][3],
                                       nb_filts=d_args['filts'][5],
                                       kernels=d_args['kernels'][5],
                                       strides=d_args['strides'][5],
                                       downsample=False)

        # self.mega_ode = mega_ode(self.device, ode.ODEfunc, niter).to(self.device)
        self.odebl = ode.ODEBlock(ode.ODEfunc(64))
        self.last_bn = nn.BatchNorm2d(num_features=d_args['filts'][-1][-1])
        self.last_lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_features=d_args['filts'][-1][-1] * 2,
                             out_features=d_args['nb_fc_node'])
        self.fc11 = nn.Linear(in_features=(d_args['filts'][-1][-1] * 2) + 400, out_features=200)
        self.fc101 = nn.Linear(in_features=200, out_features=100)

        self.fc12 = nn.Linear(in_features=100, out_features=64)

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.fc02 = nn.Linear(in_features=d_args['nb_fc_node'],
                             out_features=d_args['nb_classes'],
                             bias=False)
        self.fc002 = nn.Linear(in_features=d_args['nb_fc_node'],
                             out_features=1,
                             bias=False)

        self.fc2 = nn.Linear(in_features=64,
                             out_features=30,
                             bias=False)
        self.fc3 = nn.Linear(in_features=30,
                             out_features=d_args['nb_classes'],
                             bias=False)
        self.sftm= nn.Softmax(dim=1)
        self.register_buffer('centers', (
                torch.rand(d_args['nb_classes'], d_args['nb_fc_node']) - 0.5) * 2)

        self.fcwas_ll = nn.Linear(in_features=d_args['nb_fc_node'],
                              out_features=1,
                              bias=False)

        self.last_layer = self.bring_last_layer(target_f, vanilla_kl)
        self.forward =self.bring_forward(vanilla_kl, inp_data, use_ode,target_f)

    def  bring_last_layer(self,target_f, vanilla_kl):
        if not (vanilla_kl) and (target_f == 3):
           return self.fcwas_ll
        else:
           return  self.fc02

    def _make_layer(self, nb_blocks, nb_filts, kernels, strides, downsample=True):
        layers = []
        for _ in range(nb_blocks - 1):
            layers.append(Resnet_block(nb_filts=[nb_filts[0], nb_filts[0]],
                                         kernels=kernels,
                                         strides=1))
        if downsample:
            layers.append(Resnet_block(nb_filts=nb_filts,
                                         kernels=kernels,
                                         strides=strides,
                                         downsample=downsample))

        return nn.Sequential(*layers)



    # choose the forward
    def bring_forward(self,vanilla_kl, inp_data, use_ode,target_f):
        if (vanilla_kl == 1) or ((vanilla_kl == 0) and (target_f==3)):
           if inp_data == 1:

               if not(use_ode ):
                    return self.vanilla_forward
               else:
                    return self.vanilla_ode_forward
           elif inp_data == 3:
                if not (use_ode):
                    return self.vanilla_comb_forward
                else:
                    return self.vanilla_comb_ode_forward
        else:
            if target_f <3:
                if inp_data == 1:

                    if not (use_ode):
                        return self.bh_forward
                    else:
                        return  self.bh_ode_forward
                elif inp_data== 3:

                    if not (use_ode):
                        return self.bh_comb_forward
                    else:
                       return self.bh_comb_ode_forward

    def compute_center_loss(self, features, centers, targets):
        features = features.view(features.size(0), -1)
        target_centers = centers[targets]
        # criterion = torch.nn.MSELoss()
        # center_loss = criterion(features, target_centers)
        center_loss = torch.mean(torch.sum(torch.pow(features - target_centers, 2), dim=1)) / 2.
        return center_loss

    def get_center_delta(self, features, centers, targets, alpha):
        device = self.device
        # implementation equation (4) in the center-loss paper
        features = features.view(features.size(0), -1)
        targets, indices = torch.sort(targets)
        target_centers = centers[targets]
        features = features[indices]

        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

        uni_targets = uni_targets.to(device)
        indices = indices.to(device)

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).to(device).index_add_(0, indices, delta_centers)

        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

        delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
        result = torch.zeros_like(centers)
        result[uni_targets, :] = delta_centers
        return result

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=print):
        # if print_fn == None: printfn = print
        model = self

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        # print(type(x[0]))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print_fn("================================================================")
        print_fn("Total params: {0:,}".format(total_params))
        print_fn("Trainable params: {0:,}".format(trainable_params))
        print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print_fn("----------------------------------------------------------------")
        print_fn("Input size (MB): %0.2f" % total_input_size)
        print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print_fn("Params size (MB): %0.2f" % total_params_size)
        print_fn("Estimated Total Size (MB): %0.2f" % total_size)
        print_fn("----------------------------------------------------------------")
        return

    def downsampling_4_ode(self, x):  # used only for ode
        x = self.conv0(x)
        x = self.nrm(x)
        x = self.conv1(x)
        x = self.nrm(x)
        x = self.lrel(x)
        x = self.conv2(x)
        return x

    def bloc2_to_1d(self,x):
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.last_bn(x)
        x = self.last_lrelu(x)

        x_avg = self.global_avgpool(x)
        channel_dim = x_avg.size(1)
        x_avg = x_avg.view(-1, channel_dim)
        x_max = self.global_maxpool(x)
        x_max = x_max.view(-1, channel_dim)
        x = torch.cat((x_avg, x_max), dim=1)
        return x


    def vanilla_suffix(self, x):
        x = self.fc1(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)

        y = self.last_layer(x)
        return x, y


    def vanilla_comb_suffix (self,x):
        x = self.fc11(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)
        x = self.fc101(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)

        x = self.fc12(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)
        y = self.last_layer(x)
        return x,y







    #Forwards block
    def vanilla_forward(self, x):
        # print ("fff",x.shape)
        x = x.to(self.device)
        x= self.sinc(x)
        # print ("ggg",x.shape)
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_lrelu(x)
        # print("geeegsg", x.shape)

        x = self.block0(x)
        x = self.block1(x)
        # print("guuuugg", x.shape)

        x = self.bloc2_to_1d(x)
        x = self.fc1(x)
        x = nn.Dropout(0.1)(x)
        x = self.lrelu(x)
        y = self.fc02(x)

        # x, y = self.vanilla_suffix(x)  check

        return x, y

    def vanilla_ode_forward(self, x):
        x = x.to(self.device)
        x = self.sinc(x)

        x = self.downsampling_4_ode(x)

        x = self.odebl.forward(x)
        x = self.odebl.forward(x)
        x = self.odebl.forward(x)

        x = self.bloc2_to_1d(x)
        x,y = self.vanilla_suffix(x)

        return x, y

    def vanilla_comb_forward(self, x):
        x, z = x
        x, z = x.to(self.device), z.to(self.device)
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_lrelu(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.bloc2_to_1d(x)
        x = torch.cat((x, z), dim=1)

        x, y = self.vanilla_comb_suffix(x)

        return x, y,x

    def vanilla_comb_ode_forward(self, x):
        x, z = x
        x, z = x.to(self.device), z.to(self.device)

        x = self.downsampling_4_ode(x)

        x = self.odebl.forward(x)
        x = self.odebl.forward(x)
        x = self.odebl.forward(x)

        x = self.bloc2_to_1d(x)
        x = torch.cat((x, z), dim=1)

        x, y = self.vanilla_comb_suffix(x)

        return x, y,x

    def bh_forward(self, x):
        x, x1, _ = self.vanilla_forward(x)
        y = self.sftm(x1)
        return x, y, x1

    def bh_ode_forward(self, x):
        x, x1, _ = self.vanilla_ode_forward(x)
        y = self.sftm(x1)
        return x, y, x1

    def bh_comb_forward(self, x):
        x, x1 = self.vanilla_comb_forward(x)
        y = self.sftm(x1)
        return x, y, x1

    def bh_comb_ode_forward(self, x):
        x, x1 = self.vanilla_comb_ode_forward(x)
        y = self.sftm(x1)
        return x, y, x1

if __name__ == '__main__':
    import yaml
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % 0 if cuda else 'cpu')
    dir_yaml="C:\\Users\\picklerick\\PycharmProjects\\tochcombine\\train_pack\\train.yaml"
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)
    print (parser)
    inp = 25000

    mm= spec_CNN(parser['model'],inp ,device,1,1,0).to(device)
    # print (mm.summary((1,inp)))
    # mm = SincNet(700).to(device)
    vv = torch.tensor(np.random.rand(16, inp)).float().to(device)
    print (vv.shape)
    y,_ = mm(vv)
    print("hhhh", y)