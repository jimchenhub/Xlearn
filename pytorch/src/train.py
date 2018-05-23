import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as util_data
from tensorboardX import SummaryWriter

import network
import loss
from loss import DANLoss
import pre_process as prep
import lr_schedule
from data_list import ImageList

optim_dict = {"SGD": optim.SGD}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_classification_predict(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            for j in range(10):
                inputs[j] = torch.Tensor(inputs[j], device = device)
            outputs = []
            for j in range(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            inputs = torch.Tensor(inputs, device = device)
            outputs = model(inputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_output, predict


def image_classification_test(loader, model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]

            for j in range(10):
                inputs[j] = torch.Tensor(inputs[j], device=device)
            labels = torch.Tensor(labels, device=device)

            outputs = []
            for j in range(10):
                outputs.append(model(inputs[j]))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.float()), 0)
            all_label = torch.cat((all_label, labels.data.float()), 0)

    _, predict = torch.max(all_output, 1)
    torch.sum(torch.squeeze(predict).float() == all_label)
    accuracy = float(torch.sum(torch.squeeze(predict).float() == all_label)) / float(all_label.size()[0])
    return accuracy


def transfer_classification(config):
    # ---set log writer ---
    writer = SummaryWriter(log_dir=config["log_dir"])
    # set pre-process
    prep_dict = {}
    for prep_config in config["prep"]:
        name = prep_config["name"]
        prep_dict[name] = {}
        if prep_config["type"] == "image":
            prep_dict[name]["test_10crop"] = prep_config["test_10crop"]
            prep_dict[name]["train"] = prep.image_train(resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
            if prep_config["test_10crop"]:
                prep_dict[name]["test"] = prep.image_test_10crop(resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
            else:
                prep_dict[name]["test"] = prep.image_test(resize_size=prep_config["resize_size"], crop_size=prep_config["crop_size"])
    # print(prep_dict)

    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    loss_config = config["loss"]
    # transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}
    # print(transfer_criterion)
    transfer_criterion = DANLoss(**loss_config["params"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    for data_config in config["data"]:
        name = data_config['name']
        dsets[name] = {}
        dset_loaders[name] = {}
        ## image data
        if data_config["type"] == "image":
            dsets[name]["train"] = ImageList(open(data_config["list_path"]["train"]).readlines(), 
                                                            transform=prep_dict[name]["train"])
            dset_loaders[name]["train"] = util_data.DataLoader(dsets[name]["train"], 
                                                                batch_size=data_config["batch_size"]["train"], 
                                                                shuffle=True, num_workers=4)
            if "test" in data_config["list_path"]:
                if prep_dict[name]["test_10crop"]:
                    for i in range(10):
                        dsets[name]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["test"]).readlines(),
                            transform=prep_dict[name]["test"]["val" + str(i)])
                        dset_loaders[name]["test" + str(i)] = util_data.DataLoader(
                            dsets[name]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[name]["test"] = ImageList(
                        open(data_config["list_path"]["test"]).readlines(),
                        transform=prep_dict[name]["test"])
                    dset_loaders[name]["test"] = util_data.DataLoader(
                        dsets[name]["test"], batch_size = data_config["batch_size"]["test"],
                        shuffle=False, num_workers=4)
            else:
                if prep_dict[name]["test_10crop"]:
                    for i in range(10):
                        dsets[name]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["train"]).readlines(),
                            transform=prep_dict[name]["test"]["val" + str(i)])
                        dset_loaders[name]["test" + str(i)] = util_data.DataLoader(
                            dsets[name]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[name]["test"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
                                                                   transform=prep_dict[name]["test"])
                    dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(
                        dsets[data_config["name"]]["test"], batch_size=data_config["batch_size"]["test"],
                        shuffle=False, num_workers=4)

    class_num = 31

    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    if net_config["use_bottleneck"]:
        bottleneck_layer = nn.Linear(base_network.output_num(), net_config["bottleneck_dim"])
        classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
    else:
        classifier_layer = nn.Linear(base_network.output_num(), class_num)

    ## initialization
    if net_config["use_bottleneck"]:
        bottleneck_layer.weight.data.normal_(0, 0.005)
        bottleneck_layer.bias.data.fill_(0.1)
        bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)

    # print(base_network.state_dict())
    # print(classifier_layer.state_dict())

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        if net_config["use_bottleneck"]:
            bottleneck_layer = bottleneck_layer.to(device)
        classifier_layer = classifier_layer.to(device)
        base_network = base_network.to(device)


    ## collect parameters
    if net_config["use_bottleneck"]:
        parameter_list = [{"params":base_network.parameters(), "lr":1}, 
                          {"params":bottleneck_layer.parameters(), "lr":10}, 
                          {"params":classifier_layer.parameters(), "lr":10}]
       
    else:
        parameter_list = [{"params":base_network.parameters(), "lr":1}, 
                          {"params":classifier_layer.parameters(), "lr":10}]

    ## add additional network for some methods
    if loss_config["name"] == "JAN":
        softmax_layer = nn.Softmax()
        if use_gpu:
            softmax_layer = softmax_layer.to(device)
           
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train
    len_train_source = len(dset_loaders["source"]["train"]) - 1
    len_train_target = len(dset_loaders["target"]["train"]) - 1
    # transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    for i in range(config["num_iterations"]):
        # test in the train
        if i % config["test_interval"] == 0:
            base_network.train(False)
            classifier_layer.train(False)
            if net_config["use_bottleneck"]:
                bottleneck_layer.train(False)
                test_model = nn.Sequential(base_network, bottleneck_layer, classifier_layer)
                test_acc = image_classification_test(dset_loaders["target"],
                    test_model,
                    test_10crop=prep_dict["target"]["test_10crop"],
                    gpu=use_gpu)
                print("iteration: %d, test accuracy: %f" % (i, test_acc))

            else:
                test_model = nn.Sequential(base_network, classifier_layer)
                test_acc = image_classification_test(dset_loaders["target"],
                    test_model,
                    test_10crop=prep_dict["target"]["test_10crop"],
                    gpu=use_gpu)
                print("iteration: %d, test accuracy: %f" % (i, test_acc))
            # save model parameters
            if i % config["checkpoint_iterval"] == 0:
                torch.save(test_model.state_dict(), config["checkpoint_dir"]+"checkpoint_"+str(i).zfill(5)+".pth")

        # loss_test = nn.BCELoss()
        # train one iter
        base_network.train(True)
        if net_config["use_bottleneck"]:
            bottleneck_layer.train(True)
        classifier_layer.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"]["train"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"]["train"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        
        inputs_source = inputs_source.to(device)
        inputs_target = inputs_target.to(device)
        labels_source = labels_source.to(device)
          
        inputs = torch.cat((inputs_source, inputs_target), dim=0) ### cat the source and target 
        features = base_network(inputs)
        if net_config["use_bottleneck"]:
            features = bottleneck_layer(features)

        outputs = classifier_layer(features)

        # split the output for different loss
        num = inputs.size(0)//2
        # the premise is the batch size of source and target is the same
        classifier_loss = class_criterion(outputs.narrow(0, 0, num), labels_source)
        ## switch between different transfer loss
        if loss_config["name"] == "DAN":
            # transfer_loss = transfer_criterion(features.narrow(0, 0, num),
            #                                    features.narrow(0, num, num),
            #                                    **loss_config["params"])

            transfer_loss = transfer_criterion(features.narrow(0, 0, num),
                                               features.narrow(0, num, num))
        elif loss_config["name"] == "RTN":
            ## RTN is still under developing
            transfer_loss = 0
        elif loss_config["name"] == "JAN":
            softmax_out = softmax_layer(outputs)
            transfer_loss = transfer_criterion([features.narrow(0, 0, num), softmax_out.narrow(0, 0, num)],
                                               [features.narrow(0, num, num), softmax_out.narrow(0, num, num)],
                                               **loss_config["params"])

        total_loss = loss_config["trade_off"] * transfer_loss + classifier_loss
        if i % 100 == 0:
            print("iteration: ", str(i), "\t transfer loss: ", str(transfer_loss.item()),
                  "\t classification loss: ", str(classifier_loss.item()),
                  "\t total loss: ", str(total_loss.item()))
            for param_group in optimizer.param_groups:
                print("lr: ", param_group["lr"])
            # log for losses
            writer.add_scalar('Train/loss_classification', classifier_loss.item(), i//100)
            writer.add_scalar('Train/loss_transfer', transfer_loss.item(), i//100)
            writer.add_scalar('Train/loss_total', total_loss.item(), i//100)
        total_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Transfer Learning')
    # parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    config = {}
    config["num_iterations"] = 100000
    config["test_interval"] = 500
    config["checkpoint_iterval"] = 5000
    config["prep"] = [{"name":"source", "type":"image", "test_10crop":False, "resize_size":256, "crop_size":224},
                      {"name":"target", "type":"image", "test_10crop":False, "resize_size":256, "crop_size":224}]
    config["loss"] = {"name":"DAN", "trade_off":1.0 }
    config["data"] = [{"name":"source", "type":"image", "list_path":{"train":"../data/office/amazon_dslr_list.txt"}, 
                            "batch_size":{"train":16, "test":1} },
                      {"name":"target", "type":"image", "list_path":{"train":"../data/office/webcam_list.txt"}, 
                            "batch_size":{"train":16, "test":795}}]
    config["network"] = {"name":"AlexNet", "use_bottleneck":True, "bottleneck_dim":256}
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}, 
                                "lr_type":"inv", "lr_param":{"init_lr":0.0003, "gamma":0.001, "power":0.75} }
    config["log_dir"] = "../log/"
    config["checkpoint_dir"] = "../checkpoint/"
    transfer_classification(config)
