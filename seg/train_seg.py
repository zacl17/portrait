'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from etc.Tensor_logger import Logger
from data.dataloader import get_dataloader
import models
from etc.help_function import *
from etc.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='basic_model/segmentation/ExtC3_SINet/setting/SINet.json', help='JSON file for configuration')
    parser.add_argument('-d', '--decoder_only', type=bool, default=False, help='Decoder only training')

    args = parser.parse_args()
    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    train_config = config['train_config']
    data_config = config['data_config']

    args.optim = train_config["optim"]
    args.lrsch = train_config["lrsch"]
    args.weight_decay = train_config["weight_decay"]
    others= args.weight_decay*0.01


    if train_config["loss"] == "Lovasz":
        train_config["num_classes"] = 1
        print("Use Lovasz loss ")
        Lovasz = True

    else:
        print("Use Cross Entropy loss ")
        Lovasz = False

    if not os.path.isdir(train_config['save_dir']):
        os.mkdir(train_config['save_dir'])

    print("Run : " + train_config["Model"])
    D_ratio=[]
    if train_config["Model"].startswith('Stage1'): 
        model = models.__dict__[train_config["Model"]](
            p=train_config["p"], q=train_config["q"], classes=train_config["num_classes"])

    elif train_config["Model"].startswith('Stage2'):
        model = models.__dict__[train_config["Model"]]( classes=train_config["num_classes"],
             p=train_config["p"], q=train_config["q"], stage1_W =train_config["stage1_W"])

    elif train_config["Model"].startswith('ExtremeC3Net_small'):
        model = models.__dict__[train_config["Model"]](classes=train_config["num_classes"],
                                                       p=train_config["p"], q=train_config["q"])
    elif train_config["Model"].startswith('Enc_SINet'):
        model = models.__dict__[train_config["Model"]](
            classes=train_config["num_classes"], p=train_config["p"], q=train_config["q"],
            chnn=train_config["chnn"])

    model_name = train_config["Model"]

    print("num_classes:"+str(train_config["num_classes"]))
    color_transform = Colorize(train_config["num_classes"])  

    #################### common model setting and opt setting  #######################################


    start_epoch = 0
    Max_val_iou = 0.0
    Max_name = ''

    if train_config["resume"]:
        if os.path.isfile(train_config["resume"]):
            print("=> loading checkpoint '{}'".format(train_config["resume"]))
            checkpoint = torch.load(train_config["resume"])
            start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            Max_name =  checkpoint['Max_name']
            Max_val_iou =  checkpoint['Max_val_iou']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(train_config["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(train_config["resume"]))

    use_cuda = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()

    if use_cuda:
        print("Use gpu : %d" % torch.cuda.device_count())
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make DataParallel")
        model = model.cuda()
        print("Done")

    ###################################stage Enc setting ##############################################
    if (not args.decoder_only):
        logger, this_savedir = info_setting(train_config['save_dir'], train_config["Model"])
        logger.flush()
        logdir = this_savedir.split(train_config['save_dir'])[1]
        # my_logger = Logger(8097, './logs/' + logdir, False)

        trainLoader,  valLoader, data = get_dataloader(data_config) 

        print(data['mean'])
        print(data['std'])
        weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
        print(weight)

        if train_config["loss"] == "Lovasz":
            from etc.lovasz_losses import lovasz_hinge
            criteria = lovasz_hinge(ignore=data_config["ignore_idx"])
        else:
            from etc.Criteria import CrossEntropyLoss2d
            criteria = CrossEntropyLoss2d(weight,ignore=data_config["ignore_idx"])  # weight

        if num_gpu > 0:
            weight = weight.cuda()
            criteria = criteria.cuda()

        if args.optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), train_config['learning_rate'], (0.9, 0.999), eps=1e-08,
                                            weight_decay=args.weight_decay)
        elif args.optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), train_config["learning_rate"], momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=True)
        elif args.optim == "RMS":
            optimizer = torch.optim.RMSprop(model.parameters(), train_config["learning_rate"], alpha=0.9,
                                            momentum=0.9, eps=1, weight_decay=args.weight_decay)
        # print(str(optimizer))
        init_lr = train_config["learning_rate"]

        if args.lrsch == "multistep":
            decay1 = train_config["epochs"] // 2
            decay2 = train_config["epochs"] - train_config["epochs"] // 4
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.1)
        elif args.lrsch == "step":
            step = train_config["epochs"] // 3
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)

        print("init_lr: " + str(train_config["learning_rate"]) + "   batch_size : " + str(data_config["batch_size"]) +
              args.lrsch + " sch use weight and class " + str(train_config["num_classes"]))
        print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim +
              "\tbn-weight : " + str(others))

    ################################ start Enc train ##########################################

        print("========== Stage 1 TRAINING ===========")
        for epoch in range(start_epoch, train_config["epochs"]):
            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " + str(lr))

            # train for one epoch
            # We consider 1 epoch with all the training data (at different scales)
            start_t = time.time()

            if data_config["Edge"] :
                lossTr, ElossTr, mIOU_tr = \
                    train_edge(num_gpu, trainLoader, model, criteria, optimizer, Lovasz, epoch, train_config["epochs"])
            else:
                lossTr, mIOU_tr = \
                    train(num_gpu, trainLoader, model, criteria, optimizer, Lovasz, epoch, train_config["epochs"])

                lossVal, ElossVal, mIOU_val = val_edge(num_gpu, valLoader, model, criteria, Lovasz)
            # evaluate on validation set

            end_t = time.time()

            if num_gpu > 1:
                this_state_dict = model.module.state_dict()
            else:
                this_state_dict = model.state_dict()

            if epoch >= train_config["epochs"]*0.6 :
                model_file_name = this_savedir + '/model_' + str(epoch + 1) + '.pth'
                torch.save(this_state_dict, model_file_name)

                if (Max_val_iou < mIOU_val):
                    Max_val_iou = mIOU_val
                    Max_name = model_file_name
                    print(" new max iou : " + Max_name + '\t' + str(mIOU_val))


            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f\t\t%.2f" % (
                epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val, lr, (end_t - start_t)))
            logger.flush()
            print("Epoch : " + str(epoch+1) + ' Details')
            print("Epoch No.: %d\t mIOU(tr) = %.4f\t mIOU(val) = %.4f \n" % (
                epoch+1, mIOU_tr, mIOU_val))

            save_checkpoint({
                'epoch': epoch + 1, 'arch': str(model),
                'state_dict': this_state_dict,
                'optimizer': optimizer.state_dict(),
                'lossTr': lossTr, 'lossVal': lossVal,
                'iouTr': mIOU_tr, 'iouVal': mIOU_val,
                'lr': lr,
                'Max_name': Max_name, 'Max_val_iou': Max_val_iou
            }, this_savedir + '/checkpoint.pth.tar')

            info = {
                'S1_train_loss': lossTr,
                'S1_val_loss': lossVal,

                'S1_train_iou': mIOU_tr,
                'S1_val_iou': mIOU_val,

                'S1_lr': lr
            }
            if data_config["Edge"]:
                info["S1_train_Eloss"]: ElossTr
                info["S1_val_Eloss"]: ElossVal

            scheduler.step()

        logger.close()


        # save the model also

        print(" S1 max iou : " + Max_name + '\t' + str(Max_val_iou))

        # exit(0)
        #########################################---Decoder setting---##################################################

        print("get max iou file : " + Max_name)
        if model_name.startswith('Enc'):
            model_name = "Dnc" + train_config["Model"].split('Enc')[1]

        if model_name.startswith('Stage1'):
            model_name = "Stage2" + train_config["Model"].split('Stage1')[1]
            model = models.__dict__[model_name]( classes=train_config["num_classes"],
             p=train_config["p"], q=train_config["q"], stage1_W =Max_name)

        elif model_name.startswith('ExtremeC3Net_small'):
            model = models.__dict__[model_name](classes=train_config["num_classes"],
            p=train_config["p"], q=train_config["q"], stage2=True, enc_file = Max_name )

        elif model_name.startswith('Dnc_SINet'):
            model = models.__dict__[model_name](
                classes=train_config["num_classes"], p=train_config["p"], q=train_config["q"],
                chnn=train_config["chnn"] , encoderFile= Max_name)

        else:
            print(model_name + " \t wrong model name")
            exit(0)

        if use_cuda:
            print("Use gpu : %d" % torch.cuda.device_count())
            num_gpu = torch.cuda.device_count()
            if num_gpu > 1:
                model = torch.nn.DataParallel(model)
                print("make DataParallel")
            model = model.cuda()
            print("Done")

        start_epoch = 0
        Max_val_iou = 0.0
        Max_name = ''

        data_config["batch_size"] = train_config["dnc_batch"]
        data_config["Enc"] = False
        data_config["scaleIn"] = 1

    ####################################################################################################


    logger, this_savedir = info_setting(train_config['save_dir'], model_name)
    logger.flush()
    logdir = this_savedir.split(train_config['save_dir'])[1]
    # my_logger = Logger(8097, './logs/' + logdir, False)

    print(this_savedir)

    trainLoader, valLoader, data = get_dataloader(data_config)
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    print(weight)

    if train_config["loss"] == "Lovasz":
        from etc.lovasz_losses import lovasz_hinge

        criteria = lovasz_hinge(ignore=data_config["ignore_idx"])
    else:
        from etc.Criteria import CrossEntropyLoss2d

        criteria = CrossEntropyLoss2d(weight, ignore=data_config["ignore_idx"])  # weight

    if num_gpu > 0:
        weight = weight.cuda()
        criteria = criteria.cuda()

    params_set = []
    names_set = []

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), train_config['learning_rate'], (0.9, 0.999), eps=1e-08,
                                        weight_decay=args.weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), train_config["learning_rate"], momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), train_config["learning_rate"], alpha=0.9,
                                        momentum=0.9, eps=1, weight_decay=args.weight_decay)
    # print(str(optimizer))
    train_config["learning_rate"] *= 0.1
    init_lr = train_config["learning_rate"]

    if args.lrsch == "multistep":
        decay1 = train_config["epochs"] // 2
        decay2 = train_config["epochs"] - train_config["epochs"] // 4
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.1)
    elif args.lrsch == "step":
        step = train_config["epochs"] // 3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)

    print("init_lr: " + str(train_config["learning_rate"]) + "   batch_size : " + str(data_config["batch_size"]) +
          args.lrsch + " sch use weight and class " + str(train_config["num_classes"]))
    print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim +
         "\tbn-weight : " + str(others))


###################################---start Dnc train-----###################################################

    print("========== DECODER TRAINING ===========")

        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    for epoch in range(start_epoch, train_config["epochs"]):
        if args.lrsch == "poly":
            scheduler.step(epoch)  ## scheduler 2
        elif args.lrsch == "warmpoly":
            curr_lr = scheduler.get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
        else:
            scheduler.step()

        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        start_t = time.time()

        if data_config["Edge"]:
            lossTr, ElossTr, mIOU_tr = \
                train_edge(num_gpu, trainLoader, model, criteria, optimizer, Lovasz, epoch, train_config["epochs"])

        else:
            lossTr, mIOU_tr = \
                train(num_gpu, trainLoader, model, criteria, optimizer, Lovasz, epoch, train_config["epochs"])

        lossVal, ElossVal, mIOU_val = val_edge(num_gpu, valLoader, model, criteria, Lovasz)
        end_t = time.time()
        # save the model also
        if num_gpu > 1:
            this_state_dict = model.module.state_dict()
        else:
            this_state_dict = model.state_dict()

        if epoch >= train_config["epochs"]*0.6 :
            model_file_name = this_savedir + '/model_' + str(epoch + 1) + '.pth'
            torch.save(this_state_dict, model_file_name)

            if (Max_val_iou < mIOU_val):
                Max_val_iou = mIOU_val
                Max_name = model_file_name
                print(" new max iou : " + Max_name + '\t' + str(mIOU_val))


        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f\t\t%.2f" % (
            epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val, lr, (end_t - start_t)))
        logger.flush()
        print("Epoch : " + str(epoch+1) + ' Details')
        print("Epoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f \n" % (
            epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val))

        save_checkpoint({
            'epoch': epoch + 1, 'arch': str(model),
            'state_dict': this_state_dict,
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr, 'lossVal': lossVal,
            'iouTr': mIOU_tr, 'iouVal': mIOU_val,
            'lr': lr,
            'Max_name': Max_name, 'Max_val_iou': Max_val_iou
        }, this_savedir + '/checkpoint.pth.tar')

        info = {
            'S2_train_loss': lossTr,
            'S2_val_loss': lossVal,

            'S2_train_iou': mIOU_tr,
            'S2_val_iou': mIOU_val,

            'S2_lr': lr

        }
        if data_config["Edge"]:
            info["S2_train_Eloss"]: ElossTr
            info["S2_val_Eloss"]: ElossVal


    logger.close()
    print(" new max iou : " + Max_name + '\t' + str(Max_val_iou))

    print("========== TRAINING FINISHED ===========")
    mean = data['mean']
    std = data['std']
    print(mean)
    print(std)






