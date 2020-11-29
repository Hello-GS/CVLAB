class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 52            # set random seed
    workers = 4           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay
    # resume = "/home/majian/codes/project/cme/imgs_cls/checkpoints/resnet50/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropd_fuxian-checkpoint.pth.tar"         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    resume = None
    # resume = "./checkpoints/fuxian/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian-checkpoint.pth.tar"

    # resume = "/home/majian/codes/project/cme/imgs_cls/checkpoints/fuxian/resnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_oridifffuxian/resnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_oridifffuxian-checkpoint.pth.tar"
    evaluate = False      # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = False  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    # dataset = "/data/majian/sunstorm/solar_flare/odd_train"
    # dataset = "/data/majian/sunstorm/solar_flare/even_train"
    # dataset = "/data/majian/cme/cme_by_incident"
    # dataset = "/data/majian/cme/total_incident_diff_to_frame"
    # dataset = "/data/majian/cme/ori_total_incident_split"
    # dataset = "/data/majian/cme/total_incident_cropped_diff_to_frame"

    dataset = "/disk/11712511/tyfb/data/dataset"
    test_folder = "/disk/11712511/tyfb/data/dataset/test"
    submit_example = "/home/majian/codes/project/cme/img_cls_add_triplet/data/cme_by_incident_cropped_split_label_list.csv"

    # dataset = "/data/majian/cme/ori_total_incident_split"
    # test_folder = "/data/majian/cme/ori_total_incident_split"
    # submit_example = "/home/majian/codes/project/cme/img_cls_add_triplet/data/ori_total_incident_split_label_list.csv"

    # dataset = "/data/majian/cme/total_incident_diff_to_frame"
    # test_folder = "/data/majian/cme/total_incident_diff_to_frame"
    # submit_example = "/home/majian/codes/project/cme/img_cls_add_triplet/data/total_incident_diff_to_frame_label_list.csv"

 
    # submit_example =  "./data/test_example_key_frames.csv"
    checkpoints = "./checkpoints/fuxian/inceptionv4-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian"   # rightoridatasplit     # path to save checkpoints
    # checkpoints = "./checkpoints/dconvresnet34/dconvresnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_imgnetmean_oridata"   # rightoridatasplit     # path to save checkpoints
    # checkpoints = "./checkpoints/resnext101_32x4d/resnext101_32x4d-model-sgd_bs64_lr_1e-2_CrossEntropy_resize256x256"
    log_dir = "./logs/fuxian"                   # path to save log files
    # log_dir = "./logs/resnext101_32x4d"
    submits = "./submits/"                # path to save submission files
    bs = 64  #64 begin               # batch size
    lr = 1e-2             # learning rate
    epochs = 40           # train epochs
    input_size = (512, 512)      # model input size or image resied
    num_classes = 2       # num of classes
    dropout = 0
    gpu_id = "2"          # default gpu id
    # model_name = "dconvresnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_imgnetmean_cropdiff" 
    model_name = "inceptionv4-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian"      # model name to use # onlykeyframe
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    fp16 = True          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32??O1" means mixed??O2" means except BN??O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    loss_func = "CrossEntropy" # "CrossEntropy"??FocalLoss"??LabelSmoothCE"
    lr_scheduler = "step"  # lr scheduler method,"adjust","on_loss","on_acc","step"
    print("#" * 50)
    print(model_name)
    print("#" * 50)

    
configs = DefaultConfigs()
