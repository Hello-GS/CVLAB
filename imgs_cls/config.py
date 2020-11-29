class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 52            # set random seed
    workers = 4           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decays
    val_resume = None

    resume ='/disk/11712511/tyfb/checkpoints/fuxian/inceptionv4-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian/inceptionv4-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian-checkpoint.pth.tar'
    #resume = '/home/majian/codes/project/cme/imgs_cls/checkpoints/fuxian/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian-checkpoint.pth.tar'


    evaluate = False     # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = False  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    # dataset = "/data/majian/sunstorm/solar_flare/odd_train"
    # dataset = "/data/majian/sunstorm/solar_flare/even_train"

    #dataset = "/data/majian/cme/cme_by_incident_cropped_split"
    dataset = '/disk/event_dataset/dataset'

    #dataset = "/data/majian/cme/total_incident_diff_to_frame"
    #dataset = "/data/majian/cme/ori_total_incident_split"
    #dataset = "/data/majian/cme/ori_total_incident_split"
    submit_example =  "./data/cme_by_incident_cropped_split_label_list.csv"    # submit example file
    # submit_example =  "./data/test_example_key_frames.csv"
    checkpoints = "./checkpoints/resnet50/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian"   # rightoridatasplit     # path to save checkpoints
    # checkpoints = "./checkpoints/dconvresnet34/dconvresnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_imgnetmean_oridata"   # rightoridatasplit     # path to save checkpoints
    # checkpoints = "./checkpoints/resnext101_32x4d/resnext101_32x4d-model-sgd_bs64_lr_1e-2_CrossEntropy_resize256x256"
    log_dir = "./logs/resnet34"                   # path to save log files
    # log_dir = "./logs/resnext101_32x4d"
    submits = "./submits"                # path to saves submission files
    bs = 64              # batch size
    lr = 1e-2             # learning rate
    epochs = 25           # train epochs
    input_size = (512, 512)      # model input size or image resied
    num_classes = 2       # num of classes
    dropout = 0
    gpu_id = "1"          # default gpu id
    # model_name = "dconvresnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_imgnetmean_cropdiff"
    #model_name = "resnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_ownmean"         # model name to use # onlykeyframe
    model_name = 'resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian'
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    fp16 = True        # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32�?O1" means mixed�?O2" means except BN�?O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    loss_func = "CrossEntropy" # "CrossEntropy"�?FocalLoss"�?LabelSmoothCE"
    lr_scheduler = "step"  # lr scheduler method,"adjust","on_loss","on_acc","step"


    
configs = DefaultConfigs()
