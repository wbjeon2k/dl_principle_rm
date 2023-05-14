#/bin/bash

# CIL CONFIG
MODE="rm_iblurry" # joint, gdumb, icarl, rm, rm_iblurry ewc, rwalk, bic
# "default": If you want to use the default memory management method.
MEM_MANAGE="default" # default, random, reservoir, uncertainty, prototype.
RND_SEED=1
DATASET="cifar10" # mnist, cifar10, cifar100, imagenet
STREAM="online" # offline, online
EXP="blurry10" # disjoint, blurry10, blurry30
MEM_SIZE=500 # cifar10: k={200, 500, 1000}, mnist: k=500, cifar100: k=2,000, imagenet: k=20,000
TRANS="cutmix autoaug" # multiple choices: cutmix, cutout, randaug, autoaug

BACKBONE="basic" #basic, simplevit, vit_vanilla, vit_pretrained

N_WORKER=6
JOINT_ACC=0.0 # training all the tasks at once.
# FINISH CIL CONFIG ####################

UNCERT_METRIC="vr_randaug"
PRETRAIN="" INIT_MODEL="" INIT_OPT="--init_opt"

# iCaRL
FEAT_SIZE=2048

# BiC
distilling="--distilling" # Normal BiC. If you do not want to use distilling loss, then "".

#if [ -d "tensorboard" ]; then
#    rm -rf tensorboard
#    echo "Remove the tensorboard dir"
#fi

if [ "$DATASET" == "mnist" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1
    if [ "$BACKBONE" == "basic" ];
    then
        MODEL_NAME="mlp400"; LR=0.05
    else
        MODEL_NAME="ViT"; LR=0.001
    fi
    N_EPOCH=5; BATCHSIZE=16; OPT_NAME="sgd" SCHED_NAME="cos"
    if [ "${MODE_LIST[0]}" == "joint" ]
    then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]
    then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5
    fi
elif [ "$DATASET" == "cifar10" ]; then
    TOTAL=50000 N_VAL=250 N_CLASS=10 TOPK=1 N_RATIO=50 M_RATIO=10
    # MODEL_NAME="resnet18"
    if [ "$BACKBONE" == "basic" ];
    then
        MODEL_NAME="resnet18"; LR=0.05
    elif [ "$BACKBONE" == "vit" ];
    then
        MODEL_NAME="simplevit"; LR=0.005
    elif [ "$BACKBONE" == "vit_pretrained" ];
    then
        MODEL_NAME="vit_pretrained"; LR=0.005    
    elif [ "$BACKBONE" == "vit_vanilla" ];
    then
        MODEL_NAME="vit_vanilla"; LR=0.005       
    else
        MODEL_NAME="ViT"; LR=0.005
    fi
    N_EPOCH=256; BATCHSIZE=16; OPT_NAME="sgd" SCHED_NAME="cos"

    if [ "$MODEL_NAME"="vit_pretrained" ];
    then
        BATCHSIZE=8
    fi

    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=10 N_CLS_A_TASK=10 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=10 N_CLS_A_TASK=2 N_TASKS=5
    else
        N_INIT_CLS=2 N_CLS_A_TASK=2 N_TASKS=5

    fi
elif [ "$DATASET" == "cifar100" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=100 TOPK=1 N_RATIO=50 M_RATIO=10
    if [ "$BACKBONE" == "basic" ];
    then
        MODEL_NAME="resnet18"; LR=0.05
    elif [ "$BACKBONE" == "vit" ];
    then
        MODEL_NAME="simplevit"; LR=0.005
    elif [ "$BACKBONE" == "vit_pretrained" ];
    then
        MODEL_NAME="vit_pretrained"; LR=0.005    
    elif [ "$BACKBONE" == "vit_vanilla" ];
    then
        MODEL_NAME="vit_vanilla"; LR=0.005       
    else
        MODEL_NAME="ViT"; LR=0.005
    fi
    N_EPOCH=256; BATCHSIZE=16; OPT_NAME="sgd" SCHED_NAME="cos"

    if [ "$MODEL_NAME"="vit_pretrained" ];
    then
        BATCHSIZE=8
    fi
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=1
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=100 N_CLS_A_TASK=20 N_TASKS=5
    else
        N_INIT_CLS=20 N_CLS_A_TASK=20 N_TASKS=5
    fi

elif [ "$DATASET" == "imagenet" ]; then
    TOTAL=50000 N_VAL=0 N_CLASS=1000 TOPK=5
    if [ "$BACKBONE" == "basic" ];
    then
        MODEL_NAME="resnet34"; LR=0.05
    else
        MODEL_NAME="ViT"; LR=0.005
    fi
    N_EPOCH=100; BATCHSIZE=256; LR=0.05 OPT_NAME="sgd" SCHED_NAME="multistep"
    if [ "${MODE_LIST[0]}" == "joint" ]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    elif [[ "$EXP" == *"blurry"* ]]; then
        N_INIT_CLS=1000 N_CLS_A_TASK=100 N_TASKS=10
    else
        N_INIT_CLS=100 N_CLS_A_TASK=100 N_TASKS=10
    fi
else
    echo "Undefined setting"
    exit 1
fi

python main_optuna.py --mode $MODE --mem_manage $MEM_MANAGE --exp_name $EXP \
--dataset $DATASET \
--stream_env $STREAM  $INIT_MODEL $INIT_OPT --topk $TOPK \
--n_tasks $N_TASKS --n_cls_a_task $N_CLS_A_TASK --n_init_cls $N_INIT_CLS \
--rnd_seed $RND_SEED \
--model_name $MODEL_NAME --opt_name $OPT_NAME $PRETRAIN --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--n_worker $N_WORKER --n_epoch $N_EPOCH \
--memory_size $MEM_SIZE --transform $TRANS --uncert_metric $UNCERT_METRIC \
--feature_size $FEAT_SIZE $distilling --joint_acc $JOINT_ACC \
--backbone $BACKBONE \
--i_blurry \
--n_ratio $N_RATIO --m_ratio $M_RATIO