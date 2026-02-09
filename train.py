import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BeitForImageClassification, CLIPVisionModel, ViTMAEModel, ViTForImageClassification,BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig, AutoModel, \
    AutoConfig, AutoTokenizer, CLIPModel, AutoTokenizer, AutoModelForMaskedLM,DebertaV2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from parameters import parse_args
#from model import * 
from data_utils import *
from data_utils.utils import *
import torchvision.models as models
from torch import nn
import random
from torch.cuda.amp import autocast
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_, constant_

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args, use_modal, local_rank):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #--------------------------------- Image ---------------------------------
    Log_file.info('read items...')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_images(
        os.path.join(args.root_data_dir, args.dataset, args.images))
    before_item_id_to_dic_text, before_item_name_to_id_text = read_news_bert(
        os.path.join(args.root_data_dir, args.dataset, args.news), args, tokenizer)


    Log_file.info('read behaviors...')
    item_num, item_id_to_keys, item_id_to_keys_text, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, neg_sampling_list, pop_prob_list = \
        read_behaviors(os.path.join(args.root_data_dir, args.dataset, args.behaviors), before_item_id_to_dic, before_item_id_to_dic_text,
                       before_item_name_to_id, before_item_id_to_name, args.max_seq_len, args.min_seq_len, Log_file)
    

    Log_file.info('combine news information...')
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask = get_doc_input_bert(item_id_to_keys_text, args)

    item_content = np.concatenate([
        x for x in
        [news_title, news_title_attmask,
         news_abstract, news_abstract_attmask,
         news_body, news_body_attmask]
        if x is not None], axis=1)





    Log_file.info('build dataset...')
    if use_modal:
        train_dataset = Build_MM_Dataset(u2seq=users_train, item_content=item_content, item_num=item_num, max_seq_len=args.max_seq_len,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           item_id_to_keys=item_id_to_keys, resize=args.CV_resize,
                                           neg_sampling_list=neg_sampling_list)
    else:
        train_dataset = Build_Id_Dataset(u2seq=users_train, item_num=item_num, max_seq_len=args.max_seq_len)

    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info('build dataloader...')

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=True,sampler=train_sampler)

    Log_file.info('build model...')

    model = ModelMM(args, item_num, use_modal, cv_model,bert_model, pop_prob_list).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    if 'None' not in args.pretrained_recsys_model:
        Log_file.info('load pretrained recsys model if not None...')
        ckpt_path = get_checkpoint("../pretrained_models/", args.pretrained_recsys_model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f"Model loaded from {ckpt_path}")
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    local_rank = int(os.environ["RANK"])
    #local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    setup_seed(args.seed)
    is_use_modal = True
    model_load = args.CV_model_load.replace('.pth', '')
    dir_label = str(args.seed)+str(args.arch)+ f'{model_load}_freeze_{args.freeze_paras_before}' + f"_add_adapter_to_{args.adding_adapter_to}" + f"_adapter_cv_lr_{args.adapter_cv_lr}" + f"_adapter_down_size_{args.adapter_down_size}" + f"_cv_adapter_down_size_{args.cv_adapter_down_size}_{args.adapter_type}"
    log_paras = f'{model_load}_bs_{args.batch_size}' \
                f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                f'_L2_{args.l2_weight}_dp_{args.drop_rate}_Flr_{args.fine_tune_lr_image}_{args.fine_tune_lr_text}'
    model_dir = os.path.join('./checkpoint_' + dir_label, 'cpt_' + log_paras)
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())

    Log_file.info(args)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if 'train' in args.mode:
        train(args, is_use_modal, local_rank)
    elif 'test' in args.mode:
        test(args, is_use_modal, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
