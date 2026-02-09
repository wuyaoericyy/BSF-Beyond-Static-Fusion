import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from model.mm_td_rec import MM_TD_Rec
from model.mm_td_rec_mc import MM_TD_Rec_MC
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from data_utils import *
from data_utils.utils import *
import os
from tqdm import tqdm

def load_output(directory, item_id, prefix=''):
    file_path = os.path.join(directory, f"{prefix}_{item_id}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_num = 1
    model_list = [MM_TD_Rec, MM_TD_Rec_MC]

    config = Config(model=model_list[model_num], config_file_list=['config/config_MM_TD_Rec.yaml'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)
    

    # dataset filtering
    logger.info('create dataset...')
    dataset = create_dataset(config, db_path=os.path.join(config['root_data_dir'],config['dataset'], config['dataset'] + '.lmdb'), max_seq_len=config['max_seq_len'])
    logger.info(dataset)
    item_num = dataset.item_num
    id_to_index = dataset.field2token_id
    index_to_id = dataset.field2id_token
    index_to_keys = [u'{}'.format(i).encode('ascii') for i in index_to_id['item_id']]
    index_to_content = None
    text_emb = None
    image_emb = None
    #item_id_temp = index_to_id['item_id'][item_index_temp]
    #item_index = id_to_index['item_id'][item_id_temp]

    #--------------------------------- Text ---------------------------------
    if config['from_pretrain_transformer']:
        logger.info('load transformer pretrain...')
        empty_text_emb = load_output('stored_vectors_' + config['dataset'] + '/bert_outputs', 'pad', prefix='bert')
        blank_image_emb = load_output('stored_vectors_' + config['dataset'] + '/vit_outputs', 'blank', prefix='vit')
        
        text_emb = []
        image_emb = []
        for i in tqdm(range(item_num)):
            item_id = index_to_id['item_id'][i]
            bert_output_loaded = load_output('stored_vectors_' + config['dataset'] + '/bert_outputs', item_id, prefix='bert')
            if bert_output_loaded is None:
                bert_output_loaded = empty_text_emb
            vit_output_loaded = load_output('stored_vectors_' + config['dataset'] + '/vit_outputs', item_id, prefix='vit')
            if vit_output_loaded is None:
                vit_output_loaded = blank_image_emb
            text_emb.append(bert_output_loaded)
            image_emb.append(vit_output_loaded)
        text_emb = torch.stack(text_emb).to(config['device'])
        image_emb = torch.stack(image_emb).to(config['device'])
        
    img_size = to_2tuple(config['CV_resize'])
    patch_size = to_2tuple(config['patch_size'])
    grid_size = ((img_size[0] - patch_size[0]) // config['stride'] + 1, (img_size[1] - patch_size[1]) // config['stride'] + 1)
    num_patches = grid_size[0] * grid_size[1]

    # dataset splitting
    logger.info('dataset splitting...')
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    if model_num == 4:
        model = model_list[model_num](config, train_data.dataset).to(config['device'])
    elif model_num == 0 or model_num == 1 or model_num == 2:
        model = model_list[model_num](config, train_data.dataset, index_to_id, index_to_keys, num_patches, text_emb, image_emb)
        model = model.to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    #flops = get_flops(model, dataset, config["device"], logger, transform)
    #logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")