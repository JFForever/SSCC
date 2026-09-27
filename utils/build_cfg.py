from yacs.config import CfgNode as CN



_C = CN()



                                                                     

_C.OUTPUT_DIR = "./output"

_C.no_sim = False

_C.no_sim_weights = None

_C.cutimage = False

_C.count_list = [2, 3]

_C.RECORD_PATH = ""

                                                           

_C.RESUME = ""

                                                    

                                                

_C.SEED = -1

_C.USE_CUDA = True



                           

       

                           

_C.INPUT = CN()

_C.INPUT.SIZE = (224, 224)

_C.INPUT.TRAIN_SIZE = (224, 224)

_C.INPUT.TEST_SIZE = (224, 224)

        

_C.INPUT.CUTOUT_FACTOR = 0.5

_C.INPUT.CUTOUT_N = 1

_C.INPUT.CUTOUT_LEN = 16

               

_C.INPUT.RANDAUGMENT_N = 2

_C.INPUT.RANDAUGMENT_M = 10



                           

         

                           

_C.DATASET = CN()

                                     

_C.DATASET.ROOT = ""

_C.DATASET.NAME = ""

_C.DATASET.TRAIN_SPLIT = "train"

_C.DATASET.VAL_SPLIT = "val"

_C.DATASET.TEST_SPLIT = "test"

_C.DATASET.IMG_PATH = ""

_C.DATASET.ANNO_PATH = ""

_C.DATASET.TEXT_PATH = ""

_C.DATASET.TEXT_JSON = ""



                           

            

                           

_C.DATALOADER = CN()

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.TRAIN_X = CN()

_C.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"

_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32

_C.DATALOADER.TRAIN_X.SHUFFLE = True



                                  

_C.DATALOADER.TEST = CN()

_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"

_C.DATALOADER.TEST.BATCH_SIZE = 32

_C.DATALOADER.TEST.SHUFFLE = False



                           

       

                           

_C.MODEL = CN()

                                            

_C.MODEL.INIT_WEIGHTS = ""

_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = ""

_C.MODEL.BACKBONE.PRETRAINED = True



                           

              

                           

_C.OPTIM = CN()

_C.OPTIM.NAME = "adam"

_C.OPTIM.LR = 0.0003

_C.OPTIM.WEIGHT_DECAY = 5e-4

_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.SGD_DAMPNING = 0

_C.OPTIM.SGD_NESTEROV = False

_C.OPTIM.RMSPROP_ALPHA = 0.99

                                   

                                

_C.OPTIM.ADAM_BETA1 = 0.9

_C.OPTIM.ADAM_BETA2 = 0.999

                                           

                                            

                                           

                      

_C.OPTIM.STAGED_LR = False

_C.OPTIM.NEW_LAYERS = ()

_C.OPTIM.BASE_LR_MULT = 0.1

                         

_C.OPTIM.LR_SCHEDULER = "single_step"

                                                  

_C.OPTIM.STEPSIZE = (-1, )

_C.OPTIM.GAMMA = 0.1

_C.OPTIM.MAX_EPOCH = 10

                                                            

_C.OPTIM.WARMUP_EPOCH = -1

                           

_C.OPTIM.WARMUP_TYPE = "linear"

                                           

_C.OPTIM.WARMUP_CONS_LR = 1e-5

                                        

_C.OPTIM.WARMUP_MIN_LR = 1e-5

                                                      

                                   

_C.OPTIM.WARMUP_RECOUNT = True



                           

       

                           

_C.TRAIN = CN()

                                                 

                                                      

_C.TRAIN.CHECKPOINT_FREQ = 0

                                                 

_C.TRAIN.PRINT_FREQ = 10

                                                    

                                                       

_C.TRAIN.COUNT_ITER = "train_x"



_C.TRAIN.EVAL_PERIOD = 1

_C.TRAIN.EVAL_WAIT_FREE_MIB = 0

_C.TRAIN.EVAL_WAIT_RETRY_SEC = 60

_C.TRAIN.EARLY_STOP = CN()

_C.TRAIN.EARLY_STOP.ENABLED = False

_C.TRAIN.EARLY_STOP.MONITOR = "F1_3_unseen"

_C.TRAIN.EARLY_STOP.PATIENCE = 10

_C.TRAIN.EARLY_STOP.MIN_EPOCHS = 0

_C.TRAIN.EARLY_STOP.MIN_DELTA = 0.0

_C.TRAIN.HARD_NEG = CN()

_C.TRAIN.HARD_NEG.ENABLED = False

_C.TRAIN.HARD_NEG.WEIGHT = 1.0

_C.TRAIN.HARD_NEG.MARGIN = 0.05

_C.TRAIN.HARD_NEG.TOPK = 4

_C.TRAIN.HARD_NEG.START_EPOCH = 0

_C.TRAIN.HARD_NEG.POOL = "target_neighbors"

_C.TRAIN.HARD_NEG.NEIGHBOR_TOPK = 32

_C.TRAIN.ACCUM_STEPS = 1



                           

      

                           

_C.TEST = CN()

_C.TEST.EVALUATOR = "Classification"

_C.TEST.COMPUTE_CMAT = False

_C.TEST.SPLIT = "test"

                                                            

                                                          

                                         

_C.TEST.FINAL_MODEL = "last_step"



                           

                   

                           

_C.OT = CN()

_C.OT.ENABLED = False

_C.OT.MODE = "loss"

_C.OT.WEIGHT = 0.5

_C.OT.EPS = 0.1

_C.OT.MAX_ITER = 20

_C.OT.FP32 = True

_C.OT.ALIGN_BLEND = 0.5

_C.OT.ALIGN_BLEND_SCHEDULE = "constant"

_C.OT.ALIGN_BLEND_DECAY_EPOCHS = 0

_C.OT.SLOT_POOL = "cls_guided"

_C.OT.SRC_MASS = "uniform"

_C.OT.SRC_MASS_TEMP = 0.1

_C.OT.SRC_MASS_DETACH = True

_C.OT.SRC_MASS_SCHEDULE = "constant"

_C.OT.SRC_MASS_SWITCH_EPOCH = 0

_C.OT.SRC_MASS_AFTER = "same"



                           

                      

                           

_C.TTA = CN()

_C.TTA.ENABLED = False

_C.TTA.MODE = "multi_prompt"

_C.TTA.SCOPE = "all"

_C.TTA.INCLUDE_BASE_PROMPT = False

_C.TTA.AGGREGATION = "feature_mean"

_C.TTA.BASE_WEIGHT = 1.0

_C.TTA.TEMPLATE_WEIGHTS = []

_C.TTA.PROMPT_TEMPLATES = [

    "itap of a {}",

    "a bad photo of the {}.",

    "a origami {}.",

    "a photo of the large {}.",

    "a {} in a video game.",

    "art of the {}.",

    "a photo of the small {}.",

]



                           

                          

                           

_C.CHUNK = CN()

_C.CHUNK.ENABLED = False

_C.CHUNK.MODE = "filter_topk"

_C.CHUNK.SCOPE = "unseen_only"

_C.CHUNK.SIZE = 32

_C.CHUNK.TOPK_PER_CHUNK = 2

_C.CHUNK.FILTER_VALUE = -1e4

_C.CHUNK.BALANCED = True



                           

                                 

                           

_C.RERANK = CN()

_C.RERANK.ENABLED = False

_C.RERANK.MODE = "proto_residual"

_C.RERANK.SCOPE = "unseen_only"

_C.RERANK.TOPK = 4

_C.RERANK.SIM_TEMPERATURE = 0.2

_C.RERANK.LOGIT_TEMPERATURE = 1.0

_C.RERANK.GATE_MODE = "none"

_C.RERANK.GATE_TOPK = 0

_C.RERANK.GATE_THRESHOLD = 0.0



                           

                                          

                           

_C.CLASS_RERANK = CN()

_C.CLASS_RERANK.ENABLED = False

_C.CLASS_RERANK.SCOPE = "unseen_only"

_C.CLASS_RERANK.VISUAL_TOPK = 20

_C.CLASS_RERANK.VISUAL_CHUNK_SIZE = 256

_C.CLASS_RERANK.VISUAL_TEMPERATURE = 0.10

_C.CLASS_RERANK.PATCH_TOPK = 4

_C.CLASS_RERANK.PATCH_WEIGHT = 0.10

_C.CLASS_RERANK.MARGIN_WEIGHT = 0.08

_C.CLASS_RERANK.KNN_WEIGHT = 0.18

_C.CLASS_RERANK.RELATIVE_KNN_ENABLED = False

_C.CLASS_RERANK.RELATIVE_KNN_TOPK = 3

_C.CLASS_RERANK.RELATIVE_KNN_WEIGHT = 0.05

_C.CLASS_RERANK.PATCH_LOCAL_BLEND = 0.5

_C.CLASS_RERANK.ZSCORE_CLAMP = 5.0

_C.CLASS_RERANK.KNN_SCORE_CHUNK_SIZE = 2048



                           

                     

                           

_C.LAST = CN()

_C.LAST.ENABLED = False

_C.LAST.MODE = "replace"

_C.LAST.TOPK = 0

_C.LAST.TOPK_RATIO = 0.5

_C.LAST.FUSION_ALPHA = 0.1

_C.LAST.SIGMA = 0.0

_C.LAST.EPS = 1e-6



                           

                                 

                           

_C.EXP = CN()

_C.EXP.ORACLE_MODE = False

_C.EXP.ORACLE_TEXT_MODE = "multilabel_sentence"

_C.EXP.ORACLE_TEXT_TEMPLATE = "a photo containing {}."

_C.EXP.ORACLE_MAX_LABELS_PER_SAMPLE = 6

_C.EXP.ORACLE_INCLUDE_BASE_CAPTIONS = False

_C.EXP.ORACLE_SAMPLE_LIMIT = 0

_C.EXP.ORACLE_PROTO_WEIGHT = 1.0

_C.EXP.ORACLE_BCE_WEIGHT = 0.05

_C.EXP.ORACLE_PRIMARY_MSE_WEIGHT = 0.0

_C.EXP.EXTRA_CAPTION_JSON = ""

_C.EXP.TRAIN_CLASS_FILTER_JSON = ""

_C.EXP.TRAIN_BATCH_CLASS_SUBSET = False



_C.TRAINER = CN()



_C.TRAINER.TEXT = CN()

_C.TRAINER.TEXT.N_CLSN = 3

_C.TRAINER.TEXT.CTX_INIT = "a photo of a"

_C.TRAINER.TEXT.EXTRA_TEMPLATE = True





def get_cfg_default():

                                                                     

                                                           

                                                

  return _C.clone()



def reset_cfg(cfg, args):

    if args.output_dir is not None:

        cfg.OUTPUT_DIR = args.output_dir





def normalize_legacy_ot_cfg(cfg):

    if "OT" not in cfg:

        return cfg

    if "SRC_MASS_AFTER" not in cfg.OT:

        return cfg



    value = cfg.OT.SRC_MASS_AFTER

    if isinstance(value, bool):

        cfg.OT.SRC_MASS_AFTER = "same" if value else "off"

    return cfg





def setup_cfg(args):

    cfg = get_cfg_default()       



                                  

    if args.config_file:                            

        if str(args.config_file).endswith(".txt"):

            with open(args.config_file, "r", encoding="utf-8") as cfg_file:

                cfg_text = cfg_file.read()

            if cfg_text.startswith("- Training Parameters:"):

                cfg_text = "\n".join(cfg_text.splitlines()[1:])

            loaded_cfg = CN.load_cfg(cfg_text)

            loaded_cfg = normalize_legacy_ot_cfg(loaded_cfg)

            cfg.merge_from_other_cfg(loaded_cfg)

        else:

            cfg.merge_from_file(args.config_file)



                             

    reset_cfg(cfg, args)

    if getattr(args, "opts", None):

        cfg.merge_from_list(args.opts)



    cfg.freeze()



    return cfg

