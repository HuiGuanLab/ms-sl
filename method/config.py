import os
import time
import torch
import argparse
from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile



class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--dset_name", type=str)
        self.parser.add_argument("--eval_split_name", type=str, default="val")
        self.parser.add_argument("--debug", action="store_true",
                                 help="debug (fast) mode, break all loops, do not load all data into memory.")

        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[1], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=8,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--no_pin_memory", action="store_true", help="No use pin_memory=True for dataloader")
        # training config
        self.parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.01,
                                 help="Proportion of training to perform linear learning rate warmup.")
        self.parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=10,
                                 help="number of epochs to early stop, use -1 to disable early stop")

        self.parser.add_argument("--bsz", type=int, default=128, help="mini-batch size")
        self.parser.add_argument("--eval_query_bsz", type=int, default=50, help="minibatch size at inference for query")
        self.parser.add_argument("--eval_context_bsz", type=int, default=200,
                                 help="mini-batch size at inference, for video/sub")
        self.parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--margin", type=float, default=0.2, help="margin for hinge loss")

        self.parser.add_argument("--train_span_start_epoch", type=int, default=0,
                                 help="which epoch to start training span prediction, -1 to disable")

        self.parser.add_argument("--hard_negative_start_epoch", type=int, default=20,
                                 help="which epoch to start hard negative sampling for video-level ranking loss,"
                                      "use -1 to disable")
        self.parser.add_argument("--hard_pool_size", type=int, default=20,
                                 help="hard negatives are still sampled, but from a harder pool.")
        # Model and Data config
        self.parser.add_argument("--max_sub_l", type=int, default=50,
                                 help="max length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_desc_l", type=int, default=30, help="max length of descriptions")
        self.parser.add_argument("--max_ctx_l", type=int, default=128)
        self.parser.add_argument("--train_path", type=str, default=None)
        self.parser.add_argument("--eval_path", type=str, default=None)

        self.parser.add_argument("--sub_feat_size", type=int, default=768, help="feature dim for sub feature")
        self.parser.add_argument("--q_feat_size", type=int, default=1024, help="feature dim for query feature")


        self.parser.add_argument("--no_norm_vfeat", action="store_true",
                                 help="Do not do normalization on video feat, use it only when using resnet_i3d feat")
        self.parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalization on text feat")

        self.parser.add_argument("--vid_feat_size", type=int, help="feature dim for video feature")
        self.parser.add_argument("--max_position_embeddings", type=int, default=300)
        self.parser.add_argument("--hidden_size", type=int, default=384)
        self.parser.add_argument("--n_heads", type=int, default=4)
        self.parser.add_argument("--input_drop", type=float, default=0.1, help="Applied to all inputs")
        self.parser.add_argument("--drop", type=float, default=0.1, help="Applied to all other layers")

        self.parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range for layers")
        # post processing

        self.parser.add_argument("--model_name", type=str, default='MS-SL_Net')
        self.parser.add_argument('--root_path', type=str)
        self.parser.add_argument('--visual_feature', type=str)
        self.parser.add_argument('--collection', type=str)
        self.parser.add_argument("--map_size", type=int, default=32)
        self.parser.add_argument('--use_sub', type=bool, default=False)
        self.parser.add_argument('--clip_scale_w', type=float, default=0.7)
        self.parser.add_argument('--frame_scale_w', type=float, default=0.3)
    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print("------------ Options -------------\n{}\n-------------------".format({str(k): str(v) for k, v in
                                                                                    sorted(args.items())}))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.no_core_driver = True
            opt.num_workers = 0
            opt.eval_query_bsz = 100
        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            opt.model_dir = os.path.join(opt.root_path, opt.collection,"results",opt.model_dir)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                setattr(opt, arg, saved_options[arg])
            opt.ckpt_filepath = os.path.join(opt.model_dir, self.ckpt_filename)
            opt.train_log_filepath = os.path.join(opt.model_dir, self.train_log_filename)
            opt.eval_log_filepath = os.path.join(opt.model_dir, self.eval_log_filename)
            opt.tensorboard_log_dir = os.path.join(opt.model_dir, self.tensorboard_log_dir)
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")
            
            opt.results_dir = os.path.join(opt.root_path, opt.collection,
                                           "results", "-".join([opt.dset_name, opt.exp_id,
                                                                       time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename, enclosing_dir="code", exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"],)
            opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
            opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
            opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
            opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        self.display_save(opt)


        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        opt.h5driver = None if opt.no_core_driver else "core"
        # num_workers > 1 will only work with "core" mode, i.e., memory-mapped hdf5
        opt.num_workers = 1 if opt.no_core_driver else opt.num_workers
        opt.pin_memory = not opt.no_pin_memory

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")

