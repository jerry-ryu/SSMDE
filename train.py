# from __future__ import absolute_import, division, print_function
from trainer import Trainer
from options import MonodepthOptions
import sys
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np

options = MonodepthOptions()

def ddp_setup(dist_backend, dist_url, world_size, rank, local_rank):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    print("haha1")
    print()
    init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank = rank)
    # init_process_group(backend=dist_backend)
    print("haha2")
    torch.cuda.set_device(local_rank)
    torch.distributed.barrier()

def main(gpu, ngpus_per_node, opts):
    opts.dist_url = "tcp://127.0.0.1:15342"
    opts.rank = int(os.environ["RANK"])
    opts.world_size  = ngpus_per_node
    opts.local_rank = int(os.environ["LOCAL_RANK"])
    opts.dist_backend = "nccl"
    
    ddp_setup(opts.dist_backend, opts.dist_url, opts.world_size, opts.rank, opts.local_rank)
    print("hoho")
    if torch.distributed.get_rank()==0:
        print(f"RANK {opts.rank} WOLRD_SIZE {opts.world_size} LOCAL_RANK {opts.local_rank}")
    trainer = Trainer(opts)
    trainer.train()
    destroy_process_group()

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == "__main__":
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        opts = options.parser.parse_args([arg_filename_with_prefix])
    else:
        opts = options.parser.parse_args()
    
    # opts.world_size = 1
    # opts.rank = 0
    # local_rank = int(os.environ["LOCAL_RANK"])
    # print(local_rank)
    # nodes = ["127.0.0.1"]
    # mp.set_start_method('forkserver')

    # port = 40953
    # opts.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
    # print(opts.dist_url)
    # opts.dist_backend = 'nccl'
    # opts.gpu = None

    opts.ngpus_per_node = torch.cuda.device_count()
    print(opts.ngpus_per_node)
    # opts.num_workers = opts.num_workers
    # opts.ngpus_per_node = ngpus_per_node
    # print("haha")
    # opts.world_size = ngpus_per_node * opts.world_size
    

    mp.spawn(main, nprocs=opts.ngpus_per_node, args=(opts.ngpus_per_node, opts))
