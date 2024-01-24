from argparse import ArgumentParser
import logging
import os
import time

import scadles_py3.misc.helper_fns as misc
import scadles_py3.misc.models as models
import scadles_py3.datastreaming.kafka_subscriber as kafka_sub
import scadles_py3.datastreaming.data_partitioner as dp

# import helper.deterministic as det
# from helper import deterministic as det
# import helper.miscellaneous as misc
# import datastreamer.kafkasubscriber as subscriber
# import datastreamer.splitdata as sp

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


def parse_args():
    parser = ArgumentParser(description='training launch on streaming data')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--eval-steps', type=int, default=100, help='# training steps after which test model performance')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--val-bsz', type=int, default=128)
    parser.add_argument('--train-step', type=int, default=50)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='29500')
    parser.add_argument('--world-size', type=int, default=1, help='# overall procs to spawn across cluster')
    parser.add_argument('--local-rank', type=int, help='single node multiGPU process rank', default=0)
    parser.add_argument('--global-rank', type=int, help='multi-host single/multiGPU process rank', default=0)
    parser.add_argument("--backend", type=str, default='mpi')
    parser.add_argument('--omp-threads', type=int, default=1)
    parser.add_argument('--model-name', type=str, default='vgg19')
    parser.add_argument('--datatype', type=str, default='iid_cifar100')
    parser.add_argument('--stream-mode', type=str, default='truncate', help='either use `truncate` or `persistence`')
    parser.add_argument('--max-train-bsz', type=int, default=32, help='per rank max allowed batch-size')
    parser.add_argument('--global-bsz', type=int, default=256, help='cluster-wide aggregated batch-size')
    parser.add_argument('--determinism', type=str, default='false')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--dataset-size', type=int, default=50000, help='size of cifar10 or cifar100 for image models')
    parser.add_argument('--dir', type=str, default='/Users/sahiltyagi/Desktop/output/', help='train/val data store dir')
    parser.add_argument('--kafka-dir', type=str, default='xxx', help='')

    return parser.parse_args()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            #correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def training_accuracy(inputs, labels, output, loss, step, epoch, metrics, topk, lr):
    with torch.no_grad():
        losses, top1, topx = metrics[0], metrics[1], metrics[2]
        prec1 = accuracy(output.data, labels, topk=(1,))
        precx = accuracy(output.data, labels, topk=(topk,))
        top1.update(prec1[0], inputs.size(0))
        topx.update(precx[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        logging.info('TRAINING_METRICS logged at step %d epoch %d lr %f lossval %f lossavg %f top1val %f top1avg %f '
                     'topkused %d topxval %f topxavg %f', step, epoch, lr, losses.val, losses.avg,
                     top1.val.cpu().numpy().item(), top1.avg.cpu().numpy().item(), topk, topx.val.cpu().numpy().item(),
                     topx.avg.cpu().numpy().item())

        return losses, top1, topx


def validate_model(globalstep, epoch, val_loader, eval_step, model, criterion, topk, device, lr):
    if globalstep > 0 and globalstep % eval_step == 0:
        top1, topx, losses = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        model.eval()
        with torch.no_grad():
            for _, record in enumerate(val_loader, 0):
                inputs, labels = record
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(device), labels.cuda(device)

                output = model(inputs)
                loss = criterion(output, labels)

                prec1 = accuracy(output.data, labels, topk=(1,))
                top1.update(prec1[0], inputs.size(0))
                precx = accuracy(output.data, labels, topk=(topk,))
                topx.update(precx[0], inputs.size(0))
                losses.update(loss.item(), inputs.size(0))

            logging.info(f"VALIDATION METRICS logged on step %d epoch %d lr %f lossval %f lossavg %f top1val %f top1avg %f "
                         f"top10val %f top10avg %f", globalstep, epoch, lr, losses.val, losses.avg,
                         top1.val.cpu().numpy().item(), top1.avg.cpu().numpy().item(),
                         topx.val.cpu().numpy().item(), topx.avg.cpu().numpy().item())

        model.train()

def main():
    args = parse_args()
    # args.global_bsz = args.max_train_bsz * args.world_size
    #args.worker_datasize = args.dataset_size // args.world_size

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    misc.set_seed(args.seed)
    args.determinism = False if args.determinism == 'false' else True

    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.local_rank))
    else:
        args.device = torch.device('cpu')

    logging.basicConfig(filename=args.dir + 'trainer-' + str(args.global_rank) + '.log', level=logging.INFO)
    logging.info(f'going to use device {args.device}')
    logging.info(f'Model and execution related descriptions {args}')

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)

    dist.init_process_group(backend=args.backend, rank=args.global_rank, world_size=args.world_size)

    if args.model_name == 'resnet152' or args.model_name == 'vgg19':
        model_obj = models.initialize_models(model_name=args.model_name, lr=args.lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay, seed=args.seed, gamma=args.gamma,
                                             determinism=args.determinism)
        args.input_shape = [1, 3, 32, 32]
        args.output_shape = [1]

    if args.datatype == 'iid_cifar10' or args.datatype == 'noniid_cifar10':
        logging.info(f'using iid_cifar10 test set for model {args.model_name}')
        val_data = dp.load_CIFAR10Test(log_dir=args.dir, test_bsz=args.val_bsz, seed=args.seed,
                                       determinism=args.determinism)
    elif args.datatype == 'iid_cifar100' or args.datatype == 'noniid_cifar100':
        val_data = dp.load_CIFAR100Test(log_dir=args.dir, test_bsz=args.val_bsz, seed=args.seed,
                                        determinism=args.determinism)

    model = model_obj.return_model().to(args.device)
    opt = model_obj.return_optimizer()
    loss_fn = model_obj.return_lossfn()
    lr_scheduler = model_obj.return_lrschedule()

    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    if args.stream_mode == 'truncate':
        streaming = kafka_sub.Image_Truncated_Streamer(rank=args.global_rank, max_train_bsz=args.max_train_bsz,
                                                        input_shape=args.input_shape, output_shape=args.output_shape)
    elif args.stream_mode == 'persistence':
        streaming = kafka_sub.Image_Persistence_Streamer(rank=args.global_rank, max_train_bsz=args.max_train_bsz,
                                                          input_shape=args.input_shape, output_shape=args.output_shape)

    # broadcast the initial model among all ranks
    for _,p in model.named_parameters():
        if not p.requires_grad: continue
        dist.broadcast(tensor=p.data, src=0)

    global_step = 0
    epoch = 0
    # sum of batch-sizes to count as an epoch as individual samples processed by each worker vary based on stream-rates
    summed_batches = 0
    model.train()
    for loader in streaming.batchify_streams():

        for inputs, labels in loader:
            local_bsz = float(labels.size(dim=0))
            aggregate_bsz = torch.scalar_tensor(float(labels.size(dim=0))).to(args.device)
            inputs, labels = inputs.to(torch.float32).to(args.device), labels.to(torch.long).to(args.device)
            begin = time.time()
            out = model(inputs)
            loss = loss_fn(out, labels)
            loss.backward()
            compute_time = time.time() - begin

            if lr_scheduler is not None:
                curr_lr = lr_scheduler.get_last_lr()
            else:
                curr_lr = opt.param_groups[0]['lr']

            # sum all local batch-sizes to get current iteration's global batch-size
            dist.all_reduce(aggregate_bsz, op=ReduceOp.SUM)
            avg_rank_bsz = aggregate_bsz.item() / args.world_size
            gradient_scaling = local_bsz / avg_rank_bsz
            # scaling the gradients for the given model
            for _, param in model.named_parameters():
                param.grad *= gradient_scaling

            begin = time.time()
            for name,param in model.named_parameters():
                dist.all_reduce(param.grad, op=ReduceOp.SUM)
                param.grad = param.grad / args.world_size

            sync_time = time.time() - begin

            # log process id's consumed memory at this point
            mem_util = misc.process_memory()
            if global_step == 0:
                losses, top1, topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()

            # get batch-size scaling for current iteration to scale up the lr proportional to global_bsz
            bsz_scaling = aggregate_bsz.item() / args.global_bsz
            opt_lr = opt.param_groups[0]['lr']
            scaled_lr = opt_lr * bsz_scaling
            opt.param_groups[0]['lr'] = scaled_lr
            opt.step()
            opt.zero_grad()
            global_step += 1
            opt.param_groups[0]['lr'] = opt_lr

            if torch.cuda.is_available():
                gpu_mem = misc.get_device_memory(t_id=args.local_rank)
            else:
                gpu_mem = 0.

            disk_storage = float(misc.get_dir_size(path=args.kafka_dir)) / (1024. * 1024.)

            summed_batches += aggregate_bsz.item()
            if summed_batches >= args.dataset_size:
                summed_batches = 0
                epoch += 1
                if lr_scheduler is not None:
                    lr_scheduler.step()
                losses, top1, topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
                logging.info(f'completed epoch at step {global_step}')
                lr = curr_lr[0] if lr_scheduler is not None else curr_lr

            if args.model_name == 'resnet152' or args.model_name == 'vgg19':
                if global_step > 0 and global_step % args.train_step == 0:
                    lr = curr_lr[0] if lr_scheduler is not None else curr_lr
                    losses, top1, topx = training_accuracy(inputs=inputs, labels=labels, output=out, loss=loss,
                                                           step=global_step, epoch=epoch, metrics=[losses, top1, topx],
                                                           topk=args.topk, lr=lr)

            logging.info(f'STREAM SIMULATION training epoch {epoch} step {global_step} compute_time {compute_time} '
                         f'sync_time {sync_time} mem_util {mem_util} MB local_bsz {int(local_bsz)} aggregate_bsz '
                         f'{int(aggregate_bsz.item())} avg_bsz {avg_rank_bsz} grad_scaling {gradient_scaling} '
                         f'lr_schedule {curr_lr} opt_lr {opt_lr} bsz_scaling {bsz_scaling} bsz_scaled_lr {scaled_lr}'
                         f'gpu_mem {gpu_mem} disk_util {disk_storage} MB')

            if lr_scheduler is not None:
                lr = curr_lr[0]
            else:
                lr = curr_lr

            validate_model(global_step, epoch, val_data, args.eval_steps, model, loss_fn, args.topk, args.device, lr)


if __name__ == '__main__':
    main()