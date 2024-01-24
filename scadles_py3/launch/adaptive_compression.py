import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

import logging
import os
import time
from argparse import ArgumentParser

import scadles_py3.misc.helper_fns as misc
import scadles_py3.misc.models as models
import scadles_py3.datastreaming.iid_data as iid_data
import scadles_py3.datastreaming.noniid_data as noniid_data
import scadles_py3.datastreaming.data_partitioner as dp
import scadles_py3.misc.topkcompression as topk


class MPIOps(object):
    def __init__(self, world_size, async_op, device, param_shapes):
        self.world_size = world_size
        self.async_op = async_op
        self.device = device
        self.param_shapes = param_shapes

    def allreduce(self, model):
        for name, parameter in model.named_parameters():
            dist.all_reduce(parameter.grad, async_op=self.async_op, op=ReduceOp.SUM)

        return model

    def broadcast(self, model, rank=0):
        for _, param in model.named_parameters():
            if not param.requires_grad: continue
            dist.broadcast(tensor=param.data, src=rank, async_op=self.async_op)

        return model

    def layerwise_decompress(self, gathered_compgrads, gathered_ixs, i):
        p_shape = self.param_shapes[i]
        tensor = torch.zeros(p_shape).view(-1).to(self.device)
        for ix in range(len(gathered_compgrads)):
            tensor.data[gathered_ixs[ix]] += gathered_compgrads[ix]

        tensor = tensor.reshape(p_shape)
        return tensor

    def compression_allreduce(self, layer_values, layer_indices):
        reduced_grads = []
        for i in range(len(layer_values)):
            comp_grad = layer_values[i].to(self.device)
            comp_ixs = layer_indices[i].to(self.device)
            tensor_sizes = [torch.LongTensor([0]).to(self.device) for _ in range(self.world_size)]
            t_size = comp_grad.numel()
            dist.all_gather(tensor_sizes, torch.LongTensor([t_size]).to(self.device))

            tensor_list = []
            ix_list = []
            size_list = [int(size.item()) for size in tensor_sizes]
            max_size = max(size_list)
            if max_size > 0:
                for _ in size_list:
                    tensor_list.append(torch.zeros(size=(max_size,), dtype=torch.float32).to(self.device))
                    ix_list.append(torch.zeros(size=(max_size,), dtype=torch.long).to(self.device))
                if t_size != max_size:
                    g_padding = torch.zeros(size=(max_size - t_size,), dtype=torch.float32).to(self.device)
                    ix_padding = torch.zeros(size=(max_size - t_size,), dtype=torch.long).to(self.device)
                    comp_grad = torch.cat((comp_grad, g_padding), dim=0).to(self.device)
                    comp_ixs = torch.cat((comp_ixs, ix_padding), dim=0).to(self.device)

                dist.all_gather(tensor_list, comp_grad)
                dist.all_gather(ix_list, comp_ixs)

                reduced_grads.append(self.layerwise_decompress(tensor_list, ix_list, i))
            else:
                reduced_grads.append(None)

        return reduced_grads


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
        print(f'TRAINING_METRICS logged at step {step} epoch {epoch} lr {lr} lossval {losses.val} lossavg {losses.avg} '
              f'top1val {top1.val.cpu().numpy().item()} top1avg {top1.avg.cpu().numpy().item()} '
              f'top10val {topx.val.cpu().numpy().item()} top10avg {topx.avg.cpu().numpy().item()}')

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


def parse_args():
    parser = ArgumentParser(description='training launch on streaming data')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--eval-steps', type=int, default=200, help='# training steps after which test model performance')
    parser.add_argument('--epochs', type=int, help='# epochss to run as bulk dataloader simulator', default=50000)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--val-bsz', type=int, default=128)
    parser.add_argument('--train-bsz', type=int, default=32)
    parser.add_argument('--train-step', type=int, default=100)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='29500')
    parser.add_argument('--world-size', type=int, default=1, help='# overall procs to spawn across cluster')
    parser.add_argument('--local-rank', type=int, help='single node multiGPU process rank', default=0)
    parser.add_argument('--global-rank', type=int, help='multi-host single/multiGPU process rank', default=0)
    parser.add_argument("--backend", type=str, default='mpi')
    parser.add_argument('--omp-threads', type=int, default=1)
    parser.add_argument('--model-name', type=str, default='vgg19')
    parser.add_argument('--datatype', type=str, default='iid_cifar100')
    parser.add_argument('--global-bsz', type=int, default=256, help='cluster-wide aggregated batch-size')
    parser.add_argument('--determinism', type=str, default='false')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--dataset-size', type=int, default=50000, help='size of cifar10 or cifar100 for image models')
    parser.add_argument('--dir', type=str, default='/Users/sahiltyagi/Desktop/output/', help='train/val data store dir')
    parser.add_argument('--stream-freq', type=int, default=10)
    parser.add_argument('--kafka-dir', type=str, default='xxx', help='')
    parser.add_argument('--stream-mode', type=str, default='persistence')

    parser.add_argument('--async-op', type=int, default=0)
    parser.add_argument('--compression-ratio', type=float, default=0.2)
    parser.add_argument('--delta-threshold', type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.async_op == 0:
        args.async_op = False
    else:
        args.async_op = True

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    misc.set_seed(args.seed)
    args.determinism = False if args.determinism == 'false' else True

    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.local_rank))
    else:
        args.device = torch.device('cpu')

    args.compressor = topk.TopKCompressor(args.device)

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

    model = model_obj.return_model().to(args.device)
    opt = model_obj.return_optimizer()
    loss_fn = model_obj.return_lossfn()
    lr_scheduler = model_obj.return_lrschedule()
    num_parameters, model_size = misc.countparameters_memorysize(model)

    param_names = []
    param_shapes = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        dist.broadcast(tensor=p.data, src=0)
        param_names.append(name)
        param_shapes.append(list(p.size()))

    mpi_ops = MPIOps(args.world_size, args.async_op, args.device, param_shapes)

    if args.datatype == 'iid_cifar10' or args.datatype == 'noniid_cifar10':
        logging.info(f'using iid_cifar10 test set for model {args.model_name}')
        val_data = dp.load_CIFAR10Test(log_dir=args.dir, test_bsz=args.val_bsz, seed=args.seed,
                                       determinism=args.determinism)
    elif args.datatype == 'iid_cifar100' or args.datatype == 'noniid_cifar100':
        val_data = dp.load_CIFAR100Test(log_dir=args.dir, test_bsz=args.val_bsz, seed=args.seed,
                                        determinism=args.determinism)

    # training data..
    if args.datatype == 'iid_cifar10':
        data_loader = iid_data.bulkloader_iidcifar10(args.dir, args.train_bsz, args.world_size, args.global_rank, args.seed,
                                                args.determinism)
    elif args.datatype == 'iid_cifar100':
        data_loader = iid_data.bulkloader_iidcifar100(args.dir, args.train_bsz, args.world_size, args.global_rank, args.seed,
                                                 args.determinism)
    elif args.datatype == 'noniid_cifar10':
        args.input_shape = [3, 32, 32]
        args.output_shape = [1]
        data_loader = noniid_data.bulkloader_noniid_CIFAR10(train_dir=args.dir, t_id=args.global_rank,
                                                            world_size=args.world_size, train_bsz=args.train_bsz,
                                                            input_shape=args.input_shape, out_shape=args.output_shape,
                                                            seed=args.seed, total_labels=10, determinism=args.determinism)
    elif args.datatype == 'noniid_cifar100':
        args.input_shape = [3, 32, 32]
        args.output_shape = [1]
        data_loader = noniid_data.bulkloader_noniid_CIFAR100(train_dir=args.dir, t_id=args.global_rank,
                                                             world_size=args.world_size, train_bsz=args.train_bsz,
                                                             input_shape=args.input_shape, out_shape=args.output_shape,
                                                             seed=args.seed, total_labels=100, determinism=args.determinism)

    global_step = 0
    model.train()
    prev_itr_time = 0.0
    prev_buff_size = 0
    for epoch in range(args.epochs):
        losses, top1, topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        compress_signal = misc.AverageMeter()
        for input, label in data_loader:
            if args.stream_mode == 'persistence':
                q = args.stream_freq * prev_itr_time + prev_buff_size

                if q >= args.train_bsz:
                    wait_time = 0.0
                    prev_buff_size = q - args.train_bsz
                elif q < args.train_bsz:
                    wait_time = float(args.train_bsz - q) / float(args.stream_freq)
                    prev_buff_size = 0

            elif args.stream_mode == 'truncate':
                q = args.stream_freq * prev_itr_time
                if q >= args.train_bsz:
                    wait_time = 0.0
                elif q < args.train_bsz:
                    wait_time = float(args.train_bsz - q) / float(args.stream_freq)

            if args.datatype == 'noniid_cifar10' or args.datatype == 'noniid_cifar100':
                label = label.reshape([label.size()[0]])

            input, label = input.to(torch.float32).to(args.device), label.to(torch.long).to(args.device)
            local_bsz = float(label.size(dim=0))
            aggregate_bsz = torch.scalar_tensor(float(label.size(dim=0))).to(args.device)
            begin = time.time()
            out = model(input)
            loss = loss_fn(out, label)
            loss.backward()
            compute_time = time.time() - begin

            if lr_scheduler is not None:
                curr_lr = lr_scheduler.get_last_lr()
            else:
                curr_lr = opt.param_groups[0]['lr']

            dist.all_reduce(aggregate_bsz, op=ReduceOp.SUM)
            gradient_scaling = local_bsz / aggregate_bsz.item()
            # scaling the gradients for the given model
            for _, param in model.named_parameters():
                param.grad *= gradient_scaling

            bsz_scaling = aggregate_bsz.item() / args.global_bsz
            opt_lr = opt.param_groups[0]['lr']
            scaled_lr = opt_lr * bsz_scaling
            opt.param_groups[0]['lr'] = scaled_lr

            gradients = [p.grad for p in model.parameters()]
            og_norm = misc.gradient_norm(gradients)

            begin = time.time()
            with torch.no_grad():
                layer_values, layer_indices = [], []
                for ix in range(len(gradients)):
                    compress_tensor, _ = args.compressor.compress(gradients[ix], param_names[ix], args.compression_ratio)
                    layer_values.append(compress_tensor[0])
                    layer_indices.append(compress_tensor[1])
            compress_time = time.time() - begin

            begin = time.time()
            with torch.no_grad():
                lc_tensors = []
                for i in range(len(param_names)):
                    comp_tensors1 = layer_values[i], layer_indices[i]
                    ctx1 = gradients[i].numel(), param_shapes[i]
                    lc_tensors.append(args.compressor.decompress(tensors=comp_tensors1, ctx=ctx1))

            compression_norm = misc.gradient_norm(lc_tensors)
            compnorm_time = time.time() - begin

            # choose whether to use compression or not
            c_s = abs(og_norm - compression_norm) / og_norm
            compress_signal.update(val=c_s, n=1)

            if compress_signal.avg <= args.delta_threshold:
                cum_comp_signal = torch.scalar_tensor(int(1)).to(args.device)
            elif compress_signal.avg > args.delta_threshold:
                cum_comp_signal = torch.scalar_tensor(int(0)).to(args.device)

            dist.all_reduce(cum_comp_signal, op=ReduceOp.SUM)

            if cum_comp_signal.item() >= int(args.world_size / 2):
                print('going to apply compression')
                # sum all local batch-sizes to get current iteration's global batch-size
                begin = time.time()
                reduced_grad = mpi_ops.compression_allreduce(layer_values=layer_values, layer_indices=layer_indices)
                for p, g in zip(model.parameters(), reduced_grad):
                    p.grad = g
                sync_time = time.time() - begin

                # post aggregation gradient norms
                metric = 'compression'
                aggregated_norm = misc.gradient_norm(reduced_grad)

            elif cum_comp_signal.item() < int(args.world_size / 2):
                begin = time.time()
                for name, param in model.named_parameters():
                    dist.all_reduce(param.grad, op=ReduceOp.SUM)
                sync_time = time.time() - begin

                # aggregated gradient norm
                metric = 'no_compression'
                grads2 = [p.grad for p in model.parameters()]
                aggregated_norm = misc.gradient_norm(grads2)

            opt.step()
            opt.zero_grad()
            global_step += 1
            opt.param_groups[0]['lr'] = opt_lr
            prev_itr_time = compute_time + sync_time

            if lr_scheduler is not None:
                lr = curr_lr[0]
            else:
                lr = curr_lr

            if args.model_name == 'resnet152' or args.model_name == 'vgg19':
                if global_step > 0 and global_step % args.train_step == 0:
                    losses, top1, topx = training_accuracy(inputs=input, labels=label, output=out, loss=loss,
                                                           step=global_step, epoch=epoch, metrics=[losses, top1, topx],
                                                           topk=args.topk, lr=lr)

            validate_model(global_step, epoch, val_data, args.eval_steps, model, loss_fn, args.topk, args.device, lr)
            logging.info(
                f'ADAPTIVE_COMPRESSION training epoch {epoch} step {global_step} compute_time {compute_time} '
                f'sync_time {sync_time} compress_time {compress_time} compnorm_time {compnorm_time} og_norm {og_norm} '
                f'compression_norm {compression_norm} aggregation_norm {aggregated_norm} '
                f'local_bsz {int(local_bsz)} aggregate_bsz {int(aggregate_bsz.item())} grad_scaling {gradient_scaling} '
                f'lr_schedule {curr_lr} opt_lr {opt_lr} bsz_scaling {bsz_scaling} bsz_scaled_lr {scaled_lr} '
                f'wait_time {wait_time} bufferqueue_size {q} compression_signal {compress_signal.val} avg_compress_signal '
                f'{compress_signal.avg} metric {metric}')

        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == '__main__':
    main()