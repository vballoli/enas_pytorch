import csv
import sys
import torch
import copy
import subprocess
import time
import signal
import os


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def latency_profiler(model, sample_arc, gpu=True, tensor=(1, 3, 224, 224)):
    tensor = torch.randn(tensor, requires_grad=False)
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    if gpu:
        eval_model = eval_model.cuda(device=1)
        tensor = tensor.cuda(device=1)
    times = []
    for _ in range(100):
        with torch.autograd.profiler.profile(use_cuda=gpu) as prof:
            eval_model(tensor, sample_arc)
        time = (prof.self_cpu_time_total) / (1000.0)
        times.append(time)
    del eval_model
    del tensor
    return sum(times) / 100.0

def cuda_latency_profiler(model, sample_arc, tensor=(1, 3, 224, 224), runs=10, device=0):
    tensor = torch.randn(tensor, requires_grad=False).cuda(device=device)
    eval_model = copy.deepcopy(model)
    eval_model = eval_model.cuda(device=device)
    eval_model.eval()
    times = []
    torch.cuda.synchronize(device=device)
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        eval_model(tensor, sample_arc)
        end.record()
        torch.cuda.synchronize(device=device)
        times.append(start.elapsed_time(end))
    return sum(times) / float(runs)

def get_energy(model, sample_arc, gpu_id=0, num_gpus=2):
    model = model.cuda()
    command = "nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=100"
    times = []
    for i in range(1030):
        if i == 20:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        tensor = torch.randn(1, 3, 224, 224).cuda()
        start.record()
        model(tensor, sample_arc) 
        end.record()
        torch.cuda.synchronize()
        if i == 1010:
            os.killpg(process.pid, signal.SIGTERM)
        times.append(start.elapsed_time(end)) 
    power = 0
    num = 0
    for i, line in enumerate(process.stdout.readlines()[1:-1]):
        if i % num_gpus == gpu_id:
            power += float(str(line)[2:-1].split(' ')[0])
            num += 1
    return (power/num)*sum(times)/len(times) 