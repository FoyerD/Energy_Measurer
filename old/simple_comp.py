import torch
from torch import nn
from random import shuffle
from Logger import Logger
from time import sleep
import sys
import subprocess

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():
    assert (len(sys.argv) >= 1)
    
    job_id = str(sys.argv[1])
    prober = subprocess.Popen(["python", "./code_files/python_files/prob_nvsmi.py", job_id])
    sleep_time = 3
    list_size = 10000
    nn_iterations = 10000
    model = NeuralNetwork().to(device)
    sleep(sleep_time)

    #GPU computation
    for i in range(nn_iterations):
        X = torch.rand(1, 28, 28, device=device)
        y = torch.rand(1, 10, device=device)
        output = model(X)
        model.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(output,y)
        loss.backward()
        learning_rate = 0.01
        for f in model.parameters():
            f.data.sub_(f.grad.data * learning_rate)
        out = model(X)
        criterion = nn.MSELoss()

    #CPU compputation
    output_file = "./out_files/mesures/cpu_" + job_id + ".csv"
    logger = Logger(job_id=job_id, output_file=output_file)
    logger.setup_CPU()
    logger.log_headers()
    logger.log()
    for i in range(1):
        lst = list(range(list_size))
        shuffle(lst)
        lst.sort()
    logger.log()
    prober.kill()

if __name__ == "__main__":
    main()