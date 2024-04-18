import numpy as np

from .data_classes import Operation

def sample_start_time(resolution=1):
    return 7 + np.random.randint(10 * resolution) / resolution

def sample_duration(resolution=1):
    sample = np.exp(0.4 + np.random.randn() * 0.5)
    sample = round(sample * resolution) / resolution

    return sample

def sample_number():
    return np.random.randint(25, 50)

def sample_operation(id, cardiac=False, resolution=1):
    start_time = sample_start_time(resolution=resolution)
    duration = sample_duration(resolution=resolution)

    return Operation(id, start_time, duration, cardiac)

def sample_day():
    operations = list()

    n_operations = sample_number()

    resolution = 4
    for i in range(n_operations):
        cardiac = (np.random.rand() < 0.2)
        op = sample_operation(id=i, cardiac=cardiac, resolution=resolution)
        operations.append(op)

    return operations

def main():
    pass

if __name__ == '__main__':
    main()