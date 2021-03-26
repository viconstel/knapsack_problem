import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from utils import test_files_generator


TEST_DIR_PATH = './partial_tests'


def bitfield(n, size):
    result = np.array([int(digit) for digit in bin(n)[2:]])
    return np.pad(result, (size - len(result), 0), constant_values=0)


def knapsack(items, capacity, bin_numbers):
    weights = np.array([x[0] for x in items])
    prices = np.array([x[1] for x in items])

    max_price = 0
    max_seq = None
    for i in range(len(bin_numbers)):
        selected_items = bin_numbers[i]
        cur_weight = np.sum(np.multiply(weights, selected_items))
        cur_price = np.sum(np.multiply(prices, selected_items))

        if cur_weight <= capacity and cur_price >= max_price:
            max_price = cur_price
            max_seq = selected_items

    return max_price, max_seq


def task(args):
    pid, num_threads, data, capacity = args
    size = data.shape[0]
    total = 2 ** size
    task_size = total // num_threads
    start = task_size * pid
    count = task_size if (pid != (num_threads - 1)) else task_size + total % num_threads
    bin_numbers = [bitfield(i, size) for i in range(start, start + count)]
    price, seq = knapsack(data, capacity, bin_numbers)

    return price, seq


def main(num_threads):
    files_gen = test_files_generator(TEST_DIR_PATH)
    start = time.time()
    for name, test, answer in files_gen:
        capacity = np.sum(test[:, 0]) / 2

        with mp.Pool(num_threads) as p:
            res = p.map(task, [(i, num_threads, test, capacity) for i in
                               range(num_threads)])

            res = sorted(res, key=lambda x: x[0], reverse=True)
            print(f'Test file: {name}, result: {all(answer == res[0][1])}')

    finish = time.time() - start
    print(f'---------Threads: {num_threads}, time: {finish}---------')
    return finish


if __name__ == '__main__':
    threads = [t for t in range(1, 9)]
    times = []

    for t in threads:
        times.append(main(t))

    times = np.array(times) / times[0]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.plot(threads, times, '-o', label='Average acceleration')
    ax.set_title('Brute force approach')
    ax.set_xlabel('Threads')
    ax.legend()
    fig.savefig('./brute_force.png')
