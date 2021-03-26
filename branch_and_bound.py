import numpy as np
import time
from queue import Queue
from utils import test_files_generator


TEST_DIR_PATH = './tests'


class Node:
    def __init__(self, level, profit, weight, bound, n):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.bound = bound
        self.selected_elems = np.zeros(n, dtype=int)


def get_upper_bound(node, n, weights, prices, capacity):
    if node.weight >= capacity:
        return 0

    profit_bound = node.profit
    j = node.level + 1
    total_weight = node.weight

    while j < n and total_weight + weights[j] <= capacity:
        total_weight += weights[j]
        profit_bound += prices[j]
        j += 1

    if j < n:
        profit_bound += (capacity - total_weight) * prices[j] / weights[j]

    return profit_bound


def knapsack(items, capacity):
    item_size = len(items)
    temp = sorted(enumerate(items),
                  key=lambda k: float(k[1][1]) / k[1][0], reverse=True)
    items = [x[1] for x in temp]
    old_indices = [x[0] for x in temp]
    weights = np.array([x[0] for x in items])
    prices = np.array([x[1] for x in items])

    q = Queue()
    max_profit = 0
    max_seq = None
    root = Node(-1, 0, 0, 0.0, item_size)
    q.put(root)

    while not q.empty():
        cur_elem = q.get()

        if cur_elem.level == item_size - 1:
            continue

        next_elem = Node(0, 0, 0, 0.0, item_size)
        next_elem.level = cur_elem.level + 1
        next_elem.weight = cur_elem.weight + weights[next_elem.level]
        next_elem.profit = cur_elem.profit + prices[next_elem.level]
        next_elem.selected_elems = cur_elem.selected_elems.copy()
        next_elem.selected_elems[old_indices[next_elem.level]] = 1

        if next_elem.weight <= capacity and next_elem.profit > max_profit:
            max_profit = next_elem.profit
            max_seq = next_elem.selected_elems

        next_elem.bound = get_upper_bound(next_elem, item_size, weights,
                                          prices, capacity)

        if next_elem.bound > max_profit:
            q.put(next_elem)

        next_elem = Node(next_elem.level, 0, 0, 0.0, item_size)
        next_elem.weight = cur_elem.weight
        next_elem.profit = cur_elem.profit
        next_elem.bound = get_upper_bound(next_elem, item_size, weights,
                                          prices, capacity)
        next_elem.selected_elems = cur_elem.selected_elems.copy()

        if next_elem.bound > max_profit:
            q.put(next_elem)

    return max_profit, max_seq


if __name__ == '__main__':
    files_gen = test_files_generator(TEST_DIR_PATH)
    start = time.time()
    for name, test, answer in files_gen:
        capacity = np.sum(test[:, 0]) / 2
        _, sequence = knapsack(test, capacity)
        print(f'Test file: {name}, result: {all(answer == sequence)}')

    finish = time.time() - start
    print(f'---------Time: {finish}---------')
