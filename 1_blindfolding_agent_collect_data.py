from heapq import heappush, heappop

import math
import numpy as np


class PriorityQueue:

    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))

    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))

    def pop(self):
        priority, value = heappop(self.heap)
        return value

    def __len__(self):
        return len(self.heap)


def get_heuristic(h_fun, dim):
    def calc_h(cell):
        (i, j) = cell
        if h_fun == 'MANHATTAN':
            return (abs(dim - i) + abs(dim - j)) * 2
        elif h_fun == 'MANHATTAN_3':
            return (abs(dim - i) + abs(dim - j)) * 3
        elif h_fun == 'EUCLIDEAN':
            return math.sqrt(abs(dim - i) ** 2 + abs(dim - j) ** 2)
        elif h_fun == 'CHEBYSHEV':
            return max(abs(dim - i), abs(dim - j))
        else:
            # DEFAULT: MANHATTAN
            return abs(dim - i) + abs(dim - j)

    return calc_h


def repeated_a_star_search_q6(start, neighbors, heuristic, grid, blocked, parent):
    global cells_processed
    visited = set()
    distance = {start: 0}
    fringe = PriorityQueue()
    fringe.add(start)
    # print(start)
    while fringe:
        cell = fringe.pop()
        x, y = cell
        if cell in visited:
            continue
        cells_processed += 1
        s.add(cell)
        if cell == (inp_dim - (1 + padding), inp_dim - (1 + padding)):
            return repeated_a_star_reconstruct_path(parent, start, cell)
        if grid[x][y] == 1:
            # print('Reconstruct')
            blocked.append(cell)
            return repeated_a_star_reconstruct_path(parent, start, cell)
        visited.add(cell)
        for child in neighbors(cell, blocked):
            fringe.add(child, priority=distance[cell] + 1 + heuristic(child))
            if child not in distance or distance[cell] + 1 < distance[child]:
                distance[child] = distance[cell] + 1
                parent[child] = cell
    return []


def repeated_a_star_reconstruct_path(parent, start, end):
    global x_train, y_train, inp_dim
    if x[end[0]][end[1]] != 1:
        path = [end]
    else:
        path = []
    while end != start:
        end = parent[end]
        path.append(end)
    final_path = list(reversed(path))

    return final_path


def repeated_a_star_get_neighbors_q6(grid, dim):
    def get_adjacent_cells(cell, blocked):
        x, y = cell
        return ((x + i, y + j)
                for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]
                # (i, j) Represents movement from current cell - N,W,S,E direction eg: (1,0) means -> (x+1, y)
                # neighbor should be within grid boundary
                # neighbor should be an unblocked cell
                if 0 <= x + i < dim
                if 0 <= y + j < dim
                if (grid[x + i][y + j] == 0 and grid[x][y] != 1) or (grid[x + i][y + j] == 1 and grid[x][y] != 1)
                if (x + i, y + j) not in blocked
                )

    return get_adjacent_cells


import random


def repeated_a_star_get_shortest_path_q6(h_fun, grid):
    global trajectory_length, inp_dim, padding
    dim = len(grid[0])
    path = []
    start = (random.randint(0 + padding, dim - padding - 1), padding)
    shortest_path = []
    blocked = []
    parent = dict()

    while (inp_dim - (1 + padding), inp_dim - (1 + padding)) not in shortest_path:
        # print('repeat')
        shortest_path = repeated_a_star_search_q6(start, repeated_a_star_get_neighbors_q6(grid, dim),
                                                  get_heuristic(h_fun, dim), grid, blocked, parent)
        if len(shortest_path) == 0:
            return -1
        path.extend(shortest_path)
        try:
            start = shortest_path[len(shortest_path) - 1]
        except:
            pass
    if (inp_dim - (1 + padding), inp_dim - (1 + padding)) not in path:
        return -1
    else:
        trajectory_length = len(path)
        # print('length:', len(path))
        return path


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


if __name__ == '__main__':
    # grids: 1000, 5000, 10000, 20000, 50000
    # density - 0.3
    # grid - 25 x 25
    # window size - 7x7 ( padding - 3)
    num_grids = [50000]
    density = [0.3]
    window_sizes = [9]
    dimension = 50
    for g_num in num_grids:
        for p in density:
            for win in window_sizes:
                ct = g_num
                x_train = []
                y_train = []
                dim = dimension
                padding = int((win - 1) / 2)  # 1- for 3 window size, 2 for 5, 3 for 7..., 4-9, 5-11, 6-13
                while ct != 0:
                    print(ct)
                    cells_processed = 0
                    s = set()
                    x = np.pad(np.array(np.random.choice([0, 1], (dim * dim), p=[1 - p, p]).reshape(dim, dim)), padding, pad_with)
                    start = (random.randint(0 + padding, dim - padding - 1), padding)
                    x[start] = 0
                    inp_dim = dim + padding + padding
                    x[inp_dim - 1 - padding, inp_dim - 1 - padding] = 0
                    path = repeated_a_star_get_shortest_path_q6('MANHATTAN', x)
                    if path != -1:
                        # DATA COLLECTION
                        actual_path = []
                        for c in path:
                            if c not in actual_path or actual_path[len(actual_path) - 1] != c:
                                actual_path.append(c)

                        x_train_temp = []
                        y_train_temp = []
                        seen = dict()
                        visited = []
                        x[start] = 0
                        for pos in range(len(actual_path)):
                            visited.append(actual_path[pos])
                            (x1, y1) = actual_path[pos]
                            inp = np.zeros((inp_dim, inp_dim))
                            inp[start[0], start[1]] = 2
                            inp[inp_dim - 1 - padding, inp_dim - 1 - padding] = 3

                            for a in range(inp_dim):
                                for pad in range(0, padding):
                                    inp[a, pad] = 1
                                    inp[pad, a] = 1
                                    inp[inp_dim - 1 - pad, a] = 1
                                    inp[a, inp_dim - 1 - pad] = 1
                            # dataset preparation
                            # print(inp)

                            for coord in seen:
                                if coord not in visited:
                                    inp[coord] = x[coord]
                            for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
                                if x1 + i == 0 or x1 + i == inp_dim - 1 - padding or y1 + i == 0 or y1 + i == inp_dim - 1 - padding:
                                    pass
                                else:
                                    inp[x1 + i][y1 + j] = x[x1 + i][y1 + j]
                                    seen[(x1 + i, y1 + j)] = x[x1 + i][y1 + j]
                            for coord in visited:
                                inp[coord] = 4
                            # print(inp)
                            # print(inp)
                            temp_parent = (x1, y1)
                            if pos == len(actual_path) - 1:
                                temp_end = (inp_dim - (1 + padding), inp_dim - (1 + padding))
                            else:
                                temp_end = actual_path[pos + 1]
                            y_move = temp_end[1] - temp_parent[1]
                            x_move = temp_end[0] - temp_parent[0]
                            move = (x_move, y_move)
                            data = inp.copy()
                            data[actual_path[pos]] = 2
                            grid = data[x1 - padding:x1 + 1 + padding, y1 - padding:y1 + 1 + padding]
                            x_train_data = grid.copy()
                            x_train_temp.append(x_train_data)
                            if move == (0, -1):
                                y_train_data = 0
                            elif move == (0, 1):
                                y_train_data = 1
                            elif move == (-1, 0):
                                y_train_data = 2
                            elif move == (1, 0):
                                y_train_data = 3
                            else:
                                y_train_data = 4
                            y_train_temp.append(y_train_data)
                        y_train_temp = y_train_temp
                        x_train_temp = x_train_temp
                        x_train.extend(x_train_temp)
                        y_train.extend(y_train_temp)
                        ct -= 1
                #             break
                # for q in x_train:
                #     print(q)
                with open('dataset_window_' + str(win) + '.npy', 'wb') as f:
                    np.save(f, np.array(x_train))
                    np.save(f, np.array(y_train))
