from keras.models import model_from_json

json_file = open('models/nn_dataset_density_0_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/nn_dataset_density_0_3.h5")

json_file2 = open('models/dataset_window_9.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
# load weights into new model
model2.load_weights("models/dataset_window_9.h5")
print("Loaded model from disk")

action = ['left', 'right', 'up', 'down', 'stop']

import math
import time
from heapq import heappush, heappop

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
            return abs(dim - i) + abs(dim - j)
        elif h_fun == 'EUCLIDEAN':
            return math.sqrt(abs(dim - i) ** 2 + abs(dim - j) ** 2)
        elif h_fun == 'CHEBYSHEV':
            return max(abs(dim - i), abs(dim - j))
        else:
            return max(abs(dim - i), abs(dim - j))

    return calc_h


def repeated_a_star_search_q6(start, neighbors, heuristic, grid, blocked, parent):
    inp_dim = len(grid[0])
    visited = set()
    distance = {start: 0}
    fringe = PriorityQueue()
    fringe.add(start)
    while fringe:
        cell = fringe.pop()
        x, y = cell
        if cell in visited:
            continue
        #         s.add(cell)
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
    global x_train, y_train
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


def repeated_a_star_get_shortest_path_q6(h_fun, grid, start):
    global trajectory_length, padding
    inp_dim = len(grid[0])
    path = []
    shortest_path = []
    blocked = []
    parent = dict()

    while (inp_dim - (1 + padding), inp_dim - (1 + padding)) not in shortest_path:
        # print('repeat')
        shortest_path = repeated_a_star_search_q6(start, repeated_a_star_get_neighbors_q6(grid, inp_dim),
                                                  get_heuristic(h_fun, inp_dim), grid, blocked, parent)
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


CELL_DIM = 15


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=2):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


def draw_cell(data,img2, size):
    win_size=int(size)
    total_rows=CELL_DIM
    for i in range(0,win_size,total_rows):
        for j in range(0,win_size,total_rows):
            # Draw empty cell
            if(data[int(i/total_rows)][int(j/total_rows)]==0):
                img2=cv2.rectangle(img2,(i+1,j+1),(i+(total_rows-1),j+(total_rows-1)),(255,255,255),-1)
            # Draw block
            if(data[int(i/total_rows)][int(j/total_rows)]==1):
                img2=cv2.rectangle(img2,(i+1,j+1),(i+(total_rows-1),j+(total_rows-1)),(0,0,0),-1)
            # Draw source
            if(data[int(i/total_rows)][int(j/total_rows)]==2):
                img2=cv2.rectangle(img2,(i+1,j+1),(i+(total_rows-1),j+(total_rows-1)),(0,0,255),-1)
            # Draw target
            if(data[int(i/total_rows)][int(j/total_rows)]==3):
                img2=cv2.rectangle(img2,(i+1,j+1),(i+(total_rows-1),j+(total_rows-1)),(0,255,0),-1)
            # Draw Visited cell
            if(data[int(i/total_rows)][int(j/total_rows)]==4):
                img2=cv2.rectangle(img2,(i+1,j+1),(i+(total_rows-1),j+(total_rows-1)),(255,0,0),-1)
    return img2


import cv2


# print(x)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


iterations = 3000
print('Iteration,Agent,Failure,Path Length,Time')
while iterations != 0:
    broke = 0
    p = 0.3
    grid_dim = 50
    padding = 4
    x = np.pad(np.array(np.random.choice([0, 1], (grid_dim * grid_dim), p=[1 - p, p]).reshape(grid_dim, grid_dim)), padding, pad_with)
    dim = len(x[0])
    start = (padding, padding)
    x[start[0], start[1]] = 0
    #     print(dim - 1 - padding, dim - 1 - padding)
    x[dim - 1 - padding, dim - 1 - padding] = 0
    # print(x)
    start_manh = time.perf_counter()
    a_star_path = repeated_a_star_get_shortest_path_q6('MANHATTAN', x, start)
    end_manh = time.perf_counter()

    if a_star_path != -1:
        # print(a_star_path)
        # print(len(set(a_star_path)))
        print(str(iterations) + ',Blindfolded,NO,' + str(len(a_star_path)) + ',' + str(round(end_manh - start_manh, 5)))
    else:
        broke = 1
        #         print('Unsolvable')
        continue

    nn_path = 0
    WINDOW_SIZE = 9
    if a_star_path != -1:
        for mod in ['NN', 'CNN']:
            nn_path = 0
            broke = 0
            if mod == 'NN':
                m = model
            else:
                m = model2
            c = 0
            inp = np.zeros((dim, dim))
            inp[start[0], start[1]] = 2
            inp[dim - 1 - padding, dim - 1 - padding] = 3
            seen = dict()
            visited = []
            for a in range(dim):
                for pad in range(0, padding):
                    inp[a, pad] = 1
                    inp[pad, a] = 1
                    inp[dim - 1 - pad, a] = 1
                    inp[a, dim - 1 - pad] = 1
            for coord in seen:
                inp[coord] = x[coord]
            (x1, y1) = start
            for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
                if (x1 + i == dim - 1 - padding) and (y1 + j == dim - 1 - padding):
                    pass
                else:
                    inp[x1 + i][y1 + j] = x[x1 + i][y1 + j]
                    seen[(x1 + i, y1 + j)] = x[x1 + i][y1 + j]
            # print(inp)
            # img=np.ones((( (grid_dim + (2*padding))*CELL_DIM),( (grid_dim + (2*padding))*CELL_DIM),3))
            # img=draw_grid(img,((grid_dim + (2*padding)),(grid_dim + (2*padding))),(0, 0, 0),1)
            # img=draw(inp,img,(grid_dim + (2*padding))*CELL_DIM )
            # cv2.imshow("frmae", img)
            # cv2.waitKey(0)
            buffer = []
            if a_star_path != -1:
                data = inp.copy()
                grid = data[x1 - padding:x1 + 1 + padding, y1 - padding:y1 + 1 + padding]
                start_manh = time.perf_counter()
                a = m.predict(grid.reshape((1, WINDOW_SIZE, WINDOW_SIZE)))
                move = action[np.argmax(a)]
                pos = start
                visited.append(pos)
                nn_path += 1
                while move != 'stop':
                    visited.append(pos)
                    # print(pos)
                    data[pos] = 0
                    if move == 'left':
                        pos = (pos[0], pos[1] - 1)
                    elif move == 'right':
                        pos = (pos[0], pos[1] + 1)
                    elif move == 'up':
                        pos = (pos[0] - 1, pos[1])
                    elif move == 'down':
                        pos = (pos[0] + 1, pos[1])
                    else:
                        pos = pos
                    if pos == (dim - 1 - padding, dim - 1 - padding):
                        break

                    (x1, y1) = pos
                    for coord in seen:
                        data[coord] = x[coord]
                    for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
                        if (x1 + i == dim - 1 - padding) and (y1 + j == dim - 1 - padding):
                            pass
                        else:
                            data[x1 + i][y1 + j] = x[x1 + i][y1 + j]
                            seen[(x1 + i, y1 + j)] = x[x1 + i][y1 + j]
                    for coord in visited:
                        data[coord] = 4
                    data[pos] = 2
                    grid = data[x1 - padding:x1 + 1 + padding, y1 - padding:y1 + 1 + padding]
                    #        print(grid)
                    buffer.append(pos)
                    if len(buffer) > 12:
                        buffer.pop(0)
                    if len(set(buffer)) < 5 and len(buffer) == 12:
                        broke = 1
                        # infinite loop
                        break
                    # img=draw(data,img,(grid_dim + (2*padding))*CELL_DIM )
                    # cv2.imshow("frmae", img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    try:
                        a = m.predict(grid.reshape((1, WINDOW_SIZE, WINDOW_SIZE)))
                        move = action[np.argmax(a)]
                    except:
                        end_manh = time.perf_counter()
                        print(str(iterations) + ',' + mod + ',YES: Invalid Direction Beyond Wall,' + str(nn_path) + ',' + str(round(end_manh - start_manh, 5)))
                    nn_path += 1
                end_manh = time.perf_counter()

            if broke == 1:
                print(str(iterations) + ',' + mod + ',YES: Infinite Loop,' + str(nn_path) + ',' + str(round(end_manh - start_manh, 5)))
            else:
                print(str(iterations) + ',' + mod + ',NO,' + str(nn_path) + ',' + str(round(end_manh - start_manh, 5)))
        iterations -= 1
