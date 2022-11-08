# -*- coding: utf-8 -*-
"""tt=

@author: baboudiab
"""
print('\n Day 1')
with open('input1.txt') as f:
    lines = list(map(int, f.read().splitlines()))
    last = 3000
    counter = 0

    for i in lines:
        if i > last:
            counter += 1
        last = i

    print(counter)

with open('input1.txt') as f:
    lines = list(map(int, f.read().splitlines()))
    counter = 0

    for i in range(len(lines) - 3):
        if sum(lines[i + 1 : i + 1 + 3]) > sum(lines[i : i + 3]):
            counter += 1

    print(counter)
    
print('\n Day 1 methode 2')

import pandas as pd

df=pd.read_excel('Python Bachar.xlsx','Day 1',skiprows=0)

df.columns=['Data']

df['Sum_offset_1'] = df['Data'].diff()

df['Sum_offset_2'] = df['Data'].rolling(3).sum().diff()


print('Réponse ex 1 :', len(df[df['Sum_offset_1']>0]) )
print('Réponse ex 2 :', len(df[df['Sum_offset_2']>0]) )

print('\n Day 2')

#Part 1
inputFile = open('input2.txt', 'r')
h_pos = 0
v_pos = 0
for line in inputFile.read().splitlines():
	direction, value = line.split()
	if direction[0] == 'f':
		h_pos += int(value)
	elif direction[0] == 'u':
		v_pos -= int(value)
	elif direction[0] == 'd':
		v_pos += int(value)
print(h_pos * v_pos)
inputFile.close()


#Part 2
inputFile = open('input2.txt', 'r')
h_pos = 0
v_pos = 0
aim = 0
for line in inputFile.read().splitlines():
    direction, value = line.split()
    if direction[0] == 'f':
        h_pos += int(value)
        v_pos += aim * int(value)
    elif direction[0] == 'u':
        aim -= int(value)
    elif direction[0] == 'd':
        aim += int(value)
print(h_pos * v_pos)

print('\n Day 2 part 1 methode 2')

import pandas as pd

df = pd.read_excel('Python Bachar.xlsx', 'Day 2', skiprows=0)
df.columns=['Data']
df['Val'] = df['Data'].apply(lambda x: int(x.split(' ')[1]))
df['Data'] = df['Data'].apply(lambda x: x.split(' ')[0].strip())
df = df.groupby('Data').agg({'Val': sum}).reset_index()
df = dict(zip(df['Data'], df['Val']))
print (df)
print('Forward :', df['forward'], 
      '\nDepth :', df['down']-df['up'],
      '\nForward * Depth :', df['forward'] * (df['down']-df['up']),)


print('\n Day 3')


#Part 1:
inputFile = open('input3.txt', 'r')
l = inputFile.read().splitlines()
gamma_rate = [0] * len(l[0])
epsilon_rate = [0] * len(l[0])
for i in range(len(l[0])):
	zero = 0
	one = 0
	for j in range(len(l)):
		zero += l[j][i] == '0'
		one += l[j][i] == '1'
	if zero > one:
		gamma_rate[i] = '0'
		epsilon_rate[i] = '1'
	elif one > zero:
		gamma_rate[i] = '1'
		epsilon_rate[i] = '0'
decimal_gamma = int(''.join(gamma_rate), 2)
decimal_epsilon = int(''.join(epsilon_rate), 2)
print(decimal_gamma * decimal_epsilon)
inputFile.close()


#Part 2:
inputFile = open('input3.txt', 'r')

l = inputFile.read().splitlines()
m = l.copy()

for i in range(len(l[0])):
	zero = 0
	one = 0
	for j in range(len(l)):
		zero += l[j][i] == '0'
		one += l[j][i] == '1'
	if zero > one:
		oxygen = '0'
	elif one > zero:
		oxygen = '1'
	else:
		oxygen = '1'
	if len(l) > 1:
		nextL = []
		for j in range(len(l)):
			if l[j][i] == oxygen:
				nextL.append(l[j])
		l = nextL
	
	zero = 0
	one = 0
	for j in range(len(m)):
		zero += m[j][i] == '0'
		one += m[j][i] == '1'
	if zero > one:
		co2 = '1'
	elif one > zero:
		co2 = '0'
	else:
		co2 = '0'
	if len(m) > 1:
		nextM = []
		for j in range(len(m)):
			if m[j][i] == co2:
				nextM.append(m[j])
		m = nextM

overall_oxygen = int(l[0], 2)
overall_co2 = int(m[0], 2)
print(overall_oxygen * overall_co2)
inputFile.close()

print('\n Day 3 part 1 methode 2')

import pandas as pd

df = pd.read_excel('Python Bachar.xlsx', 'Day 3', skiprows=0).astype(str)
df.columns=['Data']
maxLgth = df['Data'].apply(lambda x: len(x)).max()
df['Data2'] = df['Data'].apply(lambda x: list(str(x).zfill(maxLgth)))
df[[i for i in range(maxLgth)]] = pd.DataFrame(df['Data2'].to_list(), index= df.index)
res = df[[i for i in range(maxLgth)]].describe()
bin_max = res.loc['top', :]
bin_min = ['1' if x=='0' else '0' for x in bin_max]
gamma = int(''.join(bin_max), 2)
epsilon = int(''.join(bin_min), 2)
print('gamma :', gamma, 
      '\nepsilon :', epsilon,
      '\ngamma * epsilon :', gamma * epsilon)

print('\n Day 4')

#Part 1:
def victory(board, move):
	x = [[board[i][j] in move for j in range(5)] for i in range(5)]
	y = [[x[j][i] for j in range(5)] for i in range(5)]
	for i in range(5):
		if sum(x[i]) == 5 or sum(y[i]) == 5:
			return True
	return False

inputFile = open('input4.txt', 'r')
data = inputFile.read().splitlines()

moves = map(int, data[0].split(','))
boards = []
for l in data[1:]:
	if not l:
		boards.append([])
		continue
	boards[-1].append(list(map(int, l.split())))

m = set()
res = -1
for move in moves:
	m.add(move)
	nextBoards = []
	for board in boards:
		if victory(board, m):
			res = 0
			for i in board:
				for j in i:
					if j not in m:
						res += j
			res *= move
			break
	if res != -1:
		break
print(res)
inputFile.close()

#Part 2:
def victory(board, move):
	x = [[board[i][j] in move for j in range(5)] for i in range(5)]
	y = [[x[j][i] for j in range(5)] for i in range(5)]
	for i in range(5):
		if sum(x[i]) == 5 or sum(y[i]) == 5:
			return True
	return False


inputFile = open('input4.txt', 'r')
data = inputFile.read().splitlines()

moves = map(int, data[0].split(','))
boards = []
for l in data[1:]:
	if not l:
		boards.append([])
		continue
	boards[-1].append(list(map(int, l.split())))

m = set()
res = -1
for move in moves:
	m.add(move)
	numboard = []
	for board in boards:
		if not victory(board, m):
			numboard.append(board)
	if len(numboard) == 0:
		res = 0
		for i in boards[0]:
			for j in i:
				if j not in m:
					res += j
		res *= move
		break
	boards = numboard
print(res)
inputFile.close()


#print('\n Day 5')


#Part 1:
def diagonals(coords):
    if coords[0][0] != coords[1][0] and coords[0][1] != coords[1][1]: 
        return True
    else:
        return False


def lines(coords):
    # horizontal line
    for p in range(coords[0][0]+1, coords[1][0]):
        coords.append([p, coords[0][1]])
    for p in range(coords[1][0]+1, coords[0][0]):
        coords.append([p, coords[0][1]])
                
    # vertical line
    for p in range(coords[0][1]+1, coords[1][1]):
        coords.append([coords[0][0], p])
    for p in range(coords[1][1]+1, coords[0][1]):
        coords.append([coords[0][0], p])
                
    return coords


inputfile = open("input.txt", mode='r').read().strip().splitlines()

coords = [[[int(k) for k in j.split(',')] for j in i.split(' -> ')] for i in inputfile]


print('\n Day 6')

#Part 1:
    
fish = dict()
days = 80
with open("input6.txt","r") as inputfile:
    counters = [ int(x) for x in inputfile.read().strip().split(',') ]
for i in range(9):
    fish[i] = counters.count(i)
for i in range(days):
    fish[(i+7)%9] += fish[i%9]
print(sum(fish.values()))


#Part 2:
    
days = 256
for i in range(80,days):
    fish[(i+7)%9] += fish[i%9]
print(sum(fish.values()))

print('\n Day 7')

#Part 1:

inputFile = open('input7.txt', 'r')
# create a list of the data within the file
data = inputFile.read().splitlines()

l = list(map(int,data[0].split(',')))
res = int(1e10)
for i in range(max(l)):
	c = 0
	for j in l:
		c += abs(i - j)
	res = min(res, c)

print(res)

#Part 2
length = len(l)
fuel = []

for v in range(length):
    diff = [abs(d-v) for d in l]
    diffs = sum([sum(list(range(dif+1))) for dif in diff])
    fuel.append(diffs)
print (min(fuel))

print('\n Day 8')
#Part 1:

inputFile = open('input.txt', 'r')
# create a list of the data within the file
data = inputFile.read().splitlines()

l = list(map(int,data[0].split(',')))
res = int(1e10)
for i in range(max(l)):
	c = 0
	for j in l:
		c += abs(i - j)
	res = min(res, c)

print(res)


print('\n Day 9')

#Part 1:

with open('input9.txt', 'r') as f:
    lines = f.read().splitlines()

height_map = {}
for y, row in enumerate(lines):
    for x, height in enumerate(row):
        height_map[(x, y)] = int(height)

low_points = []
sum = 0
for coords, height in height_map.items():
    x, y = coords
    neighbours = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    lowest = True
    for neighbour in neighbours:
        if height_map.get(neighbour, 10) <= height:
            lowest = False
            break

    if lowest:
        low_points.append(coords)
        sum += height + 1

print(sum)

#Part 2
with open('input9.txt', 'r') as f:
    lines = f.read().splitlines()

height_map = {}
for y, row in enumerate(lines):
    for x, height in enumerate(row):
        height_map[(x, y)] = int(height)

low_points = []
sum = 0
for coords, height in height_map.items():
    x, y = coords
    neighbours = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    lowest = True
    for neighbour in neighbours:
        if height_map.get(neighbour, 10) <= height:
            lowest = False
            break

    if lowest:
        low_points.append(coords)
        sum += height + 1

basin_sizes = []
for low_point in low_points:
    part_of_basin = set([low_point])

    coords_to_check_stack = [low_point]
    while coords_to_check_stack:
        x, y = coords_to_check_stack.pop()
        adjacent = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        for neighbour in adjacent:
            neighbour_height = height_map.get(neighbour, 0)
            if neighbour_height > height_map[(x, y)] and neighbour_height != 9:
                part_of_basin.add(neighbour)
                coords_to_check_stack.append(neighbour)

    basin_sizes.append(len(part_of_basin))

basin_sizes.sort(reverse=True)

largest_three_product = 1
for size in basin_sizes[:3]:
    largest_three_product *= size

print(largest_three_product)

print('\n Day 10')

#Part 1:

points = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137
}


def get_illigal_bracket(brackets):

    openings = '([{<'
    closings = ')]}>'

    stack = []
    for i in brackets:
        if i in openings:
            stack.append(i)
        elif i in closings:
            pos = closings.index(i)
            if ((len(stack) > 0) and
                    (openings[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                # print('Expected', closings[openings.index(stack[len(stack)-1])], 'but found', i, 'instead')
                return (False, i)
    if len(stack) == 0:
        return (True, None)
    else:
        return (False, None)


with open('input10.txt') as f:
    lines = f.read().splitlines()
    sum = 0

    for line in lines:
        is_valid, illigal_bracket = get_illigal_bracket(line)

        if not is_valid and illigal_bracket != None:
            sum += points.get(illigal_bracket)

    print(sum)

points = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137
}


def get_illigal_bracket(brackets):

    openings = '([{<'
    closings = ')]}>'

    stack = []
    for i in brackets:
        if i in openings:
            stack.append(i)
        elif i in closings:
            pos = closings.index(i)
            if ((len(stack) > 0) and
                    (openings[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                # print('Expected', closings[openings.index(stack[len(stack)-1])], 'but found', i, 'instead')
                return (False, i)
    if len(stack) == 0:
        return (True, None)
    else:
        return (False, None)


with open('input10.txt') as f:
    lines = f.read().splitlines()
    sum = 0

    for line in lines:
        is_valid, illigal_bracket = get_illigal_bracket(line)

        if not is_valid and illigal_bracket != None:
            sum += points.get(illigal_bracket)

    print(sum)



print('\n Day 11')


print('\n Day 12')

with open('input12.txt') as f:
	lines = f.read().splitlines()

def recur(cur, path):
	if cur == 'end':
		return 1

	out = 0
	for child in adj[cur]:
		if child.lower() != child or child not in path:
			out += recur(child, path + [child])
	return out

adj = {}

for line in lines:
	v1, v2 = line.split('-')
	adj.setdefault(v1,[]).append(v2)
	adj.setdefault(v2,[]).append(v1)

print(recur('start', ['start']))


#Part 2
with open('input12.txt') as f:
	lines = f.read().splitlines()

def recur(cur, path, twice):
	if cur == 'end':
		return 1

	out = 0
	for child in adj[cur]:
		if child == 'start':
			continue
		elif child.lower() != child or child not in path:
			out += recur(child, path + [child], twice)
		elif path.count(child) == 1 and not twice:
			out += recur(child, path + [child], True)
	return out

adj = {}

for line in lines:
	v1, v2 = line.split('-')
	adj.setdefault(v1,[]).append(v2)
	adj.setdefault(v2,[]).append(v1)

print(recur('start', ['start'], False))


print('\n Day 13')

#Part 1:

with open('input13.txt') as f:
	lines = f.read().splitlines()

grid = [[' '] * 2000 for _ in range(2000)]
folds = []

for line in lines:
	if 'fold' in line:
		folds.append(line.split()[-1].split('='))
	elif len(line) > 0:
		x, y = line.split(',')
		grid[int(x)][int(y)] = '#'

cnt = 0
for i in range(655):
	for a in range(2000):
		if grid[i][a] == '#' or grid[2 * 655 - i][a] == '#':
			cnt += 1

print(cnt)


#Part 2
with open('input13.txt') as f:
	lines = f.read().splitlines()

grid = [[' '] * 2000 for _ in range(2000)]
folds = []

for line in lines:
	if 'fold' in line:
		folds.append(line.split()[-1].split('='))
	elif len(line) > 0:
		x, y = line.split(',')
		grid[int(x)][int(y)] = '#'
		
for f in folds:
	val = int(f[1])
	if f[0] == 'x':
		for i in range(val):
			for a in range(2000):
				if grid[i][a] == '#' or grid[2 * val - i][a] == '#':
					grid[i][a] = '#'
	else:
		for i in range(2000):
			for a in range(val):
				if grid[i][a] == '#' or grid[i][2 * val - a] == '#':
					grid[i][a] = '#'	

for i in range(40):
	print(''.join(grid[i][:6][::-1]))


print('\n Day 14')


#Part 1:

with open('input14.txt') as f:
	lines = f.read().splitlines()

poly = lines[0]
mp = {}

for line in lines[2:]:
	cur = line.split(' -> ')
	mp[cur[0]] = cur[1]

for i in range(10):
	new_poly = poly[0]
	for a in range(len(poly) - 1):
		new_poly += mp[poly[a:a+2]] + poly[a+1]
	poly = new_poly

cnts = [poly.count(c) for c in poly]

print(max(cnts) - min(cnts))


#Part 2
with open('input14.txt') as f:
	lines = f.read().splitlines()

poly = lines[0]
mp = {}
cnt = {}

for line in lines[2:]:
	cur = line.split(' -> ')
	mp[cur[0]] = cur[1]

for i in range(len(poly) - 1):
	cnt[poly[i:i+2]] = cnt.get(poly[i:i+2], 0) + 1

for i in range(40):
	cnt2 = {}
	for pair in cnt:
		cnt2[pair[0] + mp[pair]] = cnt2.get(pair[0] + mp[pair], 0) + cnt[pair]
		cnt2[mp[pair] + pair[1]] = cnt2.get(mp[pair] + pair[1], 0) + cnt[pair]
	cnt = cnt2

lst = {}
for pair in cnt:
	lst[pair[0]] = lst.get(pair[0], 0) + cnt[pair]
lst[poly[-1]] += 1

print(max(lst.values()) - min(lst.values()))



print('\n Day 15')

#Part 1:

with open('input15.txt') as f:
	lines = f.read().splitlines()

row = len(lines)
col = len(lines[0])

dp = [[10 ** 100] * 1000 for _ in range(1000)]
dp[0][0] = 0

for i in range(row):
	for a in range(col):
		if i == 0 and a == 0:
			continue
		if a - 1 >= 0:
			dp[i][a] = min(dp[i][a], dp[i][a-1])
		if i - 1 >= 0:
			dp[i][a] = min(dp[i][a], dp[i - 1][a])
		dp[i][a] += int(lines[i][a])

print(dp[row - 1][col - 1])



#Part 2
from heapq import heappop, heappush

with open('input15.txt') as f:
    lines = f.read().splitlines()

row = len(lines)
col = len(lines[0])
dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]

# returns integer at position (x, y)
def val(x, y):
    num = int(lines[x % row][y % col]) + x // row + y // col
    return (num - 1) % 9 + 1

def in_bounds(x, y):
    return 0 <= x and x < 5 * row and 0 <= y and y < 5 * col

dist = [[10 ** 100] * 5 * row for i in range(5 * col)]
# (dist, x, y)
queue = [(0, 0, 0)]
dist[0][0] = 0

while len(queue) != 0:
    d, x, y = heappop(queue)
    if x == 5 * row - 1 and y == 5 * col - 1:
        print(d)

    for k in dirs:
        new_x = x + k[0]
        new_y = y + k[1]
        if in_bounds(new_x, new_y) and dist[x][y] + val(new_x, new_y) < dist[new_x][new_y]:
            dist[new_x][new_y] = dist[x][y] + val(new_x, new_y)
            heappush(queue, (dist[new_x][new_y], new_x, new_y))




