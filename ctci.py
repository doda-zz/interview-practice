# coding: utf-8
from collections import defaultdict, Counter
from pprint import pprint

# TODO: hanoi, 5.2, 5.7, 7.x, 19.9
# known bugs: 19.6 doesnt do "and"

def oneone(s):
    'Implement an algorithm to determine if a string has all unique characters.'
    chars = set()
    for c in s:
        if c in chars:
            return False
        chars.add(c)
    return True

def oneone2(s):
    'What if you can not use additional data structures?'
    if not s:
        return True
    le = len(s)
    for i in range(le-1):
        for j in range(i+1, le):
            if i != j and s[i] == s[j]:
                return False
    return True


def onetwo(s):
    '''Write code to reverse a C-Style String.
    (C-String means that “abcd” is represented as five characters, including the null character.)'''
    # lets pretend Python has mutable strings
    s = list(s)
    max_idx = len(s) - 1
    for i in range(len(s)/2):
        t = s[i]
        s[i] = s[max_idx - i]
        s[max_idx - i] = t
    return ''.join(s)


def onethree(s):
    '''
    Design an algorithm and write code to remove the duplicate characters in a string
    without using any additional buffer. NOTE: One or two additional variables are fine.
    An extra copy of the array is not.
    '''
    # note this doesnt do unicode or any other fanciness
    s = list(s)
    # pretend bit vector
    bitvector = [False] * 128
    cur_idx = 0
    for c in s:
        pos = ord(c)
        if not bitvector[pos]:
            bitvector[pos] = True
            s[cur_idx] = c
            cur_idx += 1
    return ''.join(s[:cur_idx])

def onefour(s, t):
    'Write a method to decide if two strings are anagrams or not.'
    # We could of course do
    # sorted(s) == sorted(t)
    # Counter(s) == Counter(t)
    # this is more in the spirit of the book
    d = {}
    for c in s:
        if not c in d:
            d[c] = 0
        d[c] += 1
    for c in t:
        if not c in d or not d[c]:
            return False
        d[c] -= 1
    return not any(d.values())

def onefive(s):
    'Write a method to replace all spaces in a string with ‘%20’.'
    # this is stupid
    return s.replace(' ', '%20')

def onesix(matrix):
    '''
    Given an image represented by an NxN matrix, where each pixel in the image is 4
    bytes, write a method to rotate the image by 90 degrees. Can you do this in place?
    '''
    # go in layers and do a 4-way swap on pixels
    if not matrix:
        return matrix
    n = len(matrix[0])
    for layer in range(n/2):
        first = layer
        last = n - 1 - layer
        for i in range(first, last):
            offset = i - layer
            # top = matrix[layer][i]
            # left = matrix[last-offset][layer]
            # right = matrix[i][last]
            # bottom = matrix[last][last-offset]
            # temp = top
            temp = matrix[layer][i]
            # top = left
            matrix[layer][i] = matrix[last-offset][layer]
            # left = bottom
            matrix[last-offset][layer] = matrix[last][last-offset]
            # bottom = right
            matrix[last][last-offset] = matrix[i][last]
            # right = temp
            matrix[i][last] = temp
    return matrix

def oneseven(matrix):
    '''Write an algorithm such that if an element in an MxN matrix is 0, its entire row and
    column is set to 0.'''
    if not matrix:
        return matrix
    N = len(matrix)
    M = len(matrix[0])
    null_hori = set()
    null_vert = set()
    for i in range(N):
        for j in range(M):
            if matrix[i][j] == 0:
                null_hori.add(i)
                null_vert.add(j)
    for i in range(N):
        for j in range(M):
            if i in null_hori or j in null_vert:
                matrix[i][j] = 0
    return matrix

def oneeight(s1, s2):
    '''
    Assume you have a method isSubstring which checks if one word is a substring of
    another. Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 using
    only one call to isSubstring (i.e., “waterbottle” is a rotation of “erbottlewat”).
    '''
    return s1 in s2+s2




def twoone(node):
    '''
    Write code to remove duplicates from an unsorted linked list.
    '''
    seen = set()
    prev = None
    while node is not None:
        if node.value in seen:
            prev.next = node.next
        else:
            seen.add(node.value)
        prev = node
        node = node.next


def twoone2(node):
    '''FOLLOW UP
    How would you solve this problem if a temporary buffer is not allowed?'''

    while node is not None:
        val = node.value
        run_prev = node
        run_ahead = node.next
        while run_ahead is not None:
            if run_ahead.value == val:
                run_prev.next = run_ahead.next
            run_prev = run_ahead
            run_ahead = run_ahead.next
        node = node.next


def twotwo(node, n):
    '''Implement an algorithm to find the nth to last element of a singly linked list.'''
    run_ahead = node
    for _ in range(n):
        if run_ahead is None:
            raise ValueError('Linked list is shorter than n')
        run_ahead = run_ahead.next
    while run_ahead is not None:
        node = node.next
        run_ahead = run_ahead.next
    return node


def twothree(node):
    '''
    Implement an algorithm to delete a node in the middle of a single linked list, given
    only access to that node.
    EXAMPLE
    Input: the node ‘c’ from the linked list a->b->c->d->e
    Result: nothing is returned, but the new linked list looks like a->b->d->e
    '''
    if node.next is None:
        raise ValueError('need at least one element following to delete this node')
    prev = node
    node = node.next
    while node is not None:
        prev.value = node.value
        prev = node
        node = node.next
    prev.next = None

class Node(object):
    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next

    def __repr__(self):
        return 'Node(%s, %s)' % (self.value if self.value is not None else '', self.next)

def twofour(node1, node2):
    '''
    You have two numbers represented by a linked list, where each node contains a sin-
    gle digit. The digits are stored in reverse order, such that the 1’s digit is at the head of
    the list. Write a function that adds the two numbers and returns the sum as a linked
    list.
    EXAMPLE
    Input: (3 -> 1 -> 5) + (5 -> 9 -> 2)
    Output: 8 -> 0 -> 8
    '''
    carry = 0
    start = None
    while True:
        v1 = v2 = 0
        if node1 is not None:
            v1 = node1.value
        if node2 is not None:
            v2 = node2.value
        new_value = v1+v2+carry
        if not new_value:
            break
        carry, digit = divmod(new_value, 10)

        new = Node(digit)
        if start is None:
            start = current_new = new
        else:
            current_new.next = new
        current_new = new
        if node1 is not None:
            node1 = node1.next
        if node2 is not None:
            node2 = node2.next
    current_new.next = None
    return start

# n1 = Node(value=3, next=Node(value=1, next=Node(value=5)))
# n2 = Node(value=5, next=Node(value=9, next=Node(value=2)))
# print twofour(n1, n2)

def twofive(node):
    '''Given a circular linked list, implement an algorithm which returns node at the begin-
    ning of the loop.
    DEFINITION
    Circular linked list: A (corrupt) linked list in which a node’s next pointer points to an
    earlier node, so as to make a loop in the linked list.
    EXAMPLE
    input: A -> B -> C -> D -> E -> C [the same C as earlier]
    output: C
    '''
    seen = set()
    while node is not None:
        if node in seen:
            return node
        seen.add(node)
        node = node.next
    return False


def threeone():
    '''Describe how you could use a single array to implement three stacks.'''
    # Give each stack a third of the array.
    # Alternative idea: If speed doesnt matter, grow one from the front,
    # one from the end, one from the middle, alternating between left and right :)

def threetwo():
    '''
    How would you design a stack which, in addition to push and pop, also has a function
    min which returns the minimum element? Push, pop and min should all operate in
    O(1) time.
    '''
    # for every element keep track of the min of all the elements beneath it

    class Stack(object):
        def __init__(self, values=None):
            self.stack = []
            self.mins = []
            if values is not None:
                for v in values:
                    self.push(v)

        def push(self, value):
            try:
                cur_min = self.min()
            except IndexError:
                cur_min = float('inf')
            self.mins.append(min(cur_min, value))
            return self.stack.append(value)

        def pop(self):
            self.mins.pop()
            return self.stack.pop()

        def peek(self):
            return self.stack[-1]

        def min(self):
            return self.mins[-1]

        def __repr__(self):
            return repr(self.stack)

    return Stack

def threethree():
    '''
    Imagine a (literal) stack of plates. If the stack gets too high, it might topple. There-
    fore, in real life, we would likely start a new stack when the previous stack exceeds
    some threshold. Implement a data structure SetOfStacks that mimics this. SetOf-
    Stacks should be composed of several stacks, and should create a new stack once
    the previous one exceeds capacity. SetOfStacks.push() and SetOfStacks.pop() should
    behave identically to a single stack (that is, pop() should return the same values as it
    would if there were just a single stack).
    '''
    class SetOfStacks(object):
        def __init__(self, values=None, maxlen=10):
            self.maxlen = maxlen
            self.stacks = [[]]
            if values is not None:
                for v in values:
                    self.push(v)


        def push(self, value):
            if len(self.stacks[-1]) >= self.maxlen:
                self.stacks.append([])
            self.stacks[-1].append(value)

        def pop(self):
            if len(self.stacks) == 1 and not self.stacks[0]:
                raise IndexError('pop from empty stack')
            if not self.stacks[-1]:
                self.stacks.pop()
            return self.stacks[-1].pop()

        def peek(self):
            if not self.stacks[-1]:
                return self.stacks[-2][-1]
            return self.stacks[-1][-1]


def threethree2():
    '''
    FOLLOW UP
    Implement a function popAt(int index) which performs a pop operation on a specific
    sub-stack.
    '''
    class SetOfStacks(object):
        def __init__(self, values=None, maxlen=10):
            self.maxlen = maxlen
            self.stacks = [[]]
            if values is not None:
                for v in values:
                    self.push(v)

        def push(self, value):
            for stack in self.stacks:
                if len(stack) != self.maxlen:
                    stack.append(value)
                    return
            self.stacks.append([value])

        def pop(self):
            if len(self.stacks) == 1 and not self.stacks[0]:
                raise IndexError('pop from empty stack')
            if not self.stacks[-1]:
                self.stacks.pop()
            return self.stacks[-1].pop()

        def popAt(self, stack_num):
            return self.stacks[stack_num].pop()

        def peek(self):
            if not self.stacks[-1]:
                return self.stacks[-2][-1]
            return self.stacks[-1][-1]



def threefour(A, B=None, C=None):
    '''In the classic problem of the Towers of Hanoi, you have 3 rods and N disks of different
    sizes which can slide onto any tower. The puzzle starts with disks sorted in ascending
    order of size from top to bottom (e.g., each disk sits on top of an even larger one). You
    have the following constraints:
    (A) Only one disk can be moved at a time.
    (B) A disk is slid off the top of one rod onto the next rod.
    (C) A disk can only be placed on top of a larger disk.
    Write a program to move the disks from the first rod to the last using Stacks.
    '''
    pass

def threefive():
    class MyQueue(object):
        def __init__(self, values=None):
            self.stack = []
            self.other = []
            if values is not None:
                for v in values:
                    self.put(v)

        def put(self, value):
            self.stack.append(value)

        def get(self):
            while self.stack:
                self.other.append(self.stack.pop())
            elem = self.other.pop()
            while self.other:
                self.stack.append(self.other.pop())
            return elem

        def __repr__(self):
            return repr(self.stacks[self.state])

    return MyQueue

def threesix(stack):
    '''
    Write a program to sort a stack in ascending order. You should not make any assump-
    tions about how the stack is implemented. The following are the only functions that
    should be used to write this program: push | pop | peek | isEmpty.
    '''
    Stack = threetwo()
    other = Stack()
    other.push(stack.pop())
    while not stack.isEmpty():
        elem = stack.pop()
        moved = 0
        while other.peek() > elem:
            stack.push(other.pop())
            moved += 1
        other.push(elem)
        for _ in range(moved):
            other.push(stack.pop())
    return other

# Stack = threetwo()
# class SS(Stack):
#     def isEmpty(self):
#         return not self.stack
# s = SS([5,3,2])
# print threesix(s)


def fourone(root):
    '''
    Implement a function to check if a tree is balanced. For the purposes of this question,
    a balanced tree is defined to be a tree such that no two leaf nodes differ in distance
    from the root by more than one.
    '''
    fourone.minh, fourone.maxh = float('-inf'), float('inf')
    def walk(node, depth=0):
        if node is None:
            return
        if not node.left and not node.right:
            fourone.minh = max(depth, fourone.minh)
            fourone.maxh = min(depth, fourone.maxh)
        walk(node.left, depth+1)
        walk(node.right, depth+1)
    walk(root)
    return abs(fourone.maxh - fourone.minh) <= 1

# from node import Node, tree
# print fourone(tree)

def fourtwo(n1, n2):
    '''Given a directed graph, design an algorithm to find out whether there is a route be-
    tween two nodes.
    '''
    seen = set()
    stack = [n1]
    while stack:
        node = stack.pop()
        if node in seen:
            seen.add(node)
        else:
            continue
        for c in node.children:
            if c is n2:
                return True
            stack.append(c)
    return False

def fourthree(arr):
    '''
    Given a sorted (increasing order) array, write an algorithm to create a binary tree with
    minimal height.
    '''
    class Tree:pass
    tree = Tree()
    def insert(arr):
        if not arr:
            return
        middle = len(arr)/2
        tree.insert(arr[middle])
        insert(arr[:middle])
        insert(arr[middle+1:])
    insert(arr)
    return tree

def fourfour(tree):
    '''
    Given a binary search tree, design an algorithm which creates a linked list of all the
    nodes at each depth (i.e., if you have a tree with depth D, you’ll have D linked lists).
    '''
    levels = defaultdict(list)
    def walk(node, depth):
        if node is None:
            return
        levels[depth].append(node)
        walk(node.left, depth+1)
        walk(node.right, depth+1)
    walk(tree, 0)
    return levels

def fourfive(node):
    '''
    Write an algorithm to find the ‘next’ node (i.e., in-order successor) of a given node in
    a binary search tree where each node has a link to its parent.
    '''
    '''
           5
          / \
         4  10
        /  /  \
       1  8   12
      /  / \  / \
     0  6  9 11 13
    '''

    if node.right:
        walk = node.right
        while walk.left:
            walk = walk.left
        return walk
    if node.parent.left is node:
        return node.parent
    if node.parent.right is node:
        node = node.parent
        while True:
            if node is None:
                # we got the last node
                return False
            if node.parent.left is node:
                return node.parent
            node = node.parent
    return False


def foursix(tree, n1, n2):
    '''Design an algorithm and write code to find the first common ancestor of two nodes
    in a binary tree. Avoid storing additional nodes in a data structure. NOTE: This is not
    necessarily a binary search tree.
    '''

    def depth(root, node):
        depth.found = -1
        def walk(root, dep):
            if root is None:
                return
            if depth.found != -1:
                return
            if root is node:
                depth.found = dep
            walk(root.left, dep+1)
            walk(root.right, dep+1)
        walk(root, 0)
        return depth.found

    d1 = depth(tree, n1)
    d2 = depth(tree, n2)

    lower, higher = n1, n2
    if d1 < d2:
        lower, higher = higher, lower
    print d1, d2
    for _ in range(abs(d2-d1)):
        higher = higher.parent

    while n1 is not None:
        if n1 is n2:
            return n1
        n1 = n1.parent
        n2 = n2.parent
    return False

def fourseven(t1, t2):
    '''
    You have two very large binary trees: T1, with millions of nodes, and T2, with hun-
    dreds of nodes. Create an algorithm to decide if T2 is a subtree of T1.
    '''
    def inorder_stringify(tree):
        node_strings = []
        def walk(node):
            if node is None:
                return
            walk(node.left)
            node_strings.append(str(node.value))
            walk(node.right)
        walk(tree)
        return ','.join(node_strings)

    s1 = inorder_stringify(t1)
    s2 = inorder_stringify(t2)
    return s2 in s1

def foureight(tree):
    '''You are given a binary tree in which each node contains a value. Design an algorithm
    to print all paths which sum up to that value. Note that it can be any path in the tree
    - it does not have to start at the root.
    '''
    zero_paths = []
    def walk(node, sofar=0):
        if node is None:
            return
        p = node.parent
        path = []
        sofar_up = node.value
        while p:
            sofar_up -= p.value
            path.append(p)
            if sofar_up == 0:
                zero_paths.append(path[::-1])
            p = p.parent
        walk(node.left, sofar+node.value)
        walk(node.right, sofar+node.value)
    return zero_paths

def fiveone(N, M, i, j):
    '''
    You are given two 32-bit numbers, N and M, and two bit positions, i and j. Write a
    method to set all bits between i and j in N equal to M (e.g., M becomes a substring of
    N located at i and starting at j).
    EXAMPLE:
    Input: N = 10000000000, M = 10101, i = 2, j = 6
    Output: N = 10001010100
    '''
    # all 1s
    ones = 2 ** 31 + 1
    left = ones - (1 << (j - 1))
    right = 1 << (i - 1)
    mask = left | right
    return (N & mask) | (M << i)

def fivetwo():
    '''Given a (decimal - e.g. 3.72) number that is passed in as a string, print the binary rep-
    resentation. If the number can not be represented accurately in binary, print “ERROR”
    '''

def fivethree(i):
    '''
    Given an integer, print the next smallest and next largest number that have the same
    number of 1 bits in their binary representation.
    '''
    s = bin(i)[2:]
    all_ones = all(c == '1' for c in s)
    if all_ones:
        return False, '10' + s[1:]
    big = s.rfind('01')
    if big == -1:
        bigger = s + '0'
    else:
        bigger = s[:big] + '10' + sorted(s[big+2:], reverse=True)
    small = s.rfind('10')
    if small == -1:
        smaller = False
    else:
        smaller = s[:small] + '01' + sorted(s[small+2:])

    return smaller.lstrip('0'), bigger

def fivefour():
    '''Explain what the following code does: ((n & (n-1)) == 0).'''
    # base 2

def fivefive(a, b):
    '''
    Write a function to determine the number of bits required to convert integer A to
    integer B.
    Input: 31, 14
    Output: 2
    '''
    # Levenshtein is cooler
    # from Levenshtein import distance
    # return distance(bin(a)[2:], bin(b)[2:])
    xor = a ^ b
    return sum(bool(xor & (1 << i)) for i in range(32))

def fivesix(a):
    '''
    Write a program to swap odd and even bits in an integer with as few instructions as
    possible (e.g., bit 0 and bit 1 are swapped, bit 2 and bit 3 are swapped, etc)
    '''
    to_the_left = (int('01' * 16, 2) & a) << 1
    to_the_right = (int('10' * 16, 2) & a) >> 1
    return to_the_left | to_the_right

def fiveseven():
    '''
    An array A[1...n] contains all the integers from 0 to n except for one number which is
    missing. In this problem, we cannot access an entire integer in A with a single opera-
    tion. The elements of A are represented in binary, and the only operation we can use
    to access them is “fetch the jth bit of A[i]”, which takes constant time. Write code to
    find the missing integer. Can you do it in O(n) time?
    '''
    # meh


def eightone(n):
    '''Write a method to generate the nth Fibonacci number.'''
    if not n:
        return 0
    a = b = 1
    for _ in range(n-2):
        c = a + b
        a = b
        b = c
    return b

def eighttwo(N, offlimits=None):
    '''
    Imagine a robot sitting on the upper left hand corner of an NxN grid. The robot can
    only move in two directions: right and down. How many possible paths are there for
    the robot?
    FOLLOW UP
    Imagine certain squares are “off limits”, such that the robot can not step on them.
    Design an algorithm to get all possible paths for the robot.
    '''
    paths = []
    def walk(x, y, path=None):
        if x >= N or y >= N:
            return
        if (x,y) in offlimits:
            return
        if path is None:
            path = []
        path = path[:]
        path.append((x,y))
        if x == N - 1 and y == N -1:
            paths.append(path)
        walk(x+1, y, path)
        walk(x, y+1, path)
    if offlimits is None:
        offlimits = []
    offlimits = set(offlimits)
    walk(0, 0)
    return paths

def eightthree(s):
    '''
    Write a method that returns all subsets of a set.
    '''
    if len(s) <= 1:
        return [s]
    subs = []
    for i in range(len(s)):
        li = s[:]
        li.pop(i)
        subs.append(li)
        subs.extend(eightthree(li))
    return set(map(tuple, subs))

def eightfour(s):
    '''Write a method to compute all permutations of a string.'''
    if len(s) <= 1:
        return [s]
    if len(s) == 2:
        return [s, s[::-1]]
    permuts = eightfour(s[:-1])
    result = []
    for p in permuts:
        for i in range(len(p)+1):
            result.append(p[:i] + s[-1] + p[i:])
    return result

def eightfive(N):
    '''
    Implement an algorithm to print all valid (e.g., properly opened and closed) combi-
    nations of n-pairs of parentheses.
    EXAMPLE:
    input: 3 (e.g., 3 pairs of parentheses)
    output: ()()(), ()(()), (())(), ((()))
    '''
    def walk(s, l, r):
        if l > 0:
            walk(s + '(', l-1, r)
        if r > l:
            walk(s + ')', l, r-1)
        if not l and not r:
            print s
    walk('', N, N)

def eightsix(screen, point, color):
    '''
    Implement the “paint fill” function that one might see on many image editing pro-
    grams. That is, given a screen (represented by a 2 dimensional array of Colors), a
    point, and a new color, fill in the surrounding area until you hit a border of that col-
    or.’
    '''
    max_x = len(screen[0])
    max_y = len(screen)
    seen = set()
    x, y = point
    orig_color = screen[y][x]
    def walk(x,y):
        if x < 0 or y < 0 or x >= max_x or y >= max_y:
            return
        if (x,y) in seen:
            return
        else:
            seen.add((x,y))
        if screen[y][x] != orig_color:
            return
        screen[y][x] = color
        walk(x+1,y)
        walk(x,y+1)
        walk(x-1,y)
        walk(x,y-1)
    walk(x,y)


def eightseven(n):
    '''
    Given an infinite number of quarters (25 cents), dimes (10 cents), nickels (5 cents) and
    pennies (1 cent), write code to calculate the number of ways of representing n cents.
    '''
    quantities = 25, 10, 5, 1
    all_ways = set()

    def walk(current):
        if sum(current) == n:
            all_ways.add(tuple(sorted(current)))
            return
        for q in reversed(quantities):
            if sum(current) + q > n:
                break
            walk(current[:] + [q])

    walk([])
    return all_ways


def eighteight(N=8):
    '''
    Write an algorithm to print all ways of arranging eight queens on a chess board so
    that none of them share the same row, column or diagonal.
    '''
    def xitout(board, x, y):
        def exo(board, x, y):
            if x >= 0 and y >= 0 and x < N and y < N:
                board[y][x] = 'X'
        for i in range(N):
            board[y][i] = 'X'
            board[i][x] = 'X'
        for i in range(1, N):
            # diagonal
            exo(board, x+i, y+i)
            exo(board, x+i, y-i)
            exo(board, x-i, y+i)
            exo(board, x-i, y-i)


    initial_board = [[' ' for _ in range(N)] for _ in range(N)]
    all_arrangements = []
    def walk(board):
        queens = [(x,y) for y in range(N) for x in range(N) if board[y][x] == 'Q']
        if len(queens) == N:
            all_arrangements.append(queens)
        for y in range(N):
            for x in range(N):
                if board[y][x] != ' ':
                    continue
                # make a new board
                new = [row[:] for row in board]
                # X it out
                xitout(new, x, y)
                # place a queen
                new[y][x] = 'Q'
                walk(new)
    walk(initial_board)
    return set(map(tuple, all_arrangements))

def nineone(A, B, last_valid_a):
    '''
    You are given two sorted arrays, A and B, and A has a large enough buffer at the end
    to hold B. Write a method to merge B into A in sorted order.
    '''
    idx = len(A) - 1
    a_idx = last_valid_a
    b_idx = len(B) - 1
    while a_idx >= 0 and b_idx >= 0:
        a, b = A[a_idx], B[b_idx]
        if a > b:
            A[idx] = a
            a_idx -= 1
        else:
            A[idx] = b
            b_idx -= 1
        idx -= 1
    while b_idx >= 0:
        A[idx] = B[b_idx]
        b_idx -= 1
        idx -= 1
    return A

def ninetwo(arr):
    '''
    Write a method to sort an array of strings so that all the anagrams are next to each
    other.
    '''
    return sorted(arr, cmp=lambda a,b: cmp(sorted(a), sorted(b)))


def ninethree(arr, value):
    '''
    Given a sorted array of n integers that has been rotated an unknown number of
    times, give an O(log n) algorithm that finds an element in the array. You may assume
    that the array was originally sorted in increasing order.
    EXAMPLE:
    Input: find 5 in array (15 16 19 20 25 1 3 4 5 7 10 14)
    Output: 8 (the index of 5 in the array)
    '''
    l = 0
    r = len(arr) - 1

    while l <= r:
        middle = (l + r) / 2
        if arr[middle] == value:
            return middle
        if arr[l] <= arr[middle]:
            if value > arr[middle]:
                l = middle + 1
            elif value >= arr[l]:
                r = middle - 1
            else:
                l = middle + 1
        elif value < arr[middle]:
            r = middle -1
        elif value <= arr[r]:
            l = middle + 1
        else:
            r = middle - 1
    return -1


# print ninethree(map(int, '15 16 19 20 25 1 3 4 5 7 10 14'.split()), 5)



def ninefour():
    '''
    If you have a 2 GB file with one string per line, which sorting algorithm would you use
    to sort the file and why?
    '''
    # external sort


def ninefive(arr, value):
    '''
    Given a sorted array of strings which is interspersed with empty strings, write a meth-
    od to find the location of a given string.
    Example: find “ball” in [“at”, “”, “”, “”, “ball”, “”, “”, “car”, “”, “”, “dad”, “”, “”] will return 4
    Example: find “ballcar” in [“at”, “”, “”, “”, “”, “ball”, “car”, “”, “”, “dad”, “”, “”] will return -1
    '''
    l = 0
    r = len(arr) - 1
    while l <= r:
        middle = cur = (l + r) / 2
        while arr[cur] == '':
            if cur <= 0:
                cur = middle + 1
                while arr[cur] == '':
                    if cur >= len(arr):
                        break
                    cur += 1
                break
            cur -= 1
        if arr[cur] == value:
            return cur
        if arr[cur] <= value:
            r = middle - 1
        else:
            l = middle + 1
    return -1

def ninesix(arr, value):
    '''
    Given a matrix in which each row and each column is sorted, write a method to find
    an element in it.
    '''
    n = len(arr[0])
    m = len(arr)
    x = n - 1
    y = 0
    while x >= 0 and y < m:
        if arr[y][x] == value:
            return x,y
        if arr[y][x] <= value:
            y += 1
        else:
            x -= 1
    return -1

def nineseven(people):
    '''
    A circus is designing a tower routine consisting of people standing atop one anoth-
    er’s shoulders. For practical and aesthetic reasons, each person must be both shorter
     and lighter than the person below him or her. Given the heights and weights of each
    person in the circus, write a method to compute the largest possible number of peo-
    ple in such a tower.
    EXAMPLE:
    Input (ht, wt): (65, 100) (70, 150) (56, 90) (75, 190) (60, 95) (68, 110)
    Output: The longest tower is length 6 and includes from top to bottom: (56, 90)
    (60,95) (65,100) (68,110) (70,150) (75,190)
    '''
    maxed = []
    people = sorted(people)
    coll = []
    for p in people:
        if not coll:
            coll.append(p)
            continue
        last = coll[-1]
        if p[0] >= last[0] and p[1] >= last[1]:
            coll.append(p)
        else:
            maxed = max(maxed, coll, key=len)
            coll = []
    return maxed or coll


def nineteenone(a, b):
    '''Write a function to swap a number in place without temporary variables.'''
    a = a ^ b
    b = a ^ b
    a = a ^ b
    return a,b

def nineteentwo(board):
    '''Design an algorithm to figure out if someone has won in a game of tic-tac-toe.'''
    def won(three):
        X = set(['X'])
        O = set(['O'])
        return set(three) in (X, O)
    hori = board
    verti = zip(*board)
    diago = [[
        board[0][0],
        board[1][1],
        board[2][2],
    ], [
        board[0][2],
        board[1][1],
        board[2][0],
    ]]
    return any(won(three) for threes in (hori, verti, diago) for three in threes)

def nineteenthree(n):
    '''Write an algorithm which computes the number of trailing zeros in n factorial.'''
    return n / 5 + n / 25

def nineteenfour(a, b):
    '''
    Write a method which finds the maximum of two numbers. You should not use if-
    else or any other comparison operator.  
    EXAMPLE
    Input: 5, 10
    Output: 10
    '''
    return [a, b][bool((a - b) & (1 << 31))]

def nineteenfive(guess, solution):
    '''
    The Game of Master Mind is played as follows:
    The computer has four slots containing balls that are red (R), yellow (Y), green (G) or
    blue (B). For example, the computer might have RGGB (e.g., Slot #1 is red, Slots #2 and
    #3 are green, Slot #4 is blue).
    You, the user, are trying to guess the solution. You might, for example, guess YRGB.
    When you guess the correct color for the correct slot, you get a “hit”. If you guess
    a color that exists but is in the wrong slot, you get a “pseudo-hit”. For example, the
    guess YRGB has 2 hits and one pseudo hit.
    For each guess, you are told the number of hits and pseudo-hits.
    Write a method that, given a guess and a solution, returns the number of hits and
    pseudo hits.
    '''
    pseudos = len(guess) - len(Counter(solution) - Counter(guess))
    hits = 0
    for a,b in zip(guess, solution):
        if a == b:
            pseudos -= 1
            hits += 1
    return pseudos, hits

def nineteensix(n):
    '''
    Given an integer between 0 and 999,999, print an English phrase that describes the
    integer (eg, “One Thousand, Two Hundred and Thirty Four”).
    '''
    nums = dict((i, num) for i, num in enumerate('One Two Three Four Five Six Seven Eight Nine Ten Eleven \
    Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split(), 1))
    for i, num in zip(range(20, 101, 10), 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety Hundred'.split()):
        nums[i] = num
    # 912 Nine hundred and twelve
    # 1299 One thousand two hundred 99
    # 123 One Hundred and Twenty three
    # 241,123 Two Hundred Thousand One Hundred and Twenty Three
    def thousand(n):
        if not n:
            return []
        if n in nums:
            return [nums[n]]
        if n >= 100:
            div, name, multi = 100, ['Hundred'], 1
        else:
            div, name, multi = 10, [], 10
        first, remainder = divmod(n, div)
        return [nums[first*multi]] + name + thousand(remainder)
    if not n:
        return 'Zero'
    if n < 1000:
        result = thousand(n)
    else:
        first, second = map(thousand, divmod(n, 1000))
        result = first + ['Thousand,'] + second
    r = ' '.join(result[:-1]) + ' ' + result[-1]
    return r.rstrip(',')


def nineteenseven(arr):
    '''
    You are given an array of integers (both positive and negative). Find the continuous
    sequence with the largest sum. Return the sum.
    EXAMPLE
    Input: {2, -8, 3, -2, 4, -10}
    Output: 5 (i.e., {3, -2, 4} )
    '''
    maxed = -1
    for start in range(len(arr)):
        su = arr[start]
        if su < 0:
            continue
        for curr in range(start+1, len(arr)):
            su += arr[curr]
            maxed = max(maxed, su)
    return maxed

def nineteeneight(text, word):
    '''Design a method to find the frequency of occurrences of any given word in a book.'''
    return Counter(text.split())[word]


def nineteennine():
    '''
    Since XML is very verbose, you are given a way of encoding it where each tag gets
    mapped to a pre-defined integer value. The language/grammar is as follows:

    Element --> Element Attr* END Element END [aka, encode the element
        tag, then its attributes, then tack on an END character, then
        encode its children, then another end tag]
    Attr --> Tag Value [assume all values are strings]
    END --> 01
    Tag --> some predefined mapping to int
    Value --> string value END

    Write code to print the encoded version of an xml element (passed in as string).
    FOLLOW UP
    Is there anything else you could do to (in many cases) compress this even further?

    '''

    # lol wat


def nineteenten():
    import random
    rand = lambda: random.randrange(0, 6)
    while True:
        n = rand() * 5 + rand()
        if n <= 21:
            return n % 7

def twentyone(a,b):
    '''
    Write a function that adds two numbers. You should not use + or any arithmetic op-
    erators.
    '''
    result = 0
    carry = 0
    mask = 1
    for i in range(32):
        aa = int(bool(a & mask))
        bb = int(bool(b & mask))
        here = carry ^ aa ^ bb
        anded = aa & bb
        ored = aa | bb
        carry = (anded | (carry & ored))
        result |= (here << i)
        mask = mask << 1
    return result

def twentytwo(arr):
    '''
    Write a method to shuffle a deck of cards. It must be a perfect shuffle - in other words,
    each 52! permutations of the deck has to be equally likely. Assume that you are given
    a random number generator which is perfect.
    '''
    import random
    for i in reversed(range(1, len(arr))):
        j = int(random.random() * (i+1))
        arr[i], arr[j] = arr[j], arr[i]

def twentythree(arr, m):
    '''
    Write a method to randomly generate a set of m integers from an array of size n. Each
    element must have equal probability of being chosen.
    '''
    import random
    if m > len(arr):
        return
    result = []
    used = set()
    while len(result) < m:
        idx = int(random.random() * len(arr))
        elem = arr[idx]
        if idx in used:
            continue
        else:
            result.append(elem)
            used.add(idx)
    return result

def twentyfour(n):
    '''Write a method to count the number of 2s between 0 and n.'''
    if n < 2:
        return 0
    if 2 <= n < 10:
        return 1
    first = int(str(n)[0])
    ll = 10 ** (len(str(n))-1)
    remainder = n % ll
    twos = 0
    if first == 2:
        twos += remainder
    elif first > 2:
        twos += ll
    return twos + twentyfour(ll-1) * first + twentyfour(remainder)


def twentyfive(words, a, b):
    '''
    You have a large text file containing words. Given any two words, find the shortest
    distance (in terms of number of words) between them in the file. Can you make the
    searching operation in O(1) time? What about the space complexity for your solu-
    tion?
    '''
    a_idx = -10000000
    b_idx = -10000000
    best = 10000000
    for i, word in enumerate(words):
        if word == a:
            a_idx = i
            best = min(a_idx, b_idx)
        if word == b:
            b_idx = i
            best = min(a_idx, b_idx)
    return best


def twentysix(numbers):
    '''
    Describe an algorithm to find the largest 1 million numbers in 1 billion numbers. As-
    sume that the computer memory can hold all one billion numbers.
    '''
    class MinHeap:pass
    heap = MinHeap()
    it = iter(numbers)
    for _ in range(10**6):
        heap.insert(it.next())
    for num in it:
        heap.insert(num)
        heap.remove_min()
    return list(heap)

def twentyseven(text):
    '''
    Write a program to find the longest word made of other words in a list of words.
    EXAMPLE
    Input: test, tester, testertest, testing, testingtester
    Output: testingtester
    '''
    all_words = set(text.split())
    def composed(word, accu=None):
        if word in all_words:
            return word
        for i in range(1, len(word)):
            prefix = word[:i]
            if prefix in all_words:
                suffixed = composed(word[i:])
                if suffixed:
                    return prefix + " " + suffixed

    return max([composed(word) for word in all_words], key=len)

def twentyeight(s, T):
    '''
    Given a string s and an array of smaller strings T, design a method to search s for each
    small string in T.
    '''
    class Node(object):
        def __init__(self, value):
            self.value = value
            self.children = []

        def __repr__(self):
            return repr(self.children)

    class SuffixTree(object):
        def __init__(self, string):
            self.root = Node('')
            for i in range(1, len(string)+1):
                self.insert(string[:i])

        def find(self, value, insert=False):
            try:
                self._find(value, insert)
            except IndexError:
                return False
            else:
                return True

        def _find(self, value, insert):
            def walk(root, s):
                if not s:
                    return
                found = False
                for child in root.children:
                    if child.value == s[-1]:
                        found = True
                        break
                if not found:
                    if insert:
                        child = Node(s[-1])
                        root.children.append(child)
                    else:
                        raise IndexError
                walk(child, s[:-1])
            walk(self.root, value)

        def insert(self, value):
            return self._find(value, insert=True)

    tree = SuffixTree(s)
    result = []
    for t in T:
        res = tree.find(t)
        if res:
            result.append(t)

    return result

def twentynine():
    '''
    Numbers are randomly generated and passed to a method. Write a program to find
    and maintain the median value as new values are generated.
    '''

    class Heap(object):
        pass

    class MinHeap(Heap):
        def cmp(a, b):
            return cmp(a, b)

    class MaxHeap(Heap):
        def cmp(a, b):
            return cmp(b, a)

    mini = MinHeap()
    maxi = MaxHeap()
    def fun(number):
        if number > mini.min():
            mini.insert(number)
        else:
            maxi.insert(number)
        if len(maxi) + len(mini) % 2 == 0:
            median = (maxi.max() + mini.min()) / 2
        else:
            median = maxi.max()
        return median

def twentyeleven(arr):
    '''
    Imagine you have a square matrix, where each cell is filled with either black or white.
    Design an algorithm to find the maximum subsquare such that all four borders are
    filled with black pixels.
    '''

    def is_square(x, y, size):
        ll = x, y + size
        ur = x + size, y
        lr = x + size, y + size
        return all(arr[y][x] == 'B' for x, y in ((x,y), ll, ur, lr))

    maxed = float('-inf')
    maxed_coords = []
    x = 0
    N = len(arr)
    while N - x > maxed:
        for y in range(N):
            size = N - max(x, y) - 1
            while size > maxed:
                if is_square(x, y, size):
                    maxed = size
                    maxed_coords = x, y, size
                size -= 1
        x += 1

    return maxed_coords


def twentytwelve(arr):
    '''
    Given an NxN matrix of positive and negative integers, write code to find the sub-
    matrix with the largest possible sum.
    '''

    # this is N^4 + N^4 and can be optimized to N^2 + N^4
    N = len(arr)
    d = {}
    for y1 in range(N+1):
        for x1 in range(N+1):
            d[(x1, y1)] = sum(arr[y][x] for y in range(y1) for x in range(x1))
    maxed = float('-inf')
    maxed_coords = []

    def subsum(x1, y1, x2, y2):
        return d[(x2, y2)] - d[(x1, y2)] - d[(x2, y1)] + d[(x1, y1)]

    for y1 in range(N):
        for x1 in range(N):
            for y2 in range(y1, N):
                for x2 in range(x1, N):
                    area = subsum(x1,y1,x2+1,y2+1)
                    if area > maxed:
                        maxed = area
                        maxed_coords = x1, y1, x2, y2
    return maxed_coords, maxed

# print twentytwelve([
#     [-100,2,3],
#     [4,5,6],
#     [7,8,9]])


def twentythirteen():
    '''
    Given a dictionary of millions of words, give an algorithm to find the largest possible
    rectangle of letters such that every row forms a word (reading left to right) and every
    column forms a word (reading top to bottom).
    '''
    





