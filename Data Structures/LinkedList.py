class LinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, x):
        if self.size > 0:
            self.tail.next = Node(x)
            self.tail.next.prev = self.tail
            self.tail = self.tail.next
        else:
            self.head = Node(x)
            self.tail = self.head
        self.size += 1

    def prepend(self, x):
        if self.size > 0:
            self.head.prev = Node(x)
            self.head.prev.next = self.head
            self.head = self.head.prev
        else:
            self.head = Node(x)
            self.tail = self.head
        self.size += 1

    def get(self, index):
        if index < self.size / 2:
            return self.head.forward(index)
        return self.tail.backward(self.size - index - 1)

    def rpop(self):
        data = self.tail.data
        self.tail = self.tail.prev
        self.tail.next = None
        self.size -= 1
        return data

    def lpop(self):
        data = self.head.data
        self.head = self.head.next
        self.head.prev = None
        self.size -= 1
        return data

    def __str__(self):
        return f"<{str(self.head)}>"


class Node:

    def __init__(self, x):
        self.data = x
        self.prev = None
        self.next = None

    def forward(self, steps):
        if steps == 0:
            return self.data
        return self.next.forward(steps - 1)
    
    def backward(self, steps):
        if steps == 0:
            return self.data
        return self.prev.backward(steps - 1)

    def __str__(self):
        if self.next:
            return f"{self.data}, {str(self.next)}"
        return str(self.data)
    

l = LinkedList()
l.append(0)
l.append(1)
l.prepend(2)
print(l.get(0))
print(l)
print(l.size)
print(l.rpop())
print(l)
print(l.size)
print(l.lpop())
print(l)
print(l.size)


# from time import time_ns

# def timer(func):
#     start = time_ns()
#     for iteration in range(10000):
#         func.append(iteration)
#     finish = time_ns()
#     return (finish - start) * 1e-9

# def avg_timer(func):
#     times = 0
#     for iteration in range(10):
#         times += timer(func)
#     return times / 10

# print(avg_timer([]))
# print(avg_timer(LinkedList()))

