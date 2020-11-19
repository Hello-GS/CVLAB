# coding=utf-8
import heapq

class My_PriorityQueue(object):
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        """
        队列由 (priority, index, item) 形式组成
        priority 增加 "-" 号是因为 heappush 默认是最小堆
        index 是为了当两个对象的优先级一致时，按照插入顺序排列
        """
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        """
        弹出优先级最高的对象
        """
        return heapq.heappop(self._queue)[-1]

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return True if not self._queue else False

class Car(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "{0} -- {1}".format(self.name, self.value)

if __name__ == "__main__":
    car1 = Car("BMW", 45)
    car2 = Car("Maybach", 145)
    car3 = Car("Bugatti", 85)
    car4 = Car("Cadillac", 78)
    car5 = Car("Maserati", 85)
    pq = My_PriorityQueue()
    pq.push(car1, car1.value)
    pq.push(car2, car2.value)
    pq.push(car3, car3.value)
    pq.push(car4, car4.value)
    pq.push(car5, car5.value)
    print("队列大小：{0}".format(pq.qsize()))
    # 弹出元素
    while not pq.empty():
        print(pq.pop())