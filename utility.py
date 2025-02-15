from collections import deque
import os

class Queue:

    def __init__(self):
        self.dq = deque([])
        self.sz = 0

    def push(self, x):
        self.dq.append(x)
        self.sz+=1

    def pop(self):
        if(self.sz==0): return
        self.dq.popleft()
        self.sz-=1

    def empty(self):
        if(self.sz==0): return True
        return False
    
    def front(self):
        if(self.sz==0): return -1
        x = self.dq.popleft()
        self.dq.appendleft(x)
        return x
    
    def rear(self):
        if(self.sz==0): return -1
        x = self.dq.pop()
        self.dq.append(x)
        return x
    
    def copyQueue(self, queue):
        while(not queue.empty()):
            self.push(queue.front())
            queue.pop()
        return
    

def create_directory(directory):
    print(directory)
    try:
        os.mkdir(directory)
        print("Success")
    except FileExistsError:
        print("Existed")
        return
    except PermissionError:
        print("Denied")
        return
    except Exception as e:
        print("Error")
        return
    return
