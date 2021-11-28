import uuid


class Task:
    def __init__(self, cpu_work, gpu_work):
        self.cpu_work = cpu_work
        self.gpu_work = gpu_work
        self.id == uuid.uuid4()

    def run(self):
        return RunningTask(self, self.cpu_work, self.gpu_work, self.id)


class RunningTask:
    def __init__(self, cpu_work, gpu_work, id):
        self.cpu_work = cpu_work
        self.cpu_progress = 0
        self.gpu_work = gpu_work
        self.gpu_progress = 0
        self.id = id

    def is_complete(self):
        return self.cpu_progress >= self.cpu_work and self.gpu_progress >= self.gpu_work

    def work(self, cpu_power, gpu_power):
        self.cpu_progress += cpu_power
        self.gpu_progress += gpu_power


class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def tick(self):
        pass


class Node(Entity):
    def __init__(self, x, y, cpu_power, gpu_power):
        super().__init__(x, y)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks = []
        self.results = set()

    def tick(self):
        try:
            cur_task = self.tasks[0]
        except IndexError:
            return

        if cur_task.is_complete():
            # yeah sure, popping takes a tick. whatever.
            self.results.add(cur_task.id)
            self.tasks = self.tasks[1:]
        else:
            cur_task.work(self.cpu_power, self.gpu_power)

    def estimate_time(self, task):
        return max(task.cpu_work / self.cpu_power, task.gpu_work / task.gpu_power)


def main():
    pass


if __name__ == "__main__":
    main()
