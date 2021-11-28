from uuid import UUID, uuid4
import math
from typing import List, Set


class RunningTask:
    def __init__(self, cpu_work: int, gpu_work: int, id: UUID) -> None:
        self.cpu_work = cpu_work
        self.cpu_progress = 0
        self.gpu_work = gpu_work
        self.gpu_progress = 0
        self.id = id

    def is_complete(self) -> bool:
        return self.cpu_progress >= self.cpu_work and self.gpu_progress >= self.gpu_work

    def work(self, cpu_power: int, gpu_power: int) -> None:
        self.cpu_progress += cpu_power
        self.gpu_progress += gpu_power


class Task:
    def __init__(self, cpu_work: int, gpu_work: int) -> None:
        self.cpu_work = cpu_work
        self.gpu_work = gpu_work
        self.id = uuid4()

    def run(self) -> RunningTask:
        return RunningTask(self.cpu_work, self.gpu_work, self.id)


class Entity:
    def __init__(self, x: int, y: int, range: float) -> None:
        self.x = x
        self.y = y
        self.range = range

    def tick(self) -> None:
        pass


class Node(Entity):
    def __init__(
        self, x: int, y: int, cpu_power: int, gpu_power: int, range: float
    ) -> None:
        super().__init__(x, y, range)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks: List[RunningTask] = []
        self.results: Set[UUID] = set()

    def tick(self) -> None:
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

    def estimate_time(self, task: Task) -> int:
        return math.ceil(
            max(task.cpu_work / self.cpu_power, task.gpu_work / self.gpu_power)
        )

    def spawn(self, task: Task) -> None:
        self.tasks.append(task.run())


class World:
    def __init__(self) -> None:
        self.entities: List[Entity] = []

    def tick(self) -> None:
        for entity in self.entities:
            entity.tick()

    def neighbors_of(self, entity: Entity) -> Set[Entity]:
        neighbors = set()
        for potential_neighbor in self.entities:
            dx = entity.x - potential_neighbor.x
            dy = entity.y - potential_neighbor.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= min(entity.range, potential_neighbor.range):
                neighbors.add(potential_neighbor)

        return neighbors


def main() -> None:
    pass


if __name__ == "__main__":
    main()
