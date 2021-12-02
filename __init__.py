from uuid import UUID, uuid4
import math
from typing import List, Set, Optional, Tuple, Dict
from abc import ABC, abstractmethod
from enum import Enum, auto


class EntityType(Enum):
    NODE = auto()
    DEVICE = auto()


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


class Entity(ABC):
    def __init__(self, x: int, y: int, range: float) -> None:
        self.x = x
        self.y = y
        self.range = range

    @abstractmethod
    def tick(self, world: World) -> None:
        ...

    @abstractmethod
    def entity_type(self) -> EntityType:
        ...

    def estimate_time(self, task: Task) -> int:
        assert False

    def spawn(self, task: Task) -> None:
        assert False

    def get_results(self, id: UUID) -> None:
        assert False


class Node(Entity):
    def __init__(
        self, x: int, y: int, cpu_power: int, gpu_power: int, range: float
    ) -> None:
        super().__init__(x, y, range)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks: List[RunningTask] = []
        self.results: Set[UUID] = set()
        self.time_left = 0

    def tick(self, world: World) -> None:
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
        # returns the amount of time before the node would complete the task
        return self.time_left + math.ceil(
            max(task.cpu_work / self.cpu_power, task.gpu_work / self.gpu_power)
        )

    def spawn(self, task: Task) -> None:
        self.tasks.append(task.run())
        self.time_left += math.ceil(
            max(task.cpu_work / self.cpu_power, task.gpu_work / self.gpu_power)
        )

    def get_results(self, id: UUID) -> None:
        # can be implemented to require ticks to send the data
        pass

    def entity_type(self) -> EntityType:
        return EntityType.NODE


class Device(Entity):
    def __init__(self, x: int, y: int, range: float) -> None:
        super().__init__(x, y, range)
        self.tasks: List[Tuple[Task, int]] = []
        self.runners: Dict[Task, Entity] = {}

    def tick(self, world: World) -> None:
        # A tick for a device is one iteration over its tasks (it does more than one tick of work in the tick, we can change it later if we want)
        indices_to_remove = []
        for i in range(0, len(self.tasks)):
            (task, ticksLeft) = self.tasks[i]
            if ticksLeft == -1:
                # task is not running anywhere
                # tries to find a place for it to run
                best_time = -1
                for neighbor in world.neighbors_of(self, {EntityType.NODE}):
                    time = neighbor.estimate_time(task)
                    if time < best_time or best_time == -1:
                        best_time = time
                        best_node: Entity = neighbor
                if best_time != -1:
                    best_node.spawn(task)
                    self.tasks[i] = (task, best_time)
                    self.runners.update({task: best_node})

            elif ticksLeft == 0:
                # get the computation results
                best_node.get_results(task.id)
                indices_to_remove.append(i)

            else:
                # reduce the time left
                self.tasks[i] = (task, ticksLeft - 1)

        for i in indices_to_remove:
            del self.tasks[i]

    def entity_type(self) -> EntityType:
        return EntityType.DEVICE

    def request_task(self, cpu_work: int, gpu_work: int) -> None:
        self.tasks.append((Task(cpu_work, gpu_work), -1))


class World:
    def __init__(self) -> None:
        self.entities: List[Entity] = []

    def tick(self) -> None:
        for entity in self.entities:
            entity.tick(self)

    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)

    def neighbors_of(
        self, entity: Entity, types: Optional[Set[EntityType]] = None
    ) -> Set[Entity]:
        neighbors = set()
        for potential_neighbor in self.entities:
            if potential_neighbor is entity:
                continue

            if types is not None and entity.entity_type() not in types:
                continue

            dx = entity.x - potential_neighbor.x
            dy = entity.y - potential_neighbor.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > min(entity.range, potential_neighbor.range):
                continue

            neighbors.add(potential_neighbor)

        return neighbors


def main() -> None:
    x = Node(0, 0, 1, 1, 5.0)


if __name__ == "__main__":
    main()
