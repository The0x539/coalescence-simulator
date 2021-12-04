from uuid import UUID, uuid4
import math
from typing import List, Set, Optional, Tuple, Dict, TypeVar, Type, cast, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto
import random
import matplotlib.pyplot as plt


T = TypeVar("T")


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
        self.cpu_progress = min(self.cpu_progress + cpu_power, self.cpu_work)
        self.gpu_progress = min(self.gpu_progress + gpu_power, self.gpu_work)

    def remaining_time(self, cpu_power: int, gpu_power: int) -> int:
        remaining_cpu_work = self.cpu_work - self.cpu_progress
        remaining_gpu_work = self.gpu_work - self.gpu_progress
        return (
            math.ceil(
                max(remaining_cpu_work / cpu_power, remaining_gpu_work / gpu_power)
            )
            + 1
        )


class Task:
    def __init__(self, cpu_work: int, gpu_work: int) -> None:
        self.cpu_work = cpu_work
        self.gpu_work = gpu_work
        self.id = uuid4()
        self.runners: Dict[UUID, int] = {}

    def run(self) -> RunningTask:
        return RunningTask(self.cpu_work, self.gpu_work, self.id)


class Result:
    def __init__(self, id: UUID) -> None:
        self.id = id


class Entity(ABC):
    def __init__(self, x: int, y: int, range: float) -> None:
        self.x = x
        self.y = y
        self.range = range

    @abstractmethod
    def tick(self, world: "World") -> None:
        ...


class Node(Entity):
    def __init__(
        self, x: int, y: int, cpu_power: int, gpu_power: int, range: float
    ) -> None:
        super().__init__(x, y, range)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks: List[RunningTask] = []
        self.results: Dict[UUID, Result] = {}
        self.time_left = 0
        self.id = uuid4()

    def tick(self, world: "World") -> None:
        try:
            cur_task = self.tasks[0]
        except IndexError:
            return

        if self.time_left > 0:
            self.time_left -= 1

        if cur_task.is_complete():
            # yeah sure, popping takes a tick. whatever.
            self.results[cur_task.id] = Result(cur_task.id)
            self.tasks = self.tasks[1:]
        else:
            cur_task.work(self.cpu_power, self.gpu_power)

    def estimate_time(self, task: Task) -> int:
        # returns the amount of time before the node would complete the task
        task_time = math.ceil(
            max(task.cpu_work / self.cpu_power, task.gpu_work / self.gpu_power)
        )
        return self.time_left + task_time + 1  # 1 extra tick for the "pop"

    def spawn(self, task: Task) -> None:
        self.tasks.append(task.run())
        self.time_left = self.estimate_time(task)
        task.runners[self.id] = self.time_left

    def cancel(self, task: Task) -> None:
        for (i, r_t) in enumerate(self.tasks):
            if r_t.id == task.id:
                index = i
                running_task = r_t
                break
        else:
            # ok dude, I wasn't running it anyway
            return

        self.time_left -= running_task.remaining_time(self.cpu_power, self.gpu_power)
        del self.tasks[index]

    def get_results(self, id: UUID) -> Optional[Result]:
        if id not in self.results:
            return None

        res = self.results[id]
        del self.results[id]
        return res

    def can_run(self, task: Task) -> bool:
        meets_cpu_requirement = self.cpu_power > 0 or task.cpu_work == 0
        meets_gpu_requirement = self.gpu_power > 0 or task.gpu_work == 0
        return meets_cpu_requirement and meets_gpu_requirement


class Device(Entity):
    def __init__(
        self, x: int, y: int, range: float, personal_node: Optional[Node] = None
    ) -> None:
        super().__init__(x, y, range)
        self.tasks: Dict[UUID, Task] = {}
        self.personal_node = personal_node

    def tick(self, world: "World") -> None:

        nearby_nodes = world.neighbors_of(self, Node)

        # A tick for a device is one iteration over its tasks (it does more than one tick of work in the tick, we can change it later if we want)
        for task in self.tasks.values():
            got_result = False
            cur_min_eta = min(task.runners.values())

            for node_id in task.runners.keys():
                if task.runners[node_id] > 0:
                    task.runners[node_id] -= 1

            best_neighbor_for_task = None
            for node in nearby_nodes:
                if node.id in task.runners:
                    if task.runners[node.id] == 0:
                        res = node.get_results(task.id)
                        if res is None:
                            # it should be ready, but it isn't yet. probably next cycle
                            continue

                        assert res.id == task.id, "task ID mismatch"
                        got_result = True
                        break
                elif node.can_run(task):
                    eta = node.estimate_time(task)
                    if eta < cur_min_eta:
                        cur_min_eta = eta
                        best_neighbor_for_task = node

            if got_result:
                del self.tasks[task.id]
                if self.personal_node is not None:
                    # ideally we could tell neighbors to also cancel it but whatever
                    self.personal_node.cancel(task)
            elif best_neighbor_for_task is not None:
                best_neighbor_for_task.spawn(task)

    def request_task(self, cpu_work: int, gpu_work: int) -> None:
        task = Task(cpu_work, gpu_work)
        self.tasks[task.id] = task
        if self.personal_node is not None and self.personal_node.can_run(task):
            self.personal_node.spawn(task)

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
        if self.personal_node is not None:
            self.personal_node.x += dx
            self.personal_node.y += dy


class World:
    def __init__(self) -> None:
        self.entities: List[Entity] = []

    def tick(self) -> None:
        for entity in self.entities:
            entity.tick(self)

    def add_entity(self, entity: Entity) -> None:
        for e in self.entities:
            if e is entity:
                raise ValueError("entity already in world!")
        self.entities.append(entity)

    def neighbors_of(
        self,
        entity: Entity,
        neighbor_type: Type[T],
    ) -> Set[T]:
        neighbors: Set[T] = set()
        for potential_neighbor in self.entities:
            if potential_neighbor is entity:
                continue

            # mypy what are you smoking
            pnx = potential_neighbor.x
            pny = potential_neighbor.y
            pnr = potential_neighbor.range

            if not isinstance(potential_neighbor, neighbor_type):
                continue

            dx = entity.x - pnx
            dy = entity.y - pny
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > min(entity.range, pnr):
                continue

            neighbors.add(potential_neighbor)

        return neighbors

    def show(self) -> None:
        fig, ax = plt.subplots()
        rc: Callable[[], str] = lambda: f"{random.randrange(0, 255):02x}"

        for entity in self.entities:
            linestyle = "--" if isinstance(entity, Device) else ":"

            color = f"#{rc()}{rc()}{rc()}"
            circle = plt.Circle(
                (entity.x, entity.y),
                entity.range,
                fill=False,
                color=color,
                linestyle=linestyle,
            )
            ax.add_patch(circle)
            ax.scatter(entity.x, entity.y, c=color)

        plt.show()


def main() -> None:
    w = World()
    for _ in range(10):
        n = Node(
            random.randrange(-50, 50),
            random.randrange(-50, 50),
            random.randrange(1, 10),
            random.randrange(0, 5),
            random.uniform(1.0, 15.0),
        )
        w.add_entity(n)

    for _ in range(5):
        d = Device(
            random.randrange(-50, 50),
            random.randrange(-50, 50),
            random.uniform(10.0, 25.0),
        )
        w.add_entity(d)

    w.show()


if __name__ == "__main__":
    main()
