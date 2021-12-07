from uuid import UUID, uuid4
import math
from typing import List, Set, Optional, Dict, TypeVar, Type
from abc import ABC, abstractmethod
import random
import matplotlib.pyplot as plt
import time

FPS: int = 8
SECONDS_BETWEEN_VELOCITY_UPDATES: int = 1

EXTRA_BUDGET_CONSIDERATION_FACTOR = 1.00
PERFORMANCE_VARIATION = 0.01

ANGLE_CHANGE_VARIATION = 5
MAGNITUDE_CHANGE_VARIATION = 1
MAXIMUM_MAGNITUDE = 10

TESTING = False
NUM_NODES_MU

T = TypeVar("T")


def random_color() -> str:
    r = random.randrange(0, 255)
    g = random.randrange(0, 255)
    b = random.randrange(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"


class RunningTask:
    def __init__(
        self,
        budget: int,
        cpu_work: int,
        gpu_work: int,
        task_id: UUID,
        runner_id: UUID,
        heartbeat_time: int,
    ) -> None:
        self.budget = budget
        self.progress = 0
        self.cpu_work_left = cpu_work
        self.gpu_work_left = gpu_work
        self.id = task_id
        self.runner_id = runner_id
        self.heartbeat_time = heartbeat_time
        self.heartbeat_timer = heartbeat_time

    def is_complete(self) -> bool:
        return self.cpu_work_left <= 0 and self.gpu_work_left <= 0

    # returns True iff the task is due for a heartbeat check
    def heartbeat_tick(self) -> bool:
        # this should never happen. if it does, it's a bug in the simulator.
        assert self.heartbeat_timer > 0, "Missed heartbeat check!"

        self.heartbeat_timer -= 1
        return self.heartbeat_timer == 0

    def refresh_heartbeat(self) -> None:
        self.heartbeat_timer = self.heartbeat_time

    def work(self, cpu_power: int, gpu_power: int) -> None:
        assert self.budget > 0
        self.cpu_work_left -= cpu_power * random.gauss(1, PERFORMANCE_VARIATION)
        self.gpu_work_left -= gpu_power * random.gauss(1, PERFORMANCE_VARIATION)

    def remaining_budget(self) -> int:
        return self.budget - self.progress


class Task:
    def __init__(self, cpu_budget: int, gpu_budget: int, heartbeat_time: int) -> None:
        self.cpu_budget = cpu_budget
        self.gpu_budget = gpu_budget
        self.heartbeat_time = heartbeat_time
        self.id = uuid4()
        self.runners: Dict[UUID, int] = {}

    def run(self, runner_id: UUID, cpu_power: int, gpu_power: int) -> RunningTask:
        budget = math.ceil(max(self.cpu_budget / cpu_power, self.gpu_budget / gpu_power))
        return RunningTask(
            math.ceil(budget * EXTRA_BUDGET_CONSIDERATION_FACTOR), self.id, runner_id, self.heartbeat_time
        )


class Result:
    def __init__(self, id: UUID) -> None:
        self.id = id


class Entity(ABC):
    def __init__(self, x: float, y: float, range: float) -> None:
        self.x = x
        self.y = y
        self.range = range
        self.color = random_color()
        self.id = uuid4()

    @abstractmethod
    def tick(self, world: "World") -> None:
        ...


class Node(Entity):
    def __init__(
        self, x: float, y: float, cpu_power: int, gpu_power: int, range: float
    ) -> None:
        super().__init__(x, y, range)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks: List[RunningTask] = []
        self.results: Dict[UUID, Result] = {}
        self.time_left = 0

    def tick(self, world: "World") -> None:
        try:
            cur_task = self.tasks[0]
        except IndexError:
            return

        if self.time_left > 0:
            self.time_left -= 1

        if cur_task.heartbeat_tick():
            # Check whether the device who requested it is (still) in range
            for device in world.neighbors_of(self, Device):
                if device.id == cur_task.id:
                    cur_task.refresh_heartbeat()
                    break
            else:
                self.cancel(cur_task.id)
                # at the time of writing, the rest of this function just ticks the task
                # but we just cancelled the task (which did take up a tick, which we measured)
                # so just early return, whatever
                return

        if cur_task.remaining_budget() <= 0:
            self.cancel(cur_task.id)

        elif cur_task.is_complete():
            # yeah sure, popping takes a tick. whatever.
            self.results[cur_task.id] = Result(cur_task.id)
            self.tasks = self.tasks[1:]
            # TODO: heartbeats for results?
            # how expensive are we imagining it to be to hold on to a result?
        else:
            cur_task.work(self.cpu_power, self.gpu_power)

    def estimate_time(self, task: Task) -> int:
        # returns the amount of time before the node would complete the task
        task_time = math.ceil(
            max(task.cpu_budget / self.cpu_power, task.gpu_budget / self.gpu_power)
        )
        return self.time_left + task_time + 1  # 1 extra tick for the "pop"

    def spawn(self, task: Task) -> None:
        self.tasks.append(task.run(self.id, self.cpu_power, self.gpu_power))
        self.time_left = self.estimate_time(task)
        task.runners[self.id] = self.time_left

    def cancel(self, task_id: UUID) -> None:
        for (i, r_t) in enumerate(self.tasks):
            if r_t.id == task_id:
                index = i
                running_task = r_t
                break
        else:
            # ok dude, I wasn't running it anyway
            return

        self.time_left -= running_task.remaining_budget(self.cpu_power, self.gpu_power)
        del self.tasks[index]

    def get_results(self, id: UUID) -> Optional[Result]:
        if id not in self.results:
            return None

        res = self.results[id]
        del self.results[id]
        return res

    def can_run(self, task: Task) -> bool:
        meets_cpu_requirement = self.cpu_power > 0 or task.cpu_budget == 0
        meets_gpu_requirement = self.gpu_power > 0 or task.gpu_budget == 0
        return meets_cpu_requirement and meets_gpu_requirement


class Device(Entity):
    def __init__(
        self, x: float, y: float, range: float, personal_node: Optional[Node] = None
    ) -> None:
        super().__init__(x, y, range)
        self.tasks: Dict[UUID, Task] = {}
        self.personal_node = personal_node
        self.v_ang = 0.0
        self.v_mag = 0.0

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
                    self.personal_node.cancel(task.id)
            elif best_neighbor_for_task is not None:
                best_neighbor_for_task.spawn(task)

    def request_task(self, cpu_budget: int, gpu_budget: int, heartbeat_time: int) -> None:
        task = Task(cpu_budget, gpu_budget, heartbeat_time)
        self.tasks[task.id] = task
        if self.personal_node is not None and self.personal_node.can_run(task):
            self.personal_node.spawn(task)

    def move(self) -> None:
        x_mov = math.cos(self.v_ang) * self.v_mag / FPS
        y_mov = math.sin(self.v_ang) * self.v_mag / FPS
        self.x += x_mov
        self.y += y_mov
        if self.personal_node is not None:
            self.personal_node.x += x_mov
            self.personal_node.y += y_mov

    def change_velocity(self, angle: float, magnitude: float) -> None:
        assert magnitude <= MAXIMUM_MAGNITUDE
        self.v_ang += angle
        self.v_mag += magnitude
        if self.v_mag < 0:
            self.v_mag = -self.v_mag
        if self.v_mag > MAXIMUM_MAGNITUDE:
            assert self.v_mag - MAXIMUM_MAGNITUDE <= MAXIMUM_MAGNITUDE
            self.v_mag = 2 * MAXIMUM_MAGNITUDE - self.v_mag



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

    def plot(self, ax) -> None:
        ax.clear()
        for entity in self.entities:
            linestyle = "--" if isinstance(entity, Device) else ":"

            circle = plt.Circle(
                (entity.x, entity.y),
                entity.range,
                fill=False,
                color=entity.color,
                linestyle=linestyle,
            )
            ax.add_patch(circle)
            ax.scatter(entity.x, entity.y, c=entity.color)

    def print(self) -> None:
        for e in self.entities:
            if isinstance(e, Node):
                print(f"node @ ({e.x} {e.y}) with {len(e.tasks)} tasks in queue")
            elif isinstance(e, Device):
                print(f"device @ ({e.x} {e.y}) with {len(e.tasks)} tasks active")
            else:
                print(f"entity @ ({e.x} {e.y})")


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

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    w.plot(ax)

    last_frame_time = 0.0
    last_velocity_time = 0.0

    while True:
        t = time.time()
        if t - last_frame_time > 1 / FPS:
            last_frame_time = t
            fig.canvas.draw()
            fig.canvas.flush_events()

            for e in w.entities:
                if isinstance(e, Device):
                    e.move()

            if t - last_velocity_time > SECONDS_BETWEEN_VELOCITY_UPDATES:
                last_velocity_time = t
                for e in w.entities:
                    if isinstance(e, Device):
                        e.change_velocity(random.gauss(0, ANGLE_CHANGE_VARIATION),
                                          random.gauss(0, MAGNITUDE_CHANGE_VARIATION))

            w.tick()
            w.plot(ax)


if __name__ == "__main__":
    main()
