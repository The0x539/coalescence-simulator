from uuid import UUID, uuid4
import math
from typing import List, Set, Optional, Dict, TypeVar, Type, Callable, Union
from abc import ABC, abstractmethod
import random
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

FPS: int = 8
SECONDS_BETWEEN_VELOCITY_UPDATES: int = 1

EXTRA_BUDGET_CONSIDERATION_FACTOR = 1.00
PERFORMANCE_VARIATION = 0.01

ANGLE_CHANGE_VARIATION = 5
MAGNITUDE_CHANGE_VARIATION = 1
MAXIMUM_MAGNITUDE = 10

NUM_NODES_VAR = 1
NODE_RANGE_VAR = 1
NODE_POWER_VAR = 1
TASK_SIZE_VAR = 1


T = TypeVar("T")


def random_color() -> str:
    r = random.randrange(0, 255)
    g = random.randrange(0, 255)
    b = random.randrange(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def safe_div(num: Union[int, float], den: Union[int, float]) -> float:
    if num == 0 and den == 0:
        return 0
    else:
        assert den != 0, "division by zero!"

    return num / den


class RunningTask:
    def __init__(
        self,
        time_budget: int,
        cpu_work: int,
        gpu_work: int,
        task_id: UUID,
        runner_id: UUID,
        heartbeat_time: int,
        is_local: bool,
    ) -> None:
        assert heartbeat_time > 0

        self.time_budget = time_budget
        self.time_spent = 0
        self.cpu_work_left = float(cpu_work)
        self.gpu_work_left = float(gpu_work)
        self.id = task_id
        self.runner_id = runner_id
        self.heartbeat_time = heartbeat_time
        self.heartbeat_timer = heartbeat_time
        self.is_local = is_local

    def is_complete(self) -> bool:
        return self.cpu_work_left <= 0 and self.gpu_work_left <= 0

    # returns True iff the task is due for a heartbeat check
    def heartbeat_tick(self) -> bool:
        # this should never happen. if it does, it's a bug in the simulator.
        assert self.heartbeat_timer > 0, f"Missed heartbeat check for task {self.id}!"

        self.heartbeat_timer -= 1
        return self.heartbeat_timer == 0

    def refresh_heartbeat(self) -> None:
        self.heartbeat_timer = self.heartbeat_time

    def work(self, cpu_power: float, gpu_power: float) -> None:
        self.time_spent += 1
        self.cpu_work_left -= cpu_power * random.gauss(1, PERFORMANCE_VARIATION)
        self.gpu_work_left -= gpu_power * random.gauss(1, PERFORMANCE_VARIATION)

    def has_exhausted_budget(self) -> bool:
        if self.is_local:
            # local tasks aren't subject to actual budget limitations
            return False
        else:
            return self.time_spent >= self.time_budget

    def remaining_budget(self) -> int:
        return max(self.time_budget - self.time_spent, 0)


class Task:
    def __init__(
        self, cpu_work: int, gpu_work: int, heartbeat_time: int, cur_time: int
    ) -> None:
        self.cpu_work = cpu_work
        self.gpu_work = gpu_work
        self.heartbeat_time = heartbeat_time
        self.id = uuid4()
        self.runners: Dict[UUID, int] = {}
        self.time_of_request = cur_time

    def run(
        self, runner_id: UUID, cpu_power: float, gpu_power: float, is_local: bool
    ) -> RunningTask:
        estimate = max(
            safe_div(self.cpu_work, cpu_power), safe_div(self.gpu_work, gpu_power)
        )
        budget = math.ceil(estimate * EXTRA_BUDGET_CONSIDERATION_FACTOR)

        return RunningTask(
            budget,
            self.cpu_work,
            self.gpu_work,
            self.id,
            runner_id,
            self.heartbeat_time,
            is_local,
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
    def tick(self, world: "World", cur_time: int) -> None:
        ...


class Node(Entity):
    def __init__(
        self, x: float, y: float, cpu_power: float, gpu_power: float, range: float
    ) -> None:
        super().__init__(x, y, range)
        self.cpu_power = cpu_power
        self.gpu_power = gpu_power
        self.tasks: List[RunningTask] = []
        self.results: Dict[UUID, Result] = {}
        self.time_left = 0

    def tick(self, world: "World", cur_time: int) -> None:
        try:
            cur_task = self.tasks[0]
        except IndexError:
            return

        if self.time_left > 0:
            self.time_left -= 1

        if cur_task.heartbeat_tick():
            # Check whether the device who requested it is (still) in range
            for device in world.neighbors_of(self, Device):
                if cur_task.id in device.tasks:
                    assert device.tasks[cur_task.id].id == cur_task.id
                    cur_task.refresh_heartbeat()
                    break
            else:
                self.cancel(cur_task.id)
                # at the time of writing, the rest of this function just ticks the task
                # but we just cancelled the task (which did take up a tick, which we measured)
                # so just early return, whatever
                return

        if cur_task.has_exhausted_budget():
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
            max(
                safe_div(task.cpu_work, self.cpu_power),
                safe_div(task.gpu_work, self.gpu_power),
            )
        )
        return self.time_left + task_time + 1  # 1 extra tick for the "pop"

    def spawn(self, task: Task, is_local: bool) -> None:
        self.tasks.append(task.run(self.id, self.cpu_power, self.gpu_power, is_local))
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

        self.time_left -= running_task.remaining_budget()
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
        self, x: float, y: float, range: float, personal_node: Optional[Node] = None
    ) -> None:
        super().__init__(x, y, range)
        self.tasks: Dict[UUID, Task] = {}
        if personal_node is not None:
            personal_node.x = self.x
            personal_node.y = self.y
        self.personal_node = personal_node
        self.v_ang = 0.0
        self.v_mag = 0.0

    def tick(self, world: "World", cur_time: int) -> None:

        nearby_nodes = world.neighbors_of(self, Node)

        # A tick for a device is one iteration over its tasks (it does more than one tick of work in the tick, we can change it later if we want)
        for task in self.tasks.values():
            got_result = False
            cur_min_eta = min(task.runners.values()) if len(task.runners) > 0 else None

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
                        print(f"Got result in {cur_time - task.time_of_request} ticks")
                        break
                elif node.can_run(task):
                    eta = node.estimate_time(task)
                    if cur_min_eta is None or eta < cur_min_eta:
                        cur_min_eta = eta
                        best_neighbor_for_task = node

            if got_result:
                del self.tasks[task.id]
                if self.personal_node is not None:
                    # ideally we could tell neighbors to also cancel it but whatever
                    self.personal_node.cancel(task.id)
            elif best_neighbor_for_task is not None:
                # it should have already been spawned over in request_task
                assert best_neighbor_for_task is not self.personal_node
                best_neighbor_for_task.spawn(task, False)

    def request_task(
        self, cpu_work: int, gpu_work: int, heartbeat_interval: int, cur_time: int
    ) -> None:
        task = Task(cpu_work, gpu_work, heartbeat_interval, cur_time)
        self.tasks[task.id] = task
        if self.personal_node is not None and self.personal_node.can_run(task):
            self.personal_node.spawn(task, True)

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
        self.clock = 0

    def tick(self) -> None:
        for entity in self.entities:
            entity.tick(self, self.clock)
        self.clock += 1

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

    # def plot(self, ax) -> None:
    #    ax.clear()
    #    for entity in self.entities:
    #        linestyle = "--" if isinstance(entity, Device) else ":"

    #        circle = plt.Circle(
    #            (entity.x, entity.y),
    #            entity.range,
    #            fill=False,
    #            color=entity.color,
    #            linestyle=linestyle,
    #        )
    #        ax.add_patch(circle)
    #        ax.scatter(entity.x, entity.y, c=entity.color)

    def print(self) -> None:
        for e in self.entities:
            if isinstance(e, Node):
                print(f"node @ ({e.x} {e.y}) with {len(e.tasks)} tasks in queue")
            elif isinstance(e, Device):
                print(f"device @ ({e.x} {e.y}) with {len(e.tasks)} tasks active")
            else:
                print(f"entity @ ({e.x} {e.y})")


@dataclass
class Rect:
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass
class TestingProfile:
    gen_space: Rect
    node_count: int
    node_range: Callable[["TestingProfile"], float]
    node_cpu_power: Callable[["TestingProfile"], int]
    node_gpu_power: Callable[["TestingProfile"], int]

    device_count: int
    device_range: Callable[["TestingProfile"], float]
    device_personal_node_rate: float
    device_personal_node_power_multiplier: float

    task_cpu_work: Callable[["TestingProfile"], int]
    task_gpu_work: Callable[["TestingProfile"], int]
    task_heartbeat_interval: Callable[["TestingProfile"], int]

    task_request_chance: float

    movement_direction_variation: Callable[["TestingProfile"], float]
    movement_magnitude_variation: Callable[["TestingProfile"], float]

    duration: int


INITIAL_TESTING_PROFILE = TestingProfile(
    gen_space=Rect(-50, 50, -50, 50),
    node_count=10,
    node_range=lambda *_: random.uniform(1.0, 15.0),
    node_cpu_power=lambda *_: random.randrange(1, 10),
    node_gpu_power=lambda *_: random.randrange(0, 5),
    device_count=5,
    device_range=lambda *_: random.uniform(10.0, 25.0),
    device_personal_node_rate=0.5,
    device_personal_node_power_multiplier=0.25,
    task_cpu_work=lambda *_: random.randrange(100, 1000),
    # 25% of tasks involve a GPU workload
    task_gpu_work=lambda *_: random.randrange(50, 300)
    if random.randrange(0, 4) == 0
    else 0,
    task_heartbeat_interval=lambda *_: 5,
    # every tick, for every device, 10% chance of spawning a task
    task_request_chance=0.1,
    duration=1000,
    movement_direction_variation=lambda *_: random.gauss(0, 0.05),  # radians
    movement_magnitude_variation=lambda *_: random.gauss(0, 0.1),
)


def simulate(p: TestingProfile) -> None:
    w = World()
    for _ in range(p.node_count):
        n = Node(
            x=random.randrange(p.gen_space.x0, p.gen_space.x1),
            y=random.randrange(p.gen_space.y0, p.gen_space.y1),
            cpu_power=p.node_cpu_power(),
            gpu_power=p.node_gpu_power(),
            range=p.node_range(),
        )
        w.add_entity(n)

    for _ in range(p.device_count):
        device_range = p.device_range()

        personal_node = None
        if random.random() < p.device_personal_node_rate:
            personal_node = Node(
                x=random.randrange(p.gen_space.x0, p.gen_space.x1),
                y=random.randrange(p.gen_space.y0, p.gen_space.y1),
                cpu_power=p.device_personal_node_power_multiplier * p.node_cpu_power(),
                gpu_power=p.device_personal_node_power_multiplier * p.node_gpu_power(),
                range=device_range,
            )

        d = Device(
            x=random.randrange(p.gen_space.x0, p.gen_space.x1),
            y=random.randrange(p.gen_space.y0, p.gen_space.y1),
            range=device_range,
            personal_node=personal_node,
        )
        w.add_entity(d)

    for _ in range(p.duration):
        for e in w.entities:
            if isinstance(e, Device):
                e.change_velocity(
                    angle=p.movement_direction_variation(),
                    magnitude=p.movement_magnitude_variation(),
                )
                e.move()
                if random.random() < p.task_request_chance:
                    e.request_task(
                        cpu_work=p.task_cpu_work(),
                        gpu_work=p.task_gpu_work(),
                        heartbeat_interval=p.task_heartbeat_interval(),
                        cur_time=w.clock,
                    )
        w.tick()

    print("Done simulating!")


def main() -> None:
    simulate(INITIAL_TESTING_PROFILE)


if __name__ == "__main__":
    main()
