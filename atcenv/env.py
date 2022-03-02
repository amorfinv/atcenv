"""
Environment module
"""
import gym
from typing import Dict, List
from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString

# our own packages
import numpy as np

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]

NUMBER_INTRUDERS_STATE = 5
MAX_DISTANCE = 250*u.nm
MAX_BEARING = 2*math.pi

class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 10,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 max_episode_len: Optional[int] = 300,
                 min_distance_horizontal: Optional[float] = 5.,
                 min_distance_vertical: Optional[float] = 1000,
                 distance_init_buffer: Optional[float] = 5.,
                 **kwargs):
        """
        Initialises the environment

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance_horizontal = min_distance_horizontal * u.nm
        self.min_distance_vertical = min_distance_vertical * u.ft
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.dt = dt

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        # altitude of the airspace
        self.alt = kwargs['altitude']

        self.viewer = None
        self.airspace = None
        self.flights = [] # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        self.done = set()  # set of flights that reached the target
        self.i = None

    def resolution(self, action: List) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param action: list of resolution actions assigned to each flight
        :return:
        """

        it2 = 0
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # heading, speed, climb
                new_track = f.track + action[it2][0] * MAX_BEARING/8
                f.track = (new_track + u.circle) % u.circle
                f.airspeed = (action[it2][1]) * (self.max_speed - self.min_speed) + self.min_speed
                # TODO: fix this action climb angle
                f.climb = action[it2][2]
                it2 +=1

        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

    def reward(self) -> List:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """
        # TODO: include a reward for vertical
        weight_a    = -1
        weight_b    = -1/5.
        weight_c    = -1/5.
        weight_d    = -1/5.
        weight_e    = + 1  
        
        conflicts   = self.conflict_penalties() * weight_a
        drifts      = self.drift_penalties() * weight_b
        severities  = self.conflict_severity() * weight_c 
        speed_dif   = self.speedDifference() * weight_d 
        target      = self.reachedTarget() * weight_e # can also try to just ouput negative rewards
        
        tot_reward  = conflicts + drifts + severities + speed_dif + target  

        return tot_reward

    def reachedTarget(self):
        """
        Returns a list with aircraft that just reached the target
        :return: boolean list - 1 if aircraft have reached the target
        """        
        target = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.totaldistance
                if distance < self.tol:
                    target[i] = 1
                    
        return target

    def speedDifference(self):
        """
        Returns a list with the diferent betwee the current aircraft and its optimal speed
        :return: float of the speed difference
        """
        speed_dif = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            speed_dif[i] = abs(f.airspeed - f.optimal_airspeed)
                    
        return speed_dif
        
    def conflict_penalties(self):
        """
        Returns a list with aircraft that are in conflict,
        can be used for multiplication as individual reward
        component
        :return: boolean list for conflicts
        """
        
        conflicts = np.zeros(self.num_flights)
        for i in range(self.num_flights):
            if i not in self.done:
                if i in self.conflicts:
                    conflicts[i] += 1
                    
        return conflicts
    
    def drift_penalties(self):
        """
        Returns a list with the drift angle for all aircraft,
        can be used for multiplication as individual reward
        component
        :return: float of the drift angle
        """
        
        drift = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                drift[i] = abs(f.drift)
        
        return drift

    def vertical_penalties(self):
        ...

    def conflict_severity(self):
        
        severity = np.zeros(self.num_flights)
        for i in range(self.num_flights - 1):
            if i not in self.done:
                if i in self.conflicts:
                    distances = np.array([])
                    for j in list(self.conflicts - {i}):
                        distance = dist_between_flights(self.flights[i], self.flights[j])
                        distances = np.append(distances,distance)
                    #conflict severity on a scale of 0-1
                    severity[i] = -1.*((min(distances)-self.min_distance_horizontal)/self.min_distance_horizontal)
        
        return severity

    def observation(self) -> List:
        """
        Returns the observation of each agent
        :return: observation of each agent
        """
        # observations (size = 2 * NUMBER_INTRUDERS_STATE + 4):
        # distance to closest #NUMBER_INTRUDERS_STATE intruders
        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        # current speed
        # optimal airspeed
        # distance to target
        # bearing to target
        # TODO: decide what to return for 3D airspace

        observations_all = []
        # TODO: currently this is 2d distance
        distance_all = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        bearing_all = np.ones((self.num_flights, self.num_flights))*MAX_BEARING

        for i in range(self.num_flights):
            if i not in self.done:
                for j in range(self.num_flights):
                    if j not in self.done and j != i:
                        # predicted used instead of position, so ownship can work in regard to future position and still
                        # avoid a future conflict
                        # TODO: extend this check for 3D Distance
                        distance_all[i][j] = self.flights[i].prediction.distance(self.flights[j].prediction)
                        # relative bearing
                        dx = self.flights[i].prediction.x - self.flights[j].prediction.x
                        dy = self.flights[i].prediction.y - self.flights[j].prediction.y
                        compass = math.atan2(dx, dy)
                        bearing_all[i][j] = (compass + u.circle) % u.circle

                        # TODO: extend this check for 3D? include a relative climb angle?

        for i, f in enumerate(self.flights):
            if i not in self.done:
                observations = []

                closest_intruders = np.argsort(distance_all[i])[:NUMBER_INTRUDERS_STATE]

                # distance to closest #NUMBER_INTRUDERS_STATE
                observations += np.take(distance_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                # relative bearing #NUMBER_INTRUDERS_STATE
                observations += np.take(bearing_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 2*NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                # current speed
                observations.append(f.airspeed)

                # optimal speed
                observations.append(f.optimal_airspeed)

                # distance to target
                observations.append(f.position.distance(f.target))

                # bearing to target
                observations.append(float(f.drift))

                observations_all.append(observations)
        # RDC: here you should implement your observation function
        ##########################################################
        return observations_all
        ##########################################################

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()
        # TODO:add 3d component
        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance_horizontal = self.flights[i].position.distance(self.flights[j].position)
                        distance_vertical = self.flights[i].position.vdistance(self.flights[j].position)
                        if distance_horizontal < self.min_distance_horizontal and distance_vertical < self.min_distance_vertical:
                            self.conflicts.update((i, j))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.totaldistance
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                dx, dy, dz = f.components

                # get current position
                position = f.position

                # get new position and advance one time step
                f.position._set_coords(position.x + dx * self.dt, position.y + dy * self.dt, position.z + dz * self.dt)

    def step(self, action: List) -> Tuple[List, List, bool, Dict]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """
        # apply resolution actions
        self.resolution(action)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # increase steps counter
        self.i += 1

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done = (self.i == self.max_episode_len) or (len(self.done) == self.num_flights)

        return obs, rew, done, {}

    def reset(self, number_flights_training) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # during training, the number of flights will increase from  1 to 10
        self.num_flights = number_flights_training

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance_horizontal
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol, self.alt)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if candidate.position.distance(f.position) < min_distance:  # all flights start at the same altitude
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = set()

        # return initial observation
        return self.observation()

    def render(self, mode=None) -> None:
        """
        Renders the environment
        :param mode: rendering mode
        :return:
        """
        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 600, 600

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(*BLACK)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(*WHITE)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            if i in self.conflicts:
                color = RED
            else:
                color = BLUE

            circle = rendering.make_circle(radius=self.min_distance_horizontal / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                             f.position.y)))
            circle.set_color(*BLUE)

            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)

        self.viewer.render()

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def dist_between_flights(f1: Flight, f2: Flight) -> float:
    """
    Computes the distance between two flights
    :param f1: first flight
    :param f2: second flight
    :return: distance
    """
    f1_x, f1_y, f1_z = f1.position.x, f1.position.y, f1.position.z
    f2_x, f2_y, f2_z = f2.position.x, f2.position.y, f2.position.z
    return math.sqrt((f1_x - f2_x) ** 2 + (f1_y - f2_y) ** 2 + (f1_z - f2_z) ** 2)