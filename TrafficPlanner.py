#!/usr/bin/env python3

"""
TrafficPlanner node is used for the distribution of traffic over the factory
and provides information for ETA. All path request to the path planner
trigger the creation of random points by TrafficPlanner. This are then used
to compute a path and at the same time influence the random distribution
which is used for the creation of random points. With this approach the traffic
in the factory shall be distributed equally, avoiding too much congestion.
"""

import rospy
import networkx as nx
from auto_smart_factory.msg import RoadmapGraph
from auto_smart_factory.srv import *
import random
from collections import OrderedDict
import numpy as np
from itertools import chain
from functools import partial
from collections import deque
from math import sqrt


class TrafficPlanner:
    """Holds a graph which models the traffic in the factory for several time steps.
    Since the roadmap graph is likely to have many nodes, we create boxes which are
    scattered over the graph. The probability distribution updates are carried out
    always for a whole box, if a random point from this box is chosen. For the
    computation of sensible random points, a certain number of random points is
    chosen from the roadmap graph, based on an initially uniform distribution.
    Then, a complete graph is created out of these points and Dijsktra is applied
    to this graph in order to find a shortest path. The nodes which lie on this
    shortest path are then returned as random points."""

    def __init__(self):
        self.is_ready = False

    def initGraph(self, data):
        """Build a new graph after receiving a new RoadmapGraph message from RoadmapGenerator.

        Args:
            data: Contains the data which is received by the subscriber to the topic roadmap_graph.

        Returns:

        """
        self.is_ready = False
        assert len(data.coordinates) == data.num_nodes
        assert len(data.start_nodes) == len(data.end_nodes)

        #############################
        # Parameters
        #############################
        # number of time zones is number_of_time_steps - max_offset,
        # i.e. for every start time <= max_offset the
        # same number of time zones is available
        self.number_of_time_steps = 15   # must be in int16
        self.max_offset = 5
        self.avg_speed = 1.0  # about realistic according to json
        self.number_of_random_points = 40
        # multiplicator to verify number of nodes of random graph >> path of random nodes
        self.random_node_density = 10
        # split the grid graph into boxes to have less sections for the traffic density update
        # box length is the number of nodes that a square box has in one direction,
        # i.e. one box conatains at most box_length**2 nodes
        self.box_length = 1
        ############################

        self.height = data.height
        self.width = data.width
        self.number_of_nodes = data.num_nodes
        self.resolution = data.resolution

        self.coord_list = [(p.x, p.y) for p in data.coordinates]
        self.int_coords_dict = {(x, y): i for i, (x, y) in
                                enumerate(zip(data.x_coord_int, data.y_coord_int))}
        self.int_coords_list = list(zip(data.x_coord_int, data.y_coord_int))
        self.boxes_to_nodes = {}
        for n in range(self.number_of_nodes):
            try:
                self.boxes_to_nodes[self._node_to_box(n)].append(n)
            except KeyError:
                self.boxes_to_nodes[self._node_to_box(n)] = [n]
        # we need fixed int names for the boxes
        self.list_of_boxes = list(self.boxes_to_nodes.keys())
        self.box_coords_to_index = {el: i for i, el in enumerate(self.list_of_boxes)}
        self.number_of_boxes = len(self.boxes_to_nodes)
        assert self.number_of_boxes > self.number_of_random_points  # should be >>


        # create graph that contains only the unit (vertical/horizontal) edges for time zone creation
        self.graph_reduced = nx.Graph()
        self.graph_reduced.add_nodes_from(zip(range(data.num_nodes),
                                      ({"coordinates": self.coord_list[i]} for i in range(data.num_nodes))))
        unit_edges = ((x, y) for i, (x, y, gr) in
                      enumerate(zip(data.start_nodes, data.end_nodes, data.is_grid_edge)) if gr)
        self.graph_reduced.add_edges_from(unit_edges)


        # in the beginning all nodes have equal density probabilities
        self.densities = np.empty([self.number_of_time_steps, self.number_of_boxes], dtype=np.float64)
        self.densities.fill(1 / self.number_of_boxes)

        # create an OrderedDict that maps coordinates to nodes, uses lexical ordering
        self.coord_dict = OrderedDict.fromkeys(sorted(self.coord_list))
        for node, coord in zip(range(self.number_of_nodes), self.coord_list):
            self.coord_dict[coord] = node

        # size_of_timezone is distance measured in nodes
        self.size_of_timezone = max((self.height, self.width)) / (self.resolution
                                    * (self.number_of_time_steps - self.max_offset))

        rospy.loginfo("Initialized TrafficPlanner with new graph.")
        self.is_ready = True


    def _node_to_box(self, node, asIndex=False):
        """Return the correspondig box to a node given by node name.

        Args:
            node:       int, node name
            asIndex:    bool, indicates if index of box should be returned rather
                        than tuple of coordinates. Must only be true after
                        self.box_coords_to_index is initialized.

        Returns:
            tuple (int, int) integer coordinates of the box corresponding to
            node if asIndex == False, int index of the box otherwise.
        """

        x, y = self.int_coords_list[node]
        if not asIndex:
            return x // self.box_length, y // self.box_length
        else:
            return self.box_coords_to_index[x // self.box_length, y // self.box_length]


    def _bfs_dist(self, graph, source):
        """Perform a breadth first search and compute the distances from the source node to all other nodes.

        Args:
            graph: instance of networkx.Graph
            source: a node of graph

        Returns:
            dict of distances from source for all nodes, measured in number of edges
        """

        visited = set([source])
        queue = deque([(source, graph.neighbors(source))])

        distances = {source: 0}
        while queue:
            parent, children = queue[0]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    queue.append((child, graph.neighbors(child)))
                    distances[child] = distances[parent] + 1
            except StopIteration:
                queue.popleft()
        return distances


    def getClosestNode(self, p):
        """Returns a node which is close to the coordinates given by argument p.

        Args:
            geometry_msgs/Point p: coordinates

        Returns:
             int: name of node that is close to (x,y)
        """

        int_x, int_y = int(p.x / self.resolution), int(p.y / self.resolution)
        if not (int_x, int_y) in self.int_coords_dict:
            for j in range(15):
                if (int_x + j, int_y + j) in self.int_coords_dict:
                    return self.int_coords_dict[int_x + j, int_y + j]
                if (int_x - j, int_y - j) in self.int_coords_dict:
                    return self.int_coords_dict[int_x - j, int_y - j]
                if (int_x + j, int_y - j) in self.int_coords_dict:
                    return self.int_coords_dict[int_x + j, int_y - j]
                if (int_x - j, int_y + j) in self.int_coords_dict:
                    return self.int_coords_dict[int_x - j, int_y + j]
            raise ValueError

        return self.int_coords_dict[int_x, int_y]



    def updateDistribution(self, update_nodes, time_steps, positive=True):
        """
        Makes an update of self.densities, to increase / decrease the
        probability for nodes to be selected as random points. If positive,
        then the traffic densities increase for the corresponding nodes
        and decrease otherwise.

        Args:
            update_nodes:   list / set of node names whose distribution
                                values shall be changed
            positive:       bool that indicates whether density for
                                update_nodes is increased or decreased

        Returns:
            None
            updates self.densities
        """

        # define constant c in ]0, min(len(update_nodes), self.number_of_nodes - len(update_nodes)[
        # which sets the intensity of the update

        c = 1
        mini = np.min(self.densities)
        assert update_nodes
        assert mini >= 0

        sign = 1 if positive else -1
        diff_update_nodes = sign * mini * c / len(update_nodes)
        diff_other_nodes = - sign * mini * c / (self.number_of_boxes - len(update_nodes))

        diff_array = np.empty(self.number_of_boxes)
        diff_array.fill(diff_other_nodes)
        for i in time_steps:
            self.densities[i] += diff_array
            for node in update_nodes:
                self.densities[i][self._node_to_box(node, True)] += diff_update_nodes - diff_other_nodes


    def getRandomPoints(self, req):
        """
        Args:
            req: Contains the arguments for the request as received by the service.

        Returns:
            getRandomPointsResponse

        """

        # compute number of random points, return only start and end nodes for short distances
        euclidean_dist = sqrt((req.start_coordinates.x - req.end_coordinates.x)**2
                              + (req.start_coordinates.y - req.end_coordinates.y)**2)
        if euclidean_dist <= 1.5 * self.size_of_timezone * self.resolution:
            res = getRandomPointsResponse()
            res.nodes = [self.getClosestNode(req.start_coordinates), self.getClosestNode(req.end_coordinates)]
            res.length = euclidean_dist
            res.traffic_density = 1
            return res

        duration_of_timezone = self.avg_speed * self.resolution * self.size_of_timezone
        offset = int((req.start_time.to_sec() - rospy.get_time()) // duration_of_timezone)
        if offset < 0:
            offset = 0
        if offset > self.max_offset:
            offset = self.max_offset

        # compute a distribution on the traffic densities considering the time steps
        s_node = self.getClosestNode(req.start_coordinates)
        t_node = self.getClosestNode(req.end_coordinates)
        distribution_over_time = np.zeros(self.number_of_boxes, dtype=np.float64)
        time_steps_assigned = {i: set() for i in range(self.number_of_time_steps)}

        for node, dist in self._bfs_dist(self.graph_reduced, s_node).items():
            # we multiply by 2 because distance is x+y direction, size_of_timezone just one direction
            step = int(dist // (2 * self.size_of_timezone)) + offset
            distribution_over_time[self._node_to_box(node, True)] = \
                self.densities[step, self._node_to_box(node, True)]
            time_steps_assigned[step].add(node)

        # norming to a distribution, maybe try softmax as an alternative
        sum_of_densities = np.sum(distribution_over_time)
        distribution_over_time = distribution_over_time / sum_of_densities

        nodes_from_boxes = np.array([random.choice(self.boxes_to_nodes[box]) for box in self.list_of_boxes])

        random_graph = nx.complete_graph(chain((int(_) for _ in np.nditer(
                                                    np.random.choice(nodes_from_boxes,
                                                    size=self.number_of_random_points,
                                                    replace=False,
                                                    p=distribution_over_time))),
                                                    (s_node, t_node)))

        # use Euclidean distance as heuristic, should be evaluated
        dist_dict = {(a, b): sqrt(sum((x-y)**2 for x, y in zip(self.coord_list[a], self.coord_list[b])))
                     for a, b in random_graph.edges}
        mean_distance = sum(dist_dict.values()) / len(dist_dict)
        try:
            del dist_dict[s_node, t_node]
        except KeyError:
            pass
        try:
            del dist_dict[t_node, s_node]
        except KeyError:
            pass
        # we want to make sure that in the random_graphh there is a path from s to t
        dist_dict[s_node, t_node] = 1.5 * (self.height + self.width)
        nx.set_edge_attributes(random_graph, dist_dict, "weight")
        try:
            del dist_dict[s_node, t_node]
        except KeyError:
            pass

        # remove long edges
        rem_edges = list(edge for edge, length in dist_dict.items() if length > 0.6 * mean_distance)
        random_graph.remove_edges_from(rem_edges)


        length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(random_graph, s_node, t_node)
        path_set = set(path)

        if req.path_is_real:
            for step, nodes in time_steps_assigned.items():
                to_update = path_set.intersection(nodes)
                if to_update:
                    self.updateDistribution(to_update, [step])

        rospy.loginfo(str([str(p) for p in path]) + " will be returned as random points")
        res = getRandomPointsResponse()
        res.nodes = path
        res.length = length
        res.traffic_density = sum_of_densities * sum(distribution_over_time[self._node_to_box(node, True)] for node in path)

        return res


def randomPointService(data, traffic_planner):
    """Waits until traffic planner is ready to provide getRandomPointService."""

    if traffic_planner.is_ready:
        return traffic_planner.getRandomPoints(data)
    else:
        r = rospy.Rate(20)
        while not traffic_planner.is_ready:
            r.sleep()
        return traffic_planner.getRandomPoints(data)



if __name__ == "__main__":
    tp = TrafficPlanner()
    rospy.init_node("traffic_planner")
    rospy.Subscriber("roadmap_graph", RoadmapGraph, tp.initGraph)
    handleRandomPointService = partial(randomPointService, traffic_planner=tp)
    rospy.Service("traffic_planner/get_random_points", getRandomPoints, handleRandomPointService)
    rospy.spin()
