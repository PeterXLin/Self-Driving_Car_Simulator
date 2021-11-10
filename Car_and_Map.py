import numpy as np
from math import pi
from numpy import cos, sin, arcsin
import random


class Map:
    def __init__(self, path):
        """load the map and store line and point"""
        with open(path) as f:
            # read car start position
            car_init = f.readline().split(',')
            self.car_init_position = (int(car_init[0]), int(car_init[1]))
            self.car_init_degree = int(car_init[2])

            # read destination information
            dest_1 = f.readline().split(',')
            dest_2 = f.readline().split(',')
            self.dest_up_left = (int(dest_1[0]), int(dest_1[1]))
            self.dest_botton_right = (int(dest_2[0]), int(dest_2[1]))

            self.map_point = list()

            for line in f.readlines():
                point = line.split(',')
                self.map_point.append((int(point[0]), int(point[1])))

            # ax + by = k , store [(a, b), k, point1, point2]
            self.border_linear_equations = list()
            for i in range(len(self.map_point) - 1):
                a = self.map_point[i + 1][1] - self.map_point[i][1]
                b = -(self.map_point[i + 1][0] - self.map_point[i][0])
                k = a * self.map_point[i][0] + b * self.map_point[i][1]
                self.border_linear_equations.append((np.array([[a, b]]), k, self.map_point[i], self.map_point[i+1]))

    def draw_map_and_car_start(self, subplot):
        for i in range(len(self.map_point) - 1):
            subplot.plot((self.map_point[i][0], self.map_point[i + 1][0]),
                         (self.map_point[i][1], self.map_point[i + 1][1]),
                         color='black')

        # draw destination
        a, b, c, d = self.dest_up_left[0], self.dest_up_left[1], self.dest_botton_right[0], self.dest_botton_right[1]
        subplot.plot((a, c), (b, b), color='red')
        subplot.plot((c, c), (b, d), color='red')
        subplot.plot((c, a), (d, d), color='red')
        subplot.plot((a, a), (d, b), color='red')

        car_descriptor, head_descriptor = draw_circle(subplot, self.car_init_position, 3, self.car_init_degree)
        return car_descriptor, head_descriptor


class Car:
    def __init__(self, position, head_toward):
        self.x = position[0]
        self.y = position[1]
        self.head_toward = head_toward

    def move(self, turn_degree):
        self.x = self.x + np.cos(degrees_to_radians(self.head_toward+turn_degree)) \
            + sin(degrees_to_radians(turn_degree)) * sin(degrees_to_radians(self.head_toward))

        self.y = self.y + np.sin(degrees_to_radians(self.head_toward+turn_degree)) \
            - sin(degrees_to_radians(turn_degree)) * cos(degrees_to_radians(self.head_toward))
        self.head_toward = self.head_toward - radians_to_degrees(arcsin(2 * sin(degrees_to_radians(turn_degree)) / 6))

    def sensor(self, borders) -> list:
        # detect distance between car and wall
        sensor_linear_equation_parameter = list()
        k_list = list()
        # sin*x + (-cos*y) = k
        sensor_linear_equation_parameter.append(np.array([[sin(degrees_to_radians(self.head_toward + 45)),
                                                - cos(degrees_to_radians(self.head_toward + 45))]]))
        k_list.append(sensor_linear_equation_parameter[0][0][0] * self.x
                      + sensor_linear_equation_parameter[0][0][1] * self.y)

        sensor_linear_equation_parameter.append(np.array([[sin(degrees_to_radians(self.head_toward)),
                                                - cos(degrees_to_radians(self.head_toward))]]))
        k_list.append(sensor_linear_equation_parameter[1][0][0] * self.x
                      + sensor_linear_equation_parameter[1][0][1] * self.y)

        sensor_linear_equation_parameter.append(np.array([[sin(degrees_to_radians(self.head_toward - 45)),
                                                - cos(degrees_to_radians(self.head_toward - 45))]]))
        k_list.append(sensor_linear_equation_parameter[2][0][0] * self.x
                      + sensor_linear_equation_parameter[2][0][1] * self.y)

        # set min distance to infinity
        distance_list = [10000]*3

        # three sensor: front left, front, front right
        for i in range(3):
            sensor_coefficient = sensor_linear_equation_parameter[i]
            sensor_k = k_list[i]

            for linear_function in borders:
                line_coefficient = linear_function[0]
                line_k = linear_function[1]

                # calculate determinant to check if inverse matrix exist
                if sensor_coefficient[0][0] * line_coefficient[0][1] \
                        - sensor_coefficient[0][1] * line_coefficient[0][0] == 0:
                    continue

                try:
                    # use inverse matrix to find intersection
                    inv = np.linalg.inv(np.concatenate((sensor_coefficient, line_coefficient)))
                    k_matrix = np.array([sensor_k, line_k])
                    result = inv.dot(k_matrix)

                    # in front of car
                    if same_sign(result[1] - self.y, sensor_coefficient[0][0]):
                        # intersection on border(line segment)
                        if in_range(result, linear_function[2], linear_function[3]):
                            new_distance = np.sqrt((result[0] - self.x) ** 2 + (result[1] - self.y) ** 2)
                            if new_distance < distance_list[i]:
                                distance_list[i] = new_distance
                except np.linalg.LinAlgError:
                    continue

        return distance_list

    def detect_collision(self, border):
        for linear_function in border:
            point1 = linear_function[2]
            point2 = linear_function[3]
            distance = point_to_segment_distance(self.x, self.y, point1[0], point1[1], point2[0], point2[1])
            if distance < 3:
                return True

    def draw_car(self, car_descriptor, head_descriptor):
        move_circle(car_descriptor, head_descriptor, (self.x, self.y), 3, self.head_toward)

    def arrive(self, top_left_coordinate, bottom_right_coordinate) -> bool:
        """check if the car arrive destination"""
        return in_range((self.x, self.y), top_left_coordinate, bottom_right_coordinate)

    def reset(self):
        """after collision, reset car position"""
        # TODO: change this to fit all map
        self.x = 5 * random.random() - 2.5
        self.y = 14 * random.random()
        self.head_toward = 90


def draw_circle(my_plot, center, radius, head_toward):
    angles_circle = [i*pi/180 for i in range(0, 360)]
    x = center[0] + radius*cos(angles_circle)
    y = center[1] + radius*sin(angles_circle)
    circle_descriptor, = my_plot.plot(x, y, 'r-')
    head_descriptor, = my_plot.plot((center[0], center[0] + radius * cos(degrees_to_radians(head_toward))),
                                    (center[1], center[1] + radius * sin(degrees_to_radians(head_toward))),
                                    color='blue')
    return circle_descriptor, head_descriptor


def move_circle(circle_descriptor, head_descriptor, center, radius, head_toward):
    angles_circle = [i*pi/180 for i in range(0, 360)]
    x = center[0] + radius*cos(angles_circle)
    y = center[1] + radius*sin(angles_circle)
    circle_descriptor.set_xdata(x)
    circle_descriptor.set_ydata(y)

    head_descriptor.set_xdata((center[0], center[0] + radius * cos(degrees_to_radians(head_toward))))
    head_descriptor.set_ydata((center[1], center[1] + radius * sin(degrees_to_radians(head_toward))))


def degrees_to_radians(degree):
    return degree * pi / 180


def radians_to_degrees(radian):
    return radian * 180 / pi


def in_range(intersection, point1, point2):
    x_max = max(point1[0], point2[0])
    x_min = min(point1[0], point2[0])
    y_max = max(point1[1], point2[1])
    y_min = min(point1[1], point2[1])
    if x_min <= intersection[0] <= x_max and y_min <= intersection[1] <= y_max:
        return True
    else:
        return False


def might_collision(center_x, center_y, point1, point2):
    """border is not a line but a line segment, so we need to check if collision might happen"""
    x_max = max(point1[0], point2[0])
    x_min = min(point1[0], point2[0])
    if x_min <= center_x + 3 <= x_max or x_min <= center_x - 3 <= x_max:
        return True
    else:
        return False


def point_to_segment_distance(x, y, x1, y1, x2, y2):
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)

    if cross <= 0:
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if cross >= d2:
        return np.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def same_sign(a, b):
    if a >= 0 and b >= 0:
        return True
    elif a < 0 and b < 0:
        return True
    else:
        return False


if __name__ == "__main__":
    # Map('./data/軌道座標點.txt')
    # print(degrees_to_radians(90))
    pass
