import numpy as np


def __find_scribble_point_with_minimum_distance(
        int: x_coord, 
        int: y_coord,
        ndarray: scribble_coordinates
    ) -> float():
    """"
    __find_scribble_point_with_minimum_distance(
            self, 
            x_coord, 
            y_coord, 
            scribble_coordinates
        ):
        Given a pixel's coordinates and the scribble_coordinates array
        finds the l2 distance to the closest scribble point
    """
    l2_distance = lambda x1, x2, y1, y2: ((x1 - x2)**2 + (y1 - y2)**2)**(1/2) 
    min_distance = float("inf")
    n_scribble_pixels = scribble_coordinates.shape[0] # flat vector, only one element
    for idx in range(0, n_scribble_pixels - 1, 2):
        x_coord_scribble = scribble_coordinates[idx]
        y_coord_scribble = scribble_coordinates[idx + 1]
        # l2 distance
        distance = lp_distance(
            x_coord, 
            x_coord_scribble, 
            y_coord, 
            y_coord_scribble
        )
        if distance < min_distance:
            min_distance = distance
    return min_distance
