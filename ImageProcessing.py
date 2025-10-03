import cv2 as cv
import numpy as np

# https://www.youtube.com/watch?v=uihBwtPIBxM

def get_edges(image, threshold=0.17):
    Gx_kernel = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ])
    Gy_kernel = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = []
    edge_map = np.zeros(image.shape, np.uint8)
    
    for y in range(1, grayscale_image.shape[1] - 1):
        for x in range(1, grayscale_image.shape[0] - 1):
            matrix = grayscale_image[y - 1:y + 2, x - 1:x + 2] / 255
            Gx = np.multiply(Gx_kernel, matrix).sum()
            Gy = np.multiply(Gy_kernel, matrix).sum()
            magnitude = (Gx * Gx + Gy * Gy) ** 0.5 / 32 ** 0.5
            
            if magnitude > threshold:
                edges.append((y, x))
                edge_map[y, x] = cv.cvtColor(np.array([[[np.arctan2(Gy, Gx), 255, 255]]], np.uint8), cv.COLOR_HSV2BGR)[0][0]

    return edges, edge_map

def get_lines(edges):
    thetas = (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90)
    accumulator = []

    for y, x in edges:
        for theta in thetas:
            r = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
            line = (r, theta)
            lines = [(r_, theta_) for r_, theta_, _ in accumulator]
            
            if line in lines:
                accumulator[lines.index(line)][2] += 1
            else:
                accumulator.append([r, theta, 0])
    
    for _ in sorted(accumulator, key=lambda x:x[2]):
        print(_)

def get_corners(lines):
    pass

path_to_image = "path/to/image"
image = cv.imread(path_to_image)
edges, edge_map = get_edges(image, 0.3)
# lines = get_lines(edges)

cv.imshow("image, edge_map", np.concatenate((image, edge_map), 1))
cv.waitKey(0)
cv.destroyAllWindows()