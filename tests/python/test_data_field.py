import unittest

import matplotlib.pyplot as plt
from spdm.util.numlib import np
from spdm.data.Coordinates import Coordinates
from spdm.data.Field import Field
from spdm.util.logger import logger


class TestField(unittest.TestCase):

    def test_create_1d(self):
        x = np.linspace(0.0, 10, 128)
        y = np.sin(x)  # + 0.1*np.random.random(128)-0.05
        coord = Coordinates(x, unit="m", name="x")
        f = Field(y, coordinates=coord)
        fig = plt.figure()
        plt.plot(x, np.sin(x))
        plt.plot(x, f(), "+")
        plt.plot(x, f(x), "-")
        plt.plot(x, f.derivative(), "o-")
        fig.savefig("/home/salmon/workspace/output/test_1d.png")

    def test_create_2d(self):
        x = np.linspace(0.0, 10, 128)
        y = np.linspace(0.0, 10, 128)

        coord = Coordinates(x, y, unit="m", name="r,z")

        X, Y = np.meshgrid(x, y, indexing="ij")

        z = np.sin(X) * np.sin(Y)  # + 0.1*np.random.random([128, 128])-0.05

        f = Field(z, coordinates=coord)

        fig = plt.figure()

        # plt.contourf(X, Y,  np.sin(X) * np.sin(Y))
        plt.contour(X, Y, f(X, Y))
        plt.contour(X, Y, f.derivative(x, y, dx=1, dy=1))
        print(f(1.0, 2.5))

        fig.savefig("/home/salmon/workspace/output/test_2d.png")


if __name__ == '__main__':
    unittest.main()
