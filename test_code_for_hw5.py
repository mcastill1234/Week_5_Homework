from unittest import TestCase
from code_for_hw5 import *


class Test(TestCase):
    """Unit testing for code_for_hw5"""

    def setUp(self):
        self.X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
        self.Y = np.array([[1., 2.2, 2.8, 4.1]])
        self.th = np.array([[1.], [0.05]])
        self.th0 = np.array([[2.]])

    def test_lin_reg(self):
        th0 = np.array([[0.]])
        ans1 = [[1.05, 2.05, 3.05, 4.05]]
        self.assertEqual(lin_reg(self.X, self.th, th0).tolist(), ans1)
        ans2 = [[3.05, 4.05, 5.05, 6.05]]
        self.assertEqual(lin_reg(self.X, self.th, self.th0).tolist(), ans2)

    def test_square_loss(self):
        ans = [[4.2025, 3.4224999999999985, 5.0625, 3.8025000000000007]]
        self.assertEqual(square_loss(self.X, self.Y, self.th, self.th0).tolist(), ans)

    def test_mean_square_loss(self):
        ans = [[4.1225]]
        self.assertEqual(mean_square_loss(self.X, self.Y, self.th, self.th0).tolist(), ans)

    def test_ridge_obj(self):
        ans1 = [[4.1225]]
        self.assertEqual(ridge_obj(self.X, self.Y, self.th, self.th0, 0.0).tolist(), ans1)
        ans2 = [[4.623749999999999]]
        self.assertEqual(ridge_obj(self.X, self.Y, self.th, self.th0, 0.5).tolist(), ans2)
        ans3 = [[104.37250000000002]]
        self.assertEqual(ridge_obj(self.X, self.Y, self.th, self.th0, 100.).tolist(), ans3)

    def test_d_lin_reg_th(self):
        ans1 = [[1.0], [1.0]]
        self.assertEqual(d_lin_reg_th(self.X[:, 0:1], self.th, self.th0).tolist(), ans1)
        ans2 = [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]
        self.assertEqual(d_lin_reg_th(self.X, self.th, self.th0).tolist(), ans2)

    def test_d_square_loss_th(self):
        ans1 = [[4.1], [4.1]]
        ans2 = [[4.1, 7.399999999999999, 13.5, 15.600000000000001], [4.1, 3.6999999999999993, 4.5, 3.9000000000000004]]
        self.assertEqual(d_square_loss_th(self.X[:, 0:1], self.Y[:, 0:1], self.th, self.th0).tolist(), ans1)
        self.assertEqual(d_square_loss_th(self.X, self.Y, self.th, self.th0).tolist(), ans2)

    def test_d_mean_square_loss_th(self):
        ans1 = [[4.1], [4.1]]
        ans2 = [[10.15], [4.05]]
        self.assertEqual(d_mean_square_loss_th(self.X[:, 0:1], self.Y[:, 0:1], self.th, self.th0).tolist(), ans1)
        self.assertEqual(d_mean_square_loss_th(self.X, self.Y, self.th, self.th0).tolist(), ans2)

    def test_d_lin_reg_th0(self):
        ans = [[1.0, 1.0, 1.0, 1.0]]
        self.assertEqual(d_lin_reg_th0(self.X, self.th, self.th0).tolist(), ans)

    def test_d_square_loss_th0(self):
        ans = [[4.1, 3.6999999999999993, 4.5, 3.9000000000000004]]
        self.assertEqual(d_square_loss_th0(self.X, self.Y, self.th, self.th0).tolist(), ans)

    def test_d_mean_square_loss_th0(self):
        ans = [[4.05]]
        self.assertEqual(d_mean_square_loss_th0(self.X, self.Y, self.th, self.th0).tolist(), ans)