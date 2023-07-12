from rest_framework.test import APITestCase

# from django.test import TestCase
# from menu_app.logic import operation
# class LogicTestCase(TestCase):
#     def test_multiply(self):
#         result = operation(6, 4, "*")
#         self.assertEquals(24, result)
#
#     def test_minus(self):
#         result = operation(6, 3, '-')
#         self.assertEquals(3, result)
#     def test_plus(self):
#         result = operation(5, 3, '+')
#         self.assertEquals(9, result)

class PizzaAPIViewTestCase(APITestCase):
    def test_get(self):
        p

        url = 'api/v1/pizzalist'