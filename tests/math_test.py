import math
import pytest
class TestMath:
    def test_sqrt(self):
       num = 25
       assert math.sqrt(num) == 5

    def testsquare(self):
       num = 7
       assert 7*7 == 40

    def tesequality(self):
       assert 10 == 11