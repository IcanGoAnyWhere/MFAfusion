# class A(object):
#      def set(self, a, b):
#          x = a
#          a = b
#          b = x
#          print (a, b)
#
# a = A()
# c = getattr(a, 'set')
#
# c(a='1', b='2')
#
# a.set(a='1', b='2')
#
# import numpy as np
#
# a = np.array([[1, 2, 3], [4, 5, 6]])
#
# b = a[:, [0, 1]]
#
# print(b)
#
# # c = {'test1': b, 'test2': [1, 2]}
#
# print(c)

car = {
  "brand": "Porsche",
  "model": "911",
  "year": 1963
}

x = car.get('modelo', 'no')

print(x)