#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehaiyang
@contact: 1272114024@qq.com
@file: test01.py
@time: 2019/1/11 14:25
'''

"""
函数说明：梯度上升法求解函数的近似根

"""
def Gradient_ascent_test():
    def f_prime(x_old):             #f(x) = x^-2 + 4x函数的导数公式
        return -2 * x_old + 4
    x_old = -1
    x_new = 0
    alpha = 0.01
    presision = 0.000001
    while abs(x_new - x_old) > presision:
        print(x_new)
        x_old = x_new
        #利用梯度上升算法求极值
        x_new = x_new + alpha * f_prime(x_old)
    print(x_new)

if __name__=='__main__':
    Gradient_ascent_test()