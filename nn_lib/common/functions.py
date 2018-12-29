
def translate(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


def line(m, x, b):
    return m*x + b
