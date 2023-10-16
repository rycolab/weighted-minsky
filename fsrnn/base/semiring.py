from collections import defaultdict as dd

import numpy as np


# base code from
# https://github.com/timvieira/hypergraphs/blob/master/hypergraphs/semirings/boolean.py
class Semiring:
    zero: "Semiring"
    one: "Semiring"
    idempotent = False

    def __init__(self, value):
        self.value = value

    @classmethod
    def zeros(cls, N, M):
        import numpy as np

        return np.full((N, M), cls.zero)

    @classmethod
    def chart(cls, default=None):
        if default is None:
            default = cls.zero
        return dd(lambda: default)

    @classmethod
    def diag(cls, N):
        W = cls.zeros(N, N)
        for n in range(N):
            W[n, n] = cls.one

        return W

    @classmethod
    @property
    def is_field(self):
        return False

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Boolean(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return Boolean.one

    def __add__(self, other):
        return Boolean(self.value or other.value)

    def __mul__(self, other):
        if other.value is self.one:
            return self.value
        if self.value is self.one:
            return other.value
        if other.value is self.zero:
            return self.zero
        if self.value is self.zero:
            return self.zero
        return Boolean(other.value and self.value)

    # TODO: is this correct?
    def __invert__(self):
        return Boolean.one

    def __truediv__(self, other):
        return Boolean.one

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)


Boolean.zero = Boolean(False)
Boolean.one = Boolean(True)
Boolean.idempotent = True
# TODO: check
Boolean.cancellative = True


class Real(Semiring):
    def __init__(self, value):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.value = value

    def star(self):
        return Real(1.0 / (1.0 - self.value))

    @classmethod
    @property
    def is_field(self):
        return True

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Real(self.value + other.value)

    def __sub__(self, other):
        return Real(self.value - other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Real(self.value * other.value)

    def __invert__(self):
        return Real(1.0 / self.value)

    def __pow__(self, other):
        return Real(self.value**other)

    def __truediv__(self, other):
        return Real(self.value / other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        # return f'Real({self.value})'
        return f"{round(self.value, 3)}"

    def __eq__(self, other):
        # return float(self.value) == float(other.value)
        return np.allclose(float(self.value), float(other.value), atol=1e-6)

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.value)


Real.zero = Real(0.0)
Real.one = Real(1.0)
Real.idempotent = False
Real.cancellative = True
