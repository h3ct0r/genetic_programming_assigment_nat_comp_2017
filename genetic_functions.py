import math


class OperatorFunction(object):
    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def local_add(a, b):
    return float(a + b)


def local_sub(a, b):
    return float(a - b)


def local_mul(a, b):
    return float(a * b)


def local_div(a, b):
    try:
        return float(a / b)
    except Exception as e:
        return float(1.0)


def local_sqrt(a):
    return float(math.sqrt(abs(a)))


def local_log(a):
    try:
        return float(math.log(a))
    except Exception as e:
        return .0


def local_neg(a):
    return float(a * -1)


def local_inv(a):
    try:
        return float(1.0 / float(a))
    except Exception as e:
        return .0


def local_abs(a):
    return float(abs(a))


def local_max(a, b):
    return float(max(a, b))


def local_min(a, b):
    return float(min(a, b))


def local_sin(a):
    return float(math.sin(a))


def local_cos(a):
    return float(math.cos(a))


def local_tan(a):
    return float(math.tan(a))


class GeneticFunctions(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.complete_functions = [
            {
                'name': 'add',
                'arity': 2,
                'function': OperatorFunction(local_add, 'add', 2)
            },
            {
                'name': 'sub',
                'arity': 2,
                'function': OperatorFunction(local_sub, 'sub', 2)
            },
            {
                'name': 'mul',
                'arity': 2,
                'function': OperatorFunction(local_mul, 'mul', 2)
            },
            {
                'name': 'div',
                'arity': 2,
                'function': OperatorFunction(local_div, 'div', 2)
            },
            {
                'name': 'sqrt',
                'arity': 1,
                'function': OperatorFunction(local_sqrt, 'sqrt', 1)
            },
            {
                'name': 'log',
                'arity': 1,
                'function': OperatorFunction(local_log, 'log', 1)
            },
            {
                'name': 'abs',
                'arity': 1,
                'function': OperatorFunction(local_abs, 'abs', 1)
            },
            {
                'name': 'neg',
                'arity': 1,
                'function': OperatorFunction(local_neg, 'neg', 1)
            },
            {
                'name': 'inv',
                'arity': 1,
                'function': OperatorFunction(local_inv, 'inv', 1)
            },
            {
                'name': 'max',
                'arity': 2,
                'function': OperatorFunction(local_max, 'max', 2)
            },
            {
                'name': 'min',
                'arity': 2,
                'function': OperatorFunction(local_min, 'min', 2)
            },
            {
                'name': 'sin',
                'arity': 1,
                'function': OperatorFunction(local_sin, 'sin', 1)
            },
            {
                'name': 'cos',
                'arity': 1,
                'function': OperatorFunction(local_cos, 'cos', 1)
            },
            {
                'name': 'tan',
                'arity': 1,
                'function': OperatorFunction(local_tan, 'tan', 1)
            }
        ]
        self.functions = []
        self.init_functions()

    def init_functions(self):
        if len(self.cfg.functions) <= 0:
            self.functions = self.complete_functions

        for fname in self.cfg.functions:
            f = self.get_function(fname, use_complete_function_list=True)
            self.functions.append(f)

        print '[INFO]', 'Final function list size: {}'.format(len(self.functions), self.cfg.functions)

    def get_function(self, name, use_complete_function_list=False):
        flist = self.functions
        if use_complete_function_list:
            flist = self.complete_functions

        for e in flist:
            if name == e['name']:
                return e

        raise ValueError('Operator {} is not on operator list'.format(name))