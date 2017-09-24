import random
import numpy as np
import multiprocessing as mp
import time

from warnings import warn


def pwr(x):
    return lambda y: y**x


def add(x):
    return lambda y: y + x


def gen(limit):
    for i in xrange(0, limit):
        yield i


def wt(seconds):
    def sleep(x):
        time.sleep(seconds)
        return x
    return sleep


def test(limit):
    r = Reader(gen(limit), name="1-Read")
    t1 = r.apply(wt(0.5), name="2-Wait")
    t2 = t1.apply(add(1), name="3-Add")
    t3 = t2.apply(pwr(2), name="4-Power")
    n2 = t3.run_on(2, name="5-Stitch_2")

    for i in n2:
        print i

    n2.close()


class Node(object):
    """docstring for Node"""

    def __init__(self):
        super(Node, self).__init__()
        self.designation = ""

    def close(self):
        raise NotImplementedError("Please Implement this method")

    # generator methods
    def __iter__(self):
        raise NotImplementedError("Please Implement this method")

    def next(self):
        raise NotImplementedError("Please Implement this method")

    # private methods
    def __str__(self):
        return self._pretty_print(self._print_width())

    def _print_width(self):
        raise NotImplementedError("Please Implement this method")

    def _pretty_print(self, width):
        raise NotImplementedError("Please Implement this method")

    def _print_centered(self, width, str):
        return ' ' * ((width - len(str)) / 2) + str

    def _print_arrow(self, width):
        return self._print_centered(width, '|') + \
            '\n' + self._print_centered(width, '|') + \
            '\n' + self._print_centered(width, 'V')


class Producer(Node):
    """docstring for Producer"""

    def __init__(self):
        super(Producer, self).__init__()

    # public methods
    def transform(self, transformation, name=""):
        return Transformation(self, transformation, name=name)

    def multiply(self, multiplication, name=""):
        return Multiplication(self, multiplication, name=name)

    def combine(self, producer, transformation, name=""):
        return Combination(self, producer, transformation, name=name)

    def aggregate(self, aggregation, name=""):
        return Aggregation(self, aggregation, name=name)

    def run_on(self, threads, buffer=0, num_datapoints=0, name=""):
        return Needleworker(self, threads, buffer=buffer, num_datapoints=num_datapoints, name=name)


class Reader(Producer):
    """docstring for Reader"""

    def __init__(self, source, name=""):
        super(Reader, self).__init__()
        if name != "":
            self.designation = name
        else:
            self.designation = "Reader_%06d" % np.random.randint(1, 1000000)
        self.source = source

    # public methods
    def close(self):
        pass  # nothing to do here

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        try:
            datapoint = self.source.next()
        except StopIteration:
            raise StopIteration
        return datapoint

    # private methods
    def _print_width(self):
        return len(self.designation)

    def _pretty_print(self, width):
        return self._print_centered(width, self.designation)


class Transformation(Producer):
    """docstring for Transformation"""

    def __init__(self, parent, transformation, name=""):
        super(Transformation, self).__init__()
        if name != "":
            self.designation = name
        else:
            self.designation = "Transformation_%06d" % np.random.randint(1, 1000000)
        self.transformation = transformation
        self.parent = parent

    # public methods
    def close(self):
        self.parent.close()

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        try:
            datapoint = self.parent.next()
        except StopIteration:
            raise StopIteration
        return self.transformation(datapoint)

    # private methods
    def _print_width(self):
        return max(self.parent._print_width(), len(self.designation))

    def _pretty_print(self, width):
        return self.parent._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation)


class Multiplication(Producer):
    """docstring for Multiplication"""

    def __init__(self, parent, multiplication, name=""):
        super(Multiplication, self).__init__()
        if name != "":
            self.designation = name
        else:
            self.designation = "Multiplication_%06d" % np.random.randint(1, 1000000)
        self.multiplication = multiplication
        self.parent = parent
        self.generator = None

    # public methods
    def close(self):
        self.parent.close()

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        # Make sure there is a generator
        if self.generator is None:
            try:
                data_in = self.parent.next()
            except StopIteration:
                raise StopIteration
            self.generator = self.multiplication(data_in)

        while True:
            # Use the generator until it runs out
            try:
                data_out = self.generator.next()

            # When it runs out get a new one
            except StopIteration:
                try:
                    data_in = self.parent.next()
                except StopIteration:
                    raise StopIteration
                self.generator = self.multiplication(data_in)
                continue

            # Stop if data_out has been obtained
            break

        return data_out

    # private methods
    def _print_width(self):
        return max(self.parent._print_width(), len(self.designation))

    def _pretty_print(self, width):
        return self.parent._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation)


class Aggregation(Producer):
    """docstring for Aggregation"""

    def __init__(self, parent, aggregator, name=""):
        super(Aggregation, self).__init__()
        if name != "":
            self.designation = name
        else:
            self.designation = "Aggregation_%06d" % np.random.randint(1, 1000000)
        self.aggregator = aggregator
        self.parent = parent

    # public methods
    def close(self):
        self.parent.close()

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        try:
            datapoint = self.aggregator(self.parent)
        except StopIteration:
            raise StopIteration
        return datapoint

    # private methods
    def _print_width(self):
        return max(self.parent._print_width(), len(self.designation))

    def _pretty_print(self, width):
        return self.parent._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation)


class Combination(Producer):
    """docstring for Combination"""

    def __init__(self, father, mother, transformation, name=""):
        super(Combination, self).__init__()
        if name != "":
            self.designation = name
        else:
            self.designation = "Combination_%06d" % np.random.randint(1, 1000000)
        self.transformation = transformation
        self.father = father
        self.mother = mother

    # public methods
    def close(self):
        self.father.close()
        self.mother.close()

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        try:
            left = self.father.next()
            right = self.mother.next()
        except StopIteration:
            raise StopIteration
        return self.transformation(left, right)

    # private methods
    def _print_width(self):
        return max(self.father._print_width(), self.mother._print_width(),
                   len(self.designation))

    def _pretty_print(self, width):
        return self.father._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation + ' ...') + \
            '\n' + \
            '\n' + self.mother._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation)


class Needleworker(Producer):
    """The Needleworker runs multiple parallel threads and stitches their
    output back together."""

    def __init__(self, parent, threads, buffer=0, num_datapoints=0, name=""):

        # Check if code is running in iPython notebook and raise warning
        try:
            from IPython import get_ipython
            cfg = get_ipython().config
            if "IPKernelApp" in cfg.keys():
                warn("Multiprocessing doesn't work with iPython Notebooks")
        except:
            pass

        super(Needleworker, self).__init__()

        if name != "":
            self.designation = name
        else:
            self.designation = "Needleworker_%06d" % np.random.randint(1, 1000000)
        self.parent = parent
        self.queue = mp.Queue(buffer)
        self.threads = [mp.Process(target=self._execute, args=(num_datapoints / threads,))
                        for i in xrange(threads - 1)]
        self.threads.append(mp.Process(target=self._execute, args=((num_datapoints / threads) + (num_datapoints % threads),)))
        self.counter = mp.Value('i', False)
        self.counter.value = threads
        self.mutex = mp.Lock()

        for thread in self.threads:
            thread.start()

    # public methods
    def close(self):
        self.queue.close()
        for thread in self.threads:
            thread.terminate()
            thread.join()
        self.parent.close()

    def running(self):
        self.mutex.acquire()
        value = self.counter.value
        self.mutex.release()
        return value

    # generator methods
    def __iter__(self):
        return self

    def next(self):
        self.mutex.acquire()

        if self.counter.value == 0:
            self.mutex.release()
            raise StopIteration

        datapoint = self.queue.get()

        if datapoint is None:
            self.counter.value -= 1
            self.mutex.release()
            return self.next()

        self.mutex.release()

        return datapoint

    # private methods
    def _execute(self, num_datapoints):

        # make sure every process generates different random numbers
        random.seed()
        np.random.seed()

        if num_datapoints == 0:
            while True:
                try:
                    datapoint = self.parent.next()
                except StopIteration:
                    break
                self.queue.put(datapoint)
        else:
            for _ in xrange(num_datapoints):
                try:
                    datapoint = self.parent.next()
                except StopIteration:
                    break
                self.queue.put(datapoint)

        self.queue.put(None)

    def _print_width(self):
        return max(self.parent._print_width(), len(self.designation) + len(str(len(self.threads))) + 3)

    def _pretty_print(self, width):
        return self.parent._pretty_print(width) + \
            '\n' + self._print_arrow(width) + \
            '\n' + self._print_centered(width, self.designation + ' (' + str(len(self.threads)) + ')')
