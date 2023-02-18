import multiprocessing as mp
import time


def f_mp(count, liste):
    l_ = []
    for i_ in range(count):
        l_.append(i_**2)
        l_.pop()
    liste.put(l_)


def f(count):
    l_ = []
    for i_ in range(count):
        l_.append(i_ ** 2)
        l_.pop()
    return l_


if __name__ == "__main__":
    count = 100000
    start = time.time()
    l_ = f(count)
    stop = time.time()
    print('temps en séquentiel : %.2fs' % (stop - start))

    start = time.time()
    procs = 3
    jobs = []
    res = []
    q = mp.Queue()

    for i in range(procs):
        p = mp.Process(target=f_mp, args=(count//procs, q))
        jobs.append(p)
        p.start()
    master = []
    for i in range(procs):
        master += q.get()

    for i in range(procs):
        jobs[i].join()

    stop = time.time()
    print('temps en parallele : %.2fs' % (stop - start))
