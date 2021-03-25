import numpy as np
from opfunu.cec.cec2005.F1 import Model as Func1
from opfunu.cec.cec2005.F2 import Model as Func2
from opfunu.cec.cec2005.F3 import Model as Func3
from opfunu.cec.cec2005.F4 import Model as Func4
from opfunu.cec.cec2005.F5 import Model as Func5
from opfunu.cec.cec2005.F9 import Model as Func9
from opfunu.cec.cec2005.F10 import Model as Func10

POP_SIZE = 100
F = 0.5
CR = 0.9
ITERATIONS = 100
RANGE = [(-100, 100)] * 30
RANGE5 = [(-5, 5)] * 30
MAX_EVAL = 10000

EXPERIMENTS = 25

f = open('DE_sur_f.txt', 'a')


def de(object_function_object, bounds, acc, curve, mut=F, cross_p=CR, pop_size=POP_SIZE, iterations=ITERATIONS):
    eval = 0
    current = 0
    reach = False
    dimensions = len(bounds)

    # initialize population:
    # population normalized into [0, 1]
    pop = np.random.rand(pop_size, dimensions)

    # population denormalized into real bounds
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denormalized = min_b + pop * diff

    # evaluate initial population using real values
    fitness = np.asarray([object_function_object._main__(vector) for vector in pop_denormalized])
    best_index = np.argmin(fitness)
    best_vector = pop_denormalized[best_index]

    i = 0
    while True:
        # print('= Iteration', i+1, '=')

        # x_i
        for j in range(pop_size):
            # mutate:
            # rand/1/bin
            indexs = [idx for idx in range(pop_size) if idx != j]
            # x_r1, x_r2 and x_r3
            a, b, c = pop[np.random.choice(indexs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)

            # cross:
            # binomial
            cross_points = np.random.rand(dimensions) < cross_p
            if not np.any(cross_points):
                # j_rand
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])

            # trials denormalized into real bounds
            trial_denormalized = min_b + trial * diff
            # evaluate trial using real value
            function_value = object_function_object._main__(trial_denormalized)
            if function_value < fitness[j]:
                # 实际应该选子代
                if np.random.rand() < acc:
                    # 选对: 选子代
                    eval += 1
                    # trial selected, population updated
                    fitness[j] = function_value
                    pop[j] = trial
                    # record best
                    if function_value < fitness[best_index]:
                        best_index = j
                        best_vector = trial_denormalized
                else:
                    # 选错: 选父代
                    pass
            else:
                # 实际应该选父代
                if np.random.rand() < acc:
                    # 选对: 选父代
                    pass
                else:
                    # 选错: 选子代
                    eval += 1

            if eval % 100 == 0 and eval != 0:
                if current != eval:
                    print('eval:', eval)
                    print(abs(fitness[best_index] - object_function_object.f_bias))
                    curve[eval//100-1] += abs(fitness[best_index] - object_function_object.f_bias)
                    current = eval
                    if eval == MAX_EVAL:
                        reach = True
                        break

        if reach:
            break

        i += 1

    return best_vector, fitness[best_index]


def get_output(func, ran):
    for acc in [1]:
        print('========== Accuracy', acc, '==========')
        f.write(str(acc))
        f.write('==================================================\n')

        curve = np.zeros([ITERATIONS])
        results = np.zeros([EXPERIMENTS])

        for exp in range(EXPERIMENTS):
            print('===== Experiment', exp + 1, '=====')

            result = de(func, ran, acc, curve)
            error = abs(result[1] - func.f_bias)
            print(error)
            results[exp] = error

            f.write(str(error))
            f.write(' ')
        f.write('\n')

        # 25次的结果
        f.write('mean: ')
        f.write(np.mean(results).astype(str)[:])
        f.write('\n')
        f.write('std: ')
        f.write(np.std(results).astype(str)[:])
        f.write('\n\n')

        curve /= EXPERIMENTS
        curve = curve.astype(str)
        for iteration in range(ITERATIONS):
            f.write(curve[iteration])
            f.write(' ')
        f.write('\n\n')

    print('\n')
    f.write('\n')


if __name__ == '__main__':
    print('-------------------------------------------------F1-------------------------------------------------')
    f.write('-------------------------------------------------F1-------------------------------------------------\n')
    get_output(Func1(), RANGE)

    print('-------------------------------------------------F2-------------------------------------------------')
    f.write('-------------------------------------------------F2-------------------------------------------------\n')
    get_output(Func2(), RANGE)

    print('-------------------------------------------------F3-------------------------------------------------')
    f.write('-------------------------------------------------F3-------------------------------------------------\n')
    get_output(Func3(), RANGE)

    print('-------------------------------------------------F4-------------------------------------------------')
    f.write('-------------------------------------------------F4-------------------------------------------------\n')
    get_output(Func4(), RANGE)

    print('-------------------------------------------------F5-------------------------------------------------')
    f.write('-------------------------------------------------F5-------------------------------------------------\n')
    get_output(Func5(), RANGE)

    print('-------------------------------------------------F9-------------------------------------------------')
    f.write('-------------------------------------------------F9-------------------------------------------------\n')
    get_output(Func9(), RANGE5)

    print('-------------------------------------------------F10-------------------------------------------------')
    f.write('-------------------------------------------------F10-------------------------------------------------\n')
    get_output(Func10(), RANGE5)

    f.close()
