import sys


def usage():
    print('[ERROR]\nUsage:')
    print('\tpython ML_ex4.py <module-name> <subsection>')
    print('\tmodule-name - one of {\'sgd\', \'backprop\', \'tf\'}')
    print('\tsubsection  - one of {\'a\', \'b\', \'c\', \'d\'}')


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        usage()
        sys.exit(1)

    mod = sys.argv[1]
    sub = sys.argv[2]
    ret = 0
    if mod == 'sgd':
        import sgd

        ret = sgd.main(sub)

    elif mod == 'backprop':
        import main

        ret = main.main(sub)

    elif mod == 'tf':
        if sub == 'a':
            import tf_mnist

        elif sub == 'b':
            import tf_mnist_b

        elif sub == 'c':
            import tf_mnist_c

        elif sub == 'd':
            import tf_mnist_d

        else:
            print('[ERROR]\ntf:')
            print('No subsection \'' + sub + '\' in this module')
            sys.exit(1)

    else:
        usage()
        sys.exit(1)

    if ret != 0:
        usage()

    sys.exit(ret)