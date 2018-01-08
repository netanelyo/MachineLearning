import sys

def usage():
    print('[ERROR]\nUsage:')
    print('\tpython ML_ex3.py <module-name> <subsection> [k]')
    print('\tmodule-name - one of {\'knn\', \'perceptron\', \'svm\'}')
    print('\tsubsection  - one of {\'a\', \'b\', \'c\', \'d\', \'e\'}')
    print('\tk           - [OPTIONAL] number of neighbours for knn subsection \'a\'')

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        usage()
        sys.exit(1)

    mod = sys.argv[1]
    sub = sys.argv[2]
    ret = 0
    if mod == 'knn':
        import knn
        ret = knn.main(sub, sys.argv[3:])

    elif mod == 'perceptron':
        import perceptron
        ret = perceptron.main(sub)

    elif mod == 'svm':
        import svm
        ret = svm.main(sub)

    else:
        usage()
        sys.exit(1)

    if ret != 0:
        usage()
    
    sys.exit(ret)