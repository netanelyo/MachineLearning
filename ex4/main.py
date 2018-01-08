import data
import network


def train_network(network_layers=[784, 40, 10], train_set_size=10000, test_set_size=5000,
                  b=False, d=False, batch_size=10):
    training_data, test_data = data.load(train_size=train_set_size,test_size=test_set_size)
    net = network.Network(network_layers)
    net.SGD(training_data, epochs=30, mini_batch_size=batch_size, learning_rate=0.1,
            test_data=test_data, is_subsection_b=b, is_subsection_d=d)


def q_a():
    train_network()


def q_b():
    train_network(b=True)


def q_c():
    train_network(train_set_size=50000, test_set_size=10000)


def q_d():
    train_network([784, 30, 30, 30, 30, 10], d=True, batch_size=10000)


def main(subsection):
    if subsection == 'a':
        q_a()

    elif subsection == 'b':
        q_b()

    elif subsection == 'c':
        q_c()

    elif subsection == 'd':
        q_d()

    else:
        print('[ERROR]\nbackprop:')
        print('No subsection \'' + subsection + '\' in this module')
        return 1

    return 0