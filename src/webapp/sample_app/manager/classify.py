# -*- coding: utf-8 -*-
import warnings
# warnings.filterwarnings('ignore')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import np
import time
import datetime
import random
import np
import pandas as pd
import multiprocessing as mul
import umap
import csv
import pandas as pd
from scipy.sparse.csgraph import connected_components
from notification import Notify
from scipy.special import expit
from multiprocessing import pool
from pprint import pprint
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer, compile
from numpy import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.tools.visualization import matplotlib_circuit_drawer
from numba import jit
from matplotlib.colors import ListedColormap as clp


class QVC:

    def __init__(self, qubits, cbits, target, shots, l_iteration, dimension, n_class):
        '''
        This is initial config.
        qubits, cbits: the instance of qubits, classical bits
        qc: the name of circuit
        num_q, num_c: the number of qubits, cbits
        train, test: the directory of training data, test data
        '''
        self.q = QuantumRegister(qubits)
        self.c = ClassicalRegister(cbits)
        self.qc = QuantumCircuit(self.q, self.c)
        self.num_q = qubits
        self.num_c = cbits
        self.target = target
        self.shots = shots
        self.l_iter = l_iteration
        self.dim = dimension
        self.n_class = n_class

    def _reduce_dimension(self):
        pass

    def _feature_map(self, qc, S, data_angle):
        '''
        Quantum State Mapping
        apply feature map circuit(fig1 b)
        using two qubits for making reading feature map
        1.applying feature map circuit to 0state. <-- in this part
        2.training theta
        3.measurement
        4.fitting cost function
        qc : circuit name
        S : the number of test data set
        x_angle : the angle for fitting traingin data sets
        ### In this paper, x_angle is artifitially generated.
        '''
        q = self.q
        n = self.num_q
        # TODO understand how to decide theta of u1
        for i in range(n):
            qc.h(q[i])
        qc.u1(data_angle[0], q[0])
        for j in range(S-1):
            qc.cx(q[j], q[j+1])
            qc.u1(data_angle[j], q[j+1])
            qc.cx(q[j], q[j+1])
        qc.u1(data_angle[S-1], q[S-1])

        for i in range(n):
            qc.h(q[i])
        qc.u1(data_angle[0], q[0])
        for j in range(S-1):
            qc.cx(q[j], q[j+1])
            qc.u1(data_angle[j], q[j+1])
            qc.cx(q[j], q[j+1])
        qc.u1(data_angle[S-1], q[S-1])

        return qc

    def _w_circuit(self, qc, theta_list):
        '''
        repeat this circuit for l times to classify
        qc: The name of quantum circuit
        theta : the angle for u3gate
        '''
        q = self.q
        n = self.num_q
        # TODO how to tune the theta_list
        # for ini in range(n):
        #     qc.u3(theta_list[ini], 0, theta_list[ini], q[ini]) # FIXME This part is wrong? should be one time but this part is apply many times.
        for iter in range(self.l_iter):
            for j in range(1, n):
                qc.cz(q[j-1], q[j])
            qc.cz(q[n-1], q[0])
        # TODO how to decide lambda of u3?
            for m in range(n):
                qc.u3(0, 0, theta_list[m], q[m])
        return qc

    def _R_emp(self, distribution, y, bias):
        '''
        this is cost fucntion for optimize theta(lambda)
        theta: lambda for u3 gate
        '''
        a_1 = (np.sqrt(self.shots)*(1/2-(distribution-y*bias/2)))
        a_2 = np.sqrt(abs(2*(1-distribution)*distribution))
        sig = expit(a_1/a_2)  # expit is sigmoid function
        print(sig)
        # FIXME 1/T should multiplied by the sum of emps?
        return 1/self.dim * sig

    def _multi_emp_cost(self, count, correct_class):
        binlabel = self._label2binary(correct_class)
        print(max(count, key=count.get))
        n_c = count[binlabel]
        oth_dict = count.pop(binlabel)
        at = (np.sqrt(self.shots)*max(count.values())-n_c)
        bt = np.sqrt(2*(self.shots-n_c)*n_c)
        return expit(at/bt)

    # @jit
    def _multic_cost(self, val_list, correct_class):
        # print(val_list)
        n_c = val_list[correct_class]
        _ = val_list.pop(correct_class)
        at = (np.sqrt(self.shots)*max(val_list)-n_c)
        bt = np.sqrt(2*(self.shots-n_c)*n_c)
        return expit(at/bt)

    def _label2binary(self, correct_class):
        '''
        maybe no need this function.
        input: class label ex 3
        -------------------------------------
        output: binary(String) ex.'0000000100'
        correct class -> binary label boolean
        now:10qubit # FIXME fir for any qubits.
        '''
        if correct_class == 0:
            return '0'*self.dim
        else:
            return '0'*(self.dim-correct_class)+'1'+'0'*(correct_class-1)

    def _aggregate(self, count, labels):
        '''
        input:count
        output:list(aggregate by number)
        '''
        values = []
        for k in labels:
            rc = 0
            for i, j in zip(count.keys(), count.values()):
                if list(i)[self.dim-1-k] == '1': # FIXME wrong?
                    rc += (j/self.shots)
            values.append(rc)
            
        return values

    def fit(self, x_data, y_data, labels):
        ''' training and fitting parameter
        1.applying feature map circuit to 0state.
        2.training theta <-- in this part
        3.measurement
        4.fitting cost function
        '''
        initial_theta = [0.01]*self.num_q
        b = list(np.arange(-1, 1, 0.1))
        x_data = zip(*[iter(x_data)]*3)
        y_data = zip(*[iter(y_data)]*3)
        while True:
            count = 0
            params = []
            emp_cost = [99, 99]
            theta_l = [initial_theta, initial_theta]
            for training_data, t_label in zip(x_data, y_data):  # like(1, 3, 4)
                fit_theta = self._fitting_theta(theta_l, emp_cost, count)
                # print("fit!", fit_theta)
                count_results = self._circuit(fit_theta, list(training_data))  # array
                # print(theta_l)
                theta_l.append(fit_theta)
                bias = random.choice(b)
                # print(t_label)
                for i, j in zip(count_results, t_label):
                    count_vals = self._aggregate(i, labels)
                    empirical = self._multic_cost(count_vals, list(t_label).index(j))
                    emp_cost.append(empirical)
                # print(emp_cost)
                count += 1
                print("="*25, count, "="*25)
            if self.isOptimized(min(emp_cost)):
                break
        index = np.array(emp_cost).argmin()
        # print("min 1", theta_l[-1])
        return theta_l[-1]

    def isOptimized(self, empirical_cost):
        '''
        This fucntion is for checking R_empirical is optimized or not.
        empirical_cost : the value of R_emp()
        '''
        # return True
        # if len(empirical_cost) > 3:
        #     if empirical_cost[-1] == min(empirical_cost):
        #         return True
        #     else:
        #         return False
        # if len(empirical_cost) > 5:
        #     return True
        return True

    def _fitting_theta(self, theta_list, Rempirical_cost, count):
        # print("fit_theta!", theta_list)
        # print("emps!", Rempirical_cost)
        theta_range = 2*self.dim*(self.l_iter+1)
        interval = 2*pi/theta_range
        index = np.mod(count, self.dim+1)
        sum_list = [interval if i == index else 0 for i in range(self.dim)]
        n_thetalist = np.array(theta_list[-2]) + np.array(sum_list)
        theta_list.append(list(n_thetalist))
        if Rempirical_cost[-1] < Rempirical_cost[-2]:
            return theta_list[-1]
        else:
            return theta_list[-2]

    def _circuit(self, theta_list, training_data):
        qc = self.qc
        q = self.q
        c = self.c
        # TODO we have to chenge the angle of feature map for each data.
        # TODO multi circuit
        mean = np.median(training_data, axis=0)
        # feature_angle = [((mean - (training_data[i]))**2) for i in range(self.dim)]
        # feature_angle = [(np.sin(training_data[0]))*(np.sin(training_data[1]))*(np.sin(training_data[2])) for i in range(3)]
        qc_list = []
        for data in training_data:
            # print(data)
            feature_angle = [(pi - 1/np.exp(i)) for i in data]
            self._feature_map(qc, self.dim, feature_angle)
            self._w_circuit(qc, theta_list)
            qc.measure(q, c)
            qc_list.append(qc)

        backends = ['ibmq_20_tokyo',
                    'qasm_simulator',
                    'ibmqx_hpc_qasm_simulator',
                    'statevector_simulator']

        backend_options = {'max_parallel_threads': 0,
                           'max_parallel_experiments': 0,
                           'max_parallel_shots': 0,
                           'statevector_parallel_threshold': 12}

        backend = Aer.get_backend(backends[1])
        qobj_list = [compile(qc, backend) for qc in qc_list]
        count_list = []
        job_list = [backend.run(qobj) for qobj in qobj_list]
        for job in job_list:
            counts = job.result().get_counts()
            # print([(k,v) for k, v in counts.items() if v > 10])
            count_list.append(counts)

        # print(count_list)
        return count_list

    def _test_circuit(self, theta_list, test_data):
        qc = self.qc
        q = self.q
        c = self.c
        # TODO we have to chenge the angle of feature map for each data.
        # TODO multi circuit
        # mean = np.median(training_data, axis=0)
        # feature_angle = [((mean - (training_data[i]))**2) for i in range(self.dim)]
        # feature_angle = [(np.sin(training_data[0]))*(np.sin(training_data[1]))*(np.sin(training_data[2])) for i in range(3)]
        feature_angle = [(pi - np.sin(i)) for i in test_data]
        self._feature_map(qc, self.dim, feature_angle)
        self._w_circuit(qc, theta_list)
        qc.measure(q, c)
        # qc_list.append(qc)

        backends = ['ibmq_20_tokyo',
                    'qasm_simulator',
                    'ibmqx_hpc_qasm_simulator',
                    'statevector_simulator']

        backend_options = {'max_parallel_threads': 0,
                           'max_parallel_experiments': 0,
                           'max_parallel_shots': 0,
                           'statevector_parallel_threshold': 12}

        backend = Aer.get_backend(backends[1])
        exec = execute(qc, backend, shots=self.shots, config=backend_options)
        result = exec.result()
        counts = result.get_counts(qc)
        # print([k for k, v in counts.items() if v > 10])
        return counts

    def predict(self, test_data, theta_list, label):
        # FIXME have to modify add testdata and
        # for p in parameter:
        vals = []
        # for theta in theta_list:
        count_results = self._test_circuit(theta_list, test_data)
        test_val = self._aggregate(count_results, label)
        answer = label[np.array(test_val).argmax()]
        return answer

    @staticmethod
    def calc_accuracy(labels, test_y):
        correct_answer = 0
        for i, j in zip(labels, test_y):
            if i == j:
                correct_answer += 1
        return correct_answer/len(test_y)

    def visualize(self, x, y, theta, bias, resolution=0.5):
        # print(x)
        markers = ('o', 'x')
        cmap = clp(('red', 'blue'))

        x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
        x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
        x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                       np.arange(x2_min, x2_max, resolution))

        z = self.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T, theta, bias)
        z = np.array(z)
        z = z.reshape(x1_mesh.shape)
        # print(z)

        plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap)
        plt.xlim(x1_mesh.min(), x1_mesh.max())
        plt.ylim(x2_mesh.min(), x2_mesh.max())

    @staticmethod
    def _sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def _ReLU(x):
        return max(0, x)

    @staticmethod
    def _ELU(x):
        if x > 0:
            return x
        else:
            return np.exp(x) - 1

    @staticmethod
    def circle_data(r):
        x = np.arange(-r, r, r/100)
        print(np.sqrt((r**2)-(x**2)), -np.sqrt((r**2)-(x**2)))
        return x, np.array(np.sqrt((r**2)-(x**2))), np.array(-np.sqrt((r**2)-(x**2)))

    def wrapper(self, args):
        return self.fit(*args)

    def multi_process(self, data_list):
        p = mul.Pool(8)
        output = p.map(self.wrapper, data_list)
        p.close()
        return output


if __name__ == '__main__':
    print('start')
    start = time.time()
    fig = plt.figure()

    # mnist dataset
    digits = datasets.load_digits()
    x_data = digits.data[0:100]
    y_d = digits.target[0:100]

    labels = (2, 3, 7)
    x_list = []
    y_list = []
    for i, j in zip(x_data, y_d):
        if j in labels:
            x_list.append(i)
            y_list.append(j)

    x_data = umap.UMAP(n_neighbors=20,
                       n_components=10,
                       min_dist=0.01,
                       metric='correlation').fit_transform(x_list, y=y_list)
    parameters = []
    sc = StandardScaler()
    sc.fit(x_data)
    x_data = sc.transform(x_data)
    # labels = random.sample(range(10), k=3)

    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_list,
                                                        test_size=0.1,
                                                        shuffle=False)

    dim = len(x_data[0])
    theta_list = []
    test = QVC(dim, dim, ["0"*dim, "1"*dim], 16384, 1, dim, max(y_d))
    parameter = test.fit(x_train, y_train, labels)
    # theta_list.append(parameter)
    # print("theta", theta_list)
    count = 1
    answers = []
    print("param!",parameter)
    for i in x_test:
        prob_all = []
        print("="*25, "test", count, "="*25)
        label = test.predict(i, parameter, labels)
        answers.append(label)
        count += 1
    acc = test.calc_accuracy(answers, y_test)
    Notify.notify(acc)
    print(acc)
    print(answers)
    print(y_test)
    print(parameters)
    # df = pd.DataFrame([[acc], [parameters]])
    # print(df)
    df.to_csv('./data/%sn%s.csv' % ('237', str(sys.argv[1])))
    print(time.time() - start)
