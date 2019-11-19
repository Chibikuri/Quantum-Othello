from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.compiler import transpile
import numpy as np
import collections
import random

'''
    Define class 
'''

class QuantumOthello():
    
    def __init__(self, num, turn):
        self.num = num
        self.turn = turn
        self.qc = QuantumCircuit(num)
        
    def StartTheGame(self):
        for i in range(self.num):
            if i % 2 == 0:
                x = str(input('A, instruction >'))
                y = int(input('A #qubit >'))
                self.initial(x, y)
            else:
                x = str(input('B, instruction >'))
                y = int(input('B #qubit >'))
                self.initial(x, y)
        self.end_initial()
        #q.get_cir()
        print('End of initialization')
        for i in range(self.turn):
            if i % 2 == 0:
                x = str(input('A, instruction >'))
                if x == 'CX':
                    y1 = int(input('Control qubit #> '))
                    y2 = int(input('target qubit #> '))
                    self.operation(x, [y1, y2])
                else:
                    y = int(input('A #qubit >'))
                    self.operation(x, y)
            else:
                x = str(input('B, instruction >'))
                if x == 'CX':
                    y1 = int(input('Control qubit #> '))
                    y2 = int(input('target qubit #> '))
                    self.operation(x, [y1, y2])
                else:
                    y = int(input('B #qubit >'))
                    self.operation(x, y)
        result = self.RuntheGaame()
        print(result)
        
    def get_cir(self, trans=False):
        style = { 'showindex': True, 
                 'cregbundle' : True, 'dpi' : 300}
        if trans == True:
            return transpile(self.qc, basis_gates = ['x', 'h', 'u3', 'cx']).draw(output='mpl', style = style, fold=100)
        return self.qc.draw(output='mpl', style = style, fold=100)
    
    def initial(self, instruction, num, vector=None):
        
#        ''' Instruction version '''
#        if instruction == 'G': ## General
#            self.qc.initialize(vector, [num])
#        elif instruction == '+H':
#            self.qc.initialize([1/np.sqrt(2), 1/np.sqrt(2)], [num])
#        elif instruction == '-H':
#            self.qc.initialize([1/np.sqrt(2), -1/np.sqrt(2)], [num])
#        elif instruction == '1':
#            self.qc.initialize([0, 1], [num])
#        else:
#            None
        
        ''' Normal gate version '''
        if instruction == '+H':
            self.qc.h(num)
        elif instruction == '-H':
            self.qc.x(num)
            self.qc.h(num)
        elif instruction == '1':
            self.qc.x(num)
        else:
            None
    
    def end_initial(self):
        self.qc.barrier()        
    
    def operation(self, oper, num):
        if type(num) == list and len(num) == 2:
            num_control = num[0]
            num_target = num[1]
        if oper == 'H':
            self.qc.h(num)
        if oper == 'CX':
            self.qc.cx(num_control, num_target)
        if oper == 'X':
            self.qc.x(num)
    
    def RuntheGaame(self):
        self.qc.measure_all()
        self.get_cir()
        backend = Aer.get_backend('qasm_simulator') #qasm_simulator
        job = execute(self.qc, backend=backend, shots = 8192).result().get_counts()
        List = {'0': 0, '1': 0}
        for i in job:
            if len(collections.Counter(i).most_common()) == 2:
                t, t_c = collections.Counter(i).most_common(2)[0]
                d, d_c = collections.Counter(i).most_common(2)[1]
                if t_c > d_c:
                    List[t] += job[i]
                elif t_c < d_c:
                    List[d] += job[i]
                else:
                    None
            else:
                t, _ = collections.Counter(i).most_common(1)[0]
                List[t] += job[i]
        return List
    
    def RandomGate(self):
        Gate = ['H', 'CX', 'X']
        return random.choices(Gate, k=5)