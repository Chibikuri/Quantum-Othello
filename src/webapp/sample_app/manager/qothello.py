from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.compiler import transpile
import numpy as np
import collections
import random
import matplotlib.pyplot as plt
import os
plt.ioff()
'''
    Function to create folder
'''

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
'''
    Define class 
'''

class QuantumOthello():
    
    def __init__(self, num, turn):
        self.num = num
        self.turn = turn
        self.qc = QuantumCircuit(num)
        self.turnA = int(self.turn/2)
        self.turnB = self.turn - self.turnA
        self.GateA = self.RandomGate(self.turnA)
        self.GateB = self.RandomGate(self.turnB)
        self.MeasurementBasis = self.RandomBasis()
        
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
        createFolder('./fig')
        style = { 'showindex': True, 
                 'cregbundle' : True, 'dpi' : 300}
        if trans == True:
            return transpile(self.qc, basis_gates = ['x', 'h', 'u3', 'cx']).draw(output='mpl', style = style, fold=100)
        return self.qc.draw(output='mpl', style = style, fold=100)
    
    def initial(self, instruction, num, vector=None):
        
        ''' Normal gate version '''
        if instruction == '+':
            self.qc.h(num)
        elif instruction == '-':
            self.qc.x(num)
            self.qc.h(num)
        elif instruction == '1':
            self.qc.x(num)
        elif instruction == '0':
            None
        else:
            print('invalid initialize instruction')
    
    def SeqInitial(self, instruction):
        for i in range(self.num):
            self.initial(instruction[i], i)
        
    def end_initial(self):
        self.qc.barrier()        
    
    def operation(self, oper, num):
        if type(num) == list and len(num) == 2:
            num_control = num[0]
            num_target = num[1]
        if type(num) == list and len(num) == 1:
            num = num[0]
        if oper == 'H':
            self.qc.h(num)
        if oper == 'CX':
            self.qc.cx(num_control, num_target)
        if oper == 'X':
            self.qc.x(num)
        if oper == 'Z':
            self.qc.z(num)
        if oper == 'HX':
            self.qc.h(num)
            self.qc.x(num)
        if oper == 'CZ':
            self.qc.cz(num_control, num_target)
    
    def RuntheGame(self):
        self.qc.barrier()
        
        for i in range(self.num):
            if self.MeasurementBasis[i] == 'X':
                self.qc.h(i)
            elif self.MeasurementBasis[i] == 'Y':
                self.qc.sdg(i)
                self.qc.h(i)
            else:
                None
        self.qc.measure_all()
        self.get_cir().savefig('./fig/Measurment.png')
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
    
    def RandomBasis(self):
        Basis = ['X', 'Z']
        return random.choices(Basis, k=self.num)
    
    def RandomGate(self, numTurn):
        Gate = ['X', 'Z', 'H', 'HX', 'CX', 'CZ']
        return random.choices(Gate, k=numTurn)
    
    def RemoveOper(self, player, gate):
        if player == 'A':
            self.GateA.remove(gate)
        else:
            self.GateB.remove(gate)
            