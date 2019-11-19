# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:02:51 2019

@author: User
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.compiler import transpile
import numpy as np
import collections

'''
    Define class 
'''

class QuantumOthello():
    
    def __init__(self, num):
        self.qc = QuantumCircuit(num)
        
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
    
    def RuntheGaame(self):
        self.qc.measure_all()
        backend = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, backend=backend, shots = 8192).result().get_counts()
        List = {'0': 0, '1': 0}
        for i in job:
            t, _ = collections.Counter(i).most_common(1)[0]
            List[t] += job[i]
        return List
    
q = QuantumOthello(4)
q.initial('0', 2)
q.initial('0', 3)
q.initial('1', 0)
q.initial('1', 1)
q.end_initial()
q.operation('H', 0)
q.operation('CX', [1, 3])
q.operation('H', 2)
q.operation('H', 3)
q.get_cir(trans=False)
result = q.RuntheGaame()