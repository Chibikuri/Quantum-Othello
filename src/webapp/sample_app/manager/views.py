#Create your views here.
import re
import io
import base64
import numpy as np

from io import BytesIO
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView
from django.views import generic
from .forms import PhotoForm
from .models import Photo, Post
from django.http.response import HttpResponse
from django.shortcuts import render
from .qothello import QuantumOthello
from django.views.decorators.csrf import csrf_exempt


def home(req):
    if req.method == 'GET':
        return render(req, 'apps/home.html')


def members(req):
    if req.method == 'GET':
        return render(req, 'apps/members.html')


def qtry(req):
    return render(req, 'apps/quantum_othello.html')


def four_board(req):
    return render(req, 'apps/4playboard.html')


def nine_board(req):
    return render(req, 'apps/9playboard.html')


def sixteen_board(req):
    return render(req, 'apps/16playboard.html')


def try_quantum(req):
    if req.method == 'GET':
        return render(req, 'apps/quantum_othello.html')


def technologies(req):
    if req.method == 'GET':
        return render(req, 'apps/technologies.html')


def links(req):
    if req.method == 'GET':
        return render(req, 'apps/links.html')

@csrf_exempt
def gate_ajax(req):
    """
    {"H":0 , "H":1}
    """
    opt = [req.POST.get('qubit'+str(i)) for i in range(1, 5)]
    for i, v in enumerate(opt):
        if v is not None:
            q.operation(v, i)
    circuit = q.get_cir()
    q.end_initial()
    
    figfile = BytesIO()
    circuit.savefig(figfile, format='png')
    figfile.seek(0)
    figfile_png = figfile.getvalue()
    base64_circuit = base64.b64encode(figfile_png)
    return HttpResponse(base64_circuit)

@csrf_exempt
def test_ajax_app(req):
    """
    {"H":0 , "H":1}
    """
    print(req.POST)
    initial = [req.POST.get('qubit'+str(i), 0) for i in range(1, 5)]
    print(initial)
    operations = operation(initial)
    print(operations)
    global q
    q = QuantumOthello(len(operations), 5)
    q.SeqInitial(operations)
    circuit = q.get_cir()
    # print(circuit)
    q.end_initial()
    
    figfile = BytesIO()
    circuit.savefig(figfile, format='png')
    figfile.seek(0)
    figfile_png = figfile.getvalue()
    base64_circuit = base64.b64encode(figfile_png)
    return HttpResponse(base64_circuit)


def operation(initials):
    instruction = []
    for i, v in enumerate(initials):
        if v == '0':
            instruction.append('0')
        if v == '1':
            instruction.append('1')
        if v == 'p':
            instruction.append('+')
        if v == 'm':
            instruction.append('-')
    return instruction

def detail(req):
    return render(req, 'apps/research.html')

class PostList(generic.ListView):
    model = Post
