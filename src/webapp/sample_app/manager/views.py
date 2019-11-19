#Create your views here.
import re
import io
import base64
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView
from .forms import PhotoForm
from .models import Photo
from django.http.response import HttpResponse
from django.shortcuts import render
from PIL import Image, ImageDraw, ImageFont
#from .classify import QVC
from .qothello import QuantumOthello
import umap
from sklearn import datasets


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


def ajax_response(req):
    input_text = req.POST.getlist("name_input_text")
    print('hello')
    hoge = "Ajax Response: " + input_text[0]

    return HttpResponse(hoge)


# def ajax_gate(request):
#     '''
#     operation = {'Instruction':'H', 'qubit':2}
#     '''
#     input_text = request.POST
#     print(input_text)
#     hoge = "Ajax Response: " + input_text[0]

#     return HttpResponse(hoge)


def test_ajax_app(request):
    input_text = request.POST['']
    hoge = "Ajax Response: " + input_text[0]

    return HttpResponse(hoge)


def detail(req):
    return render(req, 'apps/research.html')
