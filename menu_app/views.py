from django.shortcuts import render
from django.http import HttpResponse
from .models import Creat_pizza
from .forms import CreatePizzaForm


# Create your views here.

def home(request):
    return render(request, 'menu_app/home.html')

def about(request):
    return render(request, 'menu_app/about.html')
def pizza_constructor(request):
    form = CreatePizzaForm()
    data = {
        'form':form,

        }
    return render(request, 'menu_app/pizza_constructor.html', data)




