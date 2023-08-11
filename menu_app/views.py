from django.shortcuts import render
from django.http import HttpResponse
from .models import Creat_pizza, Pizza, Soup
from .forms import CreatePizzaForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from .forms import LoginForm
from rest_framework import generics
from .tasks import latest_parse_data

from .serializers import PizzaSerializer



class PizzaAPIView(generics.ListAPIView):
    queryset = Pizza.objects.all()
    serializer_class = PizzaSerializer

# Create your views here.
@login_required
def home(request):

    global latest_parse_data
    pizzas = Pizza.objects.all()
    soups = Soup.objects.all()


    return render(request, 'menu_app/home.html',{'pizzas': pizzas, 'soups': soups, 'latest_parse_data':latest_parse_data})
@login_required
def about(request):
    return render(request, 'menu_app/about.html')

def calculate_total_price(pizza):
    # Get the selected pizza size
    pizza_size = pizza.pizza_size

    # Get the selected dough thickness
    dought_thickness = pizza.dought_thickness

    # Get the selected ingredients
    ingredients = pizza.ingredients.all()

    # Get the selected sauce
    sauce = pizza.sauce

    # Get the selected cheese
    cheese = pizza.cheese

    # Calculate the total price based on the selected options
    total_price = pizza_size.price + dought_thickness.price

    for ingredient in ingredients:
        total_price += ingredient.price

    total_price += sauce.price + cheese.price

    return total_price
@login_required
def pizza_constructor(request):
    error = ''
    if request.method =='POST':
        form = CreatePizzaForm(request.POST)
        if form.is_valid():
            pizza = form.save(commit=False)
            pizza.total_price = 0

            pizza.save()
        # from django doocs
        # # Create a form instance with POST data.
        # >> > f = AuthorForm(request.POST)
        #
        # # Create, but don't save the new author instance.
        # >> > new_author = f.save(commit=False)
        #
        # # Modify the author in some way.
        # >> > new_author.some_field = "some_value"
        #
        # # Save the new instance.
        # >> > new_author.save()
        #
        # # Now, save the many-to-many data for the form.
        # >> > f.save_m2m()

        else:
            error = 'Форма заполнена не корректно, скорее всего остались пустые поля.'

    form = CreatePizzaForm()
    data = {
        'form': form,
        'error': error
        }
    return render(request, 'menu_app/pizza_constructor.html', data)



def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request,
                                username=cd['username'],
                                password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return HttpResponse('Authenticated successfully')
                else:
                    return HttpResponse('Disabled account')
            else:
                return HttpResponse('Invalid login')
    else:
        form = LoginForm()
        data = {
            'form': form,
            'next': 'home',
                }

        return render(request, 'registration/login.html', data)



