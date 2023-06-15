
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView

urlpatterns = [

    path('', views.home, name='home'),
    path('about', views.about, name='about'),

    path('login', views.user_login, name='login'),
    path('pizza-constructor', views.pizza_constructor, name='pizza-constructor'),
]
