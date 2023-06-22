from django.contrib.auth import views as auth_view
from django.urls import path
from . import views


urlpatterns = [
    path('api/v1/pizzalist',views.PizzaAPIView.as_view(), name='pizzalist'),
    path('', views.home, name='home'),
    path('about', views.about, name='about'),

    path('login', auth_view.LoginView.as_view(), name='login'),
    path('logout', auth_view.LogoutView.as_view(), name='logout'),
    path('pizza-constructor', views.pizza_constructor, name='pizza-constructor'),
]
