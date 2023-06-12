from django.contrib import admin
from .models import Pizza_size, Dought_thickness, Ingredients, Sauce, Cheese, Creat_pizza

# Register your models here.
admin.site.register(Pizza_size)
admin.site.register(Dought_thickness)
admin.site.register(Ingredients)
admin.site.register(Sauce)
admin.site.register(Cheese)
admin.site.register(Creat_pizza)