from decimal import Decimal
from django.conf import setting
from menu_app.models import Pizza

class Cart:
    def __init__(self, request):
        """
        create cart
        """
        self.session = request.session
        cart = self.session.get(setting.CART_SESSION_ID)
        if not cart:
            # save new cart in session
            cart = self.session[setting.CART_SESSION_ID]={}
        self.cart = cart

    def add(self, pizza, quntity=1, override_quantity=False):
        """
        add product in cart or specify quntity
        """
        pizza_id = str(pizza.id)
        if pizza_id not in self.cart:
            self.cart[pizza_id]={'quantity':0, 'price':str(pizza.price)}
        if owerride_quantity:
            self.cart[pizza_id]['quantity']=quantity
        else:
            self.cart[pizza_id][quntity]+=quantity
        self.save()

    def save(self):
        # mark the session as 'modified'
        # to ensure it is saved
        self.session.modified = True

    def remove(self, pizza):
        # to delet pizza from cart
        pizza_id = str(pizza.id)
        if pizza_id in self.cart:
            del self.cart[product.id]
            self.save()

    def __iter__(self):
        """
        scroll through the shopping cart items
        in a cycle and get items from the database
        """
        pizza_ids = self.cart.keys()
        # get product and add them to cart
        pizzas = Pizza.objects.filter(id_in=pizza_ids)
        cart = self.cart.copy()
        for pizza in pizzas:
            cart[str(pizza.id)]['pizza'] = pizza
        for item in cart.values():
            item['price'] = Decimal(item['price'])
            item['total_price'] = item['price'] * item['quantity']
            yield item

    def __len__(self):
        """ to count all amount in cart"""
        return sum(item['quantity'] for item in self.cart.values())



