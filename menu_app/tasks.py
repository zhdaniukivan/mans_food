import requests
from bs4 import BeautifulSoup
import fake_useragent
import datetime

from celery import shared_task

latest_parse_data = None #глобальная переменная для хронения данных


def time_of_function(func):
    def wrapped(*args):
        start_time = datetime.datetime.now()

        res = func(*args)
        stop_time = datetime.datetime.now()


        print(f"Время парсинга сотовляет: {stop_time-start_time} секунд")
        return res
    return wrapped


user = fake_useragent.UserAgent().random
header = {'user.agent':user}
@time_of_function
@shared_task
def do_parse():

    global latest_parse_data #Global variable for passing disparate data
    # on the main page without using a database.
    link = 'https://meteo.by/minsk/' #The address of the site from which we
    # take weather data

    responce_text = requests.get(link, headers=header).text

    soup = BeautifulSoup(responce_text, 'lxml')
    block = soup.find('div', 'weather')
    check_js = block.find_all('strong')[0].text
    """This code parse city name"""
    check_js1 = (block.find_all('strong')[1].text).replace(' ', '').replace('\n', '')
    """This code parse temperature"""
    # print(block)
    # print('--------------------------')
    latest_parse_data = {"check_js" :check_js, "check_js1" :check_js1}
    print(check_js)
    print(check_js1)
    if latest_parse_data is None:
        print("parsing don't work!!!!")



# # with open('/home/user/Work/selfeducation/mens_food/parsing/static/parsing/1.html', 'w', encoding='utf-8') as file:
# #     file.write(responce_text)
#
# @shared_task
# def schedule_parse():
#     while True:
#         do_parse.delay()
#         time.sleep(60)
#




v = do_parse()