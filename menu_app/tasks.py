import requests
from bs4 import BeautifulSoup
import fake_useragent
import time

from celery import shared_task

latest_parse_data = None #глобальная переменная для хронения данных



user = fake_useragent.UserAgent().random
header = {'user.agent':user}

@shared_task
def do_parse():
    global latest_parse_data
    link = 'https://meteo.by/minsk/'

    responce_bite = requests.get(link, headers=header).content

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