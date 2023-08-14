# import requests
# from bs4 import BeautifulSoup
# import fake_useragent
# import time
# import threading
# from celery import shared_task
#
#
#
# user = fake_useragent.UserAgent().random
# header = {'user.agent':user}
#
# @shared_task
# def do_parse():
#     link = 'https://meteo.by/minsk/'
#
#     responce_bite = requests.get(link, headers=header).content
#
#     responce_text = requests.get(link, headers=header).text
#
#     soup = BeautifulSoup(responce_text, 'lxml')
#     block = soup.find('div', 'weather')
#     check_js = block.find_all('strong')[0].text
#     """This code parse city name"""
#     check_js1 = (block.find_all('strong')[1].text).replace(' ', '').replace('\n', '')
#     """This code parse temperature"""
#     # print(block)
#     # print('--------------------------')
#     data_whethers = {"check_js" :check_js, "check_js1" :check_js1}
#     print(check_js)
#     print(check_js1)
#
# # with open('/home/user/Work/selfeducation/mens_food/parsing/static/parsing/1.html', 'w', encoding='utf-8') as file:
# #     file.write(responce_text)
# count = 0
# delay = 0
# # time.sleep(delay)
# # thread = threading.Thread(target=do_parse)
# # thread.start()
# # print(f'parsing № {count}')
#
#
#
# while True:
#     count += 1
#     if count == 1:
#         delay = 0 #time to parsing site for the first time
#     else:
#         delay = 3600 #time to parsing site for the schdule every hour
#     time.sleep(delay)
#     thread = threading.Thread(target=do_parse)
#     thread.start()
#     print(f'parsing № {count}')
#
