import configparser

import yagmail

user, password, receiver, host, result_path = '', '', '', '', ''


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, host, result_path

    user = config.get('email', 'user')
    password = config.get('email', 'password')
    receiver = config.get('email', 'receiver').split(',')
    host = config.get('email', 'host')
    result_path = config.get('data', 'result_path')


def send_email(title, content):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content)


def test_s():
    read_config()
    print(result_path.removeprefix('./'))


def test_send():
    read_config()
    send_email('附件形式发送计算结果', result_path[2:])
