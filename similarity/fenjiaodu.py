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


def send_email(title, content, attachment):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content, attachment)


if __name__ == '__main__':
    read_config()
    send_email('附件形式发送计算结果', '','/disk/11712504/fuck/similarity/result_ans.txt')
