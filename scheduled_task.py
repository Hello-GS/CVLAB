import time

import requests
import json
url = "http://dailyhealth-api.sustech.edu.cn/api/form/save"
data = {"sid": "11712504", "type": "1", "xm": "朱俊达", "deptName": "本科2017级", "curCity": "3", "dept": "ug2017",
        "fanshenDate": "", "sfQz": "", "sfZx": "", "sfQgwh": "", "sfQghb": "", "jcs": "", "bxzz": "", "tiwen": "",
        "local": "中国", "formDate": "2020-11-11", "nl": "20", "jtzz": "辽宁省-葫芦岛市-龙港区-锦葫路143-14号楼3单元36号", "xb": "男",
        "mobile": "18820964977", "ylzd22": "14", "ylzd140": "0", "ylzd117": "下午六点 吃饭", "ylzd139": "0", "ylzd21": "36.7",
        "ylzd116": "1", "ylzd23": "0"}
header = {"accessToken": "2ea632204bc5670676504d870bef9a4e0f526094", "userId": "11712504",
          "Content-Type": "application/json"}

if __name__ == '__main__':
    while True:
        res = requests.post(url, data=json.dumps(data), headers=header)
        print(res.text)
        time.sleep(3600)
