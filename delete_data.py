import os
import re

PATH_RETAINED_DATE = "retained_dates.txt"
PATH_DATA = "data"


def delete():
    dates: list[re.Pattern] = []
    with open(PATH_RETAINED_DATE, mode="r") as f:
        for line in f.readlines():
            dates.append(re.compile(line.replace("\n", "")))
            print(dates[-1])

    for file_name in [
        os.path.join(PATH_DATA, f)
        for f in os.listdir(PATH_DATA)
        if os.path.isfile(os.path.join(PATH_DATA, f))
    ]:
        is_retain = False
        for date in dates:
            if date.search(file_name):
                is_retain = True
                break
        if not is_retain:
            os.remove(file_name)


delete()
