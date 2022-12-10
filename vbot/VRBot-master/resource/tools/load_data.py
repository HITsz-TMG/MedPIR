# -*- coding: utf-8 -*-

import json
import logging
import zipfile
from tqdm import tqdm

d_logger = logging.getLogger("main")


def read_sessions_from_zip_filename(zip_filename):
    zip_file = zipfile.ZipFile(zip_filename)
    zip_namelist = list(filter(lambda x: x.endswith("json"), list(zip_file.namelist())))

    a = list(zip_namelist)
    if '/' in a[0]:
        b = [int(i.split("/")[1].split('.')[0]) for i in a]
    else:
        b = a
    tt = [(i, j) for i, j in zip(a, b)]
    tt = sorted(tt, key=lambda x: x[1])
    zip_namelist = [i[0] for i in tt]
    # zip_namelist = zip_namelist[:10]
    pbar = tqdm(zip_namelist)
    name2session = {}
    d_logger.info("\nread session from {}".format(zip_filename))

    for name in pbar:
        pbar.set_description(name)
        session = json.loads(zip_file.read(name).decode("utf-8"))
        dialogues = session["dialogues"]
        dialogues = [{"turn": dia.get("turn", None) or dia.get("turn_index", None),
                      "sentence": dia["sentence"],
                      "role": dia["role"],
                      "tokens": [word.strip("\n") for word in dia["tokens"] if word != "\n"],
                      "type": dia.get("type", None),
                      "keywords": dia["keywords"] if "keywords" in dia else "",
                      } for dia in dialogues]

        session["dialogues"] = dialogues
        session["session_name"] = name
        name2session[name] = session

    return name2session
