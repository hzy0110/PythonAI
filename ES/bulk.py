from datetime import datetime
from elasticsearch import Elasticsearch
import elasticsearch.helpers
import random
import time


def test1():
    start_time = time.time()
    es = Elasticsearch("ods17:9200")
    package = []

    for i in range(10000):
        row = {
            "@timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+0800"),
            "http_code": "404",
            "count": random.randint(1, 100)
        }
        package.append(row)
    actions = [
        {
            '_op_type': 'index',
            '_index': "http_code",  # index
            '_type': "error_code",  # type
            '_source': d
        }
        for d in package
    ]
    elasticsearch.helpers.bulk(es, actions)
    td = time.time() - start_time
    print(td)


def test():
    es = Elasticsearch("localhost:9200")
    actions = []
    for i in range(1, 10000):
        actions.append({"_index": 'nq_test2', "_type": "python", "_source": {
            "name1": "value" + str(i),
            "name2": "value" + str(i),
            "name3": "value" + str(i),
            "name4": "value" + str(i),
            "name5": "value" + str(i),
            "name6": "value" + str(i),
            "name7": "value" + str(i),
            "name8": "value" + str(i),
            "name9": "value" + str(i),
            "name10": "value" + str(i),
            "name11": "value" + str(i),
            "name12": "value" + str(i),
            "name13": "value" + str(i),
            "name14": "value" + str(i),
            "name15": "value" + str(i),
            "name16": "value" + str(i),
            "name17": "value" + str(i),
            "name18": "value" + str(i),
            "name19": "value" + str(i),
            "name20": "value" + str(i),


        }})
    starttime = time.clock()
    elasticsearch.helpers.bulk(es, actions, chunk_size=50000)
    endtime = time.clock()
    print("cost = " + str(endtime - starttime))

test()

