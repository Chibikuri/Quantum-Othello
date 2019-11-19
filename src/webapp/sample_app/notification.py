import requests
import sys


class Notify:
    def notify(acc):
        TOKEN = 'xoxp-400314354772-399843882016-493834607953-1914af04136a953ed4eae03b5328a8c8'
        CHANNEL = 'UBRQTRY0G'
        TEXT = 'finished!(poki) %s' % str(acc)
        USERNAME = 'finish bot'
        URL = 'https://slack.com/api/chat.postMessage'

        # post
        post_json = {
            'token': TOKEN,
            'text': TEXT,
            'channel': CHANNEL,
            'username': USERNAME,
            'link_names': 1
        }
        requests.post(URL, data=post_json)
