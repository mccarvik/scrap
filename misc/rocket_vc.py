"""
short script for rocket VC coding challenge
"""
import pdb
import json
import requests
# import pandas as pd
from bs4 import BeautifulSoup

TWITTER = "https://www.twitter.com/"
FACEBOOK = "https://www.facebook.com/"
PLAY_STORE = "https://play.google.com/"
IOS_STORE = "https://apps.apple.com/"


def check_href(url, soup):
    """
    check the given soup for the given URL
    """
    # pdb.set_trace()
    ret_vals = []
    href = soup.find_all("a")
    for link in href:
        if url in link.get("href"):
            ret_vals.append(link.get("href").split(url)[1])
    return list(set(ret_vals))

def main():
    """
    main function
    """
    url = "https://www.zynga.com/"
    resp = requests.get(url)
    resp_doc = resp.content
    # resp_doc = resp.read()
    soup = BeautifulSoup(resp_doc, 'html.parser')

    ret_json = {}

    # check twitter
    twitter = check_href(TWITTER, soup)
    ret_json['twitter'] = twitter

    # check facebook
    facebook = check_href(FACEBOOK, soup)
    ret_json['facebook'] = facebook

    # check ios
    ios = check_href(IOS_STORE, soup)
    ret_json['ios'] = ios

    # check play store
    play_store = check_href(PLAY_STORE, soup)
    ret_json['play_store'] = play_store

    # Serializing json
    ret_json = json.dumps(ret_json, indent = 4)
    return ret_json


if __name__ == '__main__':
    json_final = main()
    print(json_final)
