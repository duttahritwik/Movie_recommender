#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:28:36 2021

@author: hritwikdutta
"""

import pandas as pd
from gensim.models import Word2Vec

data = pd.read_csv('user_artists.dat', sep='\t')

users = list(data['userID'])
artists = list(data['artistID'])

l = list(zip(users, artists))

users_unique = list(set(users))

master_list=[]

for user_unique in users_unique:
    new = []
    for val in l:
        if user_unique == val[0]:
            new.append(val[1])
    master_list.append(new)
    
print(master_list)
m_list = []

for item in master_list:
    m_list.append(list(map(str, item)))
    
print(m_list)

model=Word2Vec(m_list,size=200, window=10, min_count=1)
model.save('embeddings_lastfm.bin')
