# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:29:21 2023

@author: gianl
"""

from dataclasses import dataclass

@dataclass(frozen = True)
class Cache:
    ### size denotes not the maximum number of elements, 
    ### but the maximum number of elements before and after a central element of interest
    ### Cache of size 0 -> 1 member; cache of size 1: 1 + 1 + 1 = 3 members, cache of size 2: 2 + 1 + 2 = 5 members.
    size: int 
    elements: list
    name: str = 'cache'
    
    @staticmethod
    def from_list(someList, name = 'cache'):
        assert len(someList) % 2 == 1, 'Building a cache from a list requires an uneven number of list members'
        cache_size = (len(someList) - 1) // 2
        return Cache(cache_size, someList, name = name) ### build a full cache from a list!
    
    def __getitem__(self, item):
        if abs(item) > self.size:
            raise KeyError(f'Key {item} does not exist in cache of size {self.size}!')
        else:
            return self.elements[item + self.size] ### if we enter 0, we want the middle element! 
                                                # if we enter -1, we want the element before the middle element                              
    def get_indices(self):
        positive_indices = list(range(1, self.size + 1))
        negative_indices = [- i for i in reversed(positive_indices)]
        return [*negative_indices, 0, *positive_indices]
    
    def is_full(self):
        return self.size * 2 + 1 == len(self.elements)
    
    def pop(self):
        return self.elements[0]
    
    def add(self, el):
        if self.is_full():
            return Cache(self.size, [*self.elements[1:], el], name = self.name)
        else:
            return Cache(self.size, [*self.elements, el], name = self.name)
        
    