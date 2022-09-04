#!/usr/bin/env python
# coding: utf-8

# In[1]:


import obspy
import obspy.taup
from obspy.taup.tau_model import TauModel
from obspy.taup.taup_create import build_taup_model


# In[2]:


build_taup_model("models/Gudkova.nd")
build_taup_model("models/TAYAK.nd")
build_taup_model("models/NewGudkova.nd")
build_taup_model("models/Combined.nd")


# In[ ]:




