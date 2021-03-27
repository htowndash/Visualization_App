#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:03:00 2021

@author: michael
"""
import os
from natsort import natsorted
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from streamlit.hashing import _CodeHasher
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors 
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from scipy.spatial import ConvexHull
import vaex




try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


#@st.cache
#def load_data():
 #   url = 'https://drive.google.com/file/d/1lqhBblZE4q8BtEvg-knQslstjkNHpr4B/view?usp=sharing'
  #  path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
   # df = pd.read_csv(path)
    #return df
    
def load_data():
    url = 'https://drive.google.com/file/d/1wjEtP7xrNBxpdi6yUF4kloMKiyjGwlkH/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path)
    return df



def load_match_data():
    url = 'https://drive.google.com/file/d/1JECDoNQlZv7oBPepbW90u29NsU4jgA0g/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path)
    return df


#df = load_data()
#df1 = pd.read_csv('/Users/michael/Documents/Python/CSV/NCAA All Matches.csv')

fontPathBold = "./EBGaramond-Bold.ttf"
fontPathNBold = "./EBGaramond-Medium.ttf"
headers = fm.FontProperties(fname=fontPathBold, size=46)
footers = fm.FontProperties(fname=fontPathNBold, size=24)
Labels = fm.FontProperties(fname=fontPathNBold, size=20)
PitchNumbers = fm.FontProperties(fname=fontPathBold, size=54)
Head = fm.FontProperties(fname=fontPathBold, size=60)
Subtitle = fm.FontProperties(fname=fontPathBold, size=34)
TableSummary = fm.FontProperties(fname=fontPathNBold, size=40)
Annotate = fm.FontProperties(fname=fontPathBold, size=24)
SmallTitle = fm.FontProperties(fname=fontPathBold, size=30)

TeamHead = fm.FontProperties(fname=fontPathBold, size=60)
GoalandxG = fm.FontProperties(fname=fontPathBold, size=48)
Summary = fm.FontProperties(fname=fontPathNBold, size=36)
TableHead = fm.FontProperties(fname=fontPathBold, size=34)
TableNum = fm.FontProperties(fname=fontPathNBold, size=30)


zo=12
def draw_pitch(pitch, line, orientation,view):
    
    orientation = orientation
    view = view
    line = line
    pitch = pitch
    
    if orientation.lower().startswith("h"):
        
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(32,18), facecolor=pitch)
            plt.xlim(40,110)
            plt.ylim(-5,73)
        else:
            fig,ax = plt.subplots(figsize=(32,18), facecolor=pitch)
            plt.xlim(-5,110)
            plt.ylim(-5,73)
        ax.axis('off') # this hides the x and y ticks
    
        # side and goal lines #
        ly1 = [0,0,68,68,0]
        lx1 = [0,105,105,0,0]

        plt.plot(lx1,ly1,color=line,zorder=5)


        # boxes, 6 yard box and goals

            #outer boxes#
        ly2 = [15.3,15.3,52.7,52.7] 
        lx2 = [105,89.25,89.25,105]
        plt.plot(lx2,ly2,color=line,zorder=5)

        ly3 = [15.3,15.3,52.7,52.7]  
        lx3 = [0,15.75,15.75,0]
        plt.plot(lx3,ly3,color=line,zorder=5)

            #goals#
        ly4 = [30.6,30.6,37.4,37.4]
        lx4 = [105,105.2,105.2,105]
        plt.plot(lx4,ly4,color=line,zorder=5)

        ly5 = [30.6,30.6,37.4,37.4]
        lx5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=5)


           #6 yard boxes#
        ly6 = [25.5,25.5,42.5,42.5]
        lx6 = [105,99.75,99.75,105]
        plt.plot(lx6,ly6,color=line,zorder=5)

        ly7 = [25.5,25.5,42.5,42.5]
        lx7 = [0,5.25,5.25,0]
        plt.plot(lx7,ly7,color=line,zorder=5)

        #Halfway line, penalty spots, and kickoff spot
        ly8 = [0,68] 
        lx8 = [52.5,52.5]
        plt.plot(lx8,ly8,color=line,zorder=5)


        plt.scatter(94.5,34,color=line,zorder=5, s=12)
        plt.scatter(10.5,34,color=line,zorder=5, s=12)
        plt.scatter(52.5,34,color=line,zorder=5, s=12)

        arc1 =  Arc((95.25,34),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color=line,zorder=2)
        arc2 = Arc((9.75,34),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color=line,zorder=2)
        circle1 = plt.Circle((52.5, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

        ## Rectangles in boxes
        rec1 = plt.Rectangle((89.25,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)
        rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)

        ## Pitch rectangle
        rec3 = plt.Rectangle((-5, -5), 115,78,ls='-',color=pitch, zorder=1,alpha=1)
        
        ## Add Direction of Play Arrow
        DoP = plt.arrow(0, -2.5, 18-2, 1-1, head_width=1.2,
            head_length=1.2,
            color=line,
            alpha=1,
            length_includes_head=True, zorder=12, width=.3)

        ax.add_artist(rec3)
        ax.add_artist(arc1)
        ax.add_artist(arc2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle1)
        ax.add_artist(DoP)
        
    else:
        if view.lower().startswith("h"):
            fig,ax = plt.subplots(figsize=(32,18), facecolor=pitch)
            plt.ylim(40,110)
            plt.xlim(-5,73)
        else:
            fig,ax = plt.subplots(figsize=(32,18), facecolor=pitch)
            plt.ylim(-5,110)
            plt.xlim(-5,73)
        ax.axis('off') # this hides the x and y ticks

        # side and goal lines #
        lx1 = [0,0,68,68,0]
        ly1 = [0,105,105,0,0]

        plt.plot(lx1,ly1,color=line,zorder=5)


        # boxes, 6 yard box and goals

            #outer boxes#
        lx2 = [15.3,15.3,52.7,52.7] 
        ly2 = [105,89.25,89.25,105]
        plt.plot(lx2,ly2,color=line,zorder=5)

        lx3 = [15.3,15.3,52.7,52.7] 
        ly3 = [0,15.75,15.75,0]
        plt.plot(lx3,ly3,color=line,zorder=5)

            #goals#
        lx4 = [30.6,30.6,37.4,37.4]
        ly4 = [105,105.2,105.2,105]
        plt.plot(lx4,ly4,color=line,zorder=5)

        lx5 = [30.6,30.6,37.4,37.4]
        ly5 = [0,-0.2,-0.2,0]
        plt.plot(lx5,ly5,color=line,zorder=5)


           #6 yard boxes#
        lx6 = [25.5,25.5,42.5,42.5]
        ly6 = [105,99.75,99.75,105]
        plt.plot(lx6,ly6,color=line,zorder=5)

        lx7 = [25.5,25.5,42.5,42.5]
        ly7 = [0,5.25,5.25,0]
        plt.plot(lx7,ly7,color=line,zorder=5)

        #Halfway line, penalty spots, and kickoff spot
        lx8 = [0,68] 
        ly8 = [52.5,52.5]
        plt.plot(lx8,ly8,color=line,zorder=5)


        plt.scatter(34,94.5,color=line,zorder=5,s=12)
        plt.scatter(34,10.5,color=line,zorder=5,s=12)
        plt.scatter(34,52.5,color=line,zorder=5,s=12)

        arc1 =  Arc((34,95.25),height=18.3,width=18.3,angle=0,theta1=220,theta2=-40,color=line, zorder=2)
        arc2 = Arc((34,9.75),height=18.3,width=18.3,angle=0,theta1=40,theta2=-220,color=line, zorder=2)
        circle1 = plt.Circle((34,52.5), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)


        ## Rectangles in boxes
        rec1 = plt.Rectangle((20, 89.25), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)
        rec2 = plt.Rectangle((20, 0), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)

        ## Pitch rectangle
        rec3 = plt.Rectangle((-5, -5), 78, 115,ls='-',color=pitch, zorder=1,alpha=1)
        
        ## Add Direction of Play Arrow
        DoP = plt.arrow(70.5, 0, 2-2, 18-1, head_width=1.2,
            head_length=1.2,
            color=line,
            alpha=1,
            length_includes_head=True, zorder=12, width=.3)

        ax.add_artist(rec3)
        ax.add_artist(arc1)
        ax.add_artist(arc2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle1)
        ax.add_artist(DoP)

def vertfull_pitch(pitch, line, ax): 
    # side and goal lines #
    ax = ax
    
    
    lx1 = [0,0,68,68,0]
    ly1 = [0,105,105,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)


    # boxes, 6 yard box and goals

        #outer boxes#
    lx2 = [15.3,15.3,52.7,52.7] 
    ly2 = [105,89.25,89.25,105]
    ax.plot(lx2,ly2,color=line,zorder=5)

    lx3 = [15.3,15.3,52.7,52.7] 
    ly3 = [0,15.75,15.75,0]
    ax.plot(lx3,ly3,color=line,zorder=5)

        #goals#
    lx4 = [30.6,30.6,37.4,37.4]
    ly4 = [105,105.2,105.2,105]
    ax.plot(lx4,ly4,color=line,zorder=5)

    lx5 = [30.6,30.6,37.4,37.4]
    ly5 = [0,-0.2,-0.2,0]
    ax.plot(lx5,ly5,color=line,zorder=5)


       #6 yard boxes#
    lx6 = [25.5,25.5,42.5,42.5]
    ly6 = [105,99.75,99.75,105]
    ax.plot(lx6,ly6,color=line,zorder=5)

    lx7 = [25.5,25.5,42.5,42.5]
    ly7 = [0,5.25,5.25,0]
    ax.plot(lx7,ly7,color=line,zorder=5)

    #Halfway line, penalty spots, and kickoff spot
    lx8 = [0,68] 
    ly8 = [52.5,52.5]
    ax.plot(lx8,ly8,color=line,zorder=5)


    ax.scatter(34,94.5,color=line,zorder=5,s=12)
    ax.scatter(34,10.5,color=line,zorder=5,s=12)
    ax.scatter(34,52.5,color=line,zorder=5,s=12)

    arc1 =  Arc((34,95.25),height=18.3,width=18.3,angle=0,theta1=220,theta2=-40,color=line, zorder=zo-8)
    arc2 = Arc((34,9.75),height=18.3,width=18.3,angle=0,theta1=40,theta2=-220,color=line, zorder=zo-8)
    circle1 = plt.Circle((34,52.5), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)


    ## Rectangles in boxes
    rec1 = plt.Rectangle((20, 89.25), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)
    rec2 = plt.Rectangle((20, 0), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)

    ## Pitch rectangle
    rec3 = plt.Rectangle((-5, -5), 78, 115,ls='-',color=pitch, zorder=1,alpha=1)
    
    ## Add Direction of Play Arrow
    DoP = plt.arrow(70.5, 0, 2-2, 18-1, head_width=1.2,
        head_length=1.2,
        color=line,
        alpha=1,
        length_includes_head=True, zorder=12, width=.3)

    ax.add_artist(rec3)
    ax.add_artist(arc1)
    ax.add_artist(arc2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle1)
    ax.add_artist(DoP)
    ax.axis('off')
    
def horizfull_pitch(pitch, line, ax): 
# side and goal lines #
    ax = ax
    
    ly1 = [0,0,68,68,0]
    lx1 = [0,105,105,0,0]
    
    ax.plot(lx1,ly1,color=line,zorder=5)
    
    
    # boxes, 6 yard box and goals
    
        #outer boxes#
    ly2 = [15.3,15.3,52.7,52.7] 
    lx2 = [105,89.25,89.25,105]
    ax.plot(lx2,ly2,color=line,zorder=5)
    
    ly3 = [15.3,15.3,52.7,52.7]  
    lx3 = [0,15.75,15.75,0]
    ax.plot(lx3,ly3,color=line,zorder=5)
    
        #goals#
    ly4 = [30.6,30.6,37.4,37.4]
    lx4 = [105,105.2,105.2,105]
    ax.plot(lx4,ly4,color=line,zorder=5)
    
    ly5 = [30.6,30.6,37.4,37.4]
    lx5 = [0,-0.2,-0.2,0]
    ax.plot(lx5,ly5,color=line,zorder=5)
    
    
       #6 yard boxes#
    ly6 = [25.5,25.5,42.5,42.5]
    lx6 = [105,99.75,99.75,105]
    ax.plot(lx6,ly6,color=line,zorder=5)
    
    ly7 = [25.5,25.5,42.5,42.5]
    lx7 = [0,5.25,5.25,0]
    ax.plot(lx7,ly7,color=line,zorder=5)
    
    #Halfway line, penalty spots, and kickoff spot
    ly8 = [0,68] 
    lx8 = [52.5,52.5]
    ax.plot(lx8,ly8,color=line,zorder=5)
    
    
    ax.scatter(94.5,34,color=line,zorder=5, s=12)
    ax.scatter(10.5,34,color=line,zorder=5, s=12)
    ax.scatter(52.5,34,color=line,zorder=5, s=12)
    
    arc1 =  Arc((95.25,34),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color=line,zorder=zo+1)
    arc2 = Arc((9.75,34),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color=line,zorder=zo+1)
    circle1 = plt.Circle((52.5, 34), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)
    
    ## Rectangles in boxes
    rec1 = plt.Rectangle((89.25,20), 16,30,ls='-',color=pitch, zorder=1,alpha=1)
    rec2 = plt.Rectangle((0, 20), 16.5,30,ls='-',color=pitch, zorder=1,alpha=1)
    
    ## Pitch rectangle
    rec3 = plt.Rectangle((-5, -5), 115,78,ls='-',color=pitch, zorder=1,alpha=1)
    
    ## Add Direction of Play Arrow
    DoP = ax.arrow(0, -2.5, 18-2, 1-1, head_width=1.2,
        head_length=1.2,
        color=line,
        alpha=1,
        length_includes_head=True, zorder=12, width=.3)
    
    ax.add_artist(rec3)
    ax.add_artist(arc1)
    ax.add_artist(arc2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle1)
    ax.add_artist(DoP)
    ax.axis('off')


def verthalf_pitch(pitch, line, ax): 
        # side and goal lines #
        ax = ax
        
        plt.ylim(40,106)
        plt.xlim(-5,73)
        
        lx1 = [0,0,68,68,0]
        ly1 = [0,105,105,0,0]
    
        plt.plot(lx1,ly1,color=line,zorder=5)
    
    
        # boxes, 6 yard box and goals
    
            #outer boxes#
        lx2 = [15.3,15.3,52.7,52.7] 
        ly2 = [105,89.25,89.25,105]
        ax.plot(lx2,ly2,color=line,zorder=5)
    
        lx3 = [15.3,15.3,52.7,52.7] 
        ly3 = [0,15.75,15.75,0]
        ax.plot(lx3,ly3,color=line,zorder=5)
    
            #goals#
        lx4 = [30.6,30.6,37.4,37.4]
        ly4 = [105,105.2,105.2,105]
        ax.plot(lx4,ly4,color=line,zorder=5)
    
        lx5 = [30.6,30.6,37.4,37.4]
        ly5 = [0,-0.2,-0.2,0]
        ax.plot(lx5,ly5,color=line,zorder=5)
    
    
           #6 yard boxes#
        lx6 = [25.5,25.5,42.5,42.5]
        ly6 = [105,99.75,99.75,105]
        ax.plot(lx6,ly6,color=line,zorder=5)
    
        lx7 = [25.5,25.5,42.5,42.5]
        ly7 = [0,5.25,5.25,0]
        ax.plot(lx7,ly7,color=line,zorder=5)
    
        #Halfway line, penalty spots, and kickoff spot
        lx8 = [0,68] 
        ly8 = [52.5,52.5]
        ax.plot(lx8,ly8,color=line,zorder=5)
    
    
        ax.scatter(34,94.5,color=line,zorder=5,s=12)
        ax.scatter(34,10.5,color=line,zorder=5,s=12)
        ax.scatter(34,52.5,color=line,zorder=5,s=12)
    
        arc1 =  Arc((34,95.25),height=18.3,width=18.3,angle=0,theta1=220,theta2=-40,color=line, zorder=zo-8)
        arc2 = Arc((34,9.75),height=18.3,width=18.3,angle=0,theta1=40,theta2=-220,color=line, zorder=zo-8)
        circle1 = plt.Circle((34,52.5), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)
    
    
        ## Rectangles in boxes
        rec1 = plt.Rectangle((20, 89.25), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)
        rec2 = plt.Rectangle((20, 0), 30,16.5,ls='-',color=pitch, zorder=1,alpha=1)
    
        ## Pitch rectangle
        rec3 = plt.Rectangle((-5, -5), 78, 115,ls='-',color=pitch, zorder=1,alpha=1)
        
        ## Add Direction of Play Arrow
        DoP = plt.arrow(70.5, 0, 2-2, 18-1, head_width=1.2,
            head_length=1.2,
            color=line,
            alpha=1,
            length_includes_head=True, zorder=12, width=.3)
    
        ax.add_artist(rec3)
        ax.add_artist(arc1)
        ax.add_artist(arc2)
        ax.add_artist(rec1)
        ax.add_artist(rec2)
        ax.add_artist(circle1)
        ax.add_artist(DoP)
        ax.axis('off')
        
        
def main():
    state = _get_state()
    pages = {
        "Team Shots": TeamShot,
        "Team Shots Against": TeamShotD,
        "Team Pass Network": TeamMatchPN,
        #"Team PassSonar": TeamPassSonar,
        #"Team Passing From/To Zones": TeamPassingEngine,
        #"Team Passing Attacking Third": PassDash,
        #"Team Defense": TeamDefensive,
        #"Team Attacking Corners": AttCorner,
        #"Team Defensive Corners": DefCorner,
        #"Team Goal Kicks": TeamGoalKicks,
        #"Opposition PassSonar": OppPassSonar,
        #"Opposition Passing From/To Zones": OppPassingEngine,
        #"Player Shots": PlayerShot,
        #"Goalkeeper Shot Map": GKShotMap,
        #"Player Pass": PlayerPass,
        #"Player Carry": PlayerCarry,
        #"Player Pass Network": PlayerMatchPN,
        #"Player PassSonar": PlayerPassSonar,
        #"Player Passing From/To Zones": PlayerPassingEngine,
        #"Player Defense": PlayerDefensive,
    }

    #st.sidebar.title("Page Filters")
    page = st.sidebar.radio("Select Page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()



def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    #st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)
    
    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()

def multiselect(label, options, default, format_func=str):
    """multiselect extension that enables default to be a subset list of the list of objects
     - not a list of strings

     Assumes that options have unique format_func representations

     cf. https://github.com/streamlit/streamlit/issues/352
     """
    options_ = {format_func(option): option for option in options}
    default_ = [format_func(option) for option in default]
    selections = st.multiselect(
        label, options=list(options_.keys()), default=default_, format_func=format_func
    )
    return [options_[format_func(selection)] for selection in selections]


#selections = multiselect("Select", options=[Option1, Option2], default=[Option2])


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

def TeamMatchPN(state):
    df = load_data()
    
    df['X'] = df['location_x'] * (105/120)
    df['Y'] = 68-(df['location_y'] * (68/80))
    df['DestX'] = df['end_location_x'] * (105/120)
    df['DestY'] = 68-(df['end_location_y'] * (68/80))
    df['xg'] = df.xg
    df['xp'] = df.xp
    df['ifOP'] = [1 if x == 'Regular Play' else 0 for x in df['play_pattern_name']]
    df['ifCA'] = [1 if x == 'From Counter' else 0 for x in df['play_pattern_name']]
    df['ifSP'] = [1 if ((x=='From Corner')or(x=='From Free Kick')or(x=='From Throw In')) else 0 for x in df['play_pattern_name']]

    pass_df = df[df.event_type_name == 'Pass']  
    def in18(pass_df):
        if (pass_df['DestX'] >= (87.25) and ((pass_df['DestY'] >= 15.3) and (pass_df['DestY'] <= (52.7)))):
            val = 1
        else:
            val = 0
        return val
    pass_df['in18'] = pass_df.apply(in18, axis=1)

    pass_df['NextTeam'] = pass_df.team_name.shift(-1)
    pass_df['SameTeam'] = (pass_df['team_name']==pass_df['NextTeam']).astype(int)
    pass_df['NextPos'] = pass_df.possession.shift(-1)
    pass_df['SamePos'] = (pass_df['possession']==pass_df['NextPos']).astype(int)
    pass_df['Retain?'] = pass_df.SameTeam + pass_df.shot_assist + pass_df.SamePos
    pass_df['ifRetain'] = [1 if x >= 2 else 0 for x in pass_df['Retain?']]
    pass_df['NextTeam'] = pass_df.team_name.shift(-1)
    pass_df['SameTeamN'] = (pass_df['team_name']==pass_df['NextTeam']).astype(int)
    pass_df['ifPass'] = 1
    pass_df['PrevTeam'] = pass_df.team_name.shift(1)
    pass_df['SameTeamP'] = (pass_df['team_name']==pass_df['PrevTeam']).astype(int)
    pass_df['DistX'] = pass_df.end_location_x - pass_df.location_x
    pass_df['DistToGoalStart'] = 120 - pass_df.location_x
    pass_df['DistToGoalEnd'] = 120 - pass_df.end_location_x
    pass_df['25DownPitch'] = pass_df.DistToGoalStart * .25
    pass_df['PP1'] = (pass_df.DistToGoalStart - pass_df.DistToGoalEnd) - pass_df['25DownPitch']
    pass_df['PP2'] = [1 if x >= 0 else 0 for x in pass_df['PP1']]
    pass_df['PP3'] = (pass_df['PP2'] + pass_df['pass_complete']) - pass_df['ifSP']
    pass_df['ProgPass'] = [1 if x == 2 else 0 for x in pass_df['PP3']]
    pass_df['DeepProg'] = [1 if x >= 80 else 0 for x in pass_df['location_x']]
    pass_df['DeepProgSum'] = pass_df.DeepProg + pass_df.pass_complete
    pass_df['DeepProg'] = [1 if x == 2 else 0 for x in pass_df['DeepProgSum']]
    pass_df['Build1'] = [1 if (x >= 0 and x<=45) else 0 for x in pass_df['location_x']]
    pass_df['Build2'] = [1 if x >= .85 else 0 for x in pass_df['xp']]
    pass_df['BuildSum'] = pass_df['Build1'] + pass_df['Build2']
    pass_df['Prog1'] = [1 if (x >= 45 and x<=85) else 0 for x in pass_df['location_x']]
    pass_df['ProgPassPrev'] = pass_df.ProgPass.shift(1)
    pass_df['ProgPassNext'] = pass_df.ProgPass.shift(-1)
    pass_df['ProgPassSum'] = pass_df.ProgPass + (pass_df.ProgPassPrev * pass_df.SameTeamP) + (pass_df.ProgPassNext * pass_df.SameTeamN)
    pass_df['Prog2'] = [1 if x>=1 else 0 for x in pass_df['ProgPassSum']]
    pass_df['ProgSum'] = pass_df['Prog1'] + pass_df['Prog2']
    pass_df['Creative1'] = [1 if x>=75 else 0 for x in pass_df['end_location_x']]
    pass_df['Creative2'] = [1 if x>=75 else 0 for x in pass_df['location_x']]
    pass_df['CreativeSum'] = pass_df['Creative1'] + pass_df['Creative2'] + pass_df['pass_complete']
    pass_df['Build'] = ["Build" if x >= 1 else 0 for x in pass_df['BuildSum']]
    pass_df['Progress'] = ["Progress" if x >= 1 else 0 for x in pass_df['ProgSum']]
    pass_df['Create'] = ["Create" if x >= 2 else 0 for x in pass_df['CreativeSum']]
    pass_df['AllAtt'] = pass_df['BuildSum'] + pass_df['ProgSum'] + pass_df['CreativeSum']
    #pass_df['Direct'] = ["Direct" if x >= 35 else 0 for x in pass_df['DistX']]
    cols = ['Build', 'Progress', 'Create']
    pass_df['PhaseofPlay'] = pass_df[cols].apply(lambda row: ' / '.join(row.values.astype(str)), axis=1)

    team = st.sidebar.selectbox("Select Team", natsorted(pass_df.team_name.unique()))    
    match_df = pass_df[pass_df['team_name'] == team]

    
    match = st.sidebar.selectbox("Select Match", natsorted(match_df.match_id.unique()))    
    match_df = match_df[match_df['match_id'] == match]

    minute = st.sidebar.slider("Select Range of Minutes", 0,120, (0,96), 1)
    minute_df = match_df[match_df.minute.between(minute[0], minute[1])]
    
    
    PoPs = ['Build', 'Progress', 'Create']

    PoP = st.sidebar.multiselect('Select Phase(s) of Play', ['Build', 'Progress', 'Create'], default=PoPs)
    PoP_df = minute_df[minute_df.PhaseofPlay.str.contains('|'.join(PoP))]



    def v_pass_networkadv(team_name1):
        Title = fm.FontProperties(fname=fontPathBold, size=32)
        Annotate = fm.FontProperties(fname=fontPathBold, size=30)
        Legend = fm.FontProperties(fname=fontPathBold, size=26)


        team_name1 = team_name1
        players = table['Players'].tolist()
        x = table['X']
        y = table['Y']
        touches_raw = table['xPR'].tolist()
        #maxpass = table['MaxPV'].tolist()
        subminute = minute[1] - minute[0]
        maxpass = 0.225 * (subminute/90)
    
        touches = []
        for item in touches_raw:
            size = 20*item
            touches.append(size)
    
        passes = {}
        for row in table.iterrows():
            index, data = row
            temp = []
            player = data[0]
            for n in range (5, len(data)):
                temp.append(data[n])
            passes[player] = temp
    
        #draw_pitch("#333333","white","vertical","full")
        fig,ax = plt.subplots(figsize=(22,32))
        vertfull_pitch('#333333', 'white', ax)
        
        pass1 = round((maxpass) * .25,3) 
        pass2 = round((maxpass) * .26,3) 
        pass3 = round((maxpass) * .44,3) 
        pass4 = round((maxpass) * .45,3) 
        pass5 = round((maxpass) * .65,3) 
        pass6 = round((maxpass) * .66,3) 
        pass7 = round((maxpass) * .84,3)
        pass8 = round((maxpass) * .85,3)
        pass9 = round((maxpass) * .99,3) 
        pass10 = round((maxpass),3)           
                   
        
        plt.plot(-2,color="darkblue",label="< "+str(pass1)+ ' GPA',zorder=0)
        plt.plot(-2,color="lightblue",label="Between " +str(pass2)+' and '+str(pass3)+ ' GPA',zorder=0)
        plt.plot(-2,color="darkkhaki",label="Between " +str(pass4)+' and '+str(pass5)+ ' GPA',zorder=0)
        plt.plot(-2,color="darkgoldenrod",label="Between " +str(pass6)+' and '+str(pass7)+ ' GPA',zorder=0)
        plt.plot(-2,color="orangered",label="Between " +str(pass8)+' and '+str(pass9)+ ' GPA',zorder=0)
        plt.plot(-2,color="darkred",label=str(pass10)+'+ GPA',zorder=0)
        leg = plt.legend(loc=1,ncol=3,frameon=False)
        plt.setp(leg.get_texts(), color='white', fontproperties=Legend)
        plt.title(str(team)+ ' Advanced Pass Network\nMatch ID: '+str(match)+'\nUp To Minute: '+str(minute), fontproperties=Title, color="black")
        for i, player in enumerate(players):
            plt.annotate(player, xy=(68-y[i],x[i]), xytext=((68-y[i]), x[i]-3), fontproperties=Annotate, color="white", ha="center", weight="bold", zorder=zo+5)
            for n in range(len(players)):
                player_passes = passes[player][n]
                width = player_passes / .175
                if player_passes >0: 
                    x_start = x[i]
                    x_length = x[n] - x[i]
                    y_start = 68-y[i]
                    y_length = (68-y[n]) - (68-y[i])
                    if x_length > 0:
                        x_start = x[i] + 1
                        x_length = x[n] - x[i] - 1
                    else:
                        x_start = x[i] - 1
                        x_length = x[n] - x[i] + 1.5
                    if y_length > 0:
                        y_start = 68-y[i] + 1.5
                        y_length = (68-y[n]) - (68-y[i]) - 2
                    else:
                        y_start = 68-y[i] - 1.5
                        y_length = (68-y[n]) - (68-y[i]) + 2
    
                    if player_passes >= pass10: 
                        color = "darkred" 
                        alpha = .8
                        zorder=zo+5
                    elif player_passes >= pass8 and player_passes <= pass9: 
                        color = "orangered"
                        alpha=.55
                        zorder=zo+4
                    elif player_passes >= pass6 and player_passes <= pass7: 
                        color = "darkgoldenrod"
                        alpha=.45
                        zorder=zo+3
                    elif player_passes >= pass4 and player_passes <= pass5: 
                        color = "darkkhaki"
                        alpha=.35
                        zorder=zo+3
                    elif player_passes >= pass2 and player_passes <= pass3: 
                        color = "lightblue"
                        alpha=.2
                        zorder=zo+2                 
                    else:
                        color = "darkblue"
                        alpha=.15
                        zorder=zo+1    
                    
                    plt.scatter(68-y, x, s = table.ifPass*25/matchcount, c=table.xPR,cmap='RdYlBu_r', linewidths=2, edgecolors='white',zorder=zo+1)
                    plt.arrow(y_start,x_start,y_length,x_length, 
                    head_length=1, color=color, alpha=alpha, width=width, head_width=width*2, length_includes_head=True,zorder=zorder)
        
        cax = plt.axes([0.15, 0.085, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=.9,vcenter=1, vmax=1.1))
        sm.A = []
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.02, pad=0.01, ticks=[.9, .95, 1, 1.05, 1.1])
        cbar.set_label('xPR - Dots', fontproperties=Annotate)
        st.pyplot(fig)

 
    Passes = PoP_df
    Passes['Players'] = Passes['player_name']

    data = Passes[(Passes['match_id'] == match) & (Passes['team_name'] == team) & (Passes['ifSP'] != 1)]#((Passes['minute'] >= Startminute) & (Passes['minute'] <= Endminute))] 
    data1 = Passes[(Passes['match_id'] == match) & (Passes['team_name'] == team) & (Passes['ifSP'] != 1) & (Passes['ifPass'] == 1)] # ((Passes['minute'] >= Startminute) & (Passes['minute'] <= Endminute))]
    
    matchcount = data.match_id.nunique()
    table = pd.pivot_table(data,index=["Players"],columns=['pass_recipient'],values=["gpa"],aggfunc=np.sum, fill_value=0).reset_index(col_level=1)
    table.columns = table.columns.droplevel()
    table['MaxGPA'] = table.iloc[:,1:].max(axis=1)
    table1 = pd.pivot_table(data1,index=["Players"],aggfunc={"X":'mean', "Y":'mean'}, fill_value=0).reset_index()
    table2 = pd.pivot_table(data1, index=["Players"], aggfunc={"ifPass":'sum', 'xp':'sum', 'pass_complete':'sum'}, fill_value=0).reset_index()
    #print(table2)
    table2['xPR'] = table2['pass_complete'] / table2['xp']    
    test = pd.merge(table1, table2, on = "Players")
    table = pd.merge(test,table, on = "Players")
    #print(table.PassesAttempted.max()*.3)
    table = table.loc[table['ifPass'] >= 1]
    idxnames = table['Players'].tolist()
    idxnames.extend(['Players', 'MaxGPA', 'X', 'Y', 'xPR', 'ifPass'])
    table = table.loc[:,table.columns.str.contains('|'.join(idxnames))].reset_index(drop=True)
    table3 = pd.pivot_table(data1, index=["pass_recipient"], aggfunc={'pass_complete':'sum'}, fill_value=0).reset_index()
    table3 = table3.rename(columns={'pass_recipient':'Players', 'pass_complete':'Received'})
    test5 = pd.merge(table,table3, on = "Players")
    test1 = test5.loc[test5['ifPass'] >= 1]
    idxnames = test1['Players'].tolist()
    idxnames.extend(['Players', 'X', 'Y', 'ifPass', 'xPR', 'MaxGPA'])
    table = test1.loc[:,test1.columns.str.contains('|'.join(idxnames))].reset_index(drop=True)


    v_pass_networkadv(PoP_df)    
    
    st.subheader("Passing Data")    
    st.text("")

    
    PNdf = PoP_df[PoP_df['ifSP'] != 1]
    PNdf = PNdf.groupby(["player_name"], as_index=False).agg({'gpa':'sum','ifPass':'sum','xp':'sum','pass_complete':'sum',
                'goal_assist':'sum', 'shot_assist':'sum', 'ifRetain':'sum','ProgPass':'sum',
                'DeepProg':'sum', 'in18':'sum'}).sort_values(by='pass_complete', ascending=False)
    PNdf = PNdf.rename(columns={'ifPass':'Passes','goal_assist':'Assists','shot_assist':'Shot Assists', 
                                  'ifRetain':'Retained','pass_complete':'Completed','xp':'xP','gpa':'GPA'})
    PNdf['xPR'] = PNdf.Completed / PNdf.xP
    PNdf = PNdf.sort_values(by='GPA', ascending=False)
    st.dataframe(PNdf)
    
    st.subheader("Passing Data By Player Connections")    
    st.text("")

    
    PNdf1 = PoP_df[((PoP_df['ifSP'] != 1))]
    PNdf1 = PNdf1.groupby(["player_name", "pass_recipient"], as_index=False).agg({'gpa':'sum','ifPass':'sum','xp':'sum','pass_complete':'sum',
                'goal_assist':'sum', 'shot_assist':'sum', 'ifRetain':'sum','ProgPass':'sum',
                'DeepProg':'sum', 'in18':'sum'}).sort_values(by='pass_complete', ascending=False)
    PNdf1 = PNdf1.rename(columns={'ifPass':'Passes','goal_assist':'Assists','shot_assist':'Shot Assists', 
                                  'ifRetain':'Retained','pass_complete':'Completed','xp':'xP','gpa':'GPA'})
    PNdf1['xPR'] = PNdf1.Completed / PNdf1.xP
    PNdf1 = PNdf1.sort_values(by='GPA', ascending=False)
    st.dataframe(PNdf1)
    
    def v_pass_networkbas(team_name1):
        Title = fm.FontProperties(fname=fontPathBold, size=32)
        Annotate = fm.FontProperties(fname=fontPathBold, size=30)
        Legend = fm.FontProperties(fname=fontPathBold, size=28)


        team_name1 = team_name1
        players = table['Players'].tolist()
        x = table['X']
        y = table['Y']
        touches_raw = table['xPR'].tolist()
        #maxpass = table['MaxPV'].tolist()
        subminute = minute[1] - minute[0]
        maxpass = 10 * (subminute/90)
    
        touches = []
        for item in touches_raw:
            size = 20*item
            touches.append(size)
    
        passes = {}
        for row in table.iterrows():
            index, data = row
            temp = []
            player = data[0]
            for n in range (5, len(data)):
                temp.append(data[n])
            passes[player] = temp
    
        #draw_pitch("#333333","white","vertical","full")
        fig,ax = plt.subplots(figsize=(22,32))
        vertfull_pitch('#333333', 'white', ax)
        
        pass1 = round((maxpass) * .25,3) 
        pass2 = round((maxpass) * .26,3) 
        pass3 = round((maxpass) * .44,3) 
        pass4 = round((maxpass) * .45,3) 
        pass5 = round((maxpass) * .65,3) 
        pass6 = round((maxpass) * .66,3) 
        pass7 = round((maxpass) * .84,3)
        pass8 = round((maxpass) * .85,3)
        pass9 = round((maxpass) * .99,3) 
        pass10 = round((maxpass),3)           
                   
        
        plt.plot(-2,color="darkblue",label="< "+str(pass1)+ ' Passes',zorder=0)
        plt.plot(-2,color="lightblue",label="Between " +str(pass2)+' and '+str(pass3)+ ' Passes',zorder=0)
        plt.plot(-2,color="darkkhaki",label="Between " +str(pass4)+' and '+str(pass5)+ ' Passes',zorder=0)
        plt.plot(-2,color="darkgoldenrod",label="Between " +str(pass6)+' and '+str(pass7)+ ' Passes',zorder=0)
        plt.plot(-2,color="orangered",label="Between " +str(pass8)+' and '+str(pass9)+ ' Passes',zorder=0)
        plt.plot(-2,color="darkred",label=str(pass10)+'+ Passes',zorder=0)
        leg = plt.legend(loc=1,ncol=3,frameon=False)
        plt.setp(leg.get_texts(), color='white', fontproperties=Legend)
        plt.title(str(team)+ ' Basic Pass Network\nMatch ID: '+str(match)+'\nUp To Minute: '+str(minute), fontproperties=Title, color="black")
        for i, player in enumerate(players):
            plt.annotate(player, xy=(68-y[i],x[i]), xytext=((68-y[i]), x[i]-3), fontproperties=Annotate, color="white", ha="center", weight="bold", zorder=zo+5)
            for n in range(len(players)):
                player_passes = passes[player][n]
                width = player_passes / 14
                if player_passes >0: 
                    x_start = x[i]
                    x_length = x[n] - x[i]
                    y_start = 68-y[i]
                    y_length = (68-y[n]) - (68-y[i])
                    if x_length > 0:
                        x_start = x[i] + 1
                        x_length = x[n] - x[i] - 1
                    else:
                        x_start = x[i] - 1
                        x_length = x[n] - x[i] + 1.5
                    if y_length > 0:
                        y_start = 68-y[i] + 1.5
                        y_length = (68-y[n]) - (68-y[i]) - 2
                    else:
                        y_start = 68-y[i] - 1.5
                        y_length = (68-y[n]) - (68-y[i]) + 2
    
                    if player_passes >= pass10: 
                        color = "darkred" 
                        alpha = .8
                        zorder=zo+5
                    elif player_passes >= pass8 and player_passes <= pass9: 
                        color = "orangered"
                        alpha=.55
                        zorder=zo+4
                    elif player_passes >= pass6 and player_passes <= pass7: 
                        color = "darkgoldenrod"
                        alpha=.45
                        zorder=zo+3
                    elif player_passes >= pass4 and player_passes <= pass5: 
                        color = "darkkhaki"
                        alpha=.35
                        zorder=zo+3
                    elif player_passes >= pass2 and player_passes <= pass3: 
                        color = "lightblue"
                        alpha=.2
                        zorder=zo+2                 
                    else:
                        color = "darkblue"
                        alpha=.15
                        zorder=zo+1    
                    
                    plt.scatter(68-y, x, s = table.ifPass*25/matchcount, c=table.xPR,cmap='RdYlBu_r', linewidths=2, edgecolors='white',zorder=zo+1)
                    plt.arrow(y_start,x_start,y_length,x_length, 
                    head_length=1, color=color, alpha=alpha, width=width, head_width=width*2, length_includes_head=True,zorder=zorder)
        
        cax = plt.axes([0.15, 0.085, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=.9,vcenter=1, vmax=1.1))
        sm.A = []
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.02, pad=0.01, ticks=[.9, .95, 1, 1.05, 1.1])
        cbar.set_label('xPR - Dots', fontproperties=Annotate)
        st.pyplot(fig)

 
    Passes = PoP_df
    Passes['Players'] = Passes['player_name']

    data = Passes[(Passes['match_id'] == match) & (Passes['team_name'] == team) & (Passes['ifSP'] != 1)]#((Passes['minute'] >= Startminute) & (Passes['minute'] <= Endminute))] 
    data1 = Passes[(Passes['match_id'] == match) & (Passes['team_name'] == team) & (Passes['ifSP'] != 1) & (Passes['ifPass'] == 1)] # ((Passes['minute'] >= Startminute) & (Passes['minute'] <= Endminute))]
    
    matchcount = data.match_id.nunique()
    table = pd.pivot_table(data,index=["Players"],columns=['pass_recipient'],values=["ifPass"],aggfunc=np.sum, fill_value=0).reset_index(col_level=1)
    table.columns = table.columns.droplevel()
    table['MaxPass'] = table.iloc[:,1:].max(axis=1)
    table1 = pd.pivot_table(data1,index=["Players"],aggfunc={"X":'mean', "Y":'mean'}, fill_value=0).reset_index()
    table2 = pd.pivot_table(data1, index=["Players"], aggfunc={"ifPass":'sum', 'xp':'sum', 'pass_complete':'sum'}, fill_value=0).reset_index()
    #print(table2)
    table2['xPR'] = table2['pass_complete'] / table2['xp']    
    test = pd.merge(table1, table2, on = "Players")
    table = pd.merge(test,table, on = "Players")
    #print(table.PassesAttempted.max()*.3)
    table = table.loc[table['ifPass'] >= 1]
    idxnames = table['Players'].tolist()
    idxnames.extend(['Players', 'MaxPass', 'X', 'Y', 'xPR', 'ifPass'])
    table = table.loc[:,table.columns.str.contains('|'.join(idxnames))].reset_index(drop=True)
    table3 = pd.pivot_table(data1, index=["pass_recipient"], aggfunc={'pass_complete':'sum'}, fill_value=0).reset_index()
    table3 = table3.rename(columns={'pass_recipient':'Players', 'pass_complete':'Received'})
    test5 = pd.merge(table,table3, on = "Players")
    test1 = test5.loc[test5['ifPass'] >= 1]
    idxnames = test1['Players'].tolist()
    idxnames.extend(['Players', 'X', 'Y', 'ifPass', 'xPR', 'MaxPass'])
    table = test1.loc[:,test1.columns.str.contains('|'.join(idxnames))].reset_index(drop=True)


    v_pass_networkbas(PoP_df)   

    
def TeamShot(state):
    df = load_data()
    mdf = load_match_data()
    cols = ['home_team', 'away_team']
    mdf['match_name'] = mdf[cols].apply(lambda row: ' - '.join(row.values.astype(str)), axis=1)
    mdf = pd.DataFrame(data=mdf, columns=['match_id', 'home_team', 'away_team', 'match_name'])
    df = pd.merge(df,mdf, on=['match_id'])
    
    df['X'] = df['location_x'] * (105/120)
    df['Y'] = 68-(df['location_y'] * (68/80))
    df['DestX'] = df['end_location_x'] * (105/120)
    df['DestY'] = 68-(df['end_location_y'] * (68/80))
    df['xg'] = df.xg
    df['ifOP'] = [1 if x == 'Regular Play' else 0 for x in df['play_pattern_name']]
    df['ifCA'] = [1 if x == 'From Counter' else 0 for x in df['play_pattern_name']]
    df['ifSP'] = [1 if ((x=='From Corner')or(x=='From Free Kick')or(x=='From Throw In')) else 0 for x in df['play_pattern_name']]

    Shots = df[df.event_type_name == 'Shot']  
    
    Shots = df[((df.event_type_name == 'Shot') | (df.event_type_name == 'Own Goal For'))]
    Shots['ifGoal'] = [1 if x == 'Goal' else 0 for x in Shots['outcome_name']]
    Shots['ifOwnGoal'] = [1 if x == 'Own Goal For' else 0 for x in Shots['event_type_name']]
    
    Shots['Shots'] = 1
    Shots['OnT'] = [1 if ((x == 'Goal')or(x=='Saved')) else 0 for x in Shots['outcome_name']]
    Shots['ifPen'] = [1 if x == 'Penalty' else 0 for x in Shots['type_name']]



    team_name = st.sidebar.selectbox("Select Team", natsorted(Shots.team_name.unique()))    
    Shots = Shots[Shots['team_name'] == team_name]

    
    matches = (Shots.match_name.unique()).tolist()
    
    match_id = st.sidebar.multiselect("Select Match(es)", natsorted(Shots.match_name.unique()), default=matches)
    Shots = Shots[Shots['match_name'].isin(match_id)]
    

    
    team_nameHead = fm.FontProperties(fname=fontPathBold, size=66)
    GoalandxG = fm.FontProperties(fname=fontPathBold, size=48)
    Summary = fm.FontProperties(fname=fontPathNBold, size=36)
    TableHead = fm.FontProperties(fname=fontPathBold, size=34)
    TableNum = fm.FontProperties(fname=fontPathNBold, size=30)
    def TeamShotMap(data):
        figsize1 = 36
        figsize2 = 18
        fig = plt.figure(figsize=(figsize1, figsize2)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[2., 1.25], wspace=0.05, right=.85)
    
        ax1 = plt.subplot(gs[:, 0])
        verthalf_pitch('#E6E6E6', 'black', ax1)
        def team_name_shot_map(data):
             shot_data = Shots[(Shots['team_name'] == team_name) & (Shots['ifPen'] != 1)]
             #stat_data = Shots[(Shots['Season'] == Season) & (Shots['team_name'] == team_name) & (Shots['match_id'] == match_id)]
            
            
             Head = shot_data[(shot_data["body_part_name"] == 'Head') & (shot_data['follows_dribble'] != 1)]
             Foot = shot_data[(shot_data["body_part_name"] != 'Head') & (shot_data['follows_dribble'] != 1)]
             Carry = shot_data[(shot_data['follows_dribble'] == 1)]
             
             HGoal = Head[Head['outcome_name'] == "Goal"]
             HMissed = Head[Head['outcome_name'] == "Off T"]
             HSaved = Head[Head['outcome_name'] == "Saved"]
             HBlocked = Head[(Head['outcome_name'] == "Blocked")]
             HSave = Head[(Head['outcome_name'] == "Wayward")]
             HPost = Head[(Head)['outcome_name'] == "Post"]
             
             FGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] != 'Free Kick')]
             FMissed = Foot[(Foot['outcome_name'] == "Off T")&(Foot['type_name'] != 'Free Kick')]
             FSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] != 'Free Kick')]
             FBlocked = Foot[(Foot['outcome_name'] == "Blocked")&(Foot['type_name'] != 'Free Kick')]
             FSave = Foot[(Foot['outcome_name'] == "Wayward")&(Foot['type_name'] != 'Free Kick')]
             FPost = Foot[(Foot['outcome_name'] == "Post")&(Foot['type_name'] != 'Free Kick')]
             
             FKGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] == 'Free Kick')]
             FKMissed = Foot[(Foot['outcome_name'] == "Off T")&(Foot['type_name'] == 'Free Kick')]
             FKSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] == 'Free Kick')]
             FKBlocked = Foot[(Foot['outcome_name'] == "Blocked")&(Foot['type_name'] == 'Free Kick')]
             FKSave = Foot[(Foot['outcome_name'] == "Wayward")&(Foot['type_name'] == 'Free Kick')]
             FKPost = Foot[(Foot['outcome_name'] == "Post")&(Foot['type_name'] == 'Free Kick')]
             
             CGoal = Carry[(Carry['outcome_name'] == "Goal")&(Carry['type_name'] != 'Free Kick')]
             CMissed = Carry[(Carry['outcome_name'] == "Off T")&(Carry['type_name'] != 'Free Kick')]
             CSaved = Carry[(Carry['outcome_name'] == "Saved")&(Carry['type_name'] != 'Free Kick')]
             CBlocked = Carry[(Carry['outcome_name'] == "Blocked")&(Carry['type_name'] != 'Free Kick')]
             CSave = Carry[(Carry['outcome_name'] == "Wayward")&(Carry['type_name'] != 'Free Kick')]
             CPost = Carry[(Carry['outcome_name'] == "Post")&(Carry['type_name'] != 'Free Kick')]
    
    
             
             #draw_pitch("#B2B2B2","white","vertical","half")
             #plt.title(str(Season)+" - "+str(team_name)+ " - " +str(Player)+" \n"+str((sum(shot_data.ifGoal)))+" Goals" " - "+str(round(sum(shot_data.xg),2))+" xG \n "+str(sum(stat_data.PenGoal))+" Goals / "+str(sum(stat_data.ifPen))+" Penalties \n "+str((sum(stat_data.Shots)))+" Shots"" - "+str(round(sum(shot_data.xg/sum(shot_data.Shots)),2))+" xG/shot", fontsize=20, weight="bold")
             
             norm = TwoSlopeNorm(vmin=0,vcenter=.2,vmax=.7)
             if len(FGoal) > 0:
                plt.scatter(68-FGoal.Y,FGoal.X,
                marker='H',c=FGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
                plt.scatter(68-FGoal.Y,FGoal.X,marker='H',c="white",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(FMissed) > 0:
                plt.scatter(68-FMissed.Y,FMissed.X,marker='H',c=FMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FSaved) > 0:
                plt.scatter(68-FSaved.Y,FSaved.X,
                marker='H',c=FSaved.xg, s=500,linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FSave) > 0:
                plt.scatter(68-FSave.Y,FSave.X,
                marker='H',c=FSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FBlocked) > 0:
                plt.scatter(68-FBlocked.Y,FBlocked.X,marker='H',c=FBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-FBlocked.Y,FBlocked.X,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(FPost) > 0:
                plt.scatter(68-FPost.Y,FPost.X,marker='H',c=FPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CGoal) > 0:
                plt.scatter(68-CGoal.Y,CGoal.X,
                marker='^',c=CGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=1.5)
                plt.scatter(68-CGoal.Y,CGoal.X,marker='^',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(CMissed) > 0:
                plt.scatter(68-CMissed.Y,CMissed.X,marker='^',c=CMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CSaved) > 0:
                plt.scatter(68-CSaved.Y,CSaved.X,
                marker='^',c=CSaved.xg, s=500,linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CSave) > 0:
                plt.scatter(68-CSave.Y,CSave.X,
                marker='^',c=CSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CBlocked) > 0:
                plt.scatter(68-CBlocked.Y,CBlocked.X,marker='^',c=CBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-CBlocked.Y,CBlocked.X,marker='^',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(CPost) > 0:
                plt.scatter(68-CPost.Y,CPost.X,marker='^',c=CPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HGoal) > 0:
                plt.scatter(68-HGoal.Y,HGoal.X,
                marker='o',c=HGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
                plt.scatter(68-HGoal.Y,HGoal.X,marker='o',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(HMissed) > 0:
                plt.scatter(68-HMissed.Y,HMissed.X,marker='o',c=HMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HSaved) > 0:
                plt.scatter(68-HSaved.Y,HSaved.X,
                marker='o',c=HSaved.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r',  norm=norm)
             if len(HSave) > 0:
                plt.scatter(68-HSave.Y,HSave.X,
                marker='o',c=HSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HBlocked) > 0:
                plt.scatter(68-HBlocked.Y,HBlocked.X,marker='H',c=HBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-HBlocked.Y,HBlocked.X,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(HPost) > 0:
                plt.scatter(68-HPost.Y,HPost.X,marker='H',c=HPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKGoal) > 0:
                plt.scatter(68-FKGoal.Y,FKGoal.X,
                marker='s',c=FKGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
                plt.scatter(68-FKGoal.Y,FKGoal.X,marker='s',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(FKMissed) > 0:
                plt.scatter(68-FKMissed.Y,FKMissed.X,marker='s',c=FKMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKSaved) > 0:
                plt.scatter(68-FKSaved.Y,FKSaved.X,
                marker='s',c=FKSaved.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKSave) > 0:
                plt.scatter(68-FKSave.Y,FKSave.X, 
                marker='s',c=FKSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKBlocked) > 0:
                plt.scatter(68-FKBlocked.Y,FKBlocked.X,marker='s',c=FKBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r',  alpha=.15, norm=norm)
                plt.scatter(68-FKBlocked.Y,FKBlocked.X,marker='s',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(FKPost) > 0:
                plt.scatter(68-FKPost.Y,FKPost.X,marker='s',c=FKPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', vmin=0, vmax=.7, norm=norm)
               
             plt.scatter(3.5,45, marker='H', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(11, 45, marker='o', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(18.5, 45, marker='^', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(26, 45, marker='s', facecolor="white", edgecolors="black", s=500, zorder=12)
             ax1.text(3.5,42.5,"Foot",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(11,42.5,"Header",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(18.5,42.5,"Carry",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(26,42.5,"FK",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
            
             plt.scatter(45,45,marker='H',c='white', s=500,
                    edgecolors="black",zorder=zo+2, linewidth=.5)
             plt.scatter(45,45,marker='H',c="white",s=750,
                    edgecolors="black",zorder=zo+1, linewidth=.5)
             plt.scatter(50.5,45, marker='H', c='white', s=500, linewidths=2,
                edgecolors="black",zorder=zo)
             plt.scatter(56,45,marker='H', facecolors="none", s=500,
                edgecolors="black",zorder=zo, alpha=.15)
             plt.scatter(56,45,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15)
             plt.scatter(61.5,45,marker='H',c='white', facecolors="none", s=500,
                edgecolors="black", linewidths=.25, zorder=zo)
             ax1.text(45,42.5,"Goal",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(50.5,42.5,"Save",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(56,42.5,"Block",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(61.5,42.5,"OffT",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             plt.ylim(40,105.5)
             plt.xlim(-.5,68.5)
        
             #a = plt.scatter(-10,-10, marker='H', facecolor="white", edgecolors="black", s=500)
             #b = plt.scatter(-10, -10, marker='o', facecolor="white", edgecolors="black", s=500)
             #c = plt.scatter(-10, -10, marker='^', facecolor="white", edgecolors="black", s=500)
             #plt.legend((a, b, c),("Foot", "Header", "Direct Free Kick"), loc='lower left', title="Types of Shots")
             #plt.annotate('Double Ring = Goal', xy=(18, 47), size = 12, color="black",ha="center")
             #plt.annotate('Black Edge = On Target', xy=(18, 45), size = 12, color="black",ha="center")
             #plt.annotate('No Edge = Off Target', xy=(18, 43), size = 12, color="black",ha="center")
             #plt.annotate('Gray Fill = Blocked', xy=(18, 41), size = 12, color="black",ha="center")
    
        team_name_shot_map(data)
            
        ax2 = plt.subplot(gs[0, 1])
        def GKMap(data, ax):
            df = Shots[ (Shots['team_name'] == team_name) & (Shots['ifPen'] != 1) & (Shots['OnT'] == 1)]
            
            ly1 = [-0.1,-0.1,2.67,2.67,-0.1]
            lx1 = [36,44,44,36,36]
            ax.plot(lx1,ly1,color='black',zorder=5, lw=6)
            ax.plot([36,44], [-0.1,-0.1], color='white', zorder=6, lw=8)
            ax.axis('off')
            
            
            Head = df[(df["body_part_name"] == 'Head') & (df['follows_dribble'] != 1)]
            Foot = df[(df["body_part_name"] != 'Head') & (df['follows_dribble'] != 1)]
            Carry = df[(df['follows_dribble'] == 1) ]
              
            HGoal = Head[Head['outcome_name'] == "Goal"]
            HSaved = Head[Head['outcome_name'] == "Saved"]
            CGoal = Carry[Carry['outcome_name'] == "Goal"]
            CSaved = Carry[Carry['outcome_name'] == "Saved"]
            FGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] != 'Free Kick')]
            FSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] != 'Free Kick')]
            FKGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] == 'Free Kick')]
            FKSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] == 'Free Kick')]
            
            norm = TwoSlopeNorm(vmin=0,vcenter=.2,vmax=.7)
            if len(FGoal) > 0:
               plt.scatter(FGoal.goal_location_x,FGoal.goal_location_y,
               marker='H',c=FGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
               plt.scatter(FGoal.goal_location_x,FGoal.goal_location_y,marker='H',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(FSaved) > 0:
               plt.scatter(FSaved.goal_location_x,FSaved.goal_location_y,
               marker='H',c=FSaved.xg, s=500,linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
            if len(CGoal) > 0:
               plt.scatter(CGoal.goal_location_x,CGoal.goal_location_y,
               marker='^',c=CGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
               plt.scatter(CGoal.goal_location_x,CGoal.goal_location_y,marker='^',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(CSaved) > 0:
               plt.scatter(CSaved.goal_location_x,CSaved.goal_location_y,
               marker='^',c=CSaved.xg, s=500,linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
            if len(HGoal) > 0:
               plt.scatter(HGoal.goal_location_x,HGoal.goal_location_y,
               marker='o',c=HGoal.xg, s=500,
               edgecolors="white",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
               plt.scatter(HGoal.goal_location_x,HGoal.goal_location_y,marker='o',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(HSaved) > 0:
               plt.scatter(HSaved.goal_location_x,HSaved.goal_location_y,
               marker='o',c=HSaved.xg, s=500, linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r',  norm=norm)
            if len(FKGoal) > 0:
               plt.scatter(FKGoal.goal_location_x,FKGoal.goal_location_y,
               marker='s',c=FKGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
               plt.scatter(FKGoal.goal_location_x,FKGoal.goal_location_y,marker='s',facecolors="none",
               s=500,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(FKSaved) > 0:
               plt.scatter(FKSaved.goal_location_x,FKSaved.goal_location_y,
               marker='s',c=FKSaved.xg, s=500, linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
        plt.xlim(36,44)
               
        GKMap(data, ax2)
        
        
        
        shot_data = Shots[ (Shots['team_name'] == team_name) & (Shots['ifPen'] != 1)]
        pen_data = Shots[(Shots['team_name'] == team_name) & (Shots['ifPen'] == 1)]
        OP_data = shot_data[(shot_data['team_name'] == team_name) & (shot_data['ifOP'] == 1)]
        CO_data = shot_data[(shot_data['team_name'] == team_name) & (shot_data['ifCA'] == 1)]
        sog_data = Shots[(Shots['team_name'] == team_name) & (Shots['ifPen'] != 1) ]
        #sOP_data = sog_data[(sog_data['team_name'] == team_name) & (sog_data['ifOP'] == 1)]
        #sCO_data = sog_data[(sog_data['team_name'] == team_name) & (sog_data['ifCA'] == 1)]
        SP_data = sog_data[(sog_data['team_name'] == team_name) & ((sog_data['ifOP'] != 1)&(sog_data['ifCA'] != 1))]        
        #SP_data = shot_data[(shot_data['TypeofPossession'] == 'Free-kick attack')or(shot_data['TypeofPossession'] == 'Throw-in attack')or(shot_data['TypeofPossession'] == 'Corner attack')]
        
    
        #ax3.text(0.10,0.85,"Open Play",fontsize=28, color='white', weight='bold', family='fantasy',
         #         horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.425,0.85,"Counter Attack",fontsize=28, color='white', weight='bold', family='fantasy',
         #        horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.750,0.85,"Set Pieces",fontsize=28, color='white', weight='bold', family='fantasy',
         #        horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.10,0.725,str(round(sum(OP_data.xg),2))+' xG\n'+str(round(sum(OP_data.ifGoal),))+' G - '+str(round(sum(OP_data.Shots),))+' S',fontsize=22, color='black', weight='bold', family='fantasy',
         #         horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', lw=3.5))
    
        
        ax3 = plt.subplot(gs[1, 1])
        ax3.axis('off')
        ax3.set_xticks([0,1])
        ax3.set_yticks([0,1])
        ax3.scatter(0,0, alpha=0)
        ax3.scatter(1,1,alpha=0)
    
        ax3.text(0.475,0.65,"xG",fontproperties=TableHead, color='white',
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.10,0.55,"Open Play",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.475,0.55,"Counter Attack",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.85,0.55,"Set Pieces",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.10,0.37,str(round(sum(OP_data.xg),2))+' xG\n'+str(round(sum(OP_data.ifGoal),))+' G - '+str(round(sum(OP_data.Shots),))+' S\n'+str(round(sum(OP_data.xg/sum(OP_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black', horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.475,0.37,str(round(sum(CO_data.xg),2))+' xG\n'+str(round(sum(CO_data.ifGoal),))+' G - '+str(round(sum(CO_data.Shots),))+' S\n'+str(round(sum(CO_data.xg/sum(CO_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black',  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.85,0.37,str(round(sum(SP_data.xg),2))+' xG\n'+str(round(sum(SP_data.ifGoal),))+' G - '+str(round(sum(SP_data.Shots),))+' S\n'+str(round(sum(SP_data.xg/sum(SP_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black', horizontalalignment='center', verticalalignment='center', zorder=12)
        
        rec1 = plt.Rectangle((-0.1, .5),1.1,.1,ls='-',color='black',zorder=6,alpha=1)
        rec2 = plt.Rectangle((-0.1, .6),1.1,.1,ls='-',color='black',zorder=6,alpha=1)
        rec3 = plt.Rectangle((-0.1, 0.2),1.1,.4,ls='-',color='white',zorder=5,alpha=.5)
        ax3.add_artist(rec1)
        ax3.add_artist(rec2)
        ax3.add_artist(rec3)
    
        
        cax = plt.axes([0.15, 0.085, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=0,vcenter=.2, vmax=.7))
        sm.A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.set_label('xG', fontproperties=TableHead)
        cbar.set_ticks([0, .03, .1, .2, .3, .4, .5, .6, .7])
        cbar.ax.set_xticklabels([0, .03, .1, .2, .3, .4, .5, .6, .7])
        cbar.ax.tick_params(labelsize=20)
        
        fig.text(0.125,0.885,str(team_name), fontproperties=team_nameHead, color='black')
        
        fig.text(0.525,0.8925,str(round(sum(shot_data.ifGoal),))+" Goals - "+str(round(sum(shot_data.xg),2))+' xG',
                 fontproperties=GoalandxG, color='black')
        fig.text(0.7,0.905,str(round(sum(pen_data.xg),2))+' xG - '+str(round(sum(pen_data.ifGoal),))+" Goals - "+str(round(sum(pen_data.ifPen),))+" Penalties",fontproperties=Summary, color='black')
        fig.text(0.7,0.88,str(round(sum(shot_data.Shots),))+'|'+str(round(sum(shot_data.OnT),))+' S|OnT - '+str(round(sum(shot_data.xg/sum(shot_data.Shots)),2))+" xGpShot",fontproperties=Summary,color='black')
        st.pyplot(fig)
    TeamShotMap(Shots)
    
def TeamShotD(state):
    df = load_data()
    mdf = load_match_data()
    cols = ['home_team', 'away_team']
    mdf['match_name'] = mdf[cols].apply(lambda row: ' - '.join(row.values.astype(str)), axis=1)
    mdf = pd.DataFrame(data=mdf, columns=['match_id', 'home_team', 'away_team', 'match_name'])
    df = pd.merge(df,mdf, on=['match_id'])
    
    
    df['X'] = df['location_x'] * (105/120)
    df['Y'] = 68-(df['location_y'] * (68/80))
    df['DestX'] = df['end_location_x'] * (105/120)
    df['DestY'] = 68-(df['end_location_y'] * (68/80))
    df['xg'] = df.xg
    df['ifOP'] = [1 if x == 'Regular Play' else 0 for x in df['play_pattern_name']]
    df['ifCA'] = [1 if x == 'From Counter' else 0 for x in df['play_pattern_name']]
    df['ifSP'] = [1 if ((x=='From Corner')or(x=='From Free Kick')or(x=='From Throw In')) else 0 for x in df['play_pattern_name']]

    Shots = df[df.event_type_name == 'Shot']  
    
    Shots = df[((df.event_type_name == 'Shot') | (df.event_type_name == 'Own Goal For'))]
    Shots['ifGoal'] = [1 if x == 'Goal' else 0 for x in Shots['outcome_name']]
    Shots['ifOwnGoal'] = [1 if x == 'Own Goal For' else 0 for x in Shots['event_type_name']]
    
    Shots['Shots'] = 1
    Shots['OnT'] = [1 if ((x == 'Goal')or(x=='Saved')) else 0 for x in Shots['outcome_name']]
    Shots['ifPen'] = [1 if x == 'Penalty' else 0 for x in Shots['type_name']]



    team_name = st.sidebar.selectbox("Select Team", natsorted(Shots.team_name.unique()))    
    Shots = Shots[((Shots['home_team'] == team_name) | (Shots['away_team'] == team_name))]
    
    Shots = Shots[Shots['team_name'] != team_name]

    
    matches = (Shots.match_name.unique()).tolist()
    
    match_id = st.sidebar.multiselect("Select Match(es)", natsorted(Shots.match_name.unique()), default=matches)
    Shots = Shots[Shots['match_name'].isin(match_id)]
    

    
    team_nameHead = fm.FontProperties(fname=fontPathBold, size=66)
    GoalandxG = fm.FontProperties(fname=fontPathBold, size=48)
    Summary = fm.FontProperties(fname=fontPathNBold, size=36)
    TableHead = fm.FontProperties(fname=fontPathBold, size=34)
    TableNum = fm.FontProperties(fname=fontPathNBold, size=30)
    def TeamShotMap(data):
        figsize1 = 36
        figsize2 = 18
        fig = plt.figure(figsize=(figsize1, figsize2)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[2., 1.25], wspace=0.05, right=.85)
    
        ax1 = plt.subplot(gs[:, 0])
        verthalf_pitch('#E6E6E6', 'black', ax1)
        def team_name_shot_map(data):
             shot_data = Shots[ (Shots['ifPen'] != 1)]
             #stat_data = Shots[(Shots['Season'] == Season) & (Shots['team_name'] == team_name) & (Shots['match_id'] == match_id)]
            
            
             Head = shot_data[(shot_data["body_part_name"] == 'Head') & (shot_data['follows_dribble'] != 1)]
             Foot = shot_data[(shot_data["body_part_name"] != 'Head') & (shot_data['follows_dribble'] != 1)]
             Carry = shot_data[(shot_data['follows_dribble'] == 1)]
             
             HGoal = Head[Head['outcome_name'] == "Goal"]
             HMissed = Head[Head['outcome_name'] == "Off T"]
             HSaved = Head[Head['outcome_name'] == "Saved"]
             HBlocked = Head[(Head['outcome_name'] == "Blocked")]
             HSave = Head[(Head['outcome_name'] == "Wayward")]
             HPost = Head[(Head)['outcome_name'] == "Post"]
             
             FGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] != 'Free Kick')]
             FMissed = Foot[(Foot['outcome_name'] == "Off T")&(Foot['type_name'] != 'Free Kick')]
             FSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] != 'Free Kick')]
             FBlocked = Foot[(Foot['outcome_name'] == "Blocked")&(Foot['type_name'] != 'Free Kick')]
             FSave = Foot[(Foot['outcome_name'] == "Wayward")&(Foot['type_name'] != 'Free Kick')]
             FPost = Foot[(Foot['outcome_name'] == "Post")&(Foot['type_name'] != 'Free Kick')]
             
             FKGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] == 'Free Kick')]
             FKMissed = Foot[(Foot['outcome_name'] == "Off T")&(Foot['type_name'] == 'Free Kick')]
             FKSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] == 'Free Kick')]
             FKBlocked = Foot[(Foot['outcome_name'] == "Blocked")&(Foot['type_name'] == 'Free Kick')]
             FKSave = Foot[(Foot['outcome_name'] == "Wayward")&(Foot['type_name'] == 'Free Kick')]
             FKPost = Foot[(Foot['outcome_name'] == "Post")&(Foot['type_name'] == 'Free Kick')]
             
             CGoal = Carry[(Carry['outcome_name'] == "Goal")&(Carry['type_name'] != 'Free Kick')]
             CMissed = Carry[(Carry['outcome_name'] == "Off T")&(Carry['type_name'] != 'Free Kick')]
             CSaved = Carry[(Carry['outcome_name'] == "Saved")&(Carry['type_name'] != 'Free Kick')]
             CBlocked = Carry[(Carry['outcome_name'] == "Blocked")&(Carry['type_name'] != 'Free Kick')]
             CSave = Carry[(Carry['outcome_name'] == "Wayward")&(Carry['type_name'] != 'Free Kick')]
             CPost = Carry[(Carry['outcome_name'] == "Post")&(Carry['type_name'] != 'Free Kick')]
    
    
             
             #draw_pitch("#B2B2B2","white","vertical","half")
             #plt.title(str(Season)+" - "+str(team_name)+ " - " +str(Player)+" \n"+str((sum(shot_data.ifGoal)))+" Goals" " - "+str(round(sum(shot_data.xg),2))+" xG \n "+str(sum(stat_data.PenGoal))+" Goals / "+str(sum(stat_data.ifPen))+" Penalties \n "+str((sum(stat_data.Shots)))+" Shots"" - "+str(round(sum(shot_data.xg/sum(shot_data.Shots)),2))+" xG/shot", fontsize=20, weight="bold")
             
             norm = TwoSlopeNorm(vmin=0,vcenter=.2,vmax=.7)
             if len(FGoal) > 0:
                plt.scatter(68-FGoal.Y,FGoal.X,
                marker='H',c=FGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
                plt.scatter(68-FGoal.Y,FGoal.X,marker='H',c="white",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(FMissed) > 0:
                plt.scatter(68-FMissed.Y,FMissed.X,marker='H',c=FMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FSaved) > 0:
                plt.scatter(68-FSaved.Y,FSaved.X,
                marker='H',c=FSaved.xg, s=500,linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FSave) > 0:
                plt.scatter(68-FSave.Y,FSave.X,
                marker='H',c=FSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FBlocked) > 0:
                plt.scatter(68-FBlocked.Y,FBlocked.X,marker='H',c=FBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-FBlocked.Y,FBlocked.X,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(FPost) > 0:
                plt.scatter(68-FPost.Y,FPost.X,marker='H',c=FPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CGoal) > 0:
                plt.scatter(68-CGoal.Y,CGoal.X,
                marker='^',c=CGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=1.5)
                plt.scatter(68-CGoal.Y,CGoal.X,marker='^',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(CMissed) > 0:
                plt.scatter(68-CMissed.Y,CMissed.X,marker='^',c=CMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CSaved) > 0:
                plt.scatter(68-CSaved.Y,CSaved.X,
                marker='^',c=CSaved.xg, s=500,linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CSave) > 0:
                plt.scatter(68-CSave.Y,CSave.X,
                marker='^',c=CSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(CBlocked) > 0:
                plt.scatter(68-CBlocked.Y,CBlocked.X,marker='^',c=CBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-CBlocked.Y,CBlocked.X,marker='^',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(CPost) > 0:
                plt.scatter(68-CPost.Y,CPost.X,marker='^',c=CPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HGoal) > 0:
                plt.scatter(68-HGoal.Y,HGoal.X,
                marker='o',c=HGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
                plt.scatter(68-HGoal.Y,HGoal.X,marker='o',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(HMissed) > 0:
                plt.scatter(68-HMissed.Y,HMissed.X,marker='o',c=HMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HSaved) > 0:
                plt.scatter(68-HSaved.Y,HSaved.X,
                marker='o',c=HSaved.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r',  norm=norm)
             if len(HSave) > 0:
                plt.scatter(68-HSave.Y,HSave.X,
                marker='o',c=HSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(HBlocked) > 0:
                plt.scatter(68-HBlocked.Y,HBlocked.X,marker='H',c=HBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', alpha=.15, norm=norm)
                plt.scatter(68-HBlocked.Y,HBlocked.X,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(HPost) > 0:
                plt.scatter(68-HPost.Y,HPost.X,marker='H',c=HPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKGoal) > 0:
                plt.scatter(68-FKGoal.Y,FKGoal.X,
                marker='s',c=FKGoal.xg, s=500,
                edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
                plt.scatter(68-FKGoal.Y,FKGoal.X,marker='s',facecolors="none",
                s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
             if len(FKMissed) > 0:
                plt.scatter(68-FKMissed.Y,FKMissed.X,marker='s',c=FKMissed.xg, facecolors="none", s=500,
                edgecolors="none",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKSaved) > 0:
                plt.scatter(68-FKSaved.Y,FKSaved.X,
                marker='s',c=FKSaved.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKSave) > 0:
                plt.scatter(68-FKSave.Y,FKSave.X, 
                marker='s',c=FKSave.xg, s=500, linewidths=2,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
             if len(FKBlocked) > 0:
                plt.scatter(68-FKBlocked.Y,FKBlocked.X,marker='s',c=FKBlocked.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r',  alpha=.15, norm=norm)
                plt.scatter(68-FKBlocked.Y,FKBlocked.X,marker='s',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15, norm=norm)
             if len(FKPost) > 0:
                plt.scatter(68-FKPost.Y,FKPost.X,marker='s',c=FKPost.xg, facecolors="none", s=500,
                edgecolors="black",zorder=zo, cmap='RdYlBu_r', vmin=0, vmax=.7, norm=norm)
               
             plt.scatter(3.5,45, marker='H', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(11, 45, marker='o', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(18.5, 45, marker='^', facecolor="white", edgecolors="black", s=500, zorder=12)
             plt.scatter(26, 45, marker='s', facecolor="white", edgecolors="black", s=500, zorder=12)
             ax1.text(3.5,42.5,"Foot",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(11,42.5,"Header",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(18.5,42.5,"Carry",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(26,42.5,"FK",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
            
             plt.scatter(45,45,marker='H',c='white', s=500,
                    edgecolors="black",zorder=zo+2, linewidth=.5)
             plt.scatter(45,45,marker='H',c="white",s=750,
                    edgecolors="black",zorder=zo+1, linewidth=.5)
             plt.scatter(50.5,45, marker='H', c='white', s=500, linewidths=2,
                edgecolors="black",zorder=zo)
             plt.scatter(56,45,marker='H', facecolors="none", s=500,
                edgecolors="black",zorder=zo, alpha=.15)
             plt.scatter(56,45,marker='H',facecolors="gray",
                s=500,edgecolors="black",zorder=zo+1, alpha=.15)
             plt.scatter(61.5,45,marker='H',c='white', facecolors="none", s=500,
                edgecolors="black", linewidths=.25, zorder=zo)
             ax1.text(45,42.5,"Goal",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(50.5,42.5,"Save",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(56,42.5,"Block",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             ax1.text(61.5,42.5,"OffT",fontsize=22, color='black', weight='bold', family='fantasy',
                      horizontalalignment='center', verticalalignment='center', zorder=12)
             plt.ylim(40,105.5)
             plt.xlim(-.5,68.5)
        
             #a = plt.scatter(-10,-10, marker='H', facecolor="white", edgecolors="black", s=500)
             #b = plt.scatter(-10, -10, marker='o', facecolor="white", edgecolors="black", s=500)
             #c = plt.scatter(-10, -10, marker='^', facecolor="white", edgecolors="black", s=500)
             #plt.legend((a, b, c),("Foot", "Header", "Direct Free Kick"), loc='lower left', title="Types of Shots")
             #plt.annotate('Double Ring = Goal', xy=(18, 47), size = 12, color="black",ha="center")
             #plt.annotate('Black Edge = On Target', xy=(18, 45), size = 12, color="black",ha="center")
             #plt.annotate('No Edge = Off Target', xy=(18, 43), size = 12, color="black",ha="center")
             #plt.annotate('Gray Fill = Blocked', xy=(18, 41), size = 12, color="black",ha="center")
    
        team_name_shot_map(data)
            
        ax2 = plt.subplot(gs[0, 1])
        def GKMap(data, ax):
            df = Shots[(Shots['ifPen'] != 1) & (Shots['OnT'] == 1)]
            
            ly1 = [-0.1,-0.1,2.67,2.67,-0.1]
            lx1 = [36,44,44,36,36]
            ax.plot(lx1,ly1,color='black',zorder=5, lw=6)
            ax.plot([36,44], [-0.1,-0.1], color='white', zorder=6, lw=8)
            ax.axis('off')
            
            
            Head = df[(df["body_part_name"] == 'Head') & (df['follows_dribble'] != 1)]
            Foot = df[(df["body_part_name"] != 'Head') & (df['follows_dribble'] != 1)]
            Carry = df[(df['follows_dribble'] == 1) ]
              
            HGoal = Head[Head['outcome_name'] == "Goal"]
            HSaved = Head[Head['outcome_name'] == "Saved"]
            CGoal = Carry[Carry['outcome_name'] == "Goal"]
            CSaved = Carry[Carry['outcome_name'] == "Saved"]
            FGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] != 'Free Kick')]
            FSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] != 'Free Kick')]
            FKGoal = Foot[(Foot['outcome_name'] == "Goal")&(Foot['type_name'] == 'Free Kick')]
            FKSaved = Foot[(Foot['outcome_name'] == "Saved")&(Foot['type_name'] == 'Free Kick')]
            
            norm = TwoSlopeNorm(vmin=0,vcenter=.2,vmax=.7)
            if len(FGoal) > 0:
               plt.scatter(FGoal.goal_location_x,FGoal.goal_location_y,
               marker='H',c=FGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
               plt.scatter(FGoal.goal_location_x,FGoal.goal_location_y,marker='H',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(FSaved) > 0:
               plt.scatter(FSaved.goal_location_x,FSaved.goal_location_y,
               marker='H',c=FSaved.xg, s=500,linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
            if len(CGoal) > 0:
               plt.scatter(CGoal.goal_location_x,CGoal.goal_location_y,
               marker='^',c=CGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', norm=norm, linewidth=.5)
               plt.scatter(CGoal.goal_location_x,CGoal.goal_location_y,marker='^',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(CSaved) > 0:
               plt.scatter(CSaved.goal_location_x,CSaved.goal_location_y,
               marker='^',c=CSaved.xg, s=500,linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
            if len(HGoal) > 0:
               plt.scatter(HGoal.goal_location_x,HGoal.goal_location_y,
               marker='o',c=HGoal.xg, s=500,
               edgecolors="white",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
               plt.scatter(HGoal.goal_location_x,HGoal.goal_location_y,marker='o',facecolors="none",
               s=750,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(HSaved) > 0:
               plt.scatter(HSaved.goal_location_x,HSaved.goal_location_y,
               marker='o',c=HSaved.xg, s=500, linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r',  norm=norm)
            if len(FKGoal) > 0:
               plt.scatter(FKGoal.goal_location_x,FKGoal.goal_location_y,
               marker='s',c=FKGoal.xg, s=500,
               edgecolors="black",zorder=zo+2, cmap='RdYlBu_r', linewidth=.5, norm=norm)
               plt.scatter(FKGoal.goal_location_x,FKGoal.goal_location_y,marker='s',facecolors="none",
               s=500,edgecolors="black",zorder=zo+1, linewidth=.5, norm=norm)
            if len(FKSaved) > 0:
               plt.scatter(FKSaved.goal_location_x,FKSaved.goal_location_y,
               marker='s',c=FKSaved.xg, s=500, linewidths=2,
               edgecolors="black",zorder=zo, cmap='RdYlBu_r', norm=norm)
        plt.xlim(36,44)
               
        GKMap(data, ax2)
        
        
        
        shot_data = Shots[  (Shots['ifPen'] != 1)]
        pen_data = Shots[(Shots['ifPen'] == 1)]
        OP_data = shot_data[ (shot_data['ifOP'] == 1)]
        CO_data = shot_data[ (shot_data['ifCA'] == 1)]
        sog_data = Shots[(Shots['ifPen'] != 1) ]
        #sOP_data = sog_data[(sog_data['team_name'] == team_name) & (sog_data['ifOP'] == 1)]
        #sCO_data = sog_data[(sog_data['team_name'] == team_name) & (sog_data['ifCA'] == 1)]
        SP_data = sog_data[((sog_data['ifOP'] != 1)&(sog_data['ifCA'] != 1))]        
        #SP_data = shot_data[(shot_data['TypeofPossession'] == 'Free-kick attack')or(shot_data['TypeofPossession'] == 'Throw-in attack')or(shot_data['TypeofPossession'] == 'Corner attack')]
        
    
        #ax3.text(0.10,0.85,"Open Play",fontsize=28, color='white', weight='bold', family='fantasy',
         #         horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.425,0.85,"Counter Attack",fontsize=28, color='white', weight='bold', family='fantasy',
         #        horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.750,0.85,"Set Pieces",fontsize=28, color='white', weight='bold', family='fantasy',
         #        horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='white', lw=3.5))
        #ax3.text(0.10,0.725,str(round(sum(OP_data.xg),2))+' xG\n'+str(round(sum(OP_data.ifGoal),))+' G - '+str(round(sum(OP_data.Shots),))+' S',fontsize=22, color='black', weight='bold', family='fantasy',
         #         horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', lw=3.5))
    
        
        ax3 = plt.subplot(gs[1, 1])
        ax3.axis('off')
        ax3.set_xticks([0,1])
        ax3.set_yticks([0,1])
        ax3.scatter(0,0, alpha=0)
        ax3.scatter(1,1,alpha=0)
    
        ax3.text(0.475,0.65,"xG",fontproperties=TableHead, color='white',
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.10,0.55,"Open Play",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.475,0.55,"Counter Attack",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.85,0.55,"Set Pieces",fontproperties=TableHead, color='white', 
                  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.10,0.37,str(round(sum(OP_data.xg),2))+' xG\n'+str(round(sum(OP_data.ifGoal),))+' G - '+str(round(sum(OP_data.Shots),))+' S\n'+str(round(sum(OP_data.xg/sum(OP_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black', horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.475,0.37,str(round(sum(CO_data.xg),2))+' xG\n'+str(round(sum(CO_data.ifGoal),))+' G - '+str(round(sum(CO_data.Shots),))+' S\n'+str(round(sum(CO_data.xg/sum(CO_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black',  horizontalalignment='center', verticalalignment='center', zorder=12)
        ax3.text(0.85,0.37,str(round(sum(SP_data.xg),2))+' xG\n'+str(round(sum(SP_data.ifGoal),))+' G - '+str(round(sum(SP_data.Shots),))+' S\n'+str(round(sum(SP_data.xg/sum(SP_data.Shots)),2))+" xGpShot",
                 fontproperties=TableNum, color='black', horizontalalignment='center', verticalalignment='center', zorder=12)
        
        rec1 = plt.Rectangle((-0.1, .5),1.1,.1,ls='-',color='black',zorder=6,alpha=1)
        rec2 = plt.Rectangle((-0.1, .6),1.1,.1,ls='-',color='black',zorder=6,alpha=1)
        rec3 = plt.Rectangle((-0.1, 0.2),1.1,.4,ls='-',color='white',zorder=5,alpha=.5)
        ax3.add_artist(rec1)
        ax3.add_artist(rec2)
        ax3.add_artist(rec3)
    
        
        cax = plt.axes([0.15, 0.085, 0.7, 0.025])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=0,vcenter=.2, vmax=.7))
        sm.A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.set_label('xG', fontproperties=TableHead)
        cbar.set_ticks([0, .03, .1, .2, .3, .4, .5, .6, .7])
        cbar.ax.set_xticklabels([0, .03, .1, .2, .3, .4, .5, .6, .7])
        cbar.ax.tick_params(labelsize=20)
        
        fig.text(0.125,0.885,str(team_name), fontproperties=team_nameHead, color='black')
        
        fig.text(0.525,0.8925,str(round(sum(shot_data.ifGoal),))+" Goals - "+str(round(sum(shot_data.xg),2))+' xG',
                 fontproperties=GoalandxG, color='black')
        fig.text(0.7,0.905,str(round(sum(pen_data.xg),2))+' xG - '+str(round(sum(pen_data.ifGoal),))+" Goals - "+str(round(sum(pen_data.ifPen),))+" Penalties",fontproperties=Summary, color='black')
        fig.text(0.7,0.88,str(round(sum(shot_data.Shots),))+'|'+str(round(sum(shot_data.OnT),))+' S|OnT - '+str(round(sum(shot_data.xg/sum(shot_data.Shots)),2))+" xGpShot",fontproperties=Summary,color='black')
        st.pyplot(fig)
    TeamShotMap(Shots)


if __name__ == "__main__":
    main()

#st.set_option('server.enableCORS', True)
# to run : streamlit run "/Users/michael/Documents/Python/Codes/Dash App.py"



