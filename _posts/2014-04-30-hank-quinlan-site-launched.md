---
layout: post
title: Clemens Schafer
date: 2014-04-30
---

#Installing SQLite on MacOS

##Open the Terminal

First of all, we need to open the MacOS terminal and then navigate to our working directory. The easiest way to open the terminal is to press
<kbd>cmd</kbd> + <kbd>space</kbd> (simultaneously, and a Spotlight search field where you can type into should open), type "Terminal" and when the the Terminal application is selected press <kbd>Enter</kbd>. Alternatively, you can find the Terminal in your Applications folder and just double click it there. In order to navigate to our working dir, we first need to find out what the path of our working directory is (those cryptic paths you have used in python before). 

$$u_{n+1} = u_n - \frac{\Delta_t}{\tau_m} (u_n - u_{res}) + \frac{\Delta_t\Delta_T}{\tau_m} \exp{(\frac{u_n- \vartheta_{RH}}{\Delta_T})}- \frac{\Delta_t}{\tau_m}Rw_n +  \frac{\Delta_t}{\tau_m} RI(t)$$

$$w_{n+1} = w_n + \frac{a\Delta_t}{\tau_w} (w_n-u_{res}) - \frac{\Delta_t}{\tau_w}w+ \Delta_t b \sum_{t^f} \delta(t-t^f)$$

To do so you can either use the Finder and navigate to your prefered working directory (then right click and figure out via properties what the path is) or you can navigate in the Terminal with the cd command to your preferred working dir. In case you figured it out through the finder you should have something like "/Users/clemens/Documents". Then type into the Terminal:

~~~
cd /Users/clemens/Documents
~~~

This will change the directory (cd) to the path you gave it.

##Setup a Virtual Environment
It is highly recommended to work with virtual environments when working with python packages (for more details see [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)). Given we are in our working directory we can set up a virtual environment very easy by executing the following commands in the Terminal:

~~~
conda create -n SP19CSE10102 python=3.5 anaconda
~~~

This command may take a while (in case it asks you to proceed, just type <kbd>y</kbd> and press <kbd>Enter</kbd>) but when it successfully ran one time we do not need to execute it again, because our virtual environment with the name SP19CSE10102 is created now. Now we have to activate our virtual environment to work in it. This is done by the following command:

~~~
source activate SP19CSE10102
~~~

In case you are done working in your environment or you want to change to another environment you need to exit the environment, this is done with the following command:

~~~
source deactivate
~~~

##Install SQLite
Given that we are in our python environment and in the right directory we can now install SQLite which we will need for the course, this is done by the following command:

~~~
conda install -n SP19CSE10102 sqlite
~~~

(in case it asks you to proceed, just type <kbd>y</kbd> and press <kbd>Enter</kbd>)


##Rescources

- [https://anaconda.org/anaconda/sqlite](https://anaconda.org/anaconda/sqlite)
- [https://www.sqlite.org/index.html](https://www.sqlite.org/index.html)
- [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)


