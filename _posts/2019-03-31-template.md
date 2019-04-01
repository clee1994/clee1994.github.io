---
layout: post
title: Clemens Schafer
date: 2014-04-30
---

# Installing SQLite on MacOS

## Open the Terminal

First of all, we need to open the MacOS terminal and then navigate to our working directory. The easiest way to open the terminal is to press
<kbd>cmd</kbd> + <kbd>space</kbd> (simultaneously, and a Spotlight search field where you can type into should open), type "Terminal" and when the the Terminal application is selected press <kbd>Enter</kbd>. Alternatively, you can find the Terminal in your Applications folder and just double click it there. In order to navigate to our working dir, we first need to find out what the path of our working directory is (those cryptic paths you have used in python before). 

$$u_{n+1} = u_n - \frac{\Delta_t}{\tau_m} (u_n - u_{res}) + \frac{\Delta_t\Delta_T}{\tau_m} \exp{(\frac{u_n- \vartheta_{RH}}{\Delta_T})}- \frac{\Delta_t}{\tau_m}Rw_n +  \frac{\Delta_t}{\tau_m} RI(t)$$

$$w_{n+1} = w_n + \frac{a\Delta_t}{\tau_w} (w_n-u_{res}) - \frac{\Delta_t}{\tau_w}w+ \Delta_t b \sum_{t^f} \delta(t-t^f)$$

To do so you can either use the Finder and navigate to your prefered working directory (then right click and figure out via properties what the path is) or you can navigate in the Terminal with the cd command to your preferred working dir. In case you figured it out through the finder you should have something like "/Users/clemens/Documents". Then type into the Terminal:

~~~
cd /Users/clemens/Documents
~~~

## Rescources

- [https://anaconda.org/anaconda/sqlite](https://anaconda.org/anaconda/sqlite)
- [https://www.sqlite.org/index.html](https://www.sqlite.org/index.html)
- [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)


