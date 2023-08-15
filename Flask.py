#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 08:13:42 2023

@author: ronishkhatiwada
"""

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

app= Flask(__name__)
Bootstrap(app)

@app.route("/")
def index():
    products = product_list()
    return render_template('index.html', products=products)

def product_list():
    product1={
        'product_name':'Wireless Headphone',
        'price':'$20',
        'stock':10,
        'buy_now':'http://www.google.com'
    }
    product2={
        'product_name':'Laptop',
        'price':'$800',
        'stock':10,
        'buy_now':'http://www.google.com'
    }
    product3={
        'product_name':'Mobile phone',
        'price':'$200',
        'stock':10,
        'buy_now':'http://www.google.com'
    }
    return [product1,product2,product3]

if __name__=='__main__':
    app.run(port=5000, debug=True)