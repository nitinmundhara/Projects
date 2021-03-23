"""
Definition of views.
"""
from django.shortcuts import render, redirect
from django.http import HttpRequest
from django.template import RequestContext
from datetime import datetime
from django.contrib import messages
from .models import Booking
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import PBKDF2PasswordHasher

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    if request.method=='POST':
        if not request.user.is_authenticated:
            return redirect('login')
        else:
            myBook=Booking()
            text=request.user.username
            u=User.objects.get(username=text)
            myBook.username=u
            myBook.fromPlace=request.POST['from']
            myBook.toPlace=request.POST['to']
            myBook.departureDate=request.POST['deparure_date']
            myBook.returnDate=request.POST['return_date']
            myBook.tripType=request.POST['trip']
            myBook.save()
            messages.info(request, f'Your ticket has been booked.')
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Contact Us at:',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'About \"Book Bus For Me\"',
            'year':datetime.now().year,
        }
    )

def signup(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    if request.method=='POST':
        text=request.POST['firstname']
        data=User()
        data.first_name=request.POST['firstname']
        data.username=request.POST['username']
        data.email=request.POST['email']
        data.date_joined=datetime.now()
        data.last_name=request.POST['lastname']
        hasher = PBKDF2PasswordHasher()
        data.password = hasher.encode(password=request.POST['password'],salt='salt',iterations=36000)
        if data.check_password:
            data.save()
        else:
            messages.error(request,f'Could not create the account for {text}!!!')
            return render(request,'app/signup.html',{'year':datetime.now().year,})
        messages.success(request, f'Account created for {text}!')
        return redirect('home')
    return render(
        request,
        'app/signup.html',
        {
            'year':datetime.now().year,
        }
    )


